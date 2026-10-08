"""Independent CNN image DP training entry; no changes to the Vanilla package."""
import argparse
import copy
from datetime import datetime
import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from .engine import AdaptiveStepEngine
from .losses import clean_actions, encode_condition, epsilon_mse
from .sampler import ActionBetaActor

OmegaConf.register_new_resolver("eval", eval, replace=True)


def configuration(task: str, dataset_path=None):
    config_dir = ROOT / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        base = compose(config_name="train_diffusion_unet_image_workspace",
                       overrides=[f"task={task}"])
    OmegaConf.set_struct(base, False)
    extra = OmegaConf.load(EXPERIMENT / "config.yaml")
    cfg = OmegaConf.merge(base, extra)
    # Keep newly obtained image data inside this experiment directory.
    image_data = pathlib.Path(dataset_path) if dataset_path else (
        EXPERIMENT / "data" / str(cfg.task.task_name) / str(cfg.task.dataset_type) / "image.hdf5")
    cfg.task.dataset.dataset_path = str(image_data)
    cfg.task.env_runner.dataset_path = str(image_data)
    if cfg.policy.noise_scheduler.num_train_timesteps != 100:
        raise ValueError("this experiment preserves the Vanilla T=100 scheduler")
    if cfg.policy.noise_scheduler.variance_type != "fixed_small":
        raise ValueError("the existing fixed_small reverse variance is required")
    if cfg.policy.noise_scheduler.beta_schedule != "squaredcos_cap_v2":
        raise ValueError("the existing cosine noise schedule is required")
    if cfg.policy.num_inference_steps != 100:
        raise ValueError("the existing 100-step inference schedule is required")
    if cfg.policy.noise_scheduler.prediction_type != "epsilon":
        raise ValueError("epsilon prediction is required")
    if not cfg.obs_as_global_cond:
        raise ValueError("the reviewed Vanilla global condition is required")
    return cfg


@torch.no_grad()
def fixed_uniform_validation(policy, loader, device, max_batches=None):
    """Shared, deterministic, uniformly cycling t/noise validation for both modes."""
    was_training = policy.training
    policy.eval()
    all_losses = []
    seen = 0
    total_t = policy.noise_scheduler.config.num_train_timesteps
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
        actions = clean_actions(policy, batch)
        condition = encode_condition(policy, batch)
        t = torch.arange(seen, seen + actions.shape[0], device=device) % total_t
        generator = torch.Generator(device=device).manual_seed(2025 + batch_index)
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype, generator=generator)
        all_losses.append(epsilon_mse(policy, actions, condition, t, noise).detach().cpu())
        seen += actions.shape[0]
    policy.train(was_training)
    return float(torch.cat(all_losses).mean()) if all_losses else None


def save_checkpoint(path, cfg, model, ema_model, ema, optimizer, scheduler,
                    actor, actor_optimizer, engine, epoch, global_step):
    payload = dict(cfg=cfg, model=model.state_dict(),
                   ema_model=ema_model.state_dict() if ema_model else None,
                   ema_step=ema.optimization_step if ema else None,
                   optimizer=optimizer.state_dict(),
                   lr_scheduler=scheduler.state_dict(),
                   actor=actor.state_dict() if actor else None,
                   actor_optimizer=actor_optimizer.state_dict() if actor_optimizer else None,
                   engine=engine.state_dict(), epoch=epoch, global_step=global_step)
    torch.save(payload, path)


def run(args):
    cfg = configuration(args.task, args.dataset_path)
    image_path = pathlib.Path(cfg.task.dataset.dataset_path)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image benchmark dataset missing: {image_path}. "
            "Obtain Can PH image.hdf5 or generate it from official demo states; "
            "low_dim.hdf5 is not an image benchmark.")
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device or cfg.training.device)
    mode = args.mode
    run_name = args.run_name or f"{args.task}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if "/" in run_name or "\\" in run_name or run_name in (".", ".."):
        raise ValueError("run_name must be a directory name")
    output = EXPERIMENT / "runs" / mode / run_name
    output.mkdir(parents=True, exist_ok=True)

    model = hydra.utils.instantiate(cfg.policy).to(device)
    ema_model = copy.deepcopy(model) if cfg.training.use_ema else None
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    actor = actor_optimizer = None
    if mode == "adaptive":
        actor = ActionBetaActor(cfg.shape_meta.action.shape[0], cfg.horizon,
                                cfg.adaptive.hidden_dim, cfg.adaptive.hidden_depth).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.adaptive.lr)
    engine = AdaptiveStepEngine(
        model, optimizer, actor, actor_optimizer, total_timesteps=100,
        accumulation=int(cfg.training.gradient_accumulate_every),
        update_every=int(cfg.adaptive.update_every),
        queue_size=int(cfg.adaptive.queue_size),
        n_select=int(cfg.adaptive.n_select),
        ent_coef=float(cfg.adaptive.ent_coef),
        probe_chunk=int(cfg.adaptive.probe_chunk))

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if not isinstance(dataset, BaseImageDataset):
        raise TypeError("image dataset required")
    val_dataset = dataset.get_validation_dataset()
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    if ema_model:
        ema_model.set_normalizer(normalizer)
    data_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(dataset, generator=data_generator, **cfg.dataloader)
    val_loader = DataLoader(val_dataset, **cfg.val_dataloader)
    planned_steps = (len(train_loader) * cfg.training.num_epochs) // engine.accumulation
    scheduler = get_scheduler(cfg.training.lr_scheduler, optimizer=optimizer,
                              num_warmup_steps=cfg.training.lr_warmup_steps,
                              num_training_steps=planned_steps, last_epoch=-1)
    ema = hydra.utils.instantiate(cfg.ema, model=ema_model) if ema_model else None
    runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=str(output))
    start_epoch = global_step = 0
    checkpoint = output / "latest.ckpt"
    if args.resume and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        if ema_model:
            ema_model.load_state_dict(state["ema_model"])
            ema.optimization_step = state["ema_step"]
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["lr_scheduler"])
        if actor:
            actor.load_state_dict(state["actor"])
            actor_optimizer.load_state_dict(state["actor_optimizer"])
        engine.load_state_dict(state["engine"])
        start_epoch, global_step = state["epoch"], state["global_step"]

    with (output / "logs.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(start_epoch, int(cfg.training.num_epochs)):
            epoch_start = time.perf_counter()
            train_losses = []
            step_times = []
            sampler_updates = 0
            model.train()
            if cfg.training.freeze_encoder:
                model.obs_encoder.eval()
                model.obs_encoder.requires_grad_(False)
            for batch_index, batch in enumerate(train_loader):
                if cfg.training.max_train_steps is not None and batch_index >= cfg.training.max_train_steps:
                    break
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                last = (batch_index == len(train_loader) - 1 or
                        cfg.training.max_train_steps is not None and
                        batch_index == cfg.training.max_train_steps - 1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = engine.process(batch, last_microbatch=last)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_times.append(time.perf_counter() - started)
                train_losses.append(result["loss"])
                if result["optimizer_stepped"]:
                    scheduler.step()
                    if ema:
                        ema.step(model)
                    sampler_updates += int(result["sampler_updated"])
                global_step += 1
            row = dict(epoch=epoch, global_step=global_step, mode=mode,
                       optimizer_steps=engine.optimizer_steps,
                       train_loss=float(np.mean(train_losses)),
                       mean_batch_seconds=float(np.mean(step_times)),
                       sampler_updates=sampler_updates,
                       epoch_seconds=time.perf_counter() - epoch_start)
            if epoch % cfg.training.val_every == 0:
                row["uniform_val_loss"] = fixed_uniform_validation(
                    model, val_loader, device, cfg.training.max_val_steps)
            if epoch % cfg.training.rollout_every == 0:
                policy = ema_model if ema_model else model
                policy.eval()
                row.update({key: value for key, value in runner.run(policy).items()
                            if isinstance(value, (float, int, np.floating, np.integer))})
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if epoch % cfg.training.checkpoint_every == 0:
                save_checkpoint(checkpoint, cfg, model, ema_model, ema, optimizer,
                                scheduler, actor, actor_optimizer, engine, epoch + 1, global_step)
    save_checkpoint(checkpoint, cfg, model, ema_model, ema, optimizer,
                    scheduler, actor, actor_optimizer, engine, int(cfg.training.num_epochs), global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vanilla", "adaptive"], required=True)
    parser.add_argument("--task", default="can_image")
    parser.add_argument("--run-name")
    parser.add_argument("--device")
    parser.add_argument("--dataset-path")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
