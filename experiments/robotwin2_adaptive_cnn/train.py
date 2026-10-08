"""Full RoboTwin 2.0 CNN DP training with the reviewed adaptive timestep method."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

POLICY_DIR = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(POLICY_DIR))

import dill
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.workspace.robotworkspace import RobotWorkspace, create_dataloader
from .engine import AdaptiveStepEngine
from .losses import clean_actions, encode_condition, epsilon_mse
from .sampler import ActionBetaActor

OmegaConf.register_new_resolver("eval", eval, replace=True)
TASKS = ("beat_block_hammer", "handover_block", "stack_bowls_three", "pick_dual_bottles")
DATA_DIR = POLICY_DIR / "data"
CHECKPOINT_INTERVAL = 25


def build_config(task, device):
    config_dir = POLICY_DIR / "diffusion_policy" / "config"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = hydra.compose(config_name="robot_dp")
    OmegaConf.set_struct(cfg, False)
    cfg.bench_name = "RoboTwin"
    cfg.task.name = task
    cfg.task.shape_meta.action.shape = [14]
    cfg.task.shape_meta.obs.agent_pos = OmegaConf.create({"shape": [14]})
    cfg.task.dataset.zarr_path = str(DATA_DIR / f"RoboTwin-{task}-aloha_agilex-joint.zarr")
    cfg.policy.use_self_condition = False
    cfg.training.device = device
    cfg.training.seed = 42
    cfg.training.resume = False
    cfg.logging.mode = "disabled"
    cfg.setting = "aloha_agilex"
    if not Path(cfg.task.dataset.zarr_path).is_dir():
        raise FileNotFoundError(cfg.task.dataset.zarr_path)
    if not (cfg.horizon == 8 and cfg.n_obs_steps == 3 and cfg.n_action_steps == 6
            and cfg.dataloader.batch_size == 128 and cfg.training.num_epochs == 600
            and cfg.policy.noise_scheduler.num_train_timesteps == 100
            and cfg.policy.num_inference_steps == 100
            and cfg.policy.noise_scheduler.prediction_type == "epsilon"
            and list(cfg.policy.down_dims) == [256, 512, 1024]):
        raise ValueError("RoboTwin official CNN DP configuration changed")
    return cfg


@torch.no_grad()
def fixed_uniform_validation(policy, dataset, device):
    was_training = policy.training
    policy.eval()
    losses = []
    seen = 0
    batch_size = int(dataset.batch_size)
    if len(dataset) == 0:
        raise ValueError("the official validation split contains no action blocks")
    for batch_index, start in enumerate(range(0, len(dataset), batch_size)):
        valid = np.arange(start, min(start + batch_size, len(dataset)), dtype=np.int64)
        indices = np.resize(valid, batch_size)
        batch = dataset.postprocess(dataset[indices], device)
        actions = clean_actions(policy, batch)
        condition = encode_condition(policy, batch)
        timesteps = torch.arange(seen, seen + batch_size, device=device) % 100
        generator = torch.Generator(device=device).manual_seed(2025 + batch_index)
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype, generator=generator)
        losses.append(epsilon_mse(policy, actions, condition, timesteps, noise)[:len(valid)].cpu())
        seen += len(valid)
    policy.train(was_training)
    return float(torch.cat(losses).mean())


def save_state(output, workspace, scheduler, ema, actor, actor_optimizer, engine,
               train_seconds, sampler_rng):
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_tmp = checkpoint_dir / "latest.tmp.ckpt"
    workspace.save_checkpoint(path=checkpoint_tmp, use_thread=False)
    os.replace(checkpoint_tmp, checkpoint_dir / "latest.ckpt")
    state = {
        "epoch": workspace.epoch,
        "global_step": workspace.global_step,
        "scheduler": scheduler.state_dict(),
        "ema_step": ema.optimization_step if ema else None,
        "actor": actor.state_dict() if actor else None,
        "actor_optimizer": actor_optimizer.state_dict() if actor_optimizer else None,
        "engine": engine.state_dict(),
        "train_seconds": train_seconds,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "sampler_rng": copy.deepcopy(sampler_rng.bit_generator.state),
    }
    state_tmp = checkpoint_dir / "trainer_state.tmp.pt"
    torch.save(state, state_tmp)
    os.replace(state_tmp, checkpoint_dir / "trainer_state.pt")
    if workspace.epoch == 600:
        final_tmp = checkpoint_dir / "600.tmp.ckpt"
        workspace.save_checkpoint(path=final_tmp, use_thread=False)
        os.replace(final_tmp, checkpoint_dir / "600.ckpt")


def restore_state(output, workspace, scheduler, ema, actor, actor_optimizer, engine,
                  device, sampler_rng):
    checkpoint_dir = output / "checkpoints"
    checkpoint = checkpoint_dir / "latest.ckpt"
    state_path = checkpoint_dir / "trainer_state.pt"
    if not checkpoint.is_file() or not state_path.is_file():
        raise FileNotFoundError("resume requires latest.ckpt and trainer_state.pt")
    payload = torch.load(checkpoint, pickle_module=dill, map_location="cpu")
    workspace.load_payload(payload)
    workspace.model.to(device)
    workspace.ema_model.to(device)
    optimizer_to(workspace.optimizer, device)
    state = torch.load(state_path, map_location=device, weights_only=False)
    if (workspace.epoch != state["epoch"] or workspace.global_step != state["global_step"]):
        raise RuntimeError("checkpoint and trainer state refer to different epochs")
    scheduler.load_state_dict(state["scheduler"])
    if ema:
        ema.optimization_step = state["ema_step"]
    if actor:
        actor.load_state_dict(state["actor"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
    engine.load_state_dict(state["engine"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    sampler_rng.bit_generator.state = state["sampler_rng"]
    return float(state["train_seconds"])


def run(args):
    if args.task not in TASKS:
        raise ValueError(args.task)
    output = EXPERIMENT_DIR / "runs" / args.run_name / args.task / args.mode
    output.mkdir(parents=True, exist_ok=True)
    if (output / "logs.jsonl").exists() and not args.resume:
        raise FileExistsError(f"existing run: {output}")
    cfg = build_config(args.task, args.device)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    device = torch.device(args.device)
    workspace = RobotWorkspace(cfg, output_dir=str(output))
    if workspace.model.use_self_condition or workspace.ema_model.use_self_condition:
        raise ValueError("both policies must use the original CNN DP without self-conditioning")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if not isinstance(dataset, BaseImageDataset):
        raise TypeError("RobotImageDataset required")
    val_dataset = dataset.get_validation_dataset()
    train_loader = create_dataloader(dataset, **cfg.dataloader)
    normalizer = dataset.get_normalizer()
    workspace.model.set_normalizer(normalizer)
    workspace.ema_model.set_normalizer(normalizer)
    workspace.model.to(device)
    workspace.ema_model.to(device)
    optimizer_to(workspace.optimizer, device)
    actor = actor_optimizer = None
    if args.mode == "adaptive":
        actor = ActionBetaActor(14, 8).to(device)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1.0e-3)
    engine = AdaptiveStepEngine(
        workspace.model, workspace.optimizer, actor, actor_optimizer,
        total_timesteps=100, accumulation=int(cfg.training.gradient_accumulate_every),
        update_every=40, queue_size=5, n_select=3, ent_coef=0.01, probe_chunk=8)
    planned_steps = len(train_loader) * int(cfg.training.num_epochs) // engine.accumulation
    scheduler = get_scheduler(
        cfg.training.lr_scheduler, optimizer=workspace.optimizer,
        num_warmup_steps=cfg.training.lr_warmup_steps,
        num_training_steps=planned_steps, last_epoch=-1)
    ema = hydra.utils.instantiate(cfg.ema, model=workspace.ema_model)
    train_seconds = 0.0
    if args.resume:
        train_seconds = restore_state(
            output, workspace, scheduler, ema, actor, actor_optimizer,
            engine, device, train_loader.sampler.rng)
    with (output / "logs.jsonl").open("a", encoding="utf-8") as log:
        for epoch in range(workspace.epoch, int(cfg.training.num_epochs)):
            epoch_start = time.perf_counter()
            losses = []
            batch_seconds = []
            sampler_updates = 0
            workspace.model.train()
            for batch_index, raw_batch in enumerate(train_loader):
                batch = dataset.postprocess(raw_batch, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                batch_start = time.perf_counter()
                result = engine.process(batch, last_microbatch=batch_index == len(train_loader) - 1)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds = time.perf_counter() - batch_start
                train_seconds += seconds
                batch_seconds.append(seconds)
                losses.append(result["loss"])
                sampler_updates += int(result["sampler_updated"])
                if result["optimizer_stepped"]:
                    scheduler.step()
                    ema.step(workspace.model)
                workspace.global_step += 1
            val_loss = fixed_uniform_validation(workspace.model, val_dataset, device)
            workspace.epoch = epoch + 1
            row = {
                "task": args.task, "mode": args.mode, "epoch": epoch + 1,
                "global_step": workspace.global_step,
                "train_loss": float(np.mean(losses)),
                "uniform_val_loss": val_loss,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_seconds": train_seconds,
                "mean_batch_seconds": float(np.mean(batch_seconds)),
                "sampler_updates": sampler_updates,
                "optimizer_steps": engine.optimizer_steps,
            }
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if workspace.epoch % CHECKPOINT_INTERVAL == 0 or workspace.epoch == 600:
                save_state(output, workspace, scheduler, ema, actor, actor_optimizer,
                           engine, train_seconds, train_loader.sampler.rng)
    (output / "status.json").write_text(
        json.dumps({"state": "training_complete", "epoch": workspace.epoch,
                    "checkpoint": str(output / "checkpoints" / "600.ckpt")}),
        encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("vanilla", "adaptive"), required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

