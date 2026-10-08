"""One synthetic action/image batch through the configured CNN image policy and adaptive update."""
import json
import pathlib
import sys
import time
import torch
from torch import nn
import hydra

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.adaptive_timestep_cnn.train import configuration
from experiments.adaptive_timestep_cnn.engine import AdaptiveStepEngine
from experiments.adaptive_timestep_cnn.sampler import ActionBetaActor


class IdentityNormalizer(nn.Module):
    def __getitem__(self, key):
        return self

    def normalize(self, data):
        return data

    def unnormalize(self, data):
        return data


def main():
    torch.manual_seed(42)
    cfg = configuration("can_image")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    policy = hydra.utils.instantiate(cfg.policy).to(device)
    policy.normalizer = IdentityNormalizer()
    obs = {}
    for key, spec in cfg.shape_meta.obs.items():
        shape = tuple(spec.shape)
        if spec.get("type", "low_dim") == "rgb":
            obs[key] = torch.rand((1, cfg.n_obs_steps) + shape, device=device)
        else:
            obs[key] = torch.randn((1, cfg.n_obs_steps) + shape, device=device)
    batch = {"action": torch.randn(1, cfg.horizon, cfg.shape_meta.action.shape[0],
                                   device=device), "obs": obs}
    actor = ActionBetaActor(cfg.shape_meta.action.shape[0], cfg.horizon).to(device)
    engine = AdaptiveStepEngine(
        policy, torch.optim.AdamW(policy.parameters(), lr=1e-4),
        actor, torch.optim.Adam(actor.parameters(), lr=1e-3),
        total_timesteps=100, update_every=40, queue_size=5, n_select=3,
        probe_chunk=4)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    result = engine.process(batch, last_microbatch=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    result["seconds"] = time.perf_counter() - started
    result["device"] = str(device)
    result["action_shape"] = list(batch["action"].shape)
    result["encoder"] = type(policy.obs_encoder).__name__
    result["denoiser"] = type(policy.model).__name__
    assert result["optimizer_stepped"] and result["sampler_updated"]
    assert torch.isfinite(torch.tensor([result["loss"], result["kl_delta_mean"]])).all()
    path = pathlib.Path(__file__).resolve().parents[1] / "runs" / "synthetic_actual_result.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
