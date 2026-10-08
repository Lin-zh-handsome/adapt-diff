import copy
import math
import pathlib
import sys

import numpy as np
import pytest
import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from experiments.adaptive_timestep_cnn.engine import AdaptiveStepEngine
from experiments.adaptive_timestep_cnn.losses import (
    clean_actions, encode_condition, epsilon_mse, feedback_kl, continuous_t0_nll)
from experiments.adaptive_timestep_cnn.sampler import (
    ActionBetaActor, DeltaQueue, reinforce_loss, sample_timesteps)


class IdentityNormalizer(nn.Module):
    def __getitem__(self, key):
        return self

    def normalize(self, data):
        return data

    def unnormalize(self, data):
        return data


class SyntheticImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 4))

    def output_shape(self):
        return (4,)

    def forward(self, obs):
        return self.layer(obs["rgb"])


def tiny_policy(total_timesteps=8):
    scheduler = DDPMScheduler(
        num_train_timesteps=total_timesteps, beta_start=0.0001,
        beta_end=0.02, beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small", prediction_type="epsilon")
    policy = DiffusionUnetImagePolicy(
        shape_meta={"action": {"shape": [2]}}, noise_scheduler=scheduler,
        obs_encoder=SyntheticImageEncoder(), horizon=16,
        n_action_steps=8, n_obs_steps=2, num_inference_steps=total_timesteps,
        obs_as_global_cond=True, diffusion_step_embed_dim=16,
        down_dims=(16, 32), kernel_size=3, n_groups=8)
    policy.normalizer = IdentityNormalizer()
    return policy


def synthetic_batch(batch_size=2):
    return {"action": torch.randn(batch_size, 16, 2),
            "obs": {"rgb": torch.rand(batch_size, 2, 3, 8, 8)}}


def test_beta_draw_and_actor_gradient():
    torch.manual_seed(1)
    actor = ActionBetaActor(2, 16, hidden_dim=32)
    draw = sample_timesteps(actor, torch.randn(32, 16, 2), 100)
    assert draw.timesteps.shape == (32,)
    assert draw.timesteps.dtype == torch.long
    assert int(draw.timesteps.min()) >= 0 and int(draw.timesteps.max()) < 100
    assert torch.isfinite(draw.log_prob).all() and torch.isfinite(draw.entropy).all()
    assert (draw.alpha > 0).all() and (draw.beta > 0).all()
    loss = reinforce_loss(draw.log_prob, draw.entropy, torch.ones(32, 3))
    loss.backward()
    assert actor.mlp[-1].weight.grad is not None
    assert actor.mlp[-1].weight.grad.abs().sum() > 0


def test_feature_selector_returns_one_dimensional_legal_indices():
    rng = np.random.default_rng(4)
    queue = DeltaQueue(total_timesteps=10, capacity=5, n_select=3)
    for _ in range(5):
        queue.add(torch.from_numpy(rng.normal(size=10)))
    chosen = queue.select()
    assert isinstance(chosen, list) and len(chosen) == 3
    assert all(type(i) is int and 0 <= i < 10 for i in chosen)
    assert len(set(chosen)) == 3
    assert queue.state_dict()["size"] == 5


def test_continuous_t0_nll_delta_matches_fixed_variance_kl_delta():
    torch.manual_seed(5)
    policy = tiny_policy()
    batch = synthetic_batch()
    actions = clean_actions(policy, batch)
    condition = encode_condition(policy, batch)
    noise = torch.randn_like(actions)
    t0 = torch.zeros(len(actions), dtype=torch.long)
    with torch.no_grad():
        before_kl = feedback_kl(policy, actions, condition, t0, noise)
        before_nll = continuous_t0_nll(policy, actions, condition, noise)
        for parameter in policy.model.parameters():
            parameter.add_(0.0001)
            break
        after_kl = feedback_kl(policy, actions, condition, t0, noise)
        after_nll = continuous_t0_nll(policy, actions, condition, noise)
    assert torch.allclose(before_kl - after_kl, before_nll - after_nll,
                          atol=1e-3, rtol=1e-3)


def test_vanilla_engine_uses_original_compute_loss():
    torch.manual_seed(9)
    original = tiny_policy()
    engine_policy = copy.deepcopy(original)
    batch = synthetic_batch()
    torch.manual_seed(123)
    expected = original.compute_loss(batch)
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(engine_policy.parameters(), lr=1e-4)
    engine = AdaptiveStepEngine(engine_policy, optimizer, total_timesteps=8)
    result = engine.process(batch, last_microbatch=True)
    assert result["optimizer_stepped"] and not result["sampler_updated"]
    assert math.isclose(result["loss"], expected.item(), rel_tol=1e-6, abs_tol=1e-6)
    assert type(engine_policy) is DiffusionUnetImagePolicy


def test_accumulation_update_feedback_and_synthetic_inference():
    torch.manual_seed(14)
    policy = tiny_policy()
    actor = ActionBetaActor(2, 16, hidden_dim=32)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    engine = AdaptiveStepEngine(policy, optimizer, actor, actor_optimizer,
                                total_timesteps=8, accumulation=2,
                                update_every=40, queue_size=5, n_select=3,
                                probe_chunk=4)
    actor_before = actor.mlp[-1].weight.detach().clone()
    model_before = next(policy.model.parameters()).detach().clone()
    first = engine.process(synthetic_batch())
    assert not first["optimizer_stepped"] and not first["sampler_updated"]
    assert engine.optimizer_steps == 0
    assert torch.equal(model_before, next(policy.model.parameters()))
    second = engine.process(synthetic_batch(), last_microbatch=True)
    assert second["optimizer_stepped"] and second["sampler_updated"]
    assert engine.optimizer_steps == 1 and engine.queue.size == 1
    assert not torch.equal(model_before, next(policy.model.parameters()))
    assert not torch.equal(actor_before, actor.mlp[-1].weight)
    assert math.isfinite(second["kl_delta_mean"])
    with torch.no_grad():
        prediction = policy.predict_action(synthetic_batch()["obs"])
    assert prediction["action"].shape == (2, 8, 2)
    assert torch.isfinite(prediction["action"]).all()


def test_timestep_boundaries_and_reward_is_sum_not_mean():
    torch.manual_seed(20)
    policy = tiny_policy()
    batch = synthetic_batch()
    actions = clean_actions(policy, batch)
    condition = encode_condition(policy, batch)
    noise = torch.randn_like(actions)
    t = torch.tensor([0, 7])
    values = feedback_kl(policy, actions, condition, t, noise)
    assert values.shape == (2,) and torch.isfinite(values).all()
    logp = torch.tensor([1.0], requires_grad=True)
    entropy = torch.zeros(1)
    delta = torch.tensor([[1.0, 2.0, 3.0]])
    loss = reinforce_loss(logp, entropy, delta, ent_coef=0)
    assert loss.item() == -6.0



def test_fixed_uniform_validation_is_repeatable():
    from experiments.adaptive_timestep_cnn.train import fixed_uniform_validation
    torch.manual_seed(31)
    policy = tiny_policy()
    batches = [synthetic_batch(3), synthetic_batch(3)]
    first = fixed_uniform_validation(policy, batches, torch.device("cpu"))
    torch.manual_seed(999)
    second = fixed_uniform_validation(policy, batches, torch.device("cpu"))
    assert first == second



def test_short_final_accumulation_window_steps_once():
    torch.manual_seed(37)
    policy = tiny_policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=1e-4)
    engine = AdaptiveStepEngine(policy, optimizer, total_timesteps=8, accumulation=3)
    assert not engine.process(synthetic_batch())["optimizer_stepped"]
    final = engine.process(synthetic_batch(), last_microbatch=True)
    assert final["optimizer_stepped"] and engine.optimizer_steps == 1
    assert engine.micro_in_window == 0
