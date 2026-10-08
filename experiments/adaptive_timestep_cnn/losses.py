"""Exact CNN-DP epsilon loss and continuous-action fixed-variance KL feedback."""
import math
import torch
import torch.nn.functional as F
from diffusion_policy.common.pytorch_util import dict_apply


def clean_actions(policy, batch):
    return policy.normalizer["action"].normalize(batch["action"])


def encode_condition(policy, batch):
    if not policy.obs_as_global_cond:
        raise ValueError("this experiment requires the Vanilla global-observation CNN configuration")
    nobs = policy.normalizer.normalize(batch["obs"])
    b = batch["action"].shape[0]
    to = policy.n_obs_steps
    obs = dict_apply(nobs, lambda x: x[:, :to, ...].reshape(-1, *x.shape[2:]))
    return policy.obs_encoder(obs).reshape(b, -1)


def epsilon_mse(policy, actions, condition, timesteps, noise):
    noisy = policy.noise_scheduler.add_noise(actions, noise, timesteps)
    predicted = policy.model(noisy, timesteps, global_cond=condition)
    if policy.noise_scheduler.config.prediction_type != "epsilon":
        raise ValueError("the method requires the original epsilon-prediction objective")
    return F.mse_loss(predicted, noise, reduction="none").flatten(1).mean(1)


def feedback_kl(policy, actions, condition, timesteps, noise):
    """KL in bits per action element; t=0 is the continuous Gaussian decoder NLL up to a constant.

    Let ab_t=prod_{s<=t}(1-beta_s), ab_prev=1 at t=0.
    q(A_prev|A_t,A0) has mean c1*A0+c2*A_t and variance v_t.
    p_theta has mean c1*A0_hat+c2*A_t and the same fixed v_t.
    For t>0, KL(q||p)=0.5*(mu_q-mu_p)^2/v_t.
    At t=0, q collapses at A0, so choose continuous
    p_theta(A0|A_t,O)=Normal(A0_hat, v_1 I). Its NLL is
    0.5*[log(2*pi*v_1)+(A0-A0_hat)^2/v_1].
    The log term cancels in before-minus-after; this matches the
    official clipped-logvar KL surrogate at t=0. No image pixel decoder is used.
    """
    scheduler = policy.noise_scheduler
    betas = scheduler.betas.to(device=actions.device, dtype=actions.dtype)
    ab = scheduler.alphas_cumprod.to(device=actions.device, dtype=actions.dtype)
    total = len(betas)
    if total < 2:
        raise ValueError("the t=0 decoder needs at least two diffusion steps")
    t = timesteps.long()
    if t.ndim != 1 or t.shape[0] != actions.shape[0]:
        raise ValueError("one timestep is required per action block")
    at = ab[t]
    prev = torch.where(t > 0, ab[(t - 1).clamp(min=0)], torch.ones_like(at))
    beta = betas[t]
    alpha = 1 - beta
    c1 = beta * prev.sqrt() / (1 - at)
    c2 = alpha.sqrt() * (1 - prev) / (1 - at)
    posterior_var = beta * (1 - prev) / (1 - at)
    variance_at_one = betas[1] * (1 - ab[0]) / (1 - ab[1])
    variance = torch.where(t == 0, variance_at_one, posterior_var)
    shape = (-1,) + (1,) * (actions.ndim - 1)
    noisy = scheduler.add_noise(actions, noise, t)
    predicted_noise = policy.model(noisy, t, global_cond=condition)
    predicted_clean = (noisy - (1 - at).sqrt().reshape(shape) * predicted_noise) / at.sqrt().reshape(shape)
    true_mean = c1.reshape(shape) * actions + c2.reshape(shape) * noisy
    model_mean = c1.reshape(shape) * predicted_clean + c2.reshape(shape) * noisy
    squared = (true_mean - model_mean).square()
    return (0.5 * squared / variance.reshape(shape) / math.log(2)).flatten(1).mean(1)


def continuous_t0_nll(policy, actions, condition, noise):
    """Full continuous Gaussian NLL in bits, used to verify the t=0 KL difference."""
    t = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.long)
    scheduler = policy.noise_scheduler
    betas = scheduler.betas.to(device=actions.device, dtype=actions.dtype)
    ab = scheduler.alphas_cumprod.to(device=actions.device, dtype=actions.dtype)
    variance = betas[1] * (1 - ab[0]) / (1 - ab[1])
    noisy = scheduler.add_noise(actions, noise, t)
    predicted = policy.model(noisy, t, global_cond=condition)
    predicted_clean = (noisy - (1 - ab[0]).sqrt() * predicted) / ab[0].sqrt()
    nll = 0.5 * ((actions - predicted_clean).square() / variance +
                 torch.log(2 * torch.pi * variance))
    return nll.flatten(1).mean(1) / math.log(2)
