"""Small, model-agnostic pieces of action-only GRPO.

The server owns the LingBot-VA-specific cache replay and transformer calls.
This module keeps the policy-ratio math and Gaussian transition accounting
separate so it can be unit-tested without loading the full model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


def scheduler_transition_mean(scheduler, model_output, timestep, sample, *, to_final=False):
    """Return the deterministic flow transition mean used by sampling/logprob.

    This intentionally delegates to ``FlowMatchScheduler.step`` so sampler and
    logprob recomputation cannot drift from the inference scheduler formula.
    """

    return scheduler.step(model_output, timestep, sample, to_final=to_final)


def sample_gaussian_transition(
    mean: torch.Tensor,
    std: float | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``Normal(mean, std)`` with optional deterministic generator."""

    if not torch.is_tensor(std):
        std = torch.tensor(float(std), device=mean.device, dtype=mean.dtype)
    noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
    return mean + noise * std.to(device=mean.device, dtype=mean.dtype)


def _expand_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(0)
    return mask.to(device=target.device, dtype=torch.bool)


def gaussian_logprob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: float | torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    normalize_action_dim: bool = True,
) -> torch.Tensor:
    """Sum Normal(mean, std).log_prob(value) over non-batch dimensions.

    Args:
        value: Sampled next state, shaped ``B, C, F, H, W`` for actions.
        mean: Transition mean with the same shape as ``value``.
        std: Scalar or tensor standard deviation.
        mask: Optional boolean mask broadcastable to action dimensions.
        normalize_action_dim: Divide by the active element count per batch.
    """

    if not torch.is_tensor(std):
        std = torch.tensor(float(std), device=mean.device, dtype=mean.dtype)
    std = std.to(device=mean.device, dtype=mean.dtype).clamp_min(1e-8)
    var = std * std
    log_scale = torch.log(std)
    logp = -0.5 * ((value - mean) ** 2 / var + 2.0 * log_scale + torch.log(torch.tensor(2.0 * torch.pi, device=mean.device, dtype=mean.dtype)))

    if mask is not None:
        mask = _expand_mask(mask, logp)
        logp = logp.masked_fill(~mask, 0.0)
        active = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1)
    else:
        active = torch.full(
            (logp.shape[0],),
            int(torch.tensor(logp.shape[1:]).prod().item()),
            device=logp.device,
            dtype=torch.long,
        )

    summed = logp.reshape(logp.shape[0], -1).sum(dim=1)
    if normalize_action_dim:
        summed = summed / active.to(summed.dtype)
    return summed


def compute_group_advantages(rewards: Sequence[float], eps: float = 1e-6) -> torch.Tensor:
    """Normalize rewards within one same-task/same-seed group."""

    reward_t = torch.as_tensor(rewards, dtype=torch.float32)
    if reward_t.numel() == 0:
        return reward_t
    std = reward_t.std(unbiased=False)
    return (reward_t - reward_t.mean()) / (std + eps)


@dataclass
class GRPOStats:
    loss: torch.Tensor
    ratio_mean: torch.Tensor
    clipfrac: torch.Tensor
    approx_kl: torch.Tensor
    entropy: torch.Tensor


def grpo_clipped_loss(
    new_logprob: torch.Tensor,
    old_logprob: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_range: float = 0.2,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> GRPOStats:
    """Clipped policy loss used by ReinFlow-style GRPO_Gaussian."""

    advantages = advantages.to(device=new_logprob.device, dtype=new_logprob.dtype)
    old_logprob = old_logprob.to(device=new_logprob.device, dtype=new_logprob.dtype)
    log_ratio = new_logprob - old_logprob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    entropy_term = torch.zeros((), device=new_logprob.device, dtype=new_logprob.dtype)
    if entropy is not None and entropy_coef:
        entropy_term = entropy.to(new_logprob).mean()
        policy_loss = policy_loss - float(entropy_coef) * entropy_term

    approx_kl = ((ratio - 1.0) - log_ratio).mean()
    clipfrac = ((ratio - 1.0).abs() > clip_range).to(new_logprob.dtype).mean()
    return GRPOStats(
        loss=policy_loss,
        ratio_mean=ratio.mean().detach(),
        clipfrac=clipfrac.detach(),
        approx_kl=approx_kl.detach(),
        entropy=entropy_term.detach(),
    )


def flatten_episode_logprobs(logprobs: Iterable[torch.Tensor]) -> torch.Tensor:
    vals = [lp.reshape(1) for lp in logprobs]
    if not vals:
        return torch.zeros(1)
    return torch.stack(vals).sum(dim=0)
