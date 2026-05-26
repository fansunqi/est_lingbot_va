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

    Returns:
        Per-batch logprob in fp32. The compute happens in fp32 regardless of
        the input dtype because the GRPO ratio amplifies tiny mean deltas by
        1/var (= 2500 at std=0.02); bf16 reduction noise alone is enough to
        produce multi-nat log_ratio outliers downstream. fp32 inside this op
        does not eliminate the rollout-vs-recompute forward delta, but it does
        eliminate the bf16-reduction-order share of it.
    """

    compute_dtype = torch.float32
    value_f = value.to(compute_dtype)
    mean_f = mean.to(compute_dtype)
    if not torch.is_tensor(std):
        std_f = torch.tensor(float(std), device=mean.device, dtype=compute_dtype)
    else:
        std_f = std.to(device=mean.device, dtype=compute_dtype)
    std_f = std_f.clamp_min(1e-8)
    var = std_f * std_f
    log_scale = torch.log(std_f)
    log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=mean.device, dtype=compute_dtype))
    logp = -0.5 * ((value_f - mean_f) ** 2 / var + 2.0 * log_scale + log_2pi)

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
    ratio_min: torch.Tensor
    ratio_max: torch.Tensor
    ratio_std: torch.Tensor
    log_ratio: torch.Tensor  # per-element (new - old) logprob diff, detached fp32
    clipfrac: torch.Tensor
    approx_kl: torch.Tensor
    entropy: torch.Tensor


APPROX_KL_LOGRATIO_CLAMP = 1.0
"""Clamp range for log_ratio when computing approx_kl.

The loss path uses PPO's ``min(unclipped, clipped)`` so a single ratio≪1
outlier contributes at most ``(1-clip_range)*advantage`` to the gradient.
approx_kl has no such protection: ``((ratio-1) - log_ratio).mean()`` is
dominated by whichever element has the largest ``|log_ratio|``, so one
chunk drifting by -7 nats due to forward-pass numerical noise (1/var=2500
amplification at std=0.02) can push the mean past ``target_kl=0.1`` and
trigger early-stop on an otherwise-healthy epoch.

Clamping to ±1.0 caps any single element's contribution to approx_kl at
``(e-1) - 1 ≈ 0.72``, which is well above ``target_kl`` so it still flags
genuinely diverged updates, but cannot be reached by a single outlier
when the rest of the batch is near zero. Raw log_ratio stays unclamped
in ``GRPOStats.log_ratio`` for diagnostics."""


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

    log_ratio_kl = log_ratio.detach().float().clamp(
        -APPROX_KL_LOGRATIO_CLAMP, APPROX_KL_LOGRATIO_CLAMP
    )
    ratio_kl = log_ratio_kl.exp()
    approx_kl = ((ratio_kl - 1.0) - log_ratio_kl).mean()
    clipfrac = ((ratio - 1.0).abs() > clip_range).to(new_logprob.dtype).mean()
    ratio_f32 = ratio.float().detach()
    log_ratio_f32 = log_ratio.float().detach()
    return GRPOStats(
        loss=policy_loss,
        ratio_mean=ratio_f32.mean(),
        ratio_min=ratio_f32.min() if ratio_f32.numel() else ratio_f32.new_zeros(()),
        ratio_max=ratio_f32.max() if ratio_f32.numel() else ratio_f32.new_zeros(()),
        ratio_std=ratio_f32.std(unbiased=False) if ratio_f32.numel() > 1 else ratio_f32.new_zeros(()),
        log_ratio=log_ratio_f32.reshape(-1),
        clipfrac=clipfrac.detach(),
        approx_kl=approx_kl.detach(),
        entropy=entropy_term.detach(),
    )


def flatten_episode_logprobs(logprobs: Iterable[torch.Tensor]) -> torch.Tensor:
    vals = [lp.reshape(1) for lp in logprobs]
    if not vals:
        return torch.zeros(1)
    return torch.stack(vals).sum(dim=0)
