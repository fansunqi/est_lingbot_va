"""Small, model-agnostic pieces of action-only GRPO.

The server owns the LingBot-VA-specific cache replay and transformer calls.
This module keeps the policy-ratio math and Gaussian transition accounting
separate so it can be unit-tested without loading the full model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch


LOGPROB_REDUCTIONS = {"sum", "mean"}


def _as_tensor_like(value, reference: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=dtype or reference.dtype)
    return torch.tensor(value, device=reference.device, dtype=dtype or reference.dtype)


def _broadcast_to_x(value, x: torch.Tensor) -> torch.Tensor:
    value = _as_tensor_like(value, x)
    if value.ndim == 0:
        return value.reshape((1,) * x.ndim)
    if value.ndim == 1 and x.ndim > 1:
        return value.reshape(value.shape[0], *((1,) * (x.ndim - 1)))
    while value.ndim < x.ndim:
        value = value.unsqueeze(-1)
    return value


def _check_logprob_reduce(logprob_reduce: str) -> str:
    logprob_reduce = str(logprob_reduce).lower()
    if logprob_reduce not in LOGPROB_REDUCTIONS:
        raise ValueError(
            f"logprob_reduce must be one of {sorted(LOGPROB_REDUCTIONS)}, got {logprob_reduce!r}"
        )
    return logprob_reduce


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


def flow_grpo_sde_transition(
    x_t: torch.Tensor,
    t: torch.Tensor | float,
    dt: torch.Tensor | float,
    v_theta: torch.Tensor,
    *,
    noise_level: float = 0.7,
    t_eps: float = 1e-5,
    std_eps: float = 1e-8,
    sigma_min: float | None = None,
    sigma_max: float | None = None,
) -> dict[str, torch.Tensor]:
    """Build the Flow-GRPO Euler-Maruyama Gaussian transition.

    ``t`` is the rectified-flow time in ``[0, 1]`` and ``dt`` is the signed
    step to the next time. The drift uses the signed ``dt``; only the diffusion
    scale uses ``sqrt(abs(dt))``.

    ``sigma_min``/``sigma_max`` optionally clamp ``sigma_t`` (the diffusion
    coefficient ``a*sqrt(t/(1-t))``) into a band. The clamp is applied to
    ``sigma_t`` itself — BEFORE it feeds both the score-correction ``drift`` and
    the transition ``std`` — so the transition stays a valid marginal-preserving
    SDE for the clamped diffusion coefficient (drift and diffusion remain
    consistent). The point is conditioning: the raw ``a*sqrt(t/(1-t))`` schedule
    spans a wide band across denoising steps, so the low-sigma (late) steps
    dominate the GRPO ratio by ~1/sigma^2; clamping into a narrow band makes
    every step contribute comparably. Clamping does introduce a small marginal
    bias near the t-endpoints, but ``t`` is already clamped via ``t_eps`` and the
    deployed policy uses deterministic ODE inference, so the conditioning win
    dominates. Leave both ``None`` to recover the unclamped schedule.
    """

    t_safe = _broadcast_to_x(t, x_t).clamp(float(t_eps), 1.0 - float(t_eps))
    dt_b = _broadcast_to_x(dt, x_t)
    sigma_t = float(noise_level) * torch.sqrt(t_safe / (1.0 - t_safe))
    if sigma_min is not None or sigma_max is not None:
        sigma_t = sigma_t.clamp(
            min=float(sigma_min) if sigma_min is not None else None,
            max=float(sigma_max) if sigma_max is not None else None,
        )
    drift = v_theta + (sigma_t ** 2) / (2.0 * t_safe) * (x_t + (1.0 - t_safe) * v_theta)
    std = (sigma_t * torch.sqrt(dt_b.abs())).clamp_min(float(std_eps))
    mean = x_t + drift * dt_b
    return {
        "mean": mean,
        "std": std,
        "v_theta": v_theta,
        "drift": drift,
        "t": _as_tensor_like(t, x_t),
        "dt": _as_tensor_like(dt, x_t),
        "t_safe": t_safe,
        "sigma_t": sigma_t,
    }


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


def flow_grpo_gaussian_logprob(
    value: torch.Tensor,
    mean: torch.Tensor,
    std: float | torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    logprob_reduce: str = "sum",
) -> torch.Tensor:
    """Normal logprob reduced over all non-batch dimensions."""

    logprob_reduce = _check_logprob_reduce(logprob_reduce)
    return gaussian_logprob(
        value,
        mean,
        std,
        mask=mask,
        normalize_action_dim=(logprob_reduce == "mean"),
    )


def _flow_grpo_same_variance_kl(
    mean: torch.Tensor,
    mean_ref: torch.Tensor,
    std: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    logprob_reduce: str = "sum",
) -> torch.Tensor:
    logprob_reduce = _check_logprob_reduce(logprob_reduce)
    compute_dtype = torch.float32
    std_f = std.to(device=mean.device, dtype=compute_dtype).clamp_min(1e-8)
    kl = ((mean.to(compute_dtype) - mean_ref.to(compute_dtype)) ** 2) / (2.0 * std_f ** 2)
    if mask is not None:
        mask = _expand_mask(mask, kl)
        kl = kl.masked_fill(~mask, 0.0)
        active = mask.reshape(mask.shape[0], -1).sum(dim=1).clamp_min(1)
    else:
        active = torch.full(
            (kl.shape[0],),
            int(torch.tensor(kl.shape[1:]).prod().item()),
            device=kl.device,
            dtype=torch.long,
        )
    reduced = kl.reshape(kl.shape[0], -1).sum(dim=1)
    if logprob_reduce == "mean":
        reduced = reduced / active.to(reduced.dtype)
    return reduced


def flow_grpo_sde_step(
    x_t: torch.Tensor,
    t: torch.Tensor | float,
    dt: torch.Tensor | float,
    model: Callable[[torch.Tensor, torch.Tensor | float, object], torch.Tensor],
    cond,
    noise_level: float = 0.7,
    eps: torch.Tensor | None = None,
    t_eps: float = 1e-5,
    return_logprob: bool = False,
    ref_model: Callable[[torch.Tensor, torch.Tensor | float, object], torch.Tensor] | None = None,
    *,
    mask: torch.Tensor | None = None,
    logprob_reduce: str = "sum",
) -> dict[str, torch.Tensor]:
    """Sample one Flow-GRPO SDE transition and optionally score it.

    The model is called as ``model(x_t, t, cond)`` and must return the
    rectified-flow velocity ``v_theta``. ``dt`` may be negative.
    """

    logprob_reduce = _check_logprob_reduce(logprob_reduce)
    v_theta = model(x_t, t, cond)
    transition = flow_grpo_sde_transition(
        x_t,
        t,
        dt,
        v_theta,
        noise_level=noise_level,
        t_eps=t_eps,
    )
    if eps is None:
        eps = torch.randn_like(x_t)
    x_next = transition["mean"] + transition["std"] * eps
    result = {
        "x_next": x_next,
        "mean": transition["mean"],
        "std": transition["std"],
        "eps": eps,
        "v_theta": transition["v_theta"],
        "drift": transition["drift"],
        "t": transition["t"],
        "dt": transition["dt"],
    }
    if return_logprob:
        result["logprob"] = flow_grpo_gaussian_logprob(
            x_next,
            transition["mean"],
            transition["std"],
            mask=mask,
            logprob_reduce=logprob_reduce,
        )
    if ref_model is not None:
        v_ref = ref_model(x_t, t, cond)
        ref_transition = flow_grpo_sde_transition(
            x_t,
            t,
            dt,
            v_ref,
            noise_level=noise_level,
            t_eps=t_eps,
        )
        result["kl_ref"] = _flow_grpo_same_variance_kl(
            transition["mean"],
            ref_transition["mean"],
            transition["std"],
            mask=mask,
            logprob_reduce=logprob_reduce,
        )
        result["v_ref"] = v_ref
        result["mean_ref"] = ref_transition["mean"]
    return result


def compute_flow_grpo_logprob(
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    t: torch.Tensor | float,
    dt: torch.Tensor | float,
    model: Callable[[torch.Tensor, torch.Tensor | float, object], torch.Tensor],
    cond,
    noise_level: float = 0.7,
    t_eps: float = 1e-5,
    *,
    mask: torch.Tensor | None = None,
    logprob_reduce: str = "sum",
) -> torch.Tensor:
    """Recompute ``log p_theta(x_next | x_t, cond)`` without resampling."""

    v_theta = model(x_t, t, cond)
    transition = flow_grpo_sde_transition(
        x_t,
        t,
        dt,
        v_theta,
        noise_level=noise_level,
        t_eps=t_eps,
    )
    return flow_grpo_gaussian_logprob(
        x_next,
        transition["mean"],
        transition["std"],
        mask=mask,
        logprob_reduce=logprob_reduce,
    )


def compute_group_advantages(rewards: Sequence[float], eps: float = 1e-8) -> torch.Tensor:
    """Normalize rewards within one same-task/same-seed group."""

    reward_t = torch.as_tensor(rewards, dtype=torch.float32)
    if reward_t.numel() == 0:
        return reward_t
    std = reward_t.std(unbiased=False)
    return (reward_t - reward_t.mean()) / (std + eps)


@dataclass
class GRPOStats:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_ref: torch.Tensor
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
    clip_range_high: float | None = None,
    clip_range_low: float | None = None,
    log_ratio_clip: float | None = 20.0,
    kl_ref: torch.Tensor | None = None,
    beta_kl: float = 0.0,
    entropy: torch.Tensor | None = None,
    entropy_coef: float = 0.0,
) -> GRPOStats:
    """Clipped policy loss used by ReinFlow-style GRPO_Gaussian.

    ``clip_range_high``/``clip_range_low`` allow an asymmetric trust region
    (PPO "dual-clip" style, as in LaST-R1's clip_ratio_high/low): the ratio is
    clamped to ``[1 - clip_range_low, 1 + clip_range_high]``. A wider high side
    gives positive-advantage samples more room to raise their probability before
    the surrogate is clipped — the symmetric default tends to throttle the
    "raise good actions" direction more than the "suppress bad actions" one.
    Both default to ``clip_range`` when ``None`` (symmetric).
    """

    advantages = advantages.to(device=new_logprob.device, dtype=new_logprob.dtype)
    old_logprob = old_logprob.to(device=new_logprob.device, dtype=new_logprob.dtype)
    chigh = float(clip_range_high) if clip_range_high is not None else float(clip_range)
    clow = float(clip_range_low) if clip_range_low is not None else float(clip_range)
    log_ratio = new_logprob - old_logprob
    if log_ratio_clip is None:
        log_ratio_for_ratio = log_ratio
    else:
        log_ratio_for_ratio = log_ratio.clamp(-float(log_ratio_clip), float(log_ratio_clip))
    ratio = torch.exp(log_ratio_for_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clow, 1.0 + chigh) * advantages
    surrogate_loss = -torch.minimum(unclipped, clipped).mean()
    total_loss = surrogate_loss

    kl_ref_term = torch.zeros((), device=new_logprob.device, dtype=new_logprob.dtype)
    if kl_ref is not None and beta_kl:
        kl_ref_term = kl_ref.to(device=new_logprob.device, dtype=new_logprob.dtype).mean()
        total_loss = total_loss + float(beta_kl) * kl_ref_term

    entropy_term = torch.zeros((), device=new_logprob.device, dtype=new_logprob.dtype)
    if entropy is not None and entropy_coef:
        entropy_term = entropy.to(new_logprob).mean()
        total_loss = total_loss - float(entropy_coef) * entropy_term

    log_ratio_kl = log_ratio.detach().float().clamp(
        -APPROX_KL_LOGRATIO_CLAMP, APPROX_KL_LOGRATIO_CLAMP
    )
    ratio_kl = log_ratio_kl.exp()
    approx_kl = ((ratio_kl - 1.0) - log_ratio_kl).mean()
    clipfrac = ((ratio > 1.0 + chigh) | (ratio < 1.0 - clow)).to(new_logprob.dtype).mean()
    ratio_f32 = ratio.float().detach()
    log_ratio_f32 = log_ratio.float().detach()
    return GRPOStats(
        loss=total_loss,
        policy_loss=surrogate_loss.detach(),
        kl_ref=kl_ref_term.detach(),
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
