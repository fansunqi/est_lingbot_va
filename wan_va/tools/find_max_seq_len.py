"""
Find the maximum total sequence length that fits in GPU memory.

Loads the real model with FSDP + activation checkpointing (identical to
``wan_va.train``), constructs synthetic inputs at increasing sequence
lengths, and runs forward + backward + optimizer step to measure peak
memory.

Packed training carries one text embedding row per packed episode and the
FlexAttention cross-attention BlockMask grows with ``N_ep``, so the probe
simulates a packed configuration: set ``--probe-n-episodes`` at the upper
envelope of what your data distribution produces, and the reported
``max_tokens`` is valid for any bin with ``N_ep <= --probe-n-episodes``.

The search has two phases:

1. **Exponential growth** — adaptive 2x/1.5x/1.25x steps, gated by
   linear extrapolation.  Transitions to phase 2 when headroom < 5%.
2. **Binary refinement** — bisects between last-ok and upper-candidate
   F, each midpoint guarded by the same linear extrapolation.

OOM is **not** caught: under FSDP a per-rank OOM mid-collective
deadlocks all subsequent collectives.  Linear extrapolation (phase 1)
and collective reserved-memory gating (both phases) prevent nearly all
OOM; if one still occurs ``torchrun`` tears down every worker cleanly.

Usage::

    torchrun --nproc_per_node=8 -m wan_va.tools.find_max_seq_len \\
        --model-path /path/to/pretrained_wan \\
        --probe-n-episodes 128
"""

import argparse
import gc
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dataset.token_math import (
    tokens_per_frame as _shared_tpf,
    TRAIN_CHUNK_SIZE_MAX,
    TRAIN_WINDOW_SIZE_MAX,
)
from distributed.fsdp import shard_model, apply_ac
from distributed.util import _configure_model, init_distributed
from modules.utils import load_transformer
from utils import get_mesh_id_packed, sample_timestep_id, data_seq_to_patch, FlowMatchScheduler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe geometry — only F varies.  Peak memory ∝ total_seq + N_ep text cost.
# ---------------------------------------------------------------------------
PROBE_H_LATENT = 16
PROBE_W_LATENT = 16
PROBE_N_ACTION = 4
PROBE_IN_CHANNELS = 48
PROBE_ACTION_DIM = 30
PROBE_TEXT_SEQ = 512
PROBE_TEXT_DIM = 4096
PATCH_SIZE = (1, 2, 2)

# Match training worst-case mask density.
PROBE_CHUNK_SIZE = TRAIN_CHUNK_SIZE_MAX
PROBE_WINDOW_SIZE = TRAIN_WINDOW_SIZE_MAX

TOKENS_PER_FRAME = _shared_tpf(
    H_lat=PROBE_H_LATENT,
    W_lat=PROBE_W_LATENT,
    patch_h=PATCH_SIZE[1],
    patch_w=PATCH_SIZE[2],
    n_action=PROBE_N_ACTION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pad128(n: int) -> int:
    return n + (128 - n % 128) % 128


def _total_seq_for_f(f: int) -> int:
    return _pad128(f * TOKENS_PER_FRAME)


def _next_f(f: int, ratio: float) -> int:
    """Next probe F with adaptive growth; guaranteed to advance by >= 1."""
    mult = 1.25 if ratio > 0.75 else 1.5 if ratio > 0.5 else 2.0
    return max(f + 1, int(f * mult))


def _episode_split(f: int, n_episodes: int) -> list[int]:
    """Split *f* frames across *n_episodes* as evenly as possible.

    Returns a list whose entries sum to *f* and differ by at most 1.
    """
    n_episodes = max(1, min(n_episodes, f))
    base, extra = divmod(f, n_episodes)
    return [base + (1 if i < extra else 0) for i in range(n_episodes)]


# ---------------------------------------------------------------------------
# Preset conversion table
# ---------------------------------------------------------------------------
_PRESETS: dict[str, tuple[int, int, str]] = {
    # (concat_latent_H, concat_latent_W, description)
    "one_primary_one_wrist_256":            (16, 32, "hconcat 2x256^2"),
    "one_primary_one_wrist_128":            (8,  16, "hconcat 2x128^2"),
    "one_primary_two_wrist_224x320":        (14, 60, "hconcat 3x224x320"),
    "one_primary_two_wrist_tshape_256x320": (24, 20, "tshape 256x320+2x128x160"),
}


def _max_f_for_budget(budget_seq: int, tpf: int) -> int:
    """Largest F whose pad128(F * tpf) <= budget_seq."""
    if tpf <= 0:
        return 0
    f = budget_seq // tpf
    while f > 0 and _pad128(f * tpf) > budget_seq:
        f -= 1
    return f


def print_conversion_table(max_seq: int):
    logger.info("")
    logger.info("=== Preset conversion (N_action = frame_stride * vae_t_downsample) ===")
    logger.info("%-44s | stride | tok/frame | max_F(100%%) | max_F(90%%)", "Preset")
    logger.info("-" * 44 + "-+-" + "-" * 6 + "-+-" + "-" * 9 + "-+-" + "-" * 11 + "-+-" + "-" * 10)
    for name, (h, w, desc) in _PRESETS.items():
        for stride in (1, 2, 4, 5):
            n_action = stride * 4
            tpf = _shared_tpf(
                H_lat=h, W_lat=w,
                patch_h=PATCH_SIZE[1], patch_w=PATCH_SIZE[2],
                n_action=n_action,
            )
            f100 = _max_f_for_budget(max_seq, tpf)
            f90 = _max_f_for_budget(int(max_seq * 0.9), tpf)
            label = f"{name} ({desc})" if stride == 1 else ""
            logger.info("%-44s | %6d | %9d | %11d | %10d", label, stride, tpf, f100, f90)
        logger.info("")
    logger.info("Formula: max_F = max_total_seq // (2*(H_l/p_h * W_l/p_w) + 2*(stride*4))")
    logger.info("Note: vae_temporal_downsample defaults to 4. 90%% column recommended for production.")


# ---------------------------------------------------------------------------
# Synthetic input — mirrors Trainer under packing.
# ---------------------------------------------------------------------------

def build_synthetic_input(
    f: int,
    n_episodes: int,
    device: torch.device,
    scheduler_latent: FlowMatchScheduler,
    scheduler_action: FlowMatchScheduler,
) -> dict:
    B, dtype = 1, torch.bfloat16
    ep_lens = _episode_split(f, n_episodes)
    n_ep = len(ep_lens)
    ep_lens_t = torch.as_tensor(ep_lens, dtype=torch.long, device=device)

    def _per_frame_timesteps(sched, *, cond=False):
        kw = {"min_timestep_bd": 0.5, "max_timestep_bd": 1.0} if cond else {}
        ids = sample_timestep_id(
            batch_size=f, num_train_timesteps=sched.num_train_timesteps, **kw,
        )
        return sched.timesteps[ids].to(device=device)

    def _per_ep_grid(h, w, *, action):
        grid_ep_lens = ep_lens if action else [L // PATCH_SIZE[0] for L in ep_lens]
        return get_mesh_id_packed(
            grid_ep_lens, h, w, t=1 if action else 0, action=action,
        ).to(device)[None].repeat(B, 1, 1)

    # Latent
    timesteps = _per_frame_timesteps(scheduler_latent)
    latent = torch.randn(B, PROBE_IN_CHANNELS, f, PROBE_H_LATENT, PROBE_W_LATENT,
                         device=device, dtype=dtype)
    noise = torch.randn_like(latent)
    noisy = scheduler_latent.add_noise(latent, noise, timesteps, t_dim=2)
    targets = scheduler_latent.training_target(latent, noise, timesteps)
    grid = _per_ep_grid(PROBE_H_LATENT // PATCH_SIZE[1],
                        PROBE_W_LATENT // PATCH_SIZE[2], action=False)

    cond_latent = latent.clone()
    if torch.rand(1).item() < 0.5:
        cond_ts = _per_frame_timesteps(scheduler_latent, cond=True)
        cond_latent = scheduler_latent.add_noise(
            latent, torch.randn_like(latent), cond_ts, t_dim=2,
        )
    else:
        cond_ts = torch.zeros_like(timesteps)

    # Action
    a_ts = _per_frame_timesteps(scheduler_action)
    action = torch.randn(B, PROBE_ACTION_DIM, f, PROBE_N_ACTION, 1,
                         device=device, dtype=dtype)
    a_noise = torch.randn_like(action)
    a_noisy = scheduler_action.add_noise(action, a_noise, a_ts, t_dim=2)
    a_targets = scheduler_action.training_target(action, a_noise, a_ts)
    a_grid = _per_ep_grid(PROBE_N_ACTION, 1, action=True)

    ep_ids = torch.arange(n_ep, dtype=torch.long, device=device)
    seq_ids_per_frame = torch.repeat_interleave(ep_ids, ep_lens_t)[None]
    ep_starts = torch.cumsum(ep_lens_t, 0) - ep_lens_t
    frame_ids_per_frame = (
        torch.arange(f, dtype=torch.long, device=device) - ep_starts.repeat_interleave(ep_lens_t)
    )[None]

    return {
        "latent_dict": {
            "noisy_latents": noisy, "latent": cond_latent, "targets": targets,
            "timesteps": timesteps[None].repeat(B, 1),
            "cond_timesteps": cond_ts[None].repeat(B, 1),
            "grid_id": grid,
            "text_emb": torch.randn(n_ep, PROBE_TEXT_SEQ, PROBE_TEXT_DIM,
                                    device=device, dtype=dtype),
            "latents_mask": torch.ones(B, f, device=device, dtype=torch.bool),
        },
        "action_dict": {
            "noisy_latents": a_noisy, "latent": action.clone(), "targets": a_targets,
            "timesteps": a_ts[None].repeat(B, 1),
            "cond_timesteps": torch.zeros_like(a_ts)[None].repeat(B, 1),
            "grid_id": a_grid,
            "actions_mask": torch.ones_like(action, dtype=torch.bool),
        },
        "chunk_size": PROBE_CHUNK_SIZE,
        "window_size": PROBE_WINDOW_SIZE,
        "seq_ids_per_frame": seq_ids_per_frame,
        "frame_ids_per_frame": frame_ids_per_frame,
        "n_episodes": n_ep,
    }


# ---------------------------------------------------------------------------
# Loss — mirrors Trainer.compute_loss
# ---------------------------------------------------------------------------

def compute_loss(input_dict, pred, sched_lat, sched_act) -> torch.Tensor:
    lat_pred, act_pred = pred
    lat_tgt = input_dict["latent_dict"]["targets"]
    act_tgt = input_dict["action_dict"]["targets"]
    act_mask = input_dict["action_dict"]["actions_mask"]

    act_pred = rearrange(act_pred, "b (f n) c -> b c f n 1", f=act_tgt.shape[-3])
    lat_pred = data_seq_to_patch(PATCH_SIZE, lat_pred,
                                  lat_tgt.shape[-3], lat_tgt.shape[-2], lat_tgt.shape[-1],
                                  batch_size=lat_pred.shape[0])

    Bn, Fn = input_dict["latent_dict"]["timesteps"].shape
    lat_w = sched_lat.training_weight(input_dict["latent_dict"]["timesteps"].flatten()).reshape(Bn, Fn)
    act_w = sched_act.training_weight(input_dict["action_dict"]["timesteps"].flatten()).reshape(Bn, Fn)
    valid = input_dict["latent_dict"]["latents_mask"].flatten().float()

    ll = F.mse_loss(lat_pred.float(), lat_tgt.float().detach(), reduction="none")
    ll = (ll * lat_w[:, None, :, None, None]).permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    ll = (ll.sum(1) / ll.shape[1] * valid).sum() / (valid.sum() + 1e-6)

    al = F.mse_loss(act_pred.float(), act_tgt.float().detach(), reduction="none")
    al = al * act_w[:, None, :, None, None] * act_mask.float()
    am = act_mask.float().permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    al = al.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
    al = (al.sum(1) / (am.sum(1) + 1e-6) * valid).sum() / (valid.sum() + 1e-6)

    return ll + al


# ---------------------------------------------------------------------------
# Single trial (local GPU) + cross-rank synchronisation
# ---------------------------------------------------------------------------

def _run_trial(f, n_episodes, transformer, optimizer, device,
               sched_lat, sched_act) -> int:
    """Forward + backward + step.  Returns peak MB.

    OOM is not caught — under FSDP it leaves NCCL rank-inconsistent,
    deadlocking any follow-up collective.  ``torchrun`` handles teardown.
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    inp = build_synthetic_input(f, n_episodes, device, sched_lat, sched_act)
    out = transformer(inp, train_mode=True)
    loss = compute_loss(inp, out, sched_lat, sched_act)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 2.0)
    optimizer.step()
    optimizer.zero_grad()
    del inp, out, loss
    torch.cuda.synchronize(device)

    return torch.cuda.max_memory_allocated(device) // (1024 * 1024)


def _sync_trial(f, n_episodes, transformer, optimizer, device,
                sched_lat, sched_act,
                ceiling_mb: int = 0) -> tuple[int | None, bool]:
    """Run one trial across all ranks.  Returns ``(peak_mb, skipped)``.

    A collective reserved-memory pre-check ensures all ranks enter or
    skip together.  OOM propagates to ``torchrun`` (see ``_run_trial``).
    """
    if ceiling_mb:
        gc.collect()
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(device) // (1 << 20)
        skip = torch.tensor(
            [int(reserved > ceiling_mb * 0.95)], device=device, dtype=torch.long,
        )
        if dist.is_initialized():
            dist.all_reduce(skip, op=dist.ReduceOp.MAX)
        if skip.item():
            return None, True

    peak = _run_trial(f, n_episodes, transformer, optimizer, device,
                      sched_lat, sched_act)

    pk = torch.tensor([peak], device=device, dtype=torch.long)
    if dist.is_initialized():
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        dist.barrier()

    return int(pk.item()), False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Find max sequence length for training")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--start-f", type=int, default=256,
                        help="Initial probe F (default 256, safe for >=90GB GPUs)")
    parser.add_argument("--mem-ceiling", type=float, default=0.90,
                        help="Fraction of GPU memory treated as ceiling (default 0.90)")
    parser.add_argument("--probe-n-episodes", type=int, default=128,
                        help=(
                            "Packed episodes to simulate per trial "
                            "(default 128, capped at F). Set to the upper "
                            "envelope of N_ep you expect in your data; the "
                            "reported max_tokens is valid for any bin with "
                            "N_ep <= this value."
                        ))
    args = parser.parse_args()

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = rank == 0

    total_mb = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
    ceiling_mb = int(total_mb * args.mem_ceiling)

    if is_main:
        logger.info("=" * 70)
        logger.info("Max sequence length probe")
        logger.info("  Model:       %s", args.model_path)
        logger.info("  World size:  %d GPUs", world_size)
        logger.info("  GPU:         %s (%d MB)", torch.cuda.get_device_name(device), total_mb)
        logger.info("  Ceiling:     %d MB (%.0f%%)", ceiling_mb, args.mem_ceiling * 100)
        logger.info("  Probe:       tok/frame=%d, chunk=%d, window=%d, N_ep=%d",
                     TOKENS_PER_FRAME, PROBE_CHUNK_SIZE, PROBE_WINDOW_SIZE,
                     args.probe_n_episodes)
        logger.info("=" * 70)

    # Load model (identical to Trainer.__init__)
    transformer = load_transformer(
        os.path.join(args.model_path, "transformer"),
        torch_dtype=torch.float32, torch_device="cpu",
        attn_mode='flex',
    )
    assert transformer.config.attn_mode == 'flex'
    apply_ac(transformer)
    transformer = _configure_model(
        model=transformer, shard_fn=shard_model,
        param_dtype=torch.bfloat16, device=device, eval_mode=False,
    )
    transformer.train()
    transformer.requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [p for p in transformer.parameters() if p.requires_grad],
        lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
        fused=True, foreach=False,
    )

    sched_lat = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    sched_lat.set_timesteps(1000, training=True)
    sched_act = FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True)
    sched_act.set_timesteps(1000, training=True)

    # Common kwargs for _sync_trial calls.
    trial_kw = dict(
        n_episodes=args.probe_n_episodes,
        transformer=transformer, optimizer=optimizer, device=device,
        sched_lat=sched_lat, sched_act=sched_act,
        ceiling_mb=ceiling_mb,
    )

    # Warmup: compile FlexAttn + allocate optimizer state.
    warmup_f = max(4, args.probe_n_episodes)
    if is_main:
        logger.info("Warmup (compiling FlexAttn + optimizer state)...")
    for i in range(2):
        peak, _ = _sync_trial(warmup_f, **trial_kw)
        if peak is None:
            if is_main:
                logger.error(
                    "Warmup %d skipped — reserved mem near ceiling. "
                    "Try reducing --probe-n-episodes (current %d) "
                    "or freeing GPU memory.",
                    i, args.probe_n_episodes,
                )
            if dist.is_initialized():
                dist.destroy_process_group()
            return
    if is_main:
        logger.info("Warmup done.\n")

    # === Phase 1: exponential growth ==================================
    results: list[tuple[int, int, int]] = []  # (F, total_seq, peak_mb)
    f = args.start_f
    upper_f: int | None = None   # upper bound for bisect (predicted / measured)
    upper_tested = False          # was upper_f a real measured failure?

    if is_main:
        logger.info(" %10s | %10s | %10s | %8s | %s",
                     "total_seq", "F_probe", "peak_MB", "headroom", "status")
        logger.info("-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-"
                     + "-" * 8 + "-+-" + "-" * 8)

    while True:
        seq = _total_seq_for_f(f)

        # Predictive gate.
        if len(results) >= 2:
            (_, s0, p0), (_, s1, p1) = results[-2], results[-1]
            if s1 != s0:
                predicted = p1 + (p1 - p0) / (s1 - s0) * (seq - s1)
                if predicted > ceiling_mb:
                    if is_main:
                        logger.info(" %10d | %10d | %10s | %8s | SKIP (predicted %.0fMB > %dMB)",
                                    seq, f, "-", "-", predicted, ceiling_mb)
                    upper_f = f
                    break

        peak, _ = _sync_trial(f, **trial_kw)

        if peak is None:
            if is_main:
                logger.info(" %10d | %10d | %10s | %8s | SKIP (reserved mem near ceiling)",
                            seq, f, "-", "-")
            upper_f = f
            break

        headroom = ceiling_mb - peak
        if peak > ceiling_mb:
            if is_main:
                logger.info(" %10d | %10d | %10d | %7dM | OVER ceiling",
                            seq, f, peak, headroom)
            upper_f, upper_tested = f, True
            break

        results.append((f, seq, peak))
        if is_main:
            logger.info(" %10d | %10d | %10d | %7dM | ok", seq, f, peak, headroom)
        if headroom < ceiling_mb * 0.05:
            # Let bisect refine the last few percent.
            upper_f = _next_f(f, peak / ceiling_mb)
            break
        f = _next_f(f, peak / ceiling_mb)

    # === Phase 2: binary refinement ===================================
    if results and upper_f is not None and upper_f > results[-1][0]:
        lo = results[-1][0]
        hi = upper_f
        hi_measured = upper_tested   # tracks whether *current* hi was measured
        if is_main:
            logger.info("\nBinary refinement between F=%d and F=%d ...", lo, hi)

        while hi - lo > 1:
            mid = (lo + hi) // 2
            mid_seq = _total_seq_for_f(mid)

            # Predictive OOM guard.
            if len(results) >= 2:
                (_, s0, p0), (_, s1, p1) = results[-2], results[-1]
                if s1 != s0:
                    predicted = p1 + (p1 - p0) / (s1 - s0) * (mid_seq - s1)
                    if predicted > ceiling_mb:
                        hi = mid
                        hi_measured = False
                        if is_main:
                            logger.info(
                                " %10d | %10d | %10s | %8s | SKIP predicted %.0fMB (bisect)",
                                mid_seq, mid, "-", "-", predicted)
                        continue

            peak, _ = _sync_trial(mid, **trial_kw)
            if peak is None or peak > ceiling_mb:
                hi = mid
                hi_measured = peak is not None  # real measurement, not a skip
                if is_main:
                    tag = f"peak {peak}MB" if peak is not None else "reserved mem"
                    logger.info(" %10d | %10d | %10s | %8s | %s (bisect)",
                                mid_seq, mid,
                                "-" if peak is None else str(peak),
                                "-", tag)
            else:
                headroom = ceiling_mb - peak
                results.append((mid, mid_seq, peak))
                lo = mid
                if is_main:
                    logger.info(" %10d | %10d | %10d | %7dM | ok (bisect)",
                                mid_seq, mid, peak, headroom)

        results.sort(key=lambda r: r[0])
        upper_tested = hi_measured

    # === Summary ======================================================
    if not results:
        if is_main:
            logger.info("\nNo successful trials. Try smaller --start-f.")
        return

    best_f, best_seq, best_peak = results[-1]
    if is_main:
        logger.info("")
        logger.info("=" * 70)
        if upper_tested:
            label = "Max F_probe"
            note = ""
        else:
            label = "Best measured F"
            note = "  (upper bound not measured, true max may be slightly higher)"
        logger.info("%-15s = %d  (total_seq = %d, peak = %d MB)%s",
                     label, best_f, best_seq, best_peak, note)
        logger.info("90%% safety      = %d tokens", int(best_seq * 0.9))
        if len(results) >= 2:
            (_, s0, p0), (_, s1, p1) = results[-2], results[-1]
            if s1 != s0:
                slope = (p1 - p0) / (s1 - s0)
                intercept = p0 - slope * s0
                logger.info("Linear fit      = %.2f MB/token + %.0f MB",
                             slope, intercept)
        logger.info("=" * 70)
        print_conversion_table(best_seq)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
