"""
Find the maximum total sequence length that fits in GPU memory.

Loads the real model with FSDP + activation checkpointing (identical to
``wan_va.train``), constructs synthetic inputs at increasing sequence
lengths, and runs forward + backward + optimizer step to measure peak
memory.

After 2+ successful trials, a linear extrapolation predicts whether the
next attempt would exceed the memory ceiling; if so, the probe stops
*before* triggering OOM (which could corrupt FSDP collective state).
A ``try/except`` around each trial acts as a last-resort safety net.

Usage::

    torchrun --nproc_per_node=8 -m wan_va.tools.find_max_seq_len \\
        --model-path /path/to/pretrained_wan
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

from distributed.fsdp import shard_model, apply_ac
from distributed.util import _configure_model, init_distributed
from modules.utils import load_transformer
from utils import get_mesh_id, sample_timestep_id, data_seq_to_patch, FlowMatchScheduler

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe geometry — only F varies.  forward_train flattens everything to
# (1, total_seq, 3072), so peak memory depends on total_seq alone.
# ---------------------------------------------------------------------------
PROBE_H_LATENT = 16
PROBE_W_LATENT = 16
PROBE_N_ACTION = 4
PROBE_IN_CHANNELS = 48
PROBE_ACTION_DIM = 30
PROBE_TEXT_SEQ = 512
PROBE_TEXT_DIM = 4096
PATCH_SIZE = (1, 2, 2)

PROBE_CHUNK_SIZE = 1   # worst-case mask density
PROBE_WINDOW_SIZE = 64

_LAT_TOK_PER_FRAME = (PROBE_H_LATENT // PATCH_SIZE[1]) * (PROBE_W_LATENT // PATCH_SIZE[2])
_ACT_TOK_PER_FRAME = PROBE_N_ACTION
TOKENS_PER_FRAME = 2 * _LAT_TOK_PER_FRAME + 2 * _ACT_TOK_PER_FRAME  # x2: noisy + cond


def _pad128(n: int) -> int:
    return n + (128 - n % 128) % 128


def _total_seq_for_f(f: int) -> int:
    return _pad128(f * TOKENS_PER_FRAME)


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
            lat = (h // PATCH_SIZE[1]) * (w // PATCH_SIZE[2])
            tpf = 2 * lat + 2 * n_action
            f100 = _max_f_for_budget(max_seq, tpf)
            f90 = _max_f_for_budget(int(max_seq * 0.9), tpf)
            label = f"{name} ({desc})" if stride == 1 else ""
            logger.info("%-44s | %6d | %9d | %11d | %10d", label, stride, tpf, f100, f90)
        logger.info("")
    logger.info("Formula: max_F = max_total_seq // (2*(H_l/p_h * W_l/p_w) + 2*(stride*4))")
    logger.info("Note: vae_temporal_downsample defaults to 4. 90%% column recommended for production.")


# ---------------------------------------------------------------------------
# Synthetic input — mirrors Trainer._add_noise + _prepare_input_dict
# ---------------------------------------------------------------------------

def build_synthetic_input(
    f: int,
    device: torch.device,
    scheduler_latent: FlowMatchScheduler,
    scheduler_action: FlowMatchScheduler,
) -> dict:
    B, dtype = 1, torch.bfloat16

    # Latent
    latent = torch.randn(B, PROBE_IN_CHANNELS, f, PROBE_H_LATENT, PROBE_W_LATENT,
                          device=device, dtype=dtype)
    noise = torch.randn_like(latent)
    ts_ids = sample_timestep_id(batch_size=f, num_train_timesteps=scheduler_latent.num_train_timesteps)
    timesteps = scheduler_latent.timesteps[ts_ids].to(device=device)
    noisy = scheduler_latent.add_noise(latent, noise, timesteps, t_dim=2)
    targets = scheduler_latent.training_target(latent, noise, timesteps)

    grid = get_mesh_id(
        f // PATCH_SIZE[0], PROBE_H_LATENT // PATCH_SIZE[1], PROBE_W_LATENT // PATCH_SIZE[2],
        t=0, f_w=1, f_shift=0, action=False,
    ).to(device)[None].repeat(B, 1, 1)

    cond_latent = latent.clone()
    if torch.rand(1).item() < 0.5:
        cond_ids = sample_timestep_id(batch_size=f, min_timestep_bd=0.5, max_timestep_bd=1.0,
                                       num_train_timesteps=scheduler_latent.num_train_timesteps)
        cond_ts = scheduler_latent.timesteps[cond_ids].to(device=device)
        cond_latent = scheduler_latent.add_noise(latent, torch.randn_like(latent), cond_ts, t_dim=2)
    else:
        cond_ts = torch.zeros_like(timesteps)

    # Action
    action = torch.randn(B, PROBE_ACTION_DIM, f, PROBE_N_ACTION, 1, device=device, dtype=dtype)
    a_noise = torch.randn_like(action)
    a_ids = sample_timestep_id(batch_size=f, num_train_timesteps=scheduler_action.num_train_timesteps)
    a_ts = scheduler_action.timesteps[a_ids].to(device=device)
    a_noisy = scheduler_action.add_noise(action, a_noise, a_ts, t_dim=2)
    a_targets = scheduler_action.training_target(action, a_noise, a_ts)

    a_grid = get_mesh_id(f, PROBE_N_ACTION, 1, t=1, f_w=1, f_shift=0, action=True,
                          ).to(device)[None].repeat(B, 1, 1)

    return {
        "latent_dict": {
            "noisy_latents": noisy, "latent": cond_latent, "targets": targets,
            "timesteps": timesteps[None].repeat(B, 1),
            "cond_timesteps": cond_ts[None].repeat(B, 1),
            "grid_id": grid,
            "text_emb": torch.randn(B, PROBE_TEXT_SEQ, PROBE_TEXT_DIM, device=device, dtype=dtype),
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
# Single trial
# ---------------------------------------------------------------------------

def run_trial(f, transformer, optimizer, device, sched_lat, sched_act, ceiling_mb) -> int | None:
    """Forward + backward + step.  Returns peak MB or None on failure."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if torch.cuda.memory_reserved(device) // (1024 * 1024) > ceiling_mb * 0.95:
        return None

    try:
        inp = build_synthetic_input(f, device, sched_lat, sched_act)
        out = transformer(inp, train_mode=True)
        loss = compute_loss(inp, out, sched_lat, sched_act)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transformer.parameters(), 2.0)
        optimizer.step()
        optimizer.zero_grad()
        del inp, out, loss
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError:
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        return None

    peak = torch.cuda.max_memory_allocated(device) // (1024 * 1024)
    return peak if peak <= ceiling_mb else None


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
        logger.info("  Probe:       tok/frame=%d, chunk=%d, window=%d",
                     TOKENS_PER_FRAME, PROBE_CHUNK_SIZE, PROBE_WINDOW_SIZE)
        logger.info("=" * 70)

    # Load model (identical to Trainer.__init__)
    transformer = load_transformer(
        os.path.join(args.model_path, "transformer"),
        torch_dtype=torch.float32, torch_device="cpu",
    )
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

    # Warmup: compile FlexAttn + establish optimizer state (momentum/variance buffers)
    if is_main:
        logger.info("Warmup (compiling FlexAttn + optimizer state)...")
    for _ in range(2):
        run_trial(4, transformer, optimizer, device, sched_lat, sched_act, ceiling_mb)
    if dist.is_initialized():
        dist.barrier()
    if is_main:
        logger.info("Warmup done.\n")

    # Probe: grow F until predicted peak exceeds ceiling
    results: list[tuple[int, int, int]] = []  # (F, total_seq, peak_mb)
    f = args.start_f

    if is_main:
        logger.info(" %10s | %10s | %10s | %8s | %s",
                     "total_seq", "F_probe", "peak_MB", "headroom", "status")
        logger.info("-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 10 + "-+-" + "-" * 8 + "-+-" + "-" * 8)

    while True:
        seq = _total_seq_for_f(f)

        # Predictive gate: skip if extrapolation says we'd exceed ceiling
        if len(results) >= 2:
            (_, s0, p0), (_, s1, p1) = results[-2], results[-1]
            if s1 != s0:
                predicted = p1 + (p1 - p0) / (s1 - s0) * (seq - s1)
                if predicted > ceiling_mb:
                    if is_main:
                        logger.info(" %10d | %10d | %10s | %8s | SKIP (predicted %.0fMB > %dMB)",
                                    seq, f, "-", "-", predicted, ceiling_mb)
                    break

        peak = run_trial(f, transformer, optimizer, device, sched_lat, sched_act, ceiling_mb)

        # Sync across ranks
        ok = torch.tensor([1 if peak is not None else 0], device=device, dtype=torch.long)
        pk = torch.tensor([peak or 0], device=device, dtype=torch.long)
        if dist.is_initialized():
            dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            dist.all_reduce(pk, op=dist.ReduceOp.MAX)

        if ok.item():
            synced_peak = int(pk.item())
            headroom = ceiling_mb - synced_peak
            results.append((f, seq, synced_peak))
            if is_main:
                logger.info(" %10d | %10d | %10d | %7dM | ok", seq, f, synced_peak, headroom)
            if headroom < ceiling_mb * 0.05:
                break
            # Adaptive growth rate based on memory usage
            ratio = synced_peak / ceiling_mb
            f = int(f * (1.25 if ratio > 0.75 else 1.5 if ratio > 0.5 else 2))
        else:
            if is_main:
                logger.info(" %10d | %10d | %10s | %8s | STOP (ceiling or OOM)", seq, f, "-", "-")
            break

        if dist.is_initialized():
            dist.barrier()

    # Summary
    if not results:
        if is_main:
            logger.info("\nNo successful trials. Try smaller --start-f.")
        return

    best_f, best_seq, best_peak = results[-1]
    if is_main:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Max F_probe     = %d  (total_seq = %d, peak = %d MB)", best_f, best_seq, best_peak)
        logger.info("90%% safety      = %d tokens", int(best_seq * 0.9))
        if len(results) >= 2:
            (_, s0, p0), (_, s1, p1) = results[-2], results[-1]
            if s1 != s0:
                logger.info("Linear fit      = %.2f MB/token + %.0f MB", (p1-p0)/(s1-s0), p0 - (p1-p0)/(s1-s0)*s0)
        logger.info("=" * 70)
        print_conversion_table(best_seq)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
