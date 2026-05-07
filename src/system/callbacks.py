# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Lightning callbacks for the VA training stack."""
from __future__ import annotations

import csv
import gc
import os
import time
from pathlib import Path
from typing import Optional

import torch
import lightning.pytorch as pl


class PeriodicGCCallback(pl.Callback):
    """Trigger CUDA cache flush + Python GC every N optimizer steps.

    Mirrors the cadence in the legacy training loop: long-running FSDP runs
    accumulate fragmented allocations across heterogeneous batch shapes, and
    a periodic ``empty_cache + gc.collect`` keeps the high-water mark in check
    without measurable throughput cost.
    """

    def __init__(self, every_n_steps: int):
        super().__init__()
        self.every_n_steps = int(every_n_steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step > 0 and step % self.every_n_steps == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()


# --------------------------------------------------------------------------- #
# FSDPMetricsCallback — measures step time + peak GPU mem for the sweep harness.
# --------------------------------------------------------------------------- #


class FSDPMetricsCallback(pl.Callback):
    """Records median step time + peak GPU memory and appends one CSV row.

    Lightweight, designed for the FSDP sweep harness:
    - Times each ``training_step`` after ``warmup_steps`` warmup iterations.
    - Resets ``torch.cuda.max_memory_allocated`` at the start of measurement,
      reads it at training end.
    - Writes a single CSV row per run on rank 0 only.

    Columns: tag, ac, reshard, block_only, world_size, step_ms_p50, step_ms_p10,
    peak_alloc_gb, peak_reserved_gb.
    """

    def __init__(self, results_path: str, tag: str, warmup_steps: int = 3):
        super().__init__()
        self.results_path = results_path
        self.tag = tag
        self.warmup_steps = int(warmup_steps)
        self._step_times_ms: list[float] = []
        self._t0: Optional[float] = None
        self._mem_reset_done = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Reset peak-mem stats once we're past warmup so we don't capture the
        # cute-dsl cold-compile spike.
        if not self._mem_reset_done and trainer.global_step >= self.warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._mem_reset_done = True
        if trainer.global_step >= self.warmup_steps:
            torch.cuda.synchronize()
            self._t0 = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._t0 is None:
            return
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - self._t0) * 1000.0
        self._step_times_ms.append(dt_ms)
        self._t0 = None

    def on_train_end(self, trainer, pl_module):
        torch.cuda.synchronize()
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

        if not self._step_times_ms:
            step_ms_p50 = float('nan')
            step_ms_p10 = float('nan')
        else:
            sorted_ms = sorted(self._step_times_ms)
            n = len(sorted_ms)
            step_ms_p50 = sorted_ms[n // 2]
            step_ms_p10 = sorted_ms[max(0, n // 10)]

        if trainer.global_rank != 0:
            return

        fsdp_cfg = (pl_module.config.get('trainer', {}) or {}).get('fsdp', {}) or {}
        row = {
            'tag': self.tag,
            'ac': fsdp_cfg.get('ac_granularity', 'every_2'),
            'reshard': fsdp_cfg.get('reshard_after_forward', True),
            'block_only': fsdp_cfg.get('block_only', True),
            'world_size': trainer.world_size,
            'step_ms_p50': f"{step_ms_p50:.2f}",
            'step_ms_p10': f"{step_ms_p10:.2f}",
            'peak_alloc_gb': f"{peak_alloc_gb:.2f}",
            'peak_reserved_gb': f"{peak_reserved_gb:.2f}",
        }
        Path(self.results_path).parent.mkdir(parents=True, exist_ok=True)
        write_header = not os.path.exists(self.results_path)
        with open(self.results_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)


# --------------------------------------------------------------------------- #
# FullStateDictCheckpoint — runs alongside Lightning's sharded ckpt and emits a
# single-file unsharded ckpt every N steps so sample.py / inference can keep
# loading a flat .pt without learning about sharded ckpts.
# --------------------------------------------------------------------------- #


class FullStateDictCheckpoint(pl.Callback):
    """Periodically gather a full (unsharded) state_dict and save to one file.

    The expensive bit is the rank-0 gather; we run it on a coarser cadence
    than the sharded ``ModelCheckpoint`` (e.g. every 5x fewer steps) so the
    gather doesn't dominate step time.
    """

    def __init__(self, dirpath: str, every_n_train_steps: int):
        super().__init__()
        self.dirpath = dirpath
        self.every_n_train_steps = int(every_n_train_steps)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step <= 0 or step % self.every_n_train_steps != 0:
            return
        self._save(trainer, pl_module, step)

    def _save(self, trainer, pl_module, step: int):
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        # All ranks must call this — collective op behind the scenes.
        sd = get_model_state_dict(pl_module, options=opts)

        if trainer.global_rank != 0:
            return

        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        path = Path(self.dirpath) / f"full_step_{step}.pt"
        # Strip the LightningModule prefix so consumers can load directly into
        # WanTransformer3DModel via ``model.load_state_dict(sd)`` if desired.
        prefix = 'transformer_wrapper.net.'
        net_sd = {
            k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
        }
        torch.save({'state_dict': sd, 'net_state_dict': net_sd, 'step': step}, path)
