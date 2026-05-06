# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Build Lightning's official ``ModelParallelStrategy`` for FSDP2 training.

We used to inherit ``DDPStrategy`` and override ``configure_ddp`` to bridge
Lightning to FSDP2 via ``fully_shard``. Lightning ≥ 2.5 ships
``ModelParallelStrategy`` which is FSDP2-native and exposes a ``device_mesh``
plus a sharded-checkpoint path; sharding now lives in
``LightningModule.configure_model`` (see ``src/system/flow_match_va.py``).
"""
from __future__ import annotations

from datetime import timedelta

import torch
from lightning.pytorch.strategies import ModelParallelStrategy


def build_strategy(config: dict) -> ModelParallelStrategy:
    """Construct ``ModelParallelStrategy`` from a YAML task config.

    Reads ``trainer.devices``, ``trainer.num_nodes`` and ``trainer.fsdp.*``.
    Returns a strategy with pure FSDP2 (``tensor_parallel_size=1``); the per-
    block ``fully_shard`` calls happen in the LightningModule.
    """
    trainer_cfg = config.get('trainer', {}) or {}
    fsdp_cfg = trainer_cfg.get('fsdp', {}) or {}

    devices = trainer_cfg.get('devices', 'auto')
    if devices == 'auto':
        devices = torch.cuda.device_count()
    devices = int(devices)
    num_nodes = int(trainer_cfg.get('num_nodes', 1))

    # 60-min timeout matches the legacy ``init_distributed`` value: FA4's first
    # per-shape cute-dsl compile can run minutes long, and the default 30-min
    # NCCL watchdog will fire monitoredBarrier and crash the run before
    # compilation finishes.
    return ModelParallelStrategy(
        data_parallel_size=devices * num_nodes,
        tensor_parallel_size=1,
        save_distributed_checkpoint=bool(fsdp_cfg.get('save_distributed_checkpoint', True)),
        timeout=timedelta(minutes=60),
    )
