# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Distributed-init helpers used by non-Lightning entry points.

The Lightning training path constructs its process group through
``ModelParallelStrategy``; only the inference server and the
``find_max_seq_len`` tool drive ``init_process_group`` directly and use
``_configure_model`` to materialize the shard.
"""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True, shard=True):
    """Apply ``shard_fn`` if dist is up and sharding is requested; otherwise materialize on a single device.

    Used by ``src.inference.server`` and ``src.tools.find_max_seq_len``. The
    Lightning training path replaces this with a ``LightningModule.configure_model``
    hook driven by the strategy's ``device_mesh``.

    ``shard=False`` is the DDP-style (full replication) path: the model goes to
    ``device`` directly even when ``dist`` is initialized. The caller is then
    responsible for synchronizing gradients across ranks.
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if shard and dist.is_initialized():
        dist.barrier()
        model = shard_fn(model)
    else:
        model.to(param_dtype)
        model.to(device)
    return model


def init_distributed(world_size, local_rank, rank):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if int(world_size) <= 1:
        return
    # 60-min default gives cute-dsl cold compiles room to finish before the
    # NCCL watchdog fires monitoredBarrier (default 30 min was already tight
    # for FA4's first-shape compile on a 24-head 128-dim attention) — this
    # matters on the *sharded* (FSDP/TP) path where the first compile happens
    # inside a collective (see _configure_model's pre-shard barrier).
    #
    # On the *replicated* (DDP) RL path FA4 compiles per-rank during rollout
    # forwards, which involve NO collective, so the first NCCL op (the GRPO
    # update barrier) is reached only after every rank has finished compiling
    # and rolling out. There, a short timeout is desirable: a crashed/hung
    # rollout client on one rank would otherwise leave its peers hanging in
    # dist.barrier() for the full hour (the "0 updates all night" failure).
    # Override with DIST_TIMEOUT_MIN so the RL launcher can fail fast (~8 min)
    # and let the supervisor restart the client, which resumes from progress.
    timeout_min = float(os.environ.get("DIST_TIMEOUT_MIN", "60"))
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size,
                            timeout=timedelta(minutes=timeout_min),
                            device_id=torch.device(f"cuda:{local_rank}"))
