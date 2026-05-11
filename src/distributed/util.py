# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""Distributed-init helpers used by non-Lightning entry points.

The Lightning training path constructs its process group through
``ModelParallelStrategy``; only the inference server and the
``find_max_seq_len`` tool drive ``init_process_group`` directly and use
``_configure_model`` to materialize the shard.
"""
from datetime import timedelta

import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """Apply ``shard_fn`` if dist is up; otherwise materialize on a single device.

    Used by ``src.inference.server`` and ``src.tools.find_max_seq_len``. The
    Lightning training path replaces this with a ``LightningModule.configure_model``
    hook driven by the strategy's ``device_mesh``.
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
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
    # 60-min timeout gives cute-dsl cold compiles room to finish before the
    # NCCL watchdog fires monitoredBarrier (default 30 min was already tight
    # for FA4's first-shape compile on a 24-head 128-dim attention).
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size,
                            timeout=timedelta(minutes=60),
                            device_id=torch.device(f"cuda:{local_rank}"))
