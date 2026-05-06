# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""FSDP2 sharding + activation checkpointing helpers.

Used both by the Lightning training path (``FlowMatchVASystem.configure_model``)
and the non-Lightning paths (``src.inference.server`` and
``src.tools.find_max_seq_len``). The non-Lightning callers keep working with
``apply_ac(model)`` / ``shard_model(model)`` because the new keyword-only args
default to the legacy behaviour (granularity='all', block_only=False,
reshard_after_forward=True, mesh=None).
"""
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


_AC_STRIDES = {'all': 1, 'every_2': 2, 'every_4': 4}


def apply_ac(model, granularity: str = 'all'):
    """Wrap selected transformer blocks in NO_REENTRANT activation checkpointing.

    granularity:
        'none'    - no AC (rely on enough memory headroom)
        'every_4' - 1 in 4 blocks
        'every_2' - 1 in 2 blocks
        'all'     - every block (legacy default; matches behaviour pre-refactor)
    """
    if granularity == 'none':
        return model
    if granularity not in _AC_STRIDES:
        raise ValueError(
            f"unknown ac granularity {granularity!r}; "
            f"expected one of {list(_AC_STRIDES) + ['none']}"
        )
    stride = _AC_STRIDES[granularity]
    for i, blk in enumerate(model.blocks):
        if i % stride == 0:
            model.blocks[i] = ptd_checkpoint_wrapper(blk, preserve_rng_state=False)
    return model


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                *,
                mesh=None,
                reshard_after_forward: bool = True,
                block_only: bool = False):
    """Apply FSDP2 sharding across the transformer blocks.

    block_only=True issues one ``fully_shard`` per block (recommended on
    NVLink-connected single-node setups; fewer all-gathers per step).
    block_only=False is the legacy 4-way granularity (attn1, attn2, ffn, block).

    mesh=None lets FSDP2 use the default global process group; pass an explicit
    ``DeviceMesh`` when running under Lightning's ``ModelParallelStrategy`` so
    sharding aligns with the strategy's mesh.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = dict(
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    )
    if mesh is not None:
        fsdp_config['mesh'] = mesh

    for block in model.blocks:
        if not block_only:
            fully_shard(block.attn1, **fsdp_config)
            fully_shard(block.attn2, **fsdp_config)
            fully_shard(block.ffn, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
