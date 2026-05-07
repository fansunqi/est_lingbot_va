# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch

from .logging import init_logger, logger
from .scheduler import FlowMatchScheduler
from .server_utils import run_async_server_mode
from .utils import (
    data_seq_to_patch,
    get_mesh_id,
    get_mesh_id_packed,
    sample_timestep_id,
    save_async,
    warmup_constant_lambda,
    warmup_linear_lambda,
)

__all__ = [
    'logger',
    'init_logger',
    'get_mesh_id',
    'get_mesh_id_packed',
    'save_async',
    'data_seq_to_patch',
    'FlowMatchScheduler',
    'run_async_server_mode',
    'sample_timestep_id',
    'warmup_constant_lambda',
    'warmup_linear_lambda',
    'dtype_from_str',
]


_DTYPE_MAP = {
    'bfloat16': torch.bfloat16,
    'bf16': torch.bfloat16,
    'float16': torch.float16,
    'fp16': torch.float16,
    'half': torch.float16,
    'float32': torch.float32,
    'fp32': torch.float32,
    'float': torch.float32,
}


def dtype_from_str(s):
    if isinstance(s, torch.dtype):
        return s
    return _DTYPE_MAP[s]
