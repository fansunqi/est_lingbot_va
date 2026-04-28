# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = '/apdcephfs_gy5/share_303588738/leoyizhang/train'

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = False

# Sequence packing.  max_episodes_per_bin must match --probe-n-episodes.
va_shared_cfg.packing = EasyDict()
va_shared_cfg.packing.max_tokens = 36224 # max 125440 on H20
va_shared_cfg.packing.max_episodes_per_bin = 128
