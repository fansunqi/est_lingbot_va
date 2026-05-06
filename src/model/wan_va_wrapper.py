# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Wrapper around the WAN transformer that owns weight loading and forward.

Module-level boundary between LightningModule (training logic) and the bare
``WanTransformer3DModel``: load weights from disk, optionally compile.
Activation checkpointing and FSDP2 sharding are applied later by
``FlowMatchVASystem.configure_model`` so granularity is config-driven.
"""
import os

import torch
import torch.nn as nn

from src.model.loaders import load_transformer


def _is_diffusers_export(path: str) -> bool:
    return os.path.isfile(os.path.join(path, 'transformer', 'config.json'))


class WanVATransformerWrapper(nn.Module):

    def __init__(
        self,
        model_name_or_path: str,
        resume_from: str | None = None,
        compile_model: bool = False,
        compile_mode: str = 'default',
    ):
        super().__init__()

        # ``resume_from`` is overloaded: it can be a diffusers-format export
        # directory (containing ``transformer/config.json``) used as a weight
        # bootstrap, or a Lightning checkpoint (file or sharded DCP directory)
        # whose weights are re-loaded later by ``Trainer.fit(ckpt_path=...)``.
        # Only treat it as a bootstrap source in the diffusers case; otherwise
        # bootstrap from ``model_name_or_path`` and let Lightning overwrite the
        # state dict.
        bootstrap_root = model_name_or_path
        if resume_from and _is_diffusers_export(resume_from):
            bootstrap_root = resume_from
        transformer_path = os.path.join(bootstrap_root, 'transformer')
        self.net = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode='flex',
        )
        if self.net.config.attn_mode != 'flex':
            raise RuntimeError("packed training requires attn_mode='flex'")

        # torch.compile pairs poorly with FA4 (see shared_config.py notes).
        # Keep the OptimizedModule swap optional and behind a flag.
        # mode='max-autotune-no-cudagraphs' searches kernel configs per matmul
        # for ~5-30% extra speed; uses no-cudagraphs because cudagraphs conflict
        # with FSDP's per-step shape variation. mode='default' is the cheap path.
        if compile_model:
            self.net = torch.compile(self.net, dynamic=True, mode=compile_mode)

    def forward(self, input_dict, train_mode: bool = True):
        return self.net(input_dict, train_mode=train_mode)
