"""Minimal LoRA adapters for GRPO fine-tuning.

This stays dependency-free so the online server can enable LoRA without adding
PEFT as a runtime requirement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Linear(
            base.in_features,
            self.rank,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.lora_B = nn.Linear(
            self.rank,
            base.out_features,
            bias=False,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out.to(dtype=base_out.dtype)


@dataclass(frozen=True)
class LoRAStats:
    wrapped_modules: tuple[str, ...]
    trainable_parameters: int
    total_parameters: int


def _matches_target(name: str, target_modules: Sequence[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in target_modules)


def _set_child_module(parent: nn.Module, child_name: str, child: nn.Module) -> None:
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
        parent[int(child_name)] = child
    else:
        setattr(parent, child_name, child)


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: Sequence[str],
    freeze_base: bool = True,
) -> LoRAStats:
    if freeze_base:
        model.requires_grad_(False)

    module_lookup = dict(model.named_modules())
    wrapped = []
    for name, module in list(model.named_modules()):
        if not name or not isinstance(module, nn.Linear) or isinstance(module, LoRALinear):
            continue
        if not _matches_target(name, target_modules):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = module_lookup[parent_name] if parent_name else model
        _set_child_module(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
        wrapped.append(name)

    if not wrapped:
        raise ValueError(f"No nn.Linear modules matched LoRA targets: {list(target_modules)}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return LoRAStats(tuple(wrapped), trainable, total)
