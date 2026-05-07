# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Export a Lightning .ckpt to diffusers transformer/ format for inference.

The training System wraps the WAN transformer inside
``self.transformer_wrapper.net``; its state_dict keys carry the
``transformer_wrapper.net.`` prefix in Lightning checkpoints. This script
strips that prefix, casts to bfloat16, writes
``transformer/diffusion_pytorch_model.safetensors`` plus a ``config.json``
copied from the original pretrained model.

Usage::

    python scripts/export_to_diffusers.py \
        --ckpt experiments/<run>/checkpoints/last.ckpt \
        --pretrained-config /path/to/lingbot-va-base \
        --output-dir experiments/<run>/exported
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


_NET_PREFIX = "transformer_wrapper.net."
_COMPILED_PREFIX = "_orig_mod."  # strip when compile_model=true was used


def _strip_prefixes(state_dict: dict) -> dict:
    out = {}
    for key, value in state_dict.items():
        if key.startswith(_NET_PREFIX):
            key = key[len(_NET_PREFIX):]
        if key.startswith(_COMPILED_PREFIX):
            key = key[len(_COMPILED_PREFIX):]
        out[key] = value
    return out


def export(ckpt_path: Path, pretrained_config_root: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    transformer_dir = output_dir / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Lightning checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    state_dict = _strip_prefixes(state_dict)

    bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    out_safetensors = transformer_dir / "diffusion_pytorch_model.safetensors"
    print(f"Writing weights ({len(bf16)} tensors) to {out_safetensors}")
    save_file(bf16, str(out_safetensors))

    src_config = pretrained_config_root / "transformer" / "config.json"
    if not src_config.exists():
        raise FileNotFoundError(f"missing pretrained transformer config: {src_config}")
    with open(src_config) as f:
        config_dict = json.load(f)
    config_dict.pop("_name_or_path", None)
    out_config = transformer_dir / "config.json"
    with open(out_config, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Wrote config.json to {out_config}")

    print(f"Export complete. Inference can load from: {transformer_dir.parent}")


def main():
    parser = argparse.ArgumentParser(description="Export Lightning .ckpt to diffusers format")
    parser.add_argument("--ckpt", required=True, type=Path, help="Path to Lightning .ckpt")
    parser.add_argument("--pretrained-config", required=True, type=Path,
                        help="Pretrained model root containing transformer/config.json")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory; will create transformer/ subdir")
    args = parser.parse_args()
    export(args.ckpt, args.pretrained_config, args.output_dir)


if __name__ == "__main__":
    main()
