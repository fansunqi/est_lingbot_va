# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Convert a Lightning sharded (DCP) checkpoint directory to a flat .ckpt.

``ModelParallelStrategy(save_distributed_checkpoint=True)`` writes
``checkpoints/<tag>.ckpt/`` directories containing ``__N_0.distcp`` shards
plus a ``meta.pt`` with the non-tensor Lightning fields (epoch, global_step,
callbacks, hparams, packing_state, ...). ``scripts/export_to_diffusers.py``
expects a flat ``.ckpt`` (``torch.load``), so this utility bridges the two:

    python scripts/dcp_to_flat_ckpt.py \
        --src experiments/<run>/checkpoints/<tag>.ckpt \
        --dst experiments/<run>/checkpoints/<tag>.flat.pt

By default only ``state_dict`` is written (sufficient for the diffusers
export). Pass ``--include-optimizer`` to also keep optimizer shards (useful
for offline analysis; the resulting file is large — tens of GB).

Notes:
- DCP saves optimizer state under raw keys like ``optimizer_0`` whereas
  Lightning's flat format expects ``optimizer_states: List[dict]``. The full
  remapping needed for a Lightning ``ckpt_path=`` resume from flat is not
  done here — for resume, point Lightning directly at the sharded directory.
- Reads happen on a single process to avoid OOM (matching
  ``dcp_to_torch_save``'s recommendation).
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save


def convert(src: Path, dst: Path, include_optimizer: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"sharded ckpt directory not found: {src}")
    distcp = sorted(src.glob("*.distcp"))
    if not distcp:
        raise FileNotFoundError(f"no *.distcp shards under {src}")
    meta_path = src / "meta.pt"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing meta.pt in {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    # dcp_to_torch_save flushes a complete dict (state_dict + optimizer_0).
    # We stage to a tmp file so we can re-save a slimmed result if requested.
    with tempfile.NamedTemporaryFile(
        prefix="dcp_flat_", suffix=".pt",
        dir=dst.parent, delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        print(f"[dcp_to_flat] {src} → tmp {tmp_path}")
        dcp_to_torch_save(str(src), str(tmp_path))

        flat = torch.load(tmp_path, map_location="cpu", weights_only=False)
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

        out: dict = dict(meta)  # epoch, global_step, callbacks, hparams, ...
        out["state_dict"] = flat.get("state_dict", {})
        if include_optimizer:
            for k, v in flat.items():
                if k != "state_dict":
                    out[k] = v
            print(f"[dcp_to_flat] keeping optimizer shards "
                  f"({sum(1 for k in flat if k != 'state_dict')} keys)")
        else:
            print(f"[dcp_to_flat] dropping optimizer shards "
                  f"({sum(1 for k in flat if k != 'state_dict')} keys)")

        print(f"[dcp_to_flat] writing {dst}")
        torch.save(out, dst)
        sd = out["state_dict"]
        print(f"[dcp_to_flat] state_dict tensors: {len(sd)}")
        if sd:
            sample = next(iter(sd.keys()))
            print(f"[dcp_to_flat] sample key: {sample}")
        print(f"[dcp_to_flat] done. step={out.get('global_step')} "
              f"epoch={out.get('epoch')}")
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Lightning DCP sharded checkpoint to flat .ckpt"
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="Sharded ckpt directory (contains *.distcp + meta.pt)")
    parser.add_argument("--dst", required=True, type=Path,
                        help="Output flat .pt path")
    parser.add_argument("--include-optimizer", action="store_true",
                        help="Keep optimizer_N shards (large file)")
    args = parser.parse_args()
    convert(args.src, args.dst, args.include_optimizer)


if __name__ == "__main__":
    main()
