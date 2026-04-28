"""Minimal multi-node NCCL sanity check.

Runs under torchrun. Each rank:
  1. init_process_group on NCCL
  2. allocates a tiny bfloat16 tensor seeded with its rank
  3. all_reduce(SUM) across the world
  4. verifies the result equals sum(0..world_size-1)
  5. barrier + destroy_process_group

Prints a single-line verdict per rank. Use stderr for warnings so stdout stays
clean enough to grep for OK/FAIL.
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.distributed as dist


def main() -> int:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    t0 = time.time()
    try:
        dist.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:  # noqa: BLE001
        print(f"[rank{rank}] FAIL init_process_group: {e}", flush=True)
        return 2
    init_s = time.time() - t0

    x = torch.full((16,), float(rank), dtype=torch.bfloat16, device="cuda")
    t1 = time.time()
    try:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"[rank{rank}] FAIL all_reduce: {e}", flush=True)
        dist.destroy_process_group()
        return 3
    ar_s = time.time() - t1

    expected = sum(range(world_size))
    got = x.float().mean().item()
    ok = abs(got - expected) < 1e-3

    print(
        f"[rank{rank}/{world_size} local{local_rank}] "
        f"{'OK' if ok else 'FAIL'} "
        f"init={init_s:.2f}s allreduce={ar_s*1000:.1f}ms "
        f"got={got:.3f} expected={expected}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
