"""Sequence-packing dataset for WanTransformer training.

Bin-packs multiple episodes (or chunks of long episodes) into sequences of
up to ``max_tokens``.  Cross-episode attention is blocked by FlexAttention
``seq_ids``; context-prefix frames are KV-visible but loss-masked via
``latents_mask``.

All frame arithmetic is in **latent-frame** space.  Oversize episodes are
split by :func:`coverage_chunking.compute_chunks`; their supervised spans
partition ``[0, latent_F)`` exactly.

Ranks recompute the plan deterministically from ``(seed, epoch)`` — no
``dist.broadcast``.  Bin order is shuffled before stride-sharding so the
trailing ``N % world_size`` drop rotates across epochs.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .bin_packing import best_fit_decreasing
from .coverage_chunking import ChunkSpec, compute_chunks
from .lerobot_latent_dataset import MultiLatentLeRobotDataset
from .token_math import worst_case_lookback_frames


@dataclass(frozen=True)
class _ChunkRef:
    """Metadata needed to materialize one chunk at ``__getitem__`` time."""
    segment_idx: int      # segment's global index (shared by all chunks of the same segment)
    dataset_idx: int
    local_idx: int        # index within sub-dataset's segments
    chunk_idx: int        # 0..N_chunks-1 within this segment
    spec: ChunkSpec
    tokens: int


class PackingDataset(Dataset):
    """Bin-packed dataset over a ``MultiLatentLeRobotDataset``.

    ``__len__`` = this rank's bin count; ``__getitem__(i)`` returns one
    packed batch dict (use ``batch_size=1`` at the DataLoader level).
    Call :meth:`set_epoch` between epochs for fresh chunking + bin plans.
    """

    def __init__(
        self,
        multi_ds: MultiLatentLeRobotDataset,
        *,
        max_tokens: int,
        max_episodes_per_bin: int = 128,
        world_size: int = 1,
        rank: int = 0,
        seed: int = 42,
        epoch: int = 0,
    ):
        if max_tokens % 128 != 0:
            raise ValueError(
                f"packing.max_tokens must be a multiple of 128 (pad128 alignment), "
                f"got {max_tokens}"
            )
        if max_episodes_per_bin <= 0:
            raise ValueError(
                f"max_episodes_per_bin must be positive, got {max_episodes_per_bin}"
            )
        self.multi_ds = multi_ds
        self.max_tokens = max_tokens
        self.overlap = worst_case_lookback_frames()
        self.max_episodes_per_bin = max_episodes_per_bin
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.tokens_per_frame = multi_ds.tokens_per_frame
        self.F_max = max_tokens // self.tokens_per_frame
        if self.F_max <= self.overlap + 1:
            raise ValueError(
                f"F_max={self.F_max} <= overlap+1={self.overlap+1}; max_tokens "
                f"({max_tokens}) too small for tokens_per_frame={self.tokens_per_frame}."
            )

        # Precompute per-segment latent_F so set_epoch can replan cheaply.
        self._segments_meta: list[tuple[int, int, int, int]] = [
            (gid, did, lid, multi_ds.resolve_segment(did, lid)[0].segment_latent_frames(seg))
            for gid, did, lid, seg in multi_ds.iter_segment_metadata()
        ]
        self.set_epoch(epoch)

    def set_epoch(self, epoch: int, start_bin: int = 0) -> None:
        """Rebuild the plan for *epoch*, optionally skipping *start_bin* bins.

        This is the **only** entry point for changing the epoch or resuming
        mid-epoch.  It rebuilds the full plan (chunking + BFD + shuffle +
        rank-shard) and then slices off the first *start_bin* bins so that
        resume does not replay already-consumed data.
        """
        self._build_epoch_plan(epoch)

        if not 0 <= start_bin <= len(self._bins):
            raise ValueError(
                f"start_bin={start_bin} out of range [0, {len(self._bins)}] "
                f"for epoch={self.epoch} on rank={self.rank}"
            )
        self._bins = self._bins[start_bin:]
        if len(self._bins) == 0:
            raise RuntimeError(
                f"After applying start_bin={start_bin}, rank={self.rank} has "
                f"0 bins in epoch={self.epoch}. Advance _packing_epoch and "
                f"call set_epoch with start_bin=0."
            )

    def _build_epoch_plan(self, epoch: int) -> None:
        """Full chunking -> BFD -> shuffle -> rank-shard.  Expensive."""
        self.epoch = epoch
        all_chunks: list[_ChunkRef] = []
        for gid, did, lid, latent_F in self._segments_meta:
            if latent_F <= 0:
                continue
            rng = np.random.default_rng(
                seed=(self.seed * 1_000_003 + epoch) * 1_000_003 + gid
            )
            specs = compute_chunks(
                F=latent_F, F_max=self.F_max, overlap=self.overlap, rng=rng
            )
            for k, spec in enumerate(specs):
                all_chunks.append(_ChunkRef(
                    segment_idx=gid, dataset_idx=did, local_idx=lid,
                    chunk_idx=k, spec=spec,
                    tokens=spec.attn_length * self.tokens_per_frame,
                ))
        self._all_chunks = all_chunks

        bins_global = best_fit_decreasing(
            [c.tokens for c in all_chunks], capacity=self.max_tokens,
            max_episodes=self.max_episodes_per_bin,
        )
        # Deterministic shuffle: break BFD's length-descending order and
        # rotate the trailing N % world_size drop across epochs.
        perm = np.random.default_rng(
            seed=self.seed * 1_000_003 + epoch * 37 + 17
        ).permutation(len(bins_global))
        bins_global = [bins_global[i] for i in perm]

        n_full = (len(bins_global) // self.world_size) * self.world_size
        if n_full == 0:
            raise RuntimeError(
                f"PackingDataset produced only {len(bins_global)} bin(s) for "
                f"world_size={self.world_size} at epoch={epoch}; every rank's "
                f"bin list is empty after trim. Reduce world_size, lower "
                f"packing.max_tokens, or provide more training segments."
            )
        self._bins = bins_global[:n_full][self.rank :: self.world_size]

    def __len__(self) -> int:
        return len(self._bins)

    def __getitem__(self, i: int) -> dict:
        refs = [self._all_chunks[r] for r in self._bins[i]]
        n_ep = len(refs)

        # Vectorised metadata in numpy.
        F_chunks = np.array([c.spec.attn_length for c in refs], dtype=np.int64)
        attn_starts = np.array([c.spec.attn_start for c in refs], dtype=np.int64)
        sup_los = np.array([c.spec.sup_start - c.spec.attn_start for c in refs], dtype=np.int64)
        sup_his = np.array([c.spec.sup_stop  - c.spec.attn_start for c in refs], dtype=np.int64)

        F_pack = int(F_chunks.sum())
        ep_of_frame = np.repeat(np.arange(n_ep, dtype=np.int64), F_chunks)
        chunk_local = np.arange(F_pack, dtype=np.int64) - (np.cumsum(F_chunks) - F_chunks)[ep_of_frame]
        frame_ids_episode = chunk_local + attn_starts[ep_of_frame]
        lmask = (chunk_local >= sup_los[ep_of_frame]) & (chunk_local < sup_his[ep_of_frame])

        # Per-chunk I/O.
        lat_list, act_list, amask_list, text_embs = [], [], [], []
        for cr in refs:
            ds, seg = self.multi_ds.resolve_segment(cr.dataset_idx, cr.local_idx)
            win = ds.load_segment_window(seg, cr.spec.attn_start, cr.spec.attn_stop)
            lat_list.append(win["latents"])
            act_list.append(win["actions"])
            amask_list.append(win["actions_mask"])
            text_embs.append(ds.sample_text_emb(seg))

        return {
            "latents":            torch.cat(lat_list,   dim=1).unsqueeze(0),
            "actions":            torch.cat(act_list,   dim=1).unsqueeze(0),
            "actions_mask":       torch.cat(amask_list, dim=1).unsqueeze(0),
            "latents_mask":       torch.from_numpy(lmask).unsqueeze(0),
            "seq_ids":            torch.from_numpy(ep_of_frame).unsqueeze(0),
            "frame_ids_episode":  torch.from_numpy(frame_ids_episode).unsqueeze(0),
            "text_emb":           torch.stack(text_embs, dim=0),
            "episode_boundaries": torch.from_numpy(F_chunks.astype(np.int32)),
        }


def packed_collate(xs: list[dict]) -> dict:
    """DataLoader collate for ``batch_size=1`` — unwrap the list."""
    if len(xs) != 1:
        raise ValueError(f"PackingDataset expects batch_size=1, got {len(xs)}")
    return xs[0]
