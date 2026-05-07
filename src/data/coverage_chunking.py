"""Coverage-preserving random chunking of long episodes.

When an episode's frame count exceeds ``F_max = max_tokens //
tokens_per_frame``, :func:`compute_chunks` splits it into overlapping
chunks such that

1. the *supervised spans* of the chunks partition ``[0, F)`` exactly — no
   frame is dropped, none is counted twice in the loss,
2. every chunk after the first carries an ``overlap``-frame context prefix
   that is KV-visible but not supervised (matches the training-time
   attention lookback ``⌊window_size_max/2⌋ * chunk_size_max``), and
3. cut points are re-randomized per epoch so the model is not biased
   towards a fixed set of boundaries.

Chunk 0 has no context prefix: it starts at the true episode start.
Context frames are loss-masked via ``latents_mask`` / ``actions_mask``.
"""

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class ChunkSpec:
    """One chunk carved out of a long episode, in episode-frame half-open indices.

    Invariants (checked by :func:`compute_chunks`):

    * ``0 <= attn_start <= sup_start < sup_stop == attn_stop <= F``
    * ``attn_stop - attn_start <= F_max``
    * chunk 0: ``attn_start == sup_start == 0``
    * chunk ``i >= 1``: ``attn_start == max(0, sup_start - overlap)``
    """

    attn_start: int
    attn_stop: int
    sup_start: int
    sup_stop: int

    @property
    def attn_length(self) -> int:
        return self.attn_stop - self.attn_start

    @property
    def sup_length(self) -> int:
        return self.sup_stop - self.sup_start


def compute_chunks(
    F: int,
    F_max: int,
    overlap: int,
    rng: np.random.Generator,
) -> list[ChunkSpec]:
    """Split an episode of ``F`` frames into coverage-preserving chunks.

    The supervised ranges of the returned chunks partition ``[0, F)``;
    each chunk's attention range is at most ``F_max`` frames wide, and for
    chunks after the first includes an ``overlap``-frame context prefix.

    Args:
        F: episode length in frames. Must be positive.
        F_max: hard cap on any chunk's attention window
            (= ``packing.max_tokens // tokens_per_frame``). Must exceed
            ``overlap`` so each chunk has at least one supervised frame.
        overlap: context-prefix length for chunks after the first.
        rng: RNG seeded deterministically from
            ``(seed, epoch, segment_global_idx)`` by the caller.
    """
    if F <= 0:
        raise ValueError(f"Episode length must be positive, got F={F}")
    if F_max <= overlap + 1:
        raise ValueError(
            f"F_max={F_max} must exceed overlap+1={overlap + 1}; "
            "packing.max_tokens is too small for the current tokens-per-frame."
        )

    if F <= F_max:
        return [ChunkSpec(0, F, 0, F)]

    # Supervised span per non-first chunk. Chunk 0 can supervise up to F_max.
    S = F_max - overlap
    N_rest = int(ceil((F - F_max) / S))
    N = 1 + N_rest

    # Pick N-1 random cut points under per-chunk capacity constraints.
    max_jitter = max(1, S // 4)
    cuts: list[int] = []
    for i in range(1, N):
        prev = cuts[-1] if cuts else 0
        gap_max = F_max if i == 1 else S
        lo = max(F - (N - i) * S, prev + 1, 1)
        hi = min(F_max + (i - 1) * S, prev + gap_max, F - 1)
        if lo > hi:
            raise RuntimeError(
                f"compute_chunks: no valid cut range at i={i} "
                f"(lo={lo}, hi={hi}, F={F}, F_max={F_max}, overlap={overlap})"
            )
        center = int(round(i * F / N))
        c = int(np.clip(center + rng.integers(-max_jitter, max_jitter + 1), lo, hi))
        cuts.append(c)

    chunks: list[ChunkSpec] = []
    for i in range(N):
        sup_start = 0 if i == 0 else cuts[i - 1]
        sup_stop = F if i == N - 1 else cuts[i]
        attn_start = 0 if i == 0 else max(0, sup_start - overlap)
        attn_stop = sup_stop  # tight: sup_stop - attn_start <= S + overlap = F_max
        assert attn_stop - attn_start <= F_max
        chunks.append(ChunkSpec(attn_start, attn_stop, sup_start, sup_stop))

    # Verify supervised partition of [0, F).
    covered = np.zeros(F, dtype=bool)
    for ch in chunks:
        if ch.sup_start >= ch.sup_stop:
            raise RuntimeError(f"Empty supervised range: {ch}")
        if covered[ch.sup_start:ch.sup_stop].any():
            raise RuntimeError(f"Supervised ranges overlap around {ch}")
        covered[ch.sup_start:ch.sup_stop] = True
    if not covered.all():
        missing = np.where(~covered)[0].tolist()
        raise RuntimeError(
            f"compute_chunks failed coverage for F={F}, F_max={F_max}, "
            f"overlap={overlap}: frames {missing[:10]} uncovered"
        )

    return chunks
