"""Best-fit-decreasing bin packing with a dual token / episode budget.

Deterministic: ranks recompute the plan from ``(seed, epoch)`` without
``dist.broadcast``, so bins and item order must be identical everywhere.

Each bin has two capacities — *tokens* (bounded by ``max_tokens``) and
*episodes* (bounded by ``max_episodes``).  Tokens drive best-fit
placement; episode count is a lex tiebreak among equally-tight bins and
a hard cap aligned with ``find_max_seq_len --probe-n-episodes``.

Complexity: O(N log N) sort + O(N M) placement (bisect is O(log M);
list splice is O(M) via C memmove).  Exhausted bins are retired from
the shelf, keeping M bounded to those still accepting items.
"""

from bisect import bisect_left, insort


def best_fit_decreasing(
    lengths: list[int],
    capacity: int,
    *,
    max_episodes: int | None = None,
) -> list[list[int]]:
    """Pack *lengths* into bins of *capacity* by best-fit-decreasing.

    Items are sorted by ``(-length, index)``.  Each is placed in the
    open bin that minimises ``(remaining_after, episodes_after)`` lex,
    or a fresh bin if none fits.

    An internal shelf sorted by ``(remaining, n_episodes, bin_id)``
    gives O(log M) lookup via ``bisect_left``.  Bins at capacity or at
    ``max_episodes`` are retired so they never slow subsequent lookups.

    All *lengths* must satisfy ``0 < length <= capacity``.
    """
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity}")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError(f"max_episodes must be positive, got {max_episodes}")
    for i, length in enumerate(lengths):
        if length <= 0:
            raise ValueError(f"lengths[{i}]={length} must be positive")
        if length > capacity:
            raise ValueError(
                f"lengths[{i}]={length} exceeds capacity={capacity}; "
                "chunking should have kept every chunk within budget."
            )

    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    bins: list[list[int]] = []
    shelf: list[tuple[int, int, int]] = []  # (remaining, n_episodes, bin_id)

    def _shelve(rem: int, n_ep: int, b: int) -> None:
        if rem > 0 and (max_episodes is None or n_ep < max_episodes):
            insort(shelf, (rem, n_ep, b))

    for idx in order:
        L = lengths[idx]
        pos = bisect_left(shelf, (L,))
        if pos < len(shelf):
            rem, n_ep, b = shelf.pop(pos)
            bins[b].append(idx)
            _shelve(rem - L, n_ep + 1, b)
        else:
            bins.append([idx])
            _shelve(capacity - L, 1, len(bins) - 1)

    return bins
