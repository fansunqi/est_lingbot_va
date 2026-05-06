"""Shared token-count math for the trainer and ``find_max_seq_len``.

Single source of truth for ``tokens_per_frame`` — the offline probe
(calibrating ``packing.max_tokens``) and the online packer both call here.

Formula::

    tpf = 2 * (H_lat / p_h) * (W_lat / p_w) + 2 * n_action

The factor 2 accounts for ``forward_train`` concatenating a noisy copy and
a clean conditioning copy along the sequence dimension.
"""


# Training-time randomisation bounds (randint(1,5), randint(4,65)).
# Widening them grows worst_case_lookback_frames → larger overlap →
# may need a re-probe of packing.max_tokens.
TRAIN_CHUNK_SIZE_MAX: int = 4
TRAIN_WINDOW_SIZE_MAX: int = 64


def tokens_per_frame(
    *,
    H_lat: int,
    W_lat: int,
    patch_h: int,
    patch_w: int,
    n_action: int,
) -> int:
    """Transformer tokens contributed by one latent frame (see module doc)."""
    if H_lat % patch_h != 0 or W_lat % patch_w != 0:
        raise ValueError(
            f"tokens_per_frame: H_lat={H_lat} / patch_h={patch_h} or "
            f"W_lat={W_lat} / patch_w={patch_w} not an integer"
        )
    return 2 * (H_lat // patch_h) * (W_lat // patch_w) + 2 * n_action


def worst_case_lookback_frames(
    *,
    chunk_size_max: int = TRAIN_CHUNK_SIZE_MAX,
    window_size_max: int = TRAIN_WINDOW_SIZE_MAX,
) -> int:
    """Largest frame gap a query can still attend to.

    ``_get_mask_mod`` maps ``frame_id = frame // chunk_size * 2`` and keeps
    ``|Δframe_id| <= window_size``, so the worst-case real-frame reach is
    ``(window_size_max // 2) * chunk_size_max + (chunk_size_max - 1)``.
    """
    return (window_size_max // 2) * chunk_size_max + (chunk_size_max - 1)
