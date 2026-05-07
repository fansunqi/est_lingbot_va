"""Shared segment building logic for extraction and training."""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

MIN_SAMPLED_FRAMES = 2

IndexColumns = Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]


@dataclass
class Segment:
    episode_index: int
    start_frame: int
    end_frame: int          # exclusive
    global_from: int        # parquet row index, inclusive
    global_to: int          # parquet row index, exclusive
    task_index: int
    subtask_index: Optional[int] = None

    @property
    def key(self) -> tuple[int, int, int]:
        """Identity triple used for latent file naming and skip-record matching."""
        return (self.episode_index, self.start_frame, self.end_frame)


def read_index_columns(hf_dataset) -> IndexColumns:
    """Read ``(frame_index, task_index, subtask_index?)`` as numpy arrays.

    Uses the Arrow table directly for speed.  Raises if the dataset has been
    filtered/shuffled/selected (``_indices`` present).
    """
    if getattr(hf_dataset, "_indices", None) is not None:
        raise NotImplementedError(
            "read_index_columns does not support datasets with _indices mapping; "
            "call flatten_indices() first or read columns manually."
        )
    cols = [c for c in ("frame_index", "task_index", "subtask_index") if c in hf_dataset.column_names]
    table = hf_dataset.select_columns(cols).data
    return (
        table.column("frame_index").to_numpy(),
        table.column("task_index").to_numpy(),
        table.column("subtask_index").to_numpy() if "subtask_index" in cols else None,
    )


def build_segments(
    episodes,
    hf_dataset=None,
    subtasks=None,
    min_segment_frames: int = 0,
    *,
    index_columns: Optional[IndexColumns] = None,
) -> list[Segment]:
    """Build training segments from LeRobot v3.0 metadata.

    Pass *index_columns* (from :func:`read_index_columns`) to reuse
    pre-computed arrays; otherwise *hf_dataset* is read on the fly.
    """
    if index_columns is not None:
        frame_col, task_col, subtask_col = index_columns
    elif hf_dataset is not None:
        frame_col, task_col, subtask_col = read_index_columns(hf_dataset)
    else:
        raise ValueError("Either hf_dataset or index_columns must be provided")

    has_subtask = (
        subtasks is not None
        and len(subtasks) > 0
        and subtask_col is not None
    )

    segments: list[Segment] = []

    for row in episodes:
        ep_idx = row["episode_index"]
        ds_from = row["dataset_from_index"]
        ds_to = row["dataset_to_index"]
        if ds_to <= ds_from:
            continue

        ep_frames = frame_col[ds_from:ds_to]
        ep_tasks = task_col[ds_from:ds_to]

        if has_subtask:
            assert subtask_col is not None
            ep_subtasks = subtask_col[ds_from:ds_to]
            new_segs = _split_by_subtask(
                ep_idx, ep_frames, ep_tasks, ep_subtasks, ds_from
            )
        else:
            new_segs = [
                Segment(
                    episode_index=ep_idx,
                    start_frame=int(ep_frames[0]),
                    end_frame=int(ep_frames[-1]) + 1,
                    global_from=ds_from,
                    global_to=ds_to,
                    task_index=int(ep_tasks[0]),
                )
            ]

        segments.extend(
            seg for seg in new_segs
            if (seg.end_frame - seg.start_frame) >= min_segment_frames
        )
    return segments


def _split_by_subtask(
    episode_index: int,
    frames: np.ndarray,
    tasks: np.ndarray,
    subtasks: np.ndarray,
    dataset_from: int,
) -> list[Segment]:
    """Split one episode into segments by contiguous subtask runs."""
    change_points = np.where(np.diff(subtasks) != 0)[0] + 1
    run_starts = np.concatenate([[0], change_points])
    run_ends = np.concatenate([change_points, [len(subtasks)]])

    segs = []
    for rs, re in zip(run_starts, run_ends):
        segs.append(Segment(
            episode_index=episode_index,
            start_frame=int(frames[rs]),
            end_frame=int(frames[re - 1]) + 1,
            global_from=dataset_from + int(rs),
            global_to=dataset_from + int(re),
            task_index=int(tasks[rs]),
            subtask_index=int(subtasks[rs]),
        ))
    return segs


def get_latent_filename(episode_index: int, start_frame: int, end_frame: int) -> str:
    """Canonical latent filename for a segment."""
    return f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
