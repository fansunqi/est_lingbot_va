# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Shared segment building logic for extraction and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

MIN_SAMPLED_FRAMES = 2  # need >= 2 sampled frames so the segment is non-degenerate after striding


@dataclass
class Segment:
    episode_index: int
    start_frame: int       # episode-local frame index (== row offset, dense)
    end_frame: int         # exclusive upper bound
    global_from: int       # parquet row index — start (inclusive)
    global_to: int         # parquet row index — end   (exclusive)
    task_index: int
    subtask_index: Optional[int] = None


def build_segments(
    episodes,
    hf_dataset,
    subtasks=None,
    min_segment_frames: int = 0,
) -> list[Segment]:
    """Build training segments from LeRobot v3.0 metadata.

    Parameters
    ----------
    episodes : datasets.Dataset
        ``meta.episodes`` — must have ``episode_index``,
        ``dataset_from_index``, ``dataset_to_index``.
    hf_dataset : datasets.Dataset
        Parquet data — must have ``frame_index``, ``task_index``, and
        optionally ``subtask_index``.
    subtasks : pandas.DataFrame | None
        When present and ``subtask_index`` exists in *hf_dataset*, episodes
        are split by contiguous runs of the same subtask.
    min_segment_frames : int
        Drop segments shorter than this.
    """
    has_subtask = (
        subtasks is not None
        and len(subtasks) > 0
        and "subtask_index" in hf_dataset.column_names
    )

    columns = ["frame_index", "task_index"]
    if has_subtask:
        columns.append("subtask_index")

    index_view = hf_dataset.select_columns(columns)
    frame_col = np.asarray(index_view["frame_index"])
    task_col = np.asarray(index_view["task_index"])
    subtask_col = np.asarray(index_view["subtask_index"]) if has_subtask else None

    segments: list[Segment] = []

    for row in episodes:
        ep_idx = row["episode_index"]
        ds_from = row["dataset_from_index"]
        ds_to = row["dataset_to_index"]
        if ds_to <= ds_from:
            continue

        ep_frames = frame_col[ds_from:ds_to]
        ep_tasks = task_col[ds_from:ds_to]
        ep_subtasks = subtask_col[ds_from:ds_to] if has_subtask else None

        if has_subtask and ep_subtasks is not None:
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
