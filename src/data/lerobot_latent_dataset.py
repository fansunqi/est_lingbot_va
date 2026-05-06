# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Latent-based LeRobot dataset for training.

Follows LATENT_DATA_DESIGN.md:
- Does NOT inherit from or call ``LeRobotDataset.__getitem__()``
- Uses only ``LeRobotDatasetMetadata`` + parquet-backed ``hf_dataset``
- Builds segments at runtime from v3.0 metadata
- Loads pre-extracted latents from flat per-camera layout
- Text embeddings cached per-dataset (task_emb / subtask_emb / empty_emb)
"""

import logging
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from scipy.spatial.transform import Rotation as R

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import load_nested_dataset

from .latent_metadata import (
    LATENT_METADATA_STATUS_COMPLETE,
    build_norm_stat_from_raw_stats,
    load_dataset_user_config,
    load_frozen_latent_metadata,
    validate_train_config_against_preprocess,
)
from .segment_builder import MIN_SAMPLED_FRAMES, Segment, build_segments, get_latent_filename

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recursive_find_dataset_roots(directory: str | Path) -> list[Path]:
    """Find all dataset roots under *directory* by locating ``meta/info.json``."""
    directory = Path(directory)
    return sorted(p.parent.parent for p in directory.rglob("meta/info.json"))


def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    relative_rot = first_rot.inv() * rot
    relative_pose = np.concatenate([pose[:, :3] - pose[:1, :3], relative_rot.as_quat()], axis=1)
    return relative_pose


def _infer_raw_action_dims(hf_dataset, action_keys: list[str]) -> list[int]:
    """Return the flattened dimension of each action key from the first row."""
    if len(hf_dataset) == 0:
        raise ValueError("dataset is empty")
    row = hf_dataset[0]
    dims: list[int] = []
    for key in action_keys:
        if key not in row:
            raise ValueError(f"action key {key!r} not found in dataset columns")
        arr = np.asarray(row[key]).flatten()
        if arr.size == 0:
            raise ValueError(f"action key {key!r} has zero-length data")
        dims.append(int(arr.size))
    return dims


# ---------------------------------------------------------------------------
# Per-dataset latent dataset
# ---------------------------------------------------------------------------

class LatentLeRobotDataset:
    """Single-dataset latent loader. Used by ``PackingDataset`` via
    ``load_segment_window`` / ``sample_text_emb``."""

    def __init__(self, root: str | Path, config):
        self.root = Path(root)
        self.cfg_prob: float = config.cfg_prob
        self.patch_size = tuple(config.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(
                f"config.patch_size must be a 3-tuple (F, H, W), got {self.patch_size}"
            )
        self.expected_model_path = (
            str(Path(config.wan22_pretrained_model_name_or_path).resolve())
            if hasattr(config, "wan22_pretrained_model_name_or_path")
            else None
        )
        self.dataset_config = load_dataset_user_config(self.root)
        self.latent_metadata = load_frozen_latent_metadata(self.root)
        if self.dataset_config.preprocess.to_dict() != self.latent_metadata.preprocess.to_dict():
            raise ValueError(
                f"{self.root.name}: meta/wan_va_config.json preprocess section no longer matches "
                "latents/metadata.json. Re-extract latents after changing preprocess settings, or "
                "restore the original preprocess section."
            )
        if self.latent_metadata.extraction_status != LATENT_METADATA_STATUS_COMPLETE:
            raise ValueError(
                f"{self.root.name}: latent extraction is marked as "
                f"{self.latent_metadata.extraction_status!r} in latents/metadata.json. "
                "Rerun src.data.extract_latents until it finishes successfully before training."
            )

        self.preprocess_config = self.latent_metadata.preprocess
        self.train_config = self.dataset_config.training
        validate_train_config_against_preprocess(
            self.preprocess_config,
            self.train_config,
            label=f"{self.root.name}: dataset config",
        )
        self.used_video_keys: list[str] = self.preprocess_config.obs_cam_keys
        self.camera_resolutions = self.preprocess_config.resolved_camera_resolutions()

        self.meta = LeRobotDatasetMetadata(repo_id="local", root=self.root)
        ds_version = self.meta.info.get("codebase_version", "unknown")
        if ds_version != "v3.0":
            raise ValueError(
                f"{self.root.name}: codebase_version='{ds_version}', "
                f"requires v3.0.  Run convert_dataset_v21_to_v30.py first."
            )
        if str(ds_version) != self.latent_metadata.codebase_version:
            raise ValueError(
                f"{self.root.name}: current codebase_version={ds_version!r} no longer matches "
                f"latents/metadata.json ({self.latent_metadata.codebase_version!r}). "
                "Re-extract latents after changing dataset metadata."
            )
        self.frame_stride = self.preprocess_config.frame_stride
        self.time_downsample = self.latent_metadata.vae_temporal_downsample
        self.action_steps_per_latent_frame = self.frame_stride * self.time_downsample
        actual_fps = self.meta.fps / self.frame_stride
        if abs(actual_fps - self.latent_metadata.actual_fps) > 1e-6:
            raise ValueError(
                f"{self.root.name}: current actual_fps={actual_fps} no longer matches "
                f"latents/metadata.json ({self.latent_metadata.actual_fps}). "
                "Re-extract latents after changing dataset fps or preprocess settings."
            )

        features = get_hf_features_from_features(self.meta.features)
        full_hf = load_nested_dataset(self.root / "data", features=features)
        raw_action_dims = _infer_raw_action_dims(full_hf, self.train_config.action_keys)
        self.train_config.validate_runtime_contract(
            raw_action_dims,
            label=f"{self.root.name}: dataset config",
        )

        if self.train_config.norm_stat is not None:
            q01 = np.array(self.train_config.norm_stat["q01"], dtype="float")
            q99 = np.array(self.train_config.norm_stat["q99"], dtype="float")
        else:
            q01, q99 = self._load_norm_stat_from_dataset_stats()
        self.q01 = q01[None]  # (1, action_dim)
        self.q99 = q99[None]  # (1, action_dim)
        if self.train_config.action_dim != config.action_dim:
            raise ValueError(
                f"{self.root.name}: dataset action_dim={self.train_config.action_dim} does not match "
                f"training config action_dim={config.action_dim}"
            )
        if (
            self.expected_model_path is not None
            and self.latent_metadata.model_path is not None
            and self.latent_metadata.model_path != self.expected_model_path
        ):
            logger.warning(
                "%s: latents were extracted with model_path=%s, but training is configured for %s. "
                "This is only a path-level heuristic; re-extract if the underlying VAE weights differ.",
                self.root.name,
                self.latent_metadata.model_path,
                self.expected_model_path,
            )

        self.segments = build_segments(
            episodes=self.meta.episodes,
            hf_dataset=full_hf,
            subtasks=self.meta.subtasks,
            min_segment_frames=self.preprocess_config.min_segment_frames(
                min_sampled_frames=MIN_SAMPLED_FRAMES,
            ),
        )

        # Exclude segments skipped during extraction (corrupt video, etc.).
        if self.latent_metadata.skipped_segments:
            skip_keys = {
                (s["episode_index"], s["start_frame"], s["end_frame"])
                for s in self.latent_metadata.skipped_segments
            }
            before = len(self.segments)
            self.segments = [seg for seg in self.segments if seg.key not in skip_keys]
            logger.warning(
                "%s: excluded %d skipped segment(s) (%d → %d)",
                self.root.name, before - len(self.segments), before, len(self.segments),
            )

        # Action-only view; avoids deep-copying the full dataset inside with_format.
        self._hf_torch_view = (
            full_hf.select_columns(self.train_config.action_keys)
                   .with_format(type="torch")
        )
        del full_hf

        text_emb_dir = self.root / "text_emb"
        self.task_emb = torch.load(text_emb_dir / "task_emb.pth", weights_only=False)
        self.empty_emb = torch.load(text_emb_dir / "empty_emb.pth", weights_only=False)
        subtask_path = text_emb_dir / "subtask_emb.pth"
        self.subtask_emb = torch.load(subtask_path, weights_only=False) if subtask_path.exists() else None
        self.text_shape = tuple(self.task_emb.shape)
        self._validate_runtime_assets()

        logger.info(
            "LatentLeRobotDataset(%s): %d segments, %d tasks",
            self.root.name, len(self.segments), len(self.task_emb),
        )

    def _latent_path_for(self, cam_key: str, seg: Segment) -> Path:
        return (
            self.root / "latents" / cam_key
            / get_latent_filename(seg.episode_index, seg.start_frame, seg.end_frame)
        )

    # ------------------------------------------------------------------
    # Text embedding lookup
    # ------------------------------------------------------------------

    def _get_text_emb(self, seg: Segment) -> torch.Tensor:
        emb = (
            self.subtask_emb[seg.subtask_index]
            if seg.subtask_index is not None
            else self.task_emb[seg.task_index]
        )
        if torch.rand(1).item() < self.cfg_prob:
            emb = self.empty_emb
        return emb

    # ------------------------------------------------------------------
    # Latent loading
    # ------------------------------------------------------------------

    def _load_cat_latents(self, seg: Segment) -> torch.Tensor:
        """Load and concatenate per-camera latents."""
        latent_lst = []
        for key in self.used_video_keys:
            data = torch.load(self._latent_path_for(key, seg), weights_only=False)
            latent = rearrange(
                data["latent"], "(f h w) c -> f h w c",
                f=data["latent_num_frames"],
                h=data["latent_height"],
                w=data["latent_width"],
            )
            latent_lst.append(latent)

        if self.train_config.latent_layout == "robotwin_tshape":
            wrist = torch.cat(latent_lst[1:], dim=2)
            return torch.cat([wrist, latent_lst[0]], dim=1)
        return torch.cat(latent_lst, dim=2)

    def _validate_runtime_assets(self):
        if not self.segments:
            raise ValueError(
                f"{self.root.name}: no segments remain after applying min_segment_frames="
                f"{self.preprocess_config.min_segment_frames(min_sampled_frames=MIN_SAMPLED_FRAMES)}"
            )
        if self.task_emb.ndim != 3:
            raise ValueError(
                f"{self.root.name}: task_emb.pth must have shape (N, seq_len, hidden_dim), "
                f"got {tuple(self.task_emb.shape)}"
            )
        if tuple(self.empty_emb.shape) != tuple(self.task_emb.shape[1:]):
            raise ValueError(
                f"{self.root.name}: empty_emb shape {tuple(self.empty_emb.shape)} does not match "
                f"task_emb tail shape {tuple(self.task_emb.shape[1:])}"
            )

        task_segments = [seg for seg in self.segments if seg.subtask_index is None]
        if task_segments:
            max_task = max(seg.task_index for seg in task_segments)
            if max_task >= len(self.task_emb):
                raise ValueError(
                    f"{self.root.name}: max task_index={max_task} exceeds task_emb size {len(self.task_emb)}"
                )

        subtask_segments = [seg for seg in self.segments if seg.subtask_index is not None]
        if not subtask_segments:
            return
        if self.subtask_emb is None:
            raise ValueError(
                f"{self.root.name}: segments use subtask_index but subtask_emb.pth is missing"
            )
        if tuple(self.subtask_emb.shape[1:]) != tuple(self.task_emb.shape[1:]):
            raise ValueError(
                f"{self.root.name}: subtask_emb tail shape {tuple(self.subtask_emb.shape[1:])} does not match "
                f"task_emb tail shape {tuple(self.task_emb.shape[1:])}"
            )
        max_subtask = max(seg.subtask_index for seg in subtask_segments)
        if max_subtask >= len(self.subtask_emb):
            raise ValueError(
                f"{self.root.name}: max subtask_index={max_subtask} exceeds subtask_emb size "
                f"{len(self.subtask_emb)}"
            )

    # ------------------------------------------------------------------
    # Action normalization stats
    # ------------------------------------------------------------------

    def _load_norm_stat_from_dataset_stats(self) -> tuple[np.ndarray, np.ndarray]:
        """Read per-key q01/q99 from ``meta/stats.json`` (identity transforms only)."""
        from lerobot.datasets.io_utils import load_stats

        stats = load_stats(self.root)
        if stats is None:
            raise ValueError(
                f"{self.root.name}: meta/stats.json is missing. "
                f"Compute dataset statistics before training."
            )
        per_key_q01: list[np.ndarray] = []
        per_key_q99: list[np.ndarray] = []
        for key, sub_map in zip(self.train_config.action_keys, self.train_config.action_key_channel_map):
            if key not in stats:
                raise ValueError(
                    f"{self.root.name}: meta/stats.json has no entry for {key!r}."
                )
            key_stats = stats[key]
            if "q01" not in key_stats or "q99" not in key_stats:
                raise ValueError(
                    f"{self.root.name}: stats for {key!r} missing 'q01'/'q99'."
                )
            q01_k = np.asarray(key_stats["q01"]).flatten()
            q99_k = np.asarray(key_stats["q99"]).flatten()
            expected_dim = len(sub_map)
            if len(q01_k) != expected_dim or len(q99_k) != expected_dim:
                raise ValueError(
                    f"{self.root.name}: stats for {key!r} have q01/q99 length "
                    f"{len(q01_k)}/{len(q99_k)}, expected {expected_dim} "
                    f"(matching used_action_channel_ids sub-list)."
                )
            per_key_q01.append(q01_k)
            per_key_q99.append(q99_k)
        ns = build_norm_stat_from_raw_stats(
            per_key_q01,
            per_key_q99,
            self.train_config.inverse_used_action_channel_ids,
            self.train_config.action_key_channel_map,
        )
        return np.array(ns["q01"], dtype="float"), np.array(ns["q99"], dtype="float")

    # ------------------------------------------------------------------
    # Action post-processing
    # ------------------------------------------------------------------

    def _action_post_process(self, latent_frame_num: int, action):
        if self.train_config.action_transform == "robotwin_relative_pose_bimanual":
            left = get_relative_pose(action[:, :7])
            right = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left, action[:, 7:8], right, action[:, 15:16]], axis=1)

        required = latent_frame_num * self.frame_stride * self.time_downsample
        action = np.pad(action, ((self.frame_stride * self.time_downsample, 0), (0, 0)), mode="constant")
        action = action[:required]
        action_mask = np.ones_like(action, dtype="bool")

        action = np.pad(action, ((0, 0), (0, 1)), mode="constant")
        action_mask = np.pad(action_mask, ((0, 0), (0, 1)), mode="constant")

        inv_ids = self.train_config.inverse_used_action_channel_ids
        action = action[:, inv_ids]
        action_mask = action_mask[:, inv_ids]
        action = (action - self.q01) / (self.q99 - self.q01 + 1e-6) * 2.0 - 1.0
        action = np.clip(action, -1.5, 1.5)
        action = rearrange(action, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask = rearrange(action_mask, "(f n) c -> c f n 1", f=latent_frame_num)
        action *= action_mask
        return torch.from_numpy(action).float(), torch.from_numpy(action_mask).bool()

    # ------------------------------------------------------------------
    # Packing support: tokens-per-frame + windowed segment loader
    # ------------------------------------------------------------------

    @property
    def tokens_per_frame(self) -> int:
        """Transformer tokens per latent frame."""
        from .token_math import tokens_per_frame as _tpf
        H_lat, W_lat = self._concat_latent_hw()
        return _tpf(
            H_lat=H_lat, W_lat=W_lat,
            patch_h=self.patch_size[1],
            patch_w=self.patch_size[2],
            n_action=self.action_steps_per_latent_frame,
        )

    def segment_latent_frames(self, seg: Segment) -> int:
        """On-disk latent frame count for *seg*."""
        n_sampled = (seg.global_to - seg.global_from + self.frame_stride - 1) // self.frame_stride
        return (n_sampled - 1) // self.time_downsample + 1

    def _concat_latent_hw(self) -> tuple[int, int]:
        """``(H_lat, W_lat)`` of the concatenated latent."""
        from .latent_metadata import DEFAULT_VAE_SPATIAL_DOWNSAMPLE as DS
        resolved = self.preprocess_config.resolved_camera_resolutions()
        latent_hws = [(H // DS, W // DS) for H, W in
                      (resolved[k] for k in self.used_video_keys)]
        if self.train_config.latent_layout == "robotwin_tshape":
            primary_H, primary_W = latent_hws[0]
            return primary_H + latent_hws[1][0], primary_W
        H_lat = latent_hws[0][0]
        W_lat = sum(w for _, w in latent_hws)
        return H_lat, W_lat

    def load_segment_window(
        self,
        seg: Segment,
        attn_start: int,
        attn_stop: int,
    ) -> dict:
        """Load latent-frame window ``[attn_start, attn_stop)`` of *seg*.

        Actions are post-processed over the *full* segment before slicing,
        so reference-frame-dependent transforms (e.g. relative pose) stay
        anchored to the true episode start.
        """
        cat_latent = self._load_cat_latents(seg)  # (F_lat, H, W, C)
        latent_F = cat_latent.shape[0]
        if not (0 <= attn_start < attn_stop <= latent_F):
            raise ValueError(
                f"load_segment_window: bad range [{attn_start}, {attn_stop}) "
                f"for segment with latent_F={latent_F}"
            )

        row_slice = self._hf_torch_view[seg.global_from : seg.global_to]
        n_rows = seg.global_to - seg.global_from
        parts = [row_slice[key].numpy().reshape(n_rows, -1)
                 for key in self.train_config.action_keys]
        action = np.concatenate(parts, axis=1)
        actions_full, actions_mask_full = self._action_post_process(latent_F, action)
        return {
            "latents": cat_latent[attn_start:attn_stop].permute(3, 0, 1, 2),  # C F H W
            "actions": actions_full[:, attn_start:attn_stop],
            "actions_mask": actions_mask_full[:, attn_start:attn_stop],
        }

    def sample_text_emb(self, seg: Segment) -> torch.Tensor:
        """Draw one text embedding for *seg* (with CFG dropout)."""
        return self._get_text_emb(seg)


# ---------------------------------------------------------------------------
# Multi-dataset wrapper
# ---------------------------------------------------------------------------

class MultiLatentLeRobotDataset:
    """Combines multiple ``LatentLeRobotDataset`` instances."""

    def __init__(self, config):
        dataset_roots = recursive_find_dataset_roots(config.dataset_path)
        if not dataset_roots:
            raise FileNotFoundError(
                f"No datasets found under {config.dataset_path} "
                "(looking for meta/info.json)"
            )

        self._datasets = [LatentLeRobotDataset(root, config) for root in dataset_roots]

        self._validate_batch_compatibility()
        total_segments = sum(len(ds.segments) for ds in self._datasets)
        logger.info(
            "MultiLatentLeRobotDataset: %d datasets, %d total segments",
            len(self._datasets), total_segments,
        )

    def _validate_batch_compatibility(self):
        if len(self._datasets) <= 1:
            return

        ref = self._datasets[0]
        for ds in self._datasets[1:]:
            if ds.used_video_keys != ref.used_video_keys:
                raise ValueError(
                    f"Camera key mismatch: {ref.root.name}={ref.used_video_keys} "
                    f"vs {ds.root.name}={ds.used_video_keys}\n"
                    f"  Mixed datasets must share the same obs_cam_keys in the same order."
                )
            if ds.preprocess_config.camera_preset != ref.preprocess_config.camera_preset:
                raise ValueError(
                    f"camera_preset mismatch: {ref.root.name}={ref.preprocess_config.camera_preset!r} "
                    f"vs {ds.root.name}={ds.preprocess_config.camera_preset!r}"
                )
            if ds.train_config.latent_layout != ref.train_config.latent_layout:
                raise ValueError(
                    f"latent_layout mismatch: {ref.root.name}='{ref.train_config.latent_layout}' "
                    f"vs {ds.root.name}='{ds.train_config.latent_layout}'"
                )
            if ds.train_config.action_transform != ref.train_config.action_transform:
                raise ValueError(
                    f"action_transform mismatch: {ref.root.name}='{ref.train_config.action_transform}' "
                    f"vs {ds.root.name}='{ds.train_config.action_transform}'"
                )
            if ds.train_config.inverse_used_action_channel_ids != ref.train_config.inverse_used_action_channel_ids:
                raise ValueError(
                    f"Action channel mapping mismatch: {ref.root.name} vs {ds.root.name} "
                    f"(inverse_used_action_channel_ids differ)"
                )
            if ds.latent_metadata.vae_temporal_downsample != ref.latent_metadata.vae_temporal_downsample:
                raise ValueError(
                    f"vae_temporal_downsample mismatch: {ref.root.name}="
                    f"{ref.latent_metadata.vae_temporal_downsample} vs {ds.root.name}="
                    f"{ds.latent_metadata.vae_temporal_downsample}"
                )
            if ds.latent_metadata.model_path != ref.latent_metadata.model_path:
                logger.warning(
                    "latent model_path mismatch: %s=%r vs %s=%r. "
                    "This is only a path-level heuristic; mixed training remains allowed.",
                    ref.root.name,
                    ref.latent_metadata.model_path,
                    ds.root.name,
                    ds.latent_metadata.model_path,
                )
            if ds.action_steps_per_latent_frame != ref.action_steps_per_latent_frame:
                raise ValueError(
                    f"action steps per latent frame mismatch: {ref.root.name}={ref.action_steps_per_latent_frame} "
                    f"vs {ds.root.name}={ds.action_steps_per_latent_frame}. "
                    "Mixed datasets must agree on frame_stride * vae_temporal_downsample for batching."
                )
            if tuple(ds.text_shape[1:]) != tuple(ref.text_shape[1:]):
                raise ValueError(
                    f"text_emb shape mismatch: {ref.root.name}={tuple(ref.text_shape[1:])} "
                    f"vs {ds.root.name}={tuple(ds.text_shape[1:])}"
                )

    # ------------------------------------------------------------------
    # Packing support
    # ------------------------------------------------------------------

    @property
    def tokens_per_frame(self) -> int:
        """Token count per latent frame (constant across sub-datasets)."""
        tpfs = {ds.tokens_per_frame for ds in self._datasets}
        if len(tpfs) != 1:
            raise RuntimeError(f"Sub-datasets disagree on tokens_per_frame: {tpfs}")
        return next(iter(tpfs))

    def iter_segment_metadata(self):
        """Yield ``(global_idx, dataset_idx, local_idx, seg)`` for every segment."""
        global_idx = 0
        for did, ds in enumerate(self._datasets):
            for local_idx, seg in enumerate(ds.segments):
                yield global_idx, did, local_idx, seg
                global_idx += 1

    def resolve_segment(self, dataset_idx: int, local_idx: int):
        """Return ``(sub_dataset, segment)`` for a metadata reference."""
        ds = self._datasets[dataset_idx]
        return ds, ds.segments[local_idx]
