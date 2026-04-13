# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""
Latent extraction and text-embedding caching.

Reads a v3.0 LeRobot dataset plus ``meta/wan_va_config.json``, encodes video
segments through the WAN VAE, and writes::

    dataset_root/
    ├── latents/{cam_key}/episode_{ep:06d}_{start}_{end}.pth
    └── text_emb/{task_emb,subtask_emb,empty_emb}.pth

Usage::

    python -m wan_va.dataset.extract_latents \\
        --dataset-root /path/to/v30_dataset \\
        --model-path   /path/to/pretrained_wan

Uses the shared ``build_segments`` logic so extraction boundaries are
guaranteed to match training-time expectations.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import load_nested_dataset

from wan_va.dataset.latent_metadata import (
    validate_train_config_against_preprocess,
    LATENT_METADATA_FILENAME,
    FrozenLatentMetadata,
    LATENT_METADATA_STATUS_COMPLETE,
    LATENT_METADATA_STATUS_EXTRACTING,
    load_dataset_user_config,
    load_frozen_latent_metadata,
    write_frozen_latent_metadata,
)
from wan_va.dataset.segment_builder import MIN_SAMPLED_FRAMES, Segment, build_segments, get_latent_filename
from wan_va.modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_vae,
)

logger = logging.getLogger(__name__)


def _save_torch_atomic(obj, path: Path):
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)


# ---------------------------------------------------------------------------
# Video frame decoding
# ---------------------------------------------------------------------------

def decode_video_segment(
    lerobot_ds: LeRobotDataset,
    cam_key: str,
    seg: Segment,
    stride: int,
    frame_index_col: np.ndarray,
    height: int,
    width: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Decode and sub-sample video frames for a segment.

    Returns ``(video, frame_ids)`` where *video* is ``(1, C, T, H, W)``
    in ``[-1, 1]`` and *frame_ids* are the actual ``frame_index`` values
    from the parquet data for the sampled rows.
    """
    row_offsets = list(range(0, seg.global_to - seg.global_from, stride))

    # TODO: batch decoding via video timestamps would be faster.
    frames = []
    for off in row_offsets:
        img = lerobot_ds[seg.global_from + off][cam_key]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        frames.append(img)

    video = torch.stack(frames, dim=1).unsqueeze(0)  # (1, C, T, H, W)
    if video.shape[-2] != height or video.shape[-1] != width:
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
        video = video.reshape(B, T, C, height, width).permute(0, 2, 1, 3, 4)
    video = video * 2.0 - 1.0

    frame_ids = np.array(
        [int(frame_index_col[seg.global_from + off]) for off in row_offsets],
        dtype=np.int64,
    )
    return video, frame_ids


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_video(vae, video: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode ``(1, C, T, H, W)`` → normalised latent ``(1, z_dim, T', H', W')``."""
    video = video.to(device=device, dtype=dtype)
    enc_out = WanVAEStreamingWrapper(vae).encode_chunk(video)
    mu, _ = torch.chunk(enc_out, 2, dim=1)
    mean = torch.tensor(vae.config.latents_mean).to(mu.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).to(mu.device).view(1, -1, 1, 1, 1)
    return ((mu.float() - mean) / std).to(dtype)


# ---------------------------------------------------------------------------
# Text embedding
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_texts(
    tokenizer, text_encoder, texts: list[str],
    device: torch.device, dtype: torch.dtype, max_seq_len: int = 512,
) -> torch.Tensor:
    """Encode texts → ``(N, max_seq_len, hidden_dim)``."""
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean

    inputs = tokenizer(
        [prompt_clean(t) for t in texts],
        padding="max_length", max_length=max_seq_len, truncation=True,
        add_special_tokens=True, return_attention_mask=True, return_tensors="pt",
    )
    enc_device = next(text_encoder.parameters()).device
    embeds = text_encoder(
        inputs.input_ids.to(enc_device), inputs.attention_mask.to(enc_device),
    ).last_hidden_state

    seq_lens = inputs.attention_mask.gt(0).sum(dim=1).long()
    result = []
    for emb, slen in zip(embeds, seq_lens):
        padded = torch.cat([emb[:slen], emb.new_zeros(max_seq_len - slen, emb.size(1))])
        result.append(padded)
    return torch.stack(result).to(dtype=dtype, device="cpu")


def _extract_indexed_texts(df, index_col: str, text_col: str | None, label: str) -> list[str]:
    """Extract texts ordered by a dense 0..N-1 index column.

    Handles text in an explicit column or in the DataFrame index.
    Validates index contiguity.
    """
    if index_col not in df.columns:
        raise ValueError(f"{label}: missing column '{index_col}'")

    indices = sorted(int(x) for x in df[index_col])
    if indices != list(range(len(indices))):
        raise ValueError(
            f"{label}: {index_col} not contiguous 0..{len(indices)-1}: {indices[:10]}"
        )

    if text_col is not None and text_col in df.columns:
        sorted_df = df.sort_values(index_col)
        return [str(sorted_df.loc[sorted_df[index_col] == i, text_col].iloc[0]) for i in indices]
    elif pd.api.types.is_string_dtype(df.index) and df.index.name != index_col:
        # Text lives in the index (tasks name it explicitly; subtasks may not).
        return [str(df.index[df[index_col] == i][0]) for i in indices]
    else:
        raise ValueError(
            f"{label}: no text found.  columns={df.columns.tolist()}, "
            f"index.name={df.index.name!r}, index.dtype={df.index.dtype}"
        )


def _load_v30_dataset(dataset_root: Path) -> tuple[LeRobotDatasetMetadata, object]:
    meta = LeRobotDatasetMetadata(repo_id="local", root=dataset_root)
    ds_version = str(meta.info.get("codebase_version", "unknown"))
    if ds_version != "v3.0":
        raise ValueError(
            f"{dataset_root.name}: codebase_version={ds_version!r}, requires v3.0. "
            "Convert the dataset before extracting latents."
        )

    features = get_hf_features_from_features(meta.features)
    hf_dataset = load_nested_dataset(dataset_root / "data", features=features)
    return meta, hf_dataset


def _build_frozen_metadata(
    model_path: Path,
    preprocess_cfg,
    meta: LeRobotDatasetMetadata,
    *,
    extraction_status: str,
) -> FrozenLatentMetadata:
    stride = preprocess_cfg.frame_stride(meta.fps)
    return FrozenLatentMetadata(
        preprocess=preprocess_cfg,
        codebase_version=str(meta.info["codebase_version"]),
        actual_fps=meta.fps / stride,
        extraction_status=extraction_status,
        model_path=str(model_path.resolve()),
    )


def _ensure_matching_frozen_metadata(
    dataset_root: Path,
    current_metadata: FrozenLatentMetadata,
):
    metadata_path = dataset_root / LATENT_METADATA_FILENAME
    latent_base = dataset_root / "latents"
    if not metadata_path.exists():
        existing_latents = list(latent_base.rglob("*.pth")) if latent_base.exists() else []
        if existing_latents:
            raise RuntimeError(
                f"{dataset_root.name}: found {len(existing_latents)} pre-existing latent files under "
                f"{latent_base} but {metadata_path} is missing. Delete {latent_base} and re-extract "
                "to produce a self-consistent latent dataset."
            )
        return metadata_path

    stored_metadata = load_frozen_latent_metadata(dataset_root)
    if stored_metadata.preprocess.to_dict() != current_metadata.preprocess.to_dict():
        raise RuntimeError(
            f"Preprocess config differs from the previous extraction at {metadata_path}.\n"
            f"  Stored : {stored_metadata.preprocess.to_dict()}\n"
            f"  Current: {current_metadata.preprocess.to_dict()}\n"
            f"Delete {latent_base} and re-extract from scratch after updating meta/wan_va_config.json."
        )
    if stored_metadata.codebase_version != current_metadata.codebase_version:
        raise RuntimeError(
            f"Dataset codebase_version changed from {stored_metadata.codebase_version!r} "
            f"to {current_metadata.codebase_version!r}. Delete {latent_base} and re-extract."
        )
    if abs(stored_metadata.actual_fps - current_metadata.actual_fps) > 1e-6:
        raise RuntimeError(
            f"Stored actual_fps={stored_metadata.actual_fps} does not match "
            f"current actual_fps={current_metadata.actual_fps}. Delete {latent_base} and re-extract."
        )
    if stored_metadata.vae_temporal_downsample != current_metadata.vae_temporal_downsample:
        raise RuntimeError(
            f"Stored vae_temporal_downsample={stored_metadata.vae_temporal_downsample} does not match "
            f"current value {current_metadata.vae_temporal_downsample}. Delete {latent_base} and re-extract."
        )
    if stored_metadata.extraction_status == LATENT_METADATA_STATUS_EXTRACTING:
        logger.info(
            "%s: found in-progress latent metadata at %s; resuming extraction.",
            dataset_root.name, metadata_path,
        )
    if stored_metadata.model_path not in (None, current_metadata.model_path):
        logger.warning(
            "model_path changed since last extraction.\n"
            "  Stored : %s\n  Current: %s\n"
            "Latents were extracted with a different VAE — delete %s to re-extract.",
            stored_metadata.model_path, current_metadata.model_path, latent_base,
        )
    return metadata_path


def _segment_output_paths(latent_base: Path, cam_keys: list[str], seg: Segment) -> list[Path]:
    filename = get_latent_filename(seg.episode_index, seg.start_frame, seg.end_frame)
    return [latent_base / cam_key / filename for cam_key in cam_keys]


def _validate_segment_latents(
    dataset_root: Path,
    seg: Segment,
    cam_key: str,
    frame_ids: np.ndarray,
    latent: torch.Tensor,
    time_downsample: int,
    ref_frame_ids: np.ndarray | None,
    ref_latent_frames: int | None,
) -> tuple[np.ndarray, int]:
    if len(frame_ids) < MIN_SAMPLED_FRAMES:
        raise ValueError(
            f"{dataset_root.name}: segment ep={seg.episode_index} [{seg.start_frame}, {seg.end_frame}) "
            f"produced only {len(frame_ids)} sampled frames"
        )

    latent_frames = int(latent.shape[1])
    expected_latent_frames = (len(frame_ids) - 1) // time_downsample + 1
    if latent_frames != expected_latent_frames:
        raise ValueError(
            f"{dataset_root.name}: {cam_key} ep={seg.episode_index} [{seg.start_frame}, {seg.end_frame}) "
            f"latent_num_frames={latent_frames} but expected {expected_latent_frames} from frame_ids"
        )

    if ref_frame_ids is None:
        return frame_ids, latent_frames
    if not np.array_equal(frame_ids, ref_frame_ids):
        raise ValueError(
            f"{dataset_root.name}: frame_ids mismatch across cameras for ep={seg.episode_index} "
            f"[{seg.start_frame}, {seg.end_frame})"
        )
    if latent_frames != ref_latent_frames:
        raise ValueError(
            f"{dataset_root.name}: latent_num_frames mismatch across cameras for ep={seg.episode_index} "
            f"[{seg.start_frame}, {seg.end_frame})"
        )
    return ref_frame_ids, ref_latent_frames


def _validate_dataset_convention(
    dataset_root: Path,
    meta: LeRobotDatasetMetadata,
    hf_dataset,
    preprocess_cfg,
):
    missing_cam_keys = [cam_key for cam_key in preprocess_cfg.obs_cam_keys if cam_key not in meta.features]
    if missing_cam_keys:
        raise ValueError(
            f"{dataset_root.name}: obs_cam_keys not found in dataset features: {missing_cam_keys}"
        )

    has_subtask = (
        meta.subtasks is not None
        and len(meta.subtasks) > 0
        and "subtask_index" in hf_dataset.column_names
    )

    columns = ["frame_index", "task_index"]
    if has_subtask:
        columns.append("subtask_index")
    index_view = hf_dataset.select_columns(columns)
    frame_col = np.asarray(index_view["frame_index"])
    task_col = np.asarray(index_view["task_index"])
    subtask_col = np.asarray(index_view["subtask_index"]) if has_subtask else None

    if len(meta.tasks) == 0:
        raise ValueError(f"{dataset_root.name}: meta/tasks is empty")
    if np.any(task_col < 0) or np.any(task_col >= len(meta.tasks)):
        bad = np.unique(task_col[(task_col < 0) | (task_col >= len(meta.tasks))]).tolist()
        raise ValueError(
            f"{dataset_root.name}: task_index values out of bounds for meta/tasks ({len(meta.tasks)} rows): {bad}"
        )

    if has_subtask and subtask_col is not None:
        if np.any(subtask_col < 0) or np.any(subtask_col >= len(meta.subtasks)):
            bad = np.unique(
                subtask_col[(subtask_col < 0) | (subtask_col >= len(meta.subtasks))]
            ).tolist()
            raise ValueError(
                f"{dataset_root.name}: subtask_index values out of bounds for meta/subtasks "
                f"({len(meta.subtasks)} rows): {bad}"
            )

    for row in meta.episodes:
        ep_idx = row["episode_index"]
        ds_from = row["dataset_from_index"]
        ds_to = row["dataset_to_index"]
        if ds_to <= ds_from:
            continue

        ep_frames = frame_col[ds_from:ds_to]
        expected = np.arange(ds_to - ds_from)
        if not np.array_equal(ep_frames, expected):
            raise ValueError(
                f"{dataset_root.name}: episode {ep_idx} frame_index is not dense 0-based "
                f"(got {ep_frames[:10].tolist()})"
            )

        if not has_subtask:
            ep_tasks = task_col[ds_from:ds_to]
            if not np.all(ep_tasks == ep_tasks[0]):
                raise ValueError(
                    f"{dataset_root.name}: episode {ep_idx} task_index is not constant without subtask_index "
                    f"(values={np.unique(ep_tasks).tolist()})"
                )


def _save_text_embeddings(
    dataset_root: Path,
    model_path: Path,
    meta: LeRobotDatasetMetadata,
    device: torch.device,
    dtype: torch.dtype,
):
    tokenizer = load_tokenizer(str(model_path / "tokenizer"))
    text_encoder = load_text_encoder(str(model_path / "text_encoder"), torch_dtype=dtype, torch_device=device)

    text_emb_dir = dataset_root / "text_emb"
    text_emb_dir.mkdir(parents=True, exist_ok=True)

    task_texts = _extract_indexed_texts(meta.tasks, "task_index", "task", "tasks")
    _save_torch_atomic(
        encode_texts(tokenizer, text_encoder, task_texts, device, dtype),
        text_emb_dir / "task_emb.pth",
    )

    if meta.subtasks is not None and len(meta.subtasks) > 0:
        subtask_texts = _extract_indexed_texts(meta.subtasks, "subtask_index", "subtask", "subtasks")
        _save_torch_atomic(
            encode_texts(tokenizer, text_encoder, subtask_texts, device, dtype),
            text_emb_dir / "subtask_emb.pth",
        )

    _save_torch_atomic(
        encode_texts(tokenizer, text_encoder, [""], device, dtype).squeeze(0),
        text_emb_dir / "empty_emb.pth",
    )


def _validate_completed_outputs(
    dataset_root: Path,
    latent_base: Path,
    text_emb_dir: Path,
    cam_keys: list[str],
    segments: list[Segment],
    *,
    require_subtask_emb: bool,
):
    missing_latents: list[Path] = []
    for seg in segments:
        for path in _segment_output_paths(latent_base, cam_keys, seg):
            if not path.exists():
                missing_latents.append(path)
                if len(missing_latents) >= 3:
                    break
        if missing_latents:
            break
    if missing_latents:
        raise RuntimeError(
            f"{dataset_root.name}: extraction finished but some latent files are missing, e.g. "
            f"{[str(path) for path in missing_latents]}"
        )

    required_text_paths = [text_emb_dir / "task_emb.pth", text_emb_dir / "empty_emb.pth"]
    if require_subtask_emb:
        required_text_paths.append(text_emb_dir / "subtask_emb.pth")
    missing_text = [str(path) for path in required_text_paths if not path.exists()]
    if missing_text:
        raise RuntimeError(
            f"{dataset_root.name}: extraction finished but some text embedding files are missing: {missing_text}"
        )


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_latents(
    dataset_root: Path,
    model_path: Path,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    dataset_root = Path(dataset_root)
    device = torch.device(device)
    dataset_config = load_dataset_user_config(dataset_root)
    preprocess_cfg = dataset_config.preprocess
    validate_train_config_against_preprocess(
        preprocess_cfg,
        dataset_config.training,
        label=f"{dataset_root.name}: dataset config",
    )
    cam_keys = preprocess_cfg.obs_cam_keys

    meta, hf_dataset = _load_v30_dataset(dataset_root)
    _validate_dataset_convention(dataset_root, meta, hf_dataset, preprocess_cfg)
    stride = preprocess_cfg.frame_stride(meta.fps)

    segments = build_segments(
        episodes=meta.episodes,
        hf_dataset=hf_dataset,
        subtasks=meta.subtasks,
        min_segment_frames=preprocess_cfg.min_segment_frames(meta.fps, min_sampled_frames=MIN_SAMPLED_FRAMES),
    )
    frame_index_col = np.asarray(hf_dataset.select_columns(["frame_index"])["frame_index"])
    if not segments:
        raise ValueError(
            f"{dataset_root.name}: no segments remain after applying min_segment_frames="
            f"{preprocess_cfg.min_segment_frames(meta.fps, min_sampled_frames=MIN_SAMPLED_FRAMES)}. "
            "Adjust target_fps or dataset annotations before extracting latents."
        )
    logger.info("Extracting %d segments (stride=%d)", len(segments), stride)

    lerobot_ds = LeRobotDataset(repo_id="local", root=dataset_root, download_videos=False)
    vae = load_vae(str(model_path / "vae"), torch_dtype=dtype, torch_device=device)

    latent_base = dataset_root / "latents"
    for cam_key in cam_keys:
        (latent_base / cam_key).mkdir(parents=True, exist_ok=True)

    in_progress_metadata = _build_frozen_metadata(
        model_path,
        preprocess_cfg,
        meta,
        extraction_status=LATENT_METADATA_STATUS_EXTRACTING,
    )
    metadata_path = _ensure_matching_frozen_metadata(dataset_root, in_progress_metadata)
    write_frozen_latent_metadata(dataset_root, in_progress_metadata)
    logger.info("Marked latent metadata as in-progress at %s", metadata_path)

    for seg in tqdm(segments, desc="Extracting latents"):
        seg_paths = _segment_output_paths(latent_base, cam_keys, seg)
        if all(p.exists() for p in seg_paths):
            continue

        # Remove partial files so all cameras in a segment come from the same run.
        for p in seg_paths:
            p.unlink(missing_ok=True)

        ref_frame_ids: np.ndarray | None = None
        ref_latent_frames: int | None = None
        for cam_key, out_path in zip(cam_keys, seg_paths):
            h_i, w_i = preprocess_cfg.resolution_for(cam_key)
            video, frame_ids = decode_video_segment(
                lerobot_ds, cam_key, seg, stride, frame_index_col, h_i, w_i,
            )
            latent = encode_video(vae, video, device, dtype).squeeze(0)
            ref_frame_ids, ref_latent_frames = _validate_segment_latents(
                dataset_root,
                seg,
                cam_key,
                frame_ids,
                latent,
                in_progress_metadata.vae_temporal_downsample,
                ref_frame_ids,
                ref_latent_frames,
            )
            _, lt, lh, lw = latent.shape

            _save_torch_atomic(
                {
                    "latent": latent.permute(1, 2, 3, 0).reshape(-1, latent.shape[0]).cpu(),
                    "frame_ids": frame_ids,
                    "latent_num_frames": lt,
                    "latent_height": lh,
                    "latent_width": lw,
                    "fps": in_progress_metadata.actual_fps,
                },
                out_path,
            )

    logger.info("Latent extraction complete.")
    text_emb_dir = dataset_root / "text_emb"
    _save_text_embeddings(dataset_root, model_path, meta, device, dtype)
    _validate_completed_outputs(
        dataset_root,
        latent_base,
        text_emb_dir,
        cam_keys,
        segments,
        require_subtask_emb=meta.subtasks is not None and len(meta.subtasks) > 0,
    )
    write_frozen_latent_metadata(
        dataset_root,
        replace(in_progress_metadata, extraction_status=LATENT_METADATA_STATUS_COMPLETE),
    )
    logger.info("Marked latent metadata as complete at %s", metadata_path)
    logger.info("All done. Output at %s", dataset_root)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract latents & text embeddings for a v3.0 LeRobot dataset")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_latents(
        dataset_root=Path(args.dataset_root), model_path=Path(args.model_path),
        device=args.device,
    )


if __name__ == "__main__":
    main()
