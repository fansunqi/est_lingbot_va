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
        --model-path   /path/to/pretrained_wan \\
        --num-gpus 6

``--num-gpus 0`` (default) uses all visible GPUs.  If you observe CPU
utilisation pinned near 100 % and some GPUs idling at 0 %, reduce
``--num-gpus`` until CPU headroom reappears — too many parallel workers
can saturate CPU / IO and hurt overall throughput.
"""

import argparse
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from diffusers import AutoencoderKLWan
from lerobot.datasets.feature_utils import get_hf_features_from_features
from lerobot.datasets.io_utils import load_nested_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.video_utils import decode_video_frames

from wan_va.dataset.latent_metadata import (
    LATENT_METADATA_FILENAME,
    LATENT_METADATA_STATUS_COMPLETE,
    LATENT_METADATA_STATUS_EXTRACTING,
    FrozenLatentMetadata,
    load_dataset_user_config,
    load_frozen_latent_metadata,
    validate_train_config_against_preprocess,
    write_frozen_latent_metadata,
)
from wan_va.dataset.segment_builder import (
    MIN_SAMPLED_FRAMES,
    Segment,
    build_segments,
    get_latent_filename,
    read_index_columns,
)
from wan_va.modules.utils import load_text_encoder, load_tokenizer, load_vae

logger = logging.getLogger(__name__)


def _save_torch_atomic(obj, path: Path):
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def decode_video_segment(
    lerobot_ds: LeRobotDataset,
    cam_key: str,
    seg: Segment,
    stride: int,
    frame_index_col: np.ndarray,
    timestamp_col: np.ndarray,
    height: int,
    width: int,
) -> tuple[torch.Tensor, np.ndarray]:
    """Batch-decode and sub-sample video frames for one segment.

    Returns ``(video, frame_ids)`` where *video* is ``(1, C, T, H, W)``
    in ``[-1, 1]`` and *frame_ids* are the parquet ``frame_index`` values.
    """
    global_indices = list(range(seg.global_from, seg.global_to, stride))
    frame_ids = frame_index_col[global_indices].astype(np.int64)

    ep_meta = lerobot_ds.meta.episodes[seg.episode_index]
    from_ts = ep_meta[f"videos/{cam_key}/from_timestamp"]
    timestamps = (from_ts + timestamp_col[global_indices]).tolist()
    video_path = lerobot_ds.root / lerobot_ds.meta.get_video_file_path(seg.episode_index, cam_key)

    frames = decode_video_frames(
        video_path, timestamps, tolerance_s=1e-3, backend=lerobot_ds._video_backend,
    )

    video = frames.permute(1, 0, 2, 3).unsqueeze(0)
    if video.shape[-2] != height or video.shape[-1] != width:
        B, C, T, H, W = video.shape
        video = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video = F.interpolate(video, size=(height, width), mode="bilinear", align_corners=False)
        video = video.reshape(B, T, C, height, width).permute(0, 2, 1, 3, 4)
    video = video * 2.0 - 1.0
    return video, frame_ids


@torch.no_grad()
def encode_video(
    vae: AutoencoderKLWan,
    video: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode ``(B, C, T, H, W)`` → normalised latent ``(B, z, T', H', W')``."""
    mu = vae.encode(video.to(device=device, dtype=dtype)).latent_dist.mean  # type: ignore[union-attr]
    mean = torch.tensor(vae.config.latents_mean, device=mu.device).view(1, -1, 1, 1, 1)  # type: ignore[attr-defined]
    std = torch.tensor(vae.config.latents_std, device=mu.device).view(1, -1, 1, 1, 1)  # type: ignore[attr-defined]
    return ((mu.float() - mean) / std).to(dtype)


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
    """Extract texts ordered by a dense 0..N-1 index column."""
    if index_col not in df.columns:
        raise ValueError(f"{label}: missing column '{index_col}'")

    indices = sorted(int(x) for x in df[index_col])
    if indices != list(range(len(indices))):
        raise ValueError(f"{label}: {index_col} not contiguous 0..{len(indices)-1}: {indices[:10]}")

    if text_col is not None and text_col in df.columns:
        sorted_df = df.sort_values(index_col)
        return [str(sorted_df.loc[sorted_df[index_col] == i, text_col].iloc[0]) for i in indices]
    elif pd.api.types.is_string_dtype(df.index) and df.index.name != index_col:
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
    model_path: Path, preprocess_cfg, meta: LeRobotDatasetMetadata,
    *, extraction_status: str,
) -> FrozenLatentMetadata:
    return FrozenLatentMetadata(
        preprocess=preprocess_cfg,
        codebase_version=str(meta.info["codebase_version"]),
        actual_fps=meta.fps / preprocess_cfg.frame_stride,
        extraction_status=extraction_status,
        model_path=str(model_path.resolve()),
    )


def _ensure_matching_frozen_metadata(
    dataset_root: Path, current_metadata: FrozenLatentMetadata,
):
    metadata_path = dataset_root / LATENT_METADATA_FILENAME
    latent_base = dataset_root / "latents"
    if not metadata_path.exists():
        existing_latents = list(latent_base.rglob("*.pth")) if latent_base.exists() else []
        if existing_latents:
            raise RuntimeError(
                f"{dataset_root.name}: found {len(existing_latents)} pre-existing latent files under "
                f"{latent_base} but {metadata_path} is missing. Delete {latent_base} and re-extract."
            )
        return metadata_path

    stored = load_frozen_latent_metadata(dataset_root)
    if stored.preprocess.to_dict() != current_metadata.preprocess.to_dict():
        raise RuntimeError(
            f"Preprocess config differs from the previous extraction at {metadata_path}.\n"
            f"  Stored : {stored.preprocess.to_dict()}\n"
            f"  Current: {current_metadata.preprocess.to_dict()}\n"
            f"Delete {latent_base} and re-extract."
        )
    if stored.codebase_version != current_metadata.codebase_version:
        raise RuntimeError(
            f"codebase_version changed: {stored.codebase_version!r} → "
            f"{current_metadata.codebase_version!r}. Delete {latent_base} and re-extract."
        )
    if abs(stored.actual_fps - current_metadata.actual_fps) > 1e-6:
        raise RuntimeError(
            f"actual_fps changed: {stored.actual_fps} → {current_metadata.actual_fps}. "
            f"Delete {latent_base} and re-extract."
        )
    if stored.vae_temporal_downsample != current_metadata.vae_temporal_downsample:
        raise RuntimeError(
            f"vae_temporal_downsample changed: {stored.vae_temporal_downsample} → "
            f"{current_metadata.vae_temporal_downsample}. Delete {latent_base} and re-extract."
        )
    if stored.extraction_status == LATENT_METADATA_STATUS_EXTRACTING:
        logger.info("%s: resuming in-progress extraction.", dataset_root.name)
    if stored.model_path not in (None, current_metadata.model_path):
        logger.warning(
            "model_path changed: %s → %s. "
            "Delete %s to re-extract if the VAE weights differ.",
            stored.model_path, current_metadata.model_path, latent_base,
        )
    return metadata_path


def _segment_output_paths(latent_base: Path, cam_keys: list[str], seg: Segment) -> list[Path]:
    filename = get_latent_filename(seg.episode_index, seg.start_frame, seg.end_frame)
    return [latent_base / cam_key / filename for cam_key in cam_keys]


def _validate_segment_latents(
    dataset_root: Path, seg: Segment, cam_key: str,
    frame_ids: np.ndarray, latent: torch.Tensor, time_downsample: int,
    ref_frame_ids: np.ndarray | None, ref_latent_frames: int | None,
) -> tuple[np.ndarray, int]:
    if len(frame_ids) < MIN_SAMPLED_FRAMES:
        raise ValueError(
            f"{dataset_root.name}: ep={seg.episode_index} [{seg.start_frame}, {seg.end_frame}) "
            f"only {len(frame_ids)} sampled frames"
        )

    latent_frames = int(latent.shape[1])
    expected = (len(frame_ids) - 1) // time_downsample + 1
    if latent_frames != expected:
        raise ValueError(
            f"{dataset_root.name}: {cam_key} ep={seg.episode_index} [{seg.start_frame}, {seg.end_frame}) "
            f"latent_num_frames={latent_frames}, expected {expected}"
        )

    if ref_frame_ids is None:
        return frame_ids, latent_frames
    if not np.array_equal(frame_ids, ref_frame_ids):
        raise ValueError(
            f"{dataset_root.name}: frame_ids mismatch across cameras for "
            f"ep={seg.episode_index} [{seg.start_frame}, {seg.end_frame})"
        )
    if latent_frames != ref_latent_frames:
        raise ValueError(
            f"{dataset_root.name}: latent_num_frames mismatch across cameras for "
            f"ep={seg.episode_index} [{seg.start_frame}, {seg.end_frame})"
        )
    return ref_frame_ids, ref_latent_frames


def _validate_dataset_convention(
    dataset_root: Path, meta: LeRobotDatasetMetadata, preprocess_cfg,
    frame_col: np.ndarray, task_col: np.ndarray, subtask_col: np.ndarray | None,
):
    missing = [k for k in preprocess_cfg.obs_cam_keys if k not in meta.features]
    if missing:
        raise ValueError(f"{dataset_root.name}: obs_cam_keys not in dataset features: {missing}")

    has_subtask = meta.subtasks is not None and len(meta.subtasks) > 0 and subtask_col is not None

    if len(meta.tasks) == 0:
        raise ValueError(f"{dataset_root.name}: meta/tasks is empty")
    if np.any(task_col < 0) or np.any(task_col >= len(meta.tasks)):
        bad = np.unique(task_col[(task_col < 0) | (task_col >= len(meta.tasks))]).tolist()
        raise ValueError(f"{dataset_root.name}: task_index out of bounds ({len(meta.tasks)} tasks): {bad}")

    if has_subtask and subtask_col is not None:
        if np.any(subtask_col < 0) or np.any(subtask_col >= len(meta.subtasks)):
            bad = np.unique(subtask_col[(subtask_col < 0) | (subtask_col >= len(meta.subtasks))]).tolist()
            raise ValueError(
                f"{dataset_root.name}: subtask_index out of bounds ({len(meta.subtasks)} subtasks): {bad}"
            )

    for row in meta.episodes:
        ep_idx, ds_from, ds_to = row["episode_index"], row["dataset_from_index"], row["dataset_to_index"]
        if ds_to <= ds_from:
            continue
        if not np.array_equal(frame_col[ds_from:ds_to], np.arange(ds_to - ds_from)):
            raise ValueError(
                f"{dataset_root.name}: episode {ep_idx} frame_index is not dense 0-based "
                f"(got {frame_col[ds_from:ds_from+10].tolist()})"
            )
        if not has_subtask and not np.all(task_col[ds_from:ds_to] == task_col[ds_from]):
            raise ValueError(
                f"{dataset_root.name}: episode {ep_idx} task_index is not constant "
                f"(values={np.unique(task_col[ds_from:ds_to]).tolist()})"
            )


def _save_text_embeddings(
    dataset_root: Path, model_path: Path,
    meta: LeRobotDatasetMetadata, device: torch.device, dtype: torch.dtype,
):
    tokenizer = load_tokenizer(str(model_path / "tokenizer"))
    text_encoder = load_text_encoder(str(model_path / "text_encoder"), torch_dtype=dtype, torch_device=device)

    text_emb_dir = dataset_root / "text_emb"
    text_emb_dir.mkdir(parents=True, exist_ok=True)

    task_texts = _extract_indexed_texts(meta.tasks, "task_index", "task", "tasks")
    _save_torch_atomic(encode_texts(tokenizer, text_encoder, task_texts, device, dtype),
                       text_emb_dir / "task_emb.pth")

    if meta.subtasks is not None and len(meta.subtasks) > 0:
        subtask_texts = _extract_indexed_texts(meta.subtasks, "subtask_index", "subtask", "subtasks")
        _save_torch_atomic(encode_texts(tokenizer, text_encoder, subtask_texts, device, dtype),
                           text_emb_dir / "subtask_emb.pth")

    _save_torch_atomic(encode_texts(tokenizer, text_encoder, [""], device, dtype).squeeze(0),
                       text_emb_dir / "empty_emb.pth")


def _validate_completed_outputs(
    dataset_root: Path, latent_base: Path, text_emb_dir: Path,
    cam_keys: list[str], segments: list[Segment], *, require_subtask_emb: bool,
):
    missing = []
    for seg in segments:
        for p in _segment_output_paths(latent_base, cam_keys, seg):
            if not p.exists():
                missing.append(p)
                if len(missing) >= 3:
                    break
        if missing:
            break
    if missing:
        raise RuntimeError(
            f"{dataset_root.name}: extraction finished but latent files missing, "
            f"e.g. {[str(p) for p in missing]}"
        )

    required = [text_emb_dir / "task_emb.pth", text_emb_dir / "empty_emb.pth"]
    if require_subtask_emb:
        required.append(text_emb_dir / "subtask_emb.pth")
    missing_text = [str(p) for p in required if not p.exists()]
    if missing_text:
        raise RuntimeError(f"{dataset_root.name}: text embedding files missing: {missing_text}")


# ---------------------------------------------------------------------------
# Multi-GPU worker
# ---------------------------------------------------------------------------

def _extraction_worker(
    rank: int, world_size: int, gpu_ids: list[int],
    dataset_root: Path, model_path: Path, dtype: torch.dtype,
    seg_work: list, stride: int, vae_temporal_downsample: int, actual_fps: float,
    progress_counter,
):
    """Encode latents for ``seg_work[rank::world_size]`` on ``cuda:{gpu_ids[rank]}``."""
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)

    lerobot_ds = LeRobotDataset(repo_id="local", root=dataset_root, download_videos=False)
    _, hf_dataset = _load_v30_dataset(dataset_root)
    frame_index_col, _, _ = read_index_columns(hf_dataset)
    timestamp_col = hf_dataset.select_columns(["timestamp"]).data.column("timestamp").to_numpy()
    vae = load_vae(str(model_path / "vae"), torch_dtype=dtype, torch_device=device)

    my_work = seg_work[rank::world_size]

    def _decode_seg(seg, cams):
        return [
            (*cam, *decode_video_segment(
                lerobot_ds, cam[0], seg, stride,
                frame_index_col, timestamp_col, cam[2], cam[3],
            ))
            for cam in cams
        ]

    try:
        # max_workers=1: torchcodec's VideoDecoder is not thread-safe.
        prefetch: Future | None = None
        bar = tqdm(total=len(seg_work), desc="Extracting latents",
                   smoothing=0.05, disable=(rank != 0))
        with ThreadPoolExecutor(max_workers=1) as pool:
            for si, (seg, cams) in enumerate(my_work):
                decoded = prefetch.result() if prefetch is not None else _decode_seg(seg, cams)
                prefetch = (
                    pool.submit(_decode_seg, *my_work[si + 1])
                    if si + 1 < len(my_work) else None
                )

                res_groups: dict[tuple[int, int], list[int]] = {}
                for i, (_, _, h, w, _, _) in enumerate(decoded):
                    res_groups.setdefault((h, w), []).append(i)

                latents: list[torch.Tensor | None] = [None] * len(decoded)
                for indices in res_groups.values():
                    batch = torch.cat([decoded[i][4] for i in indices], dim=0)
                    out = encode_video(vae, batch, device, dtype)
                    del batch
                    for bi, ci in enumerate(indices):
                        latents[ci] = out[bi]
                    del out

                ref_fids: np.ndarray | None = None
                ref_lt: int | None = None
                for i, (cam_key, out_path, _, _, _, frame_ids) in enumerate(decoded):
                    latent = latents[i]
                    assert latent is not None
                    ref_fids, ref_lt = _validate_segment_latents(
                        dataset_root, seg, cam_key, frame_ids, latent,
                        vae_temporal_downsample, ref_fids, ref_lt,
                    )
                    _, lt, lh, lw = latent.shape
                    _save_torch_atomic({
                        "latent": latent.permute(1, 2, 3, 0).reshape(-1, latent.shape[0]).cpu(),
                        "frame_ids": frame_ids,
                        "latent_num_frames": lt, "latent_height": lh, "latent_width": lw,
                        "fps": actual_fps,
                    }, out_path)
                    del latent

                with progress_counter.get_lock():
                    progress_counter.value += 1
                    count = progress_counter.value
                if rank == 0:
                    bar.update(count - bar.n)
    finally:
        bar.close()
        del vae
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_latents(
    dataset_root: Path, model_path: Path,
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16, num_gpus: int = 0,
):
    dataset_root = Path(dataset_root)
    dataset_config = load_dataset_user_config(dataset_root)
    preprocess_cfg = dataset_config.preprocess
    validate_train_config_against_preprocess(
        preprocess_cfg, dataset_config.training,
        label=f"{dataset_root.name}: dataset config",
    )
    cam_keys = preprocess_cfg.obs_cam_keys

    meta, hf_dataset = _load_v30_dataset(dataset_root)
    idx_cols = read_index_columns(hf_dataset)
    frame_index_col, task_col, subtask_col = idx_cols
    _validate_dataset_convention(dataset_root, meta, preprocess_cfg, frame_index_col, task_col, subtask_col)

    stride = preprocess_cfg.frame_stride
    segments = build_segments(
        episodes=meta.episodes, subtasks=meta.subtasks,
        min_segment_frames=preprocess_cfg.min_segment_frames(min_sampled_frames=MIN_SAMPLED_FRAMES),
        index_columns=idx_cols,
    )
    if not segments:
        raise ValueError(
            f"{dataset_root.name}: no segments after min_segment_frames="
            f"{preprocess_cfg.min_segment_frames(min_sampled_frames=MIN_SAMPLED_FRAMES)}"
        )
    logger.info("Extracting %d segments (stride=%d)", len(segments), stride)

    latent_base = dataset_root / "latents"
    for cam_key in cam_keys:
        (latent_base / cam_key).mkdir(parents=True, exist_ok=True)

    in_progress_metadata = _build_frozen_metadata(
        model_path, preprocess_cfg, meta,
        extraction_status=LATENT_METADATA_STATUS_EXTRACTING,
    )
    metadata_path = _ensure_matching_frozen_metadata(dataset_root, in_progress_metadata)
    write_frozen_latent_metadata(dataset_root, in_progress_metadata)

    # Build work list: skip completed segments, clean partial ones.
    seg_work: list[tuple[Segment, list[tuple[str, Path, int, int]]]] = []
    for seg in segments:
        seg_paths = _segment_output_paths(latent_base, cam_keys, seg)
        if all(p.exists() for p in seg_paths):
            continue
        for p in seg_paths:
            p.unlink(missing_ok=True)
        seg_work.append((seg, [
            (ck, op, *preprocess_cfg.resolution_for(ck))
            for ck, op in zip(cam_keys, seg_paths)
        ]))

    default_gpu = torch.device(device).index or 0

    if not seg_work:
        logger.info("All segments already extracted; skipping to text embeddings.")
    else:
        available = torch.cuda.device_count()
        if available == 0:
            raise RuntimeError("No CUDA devices available.")
        gpu_ids = (
            [default_gpu] if num_gpus == 1
            else list(range(min(num_gpus, available) if num_gpus > 0 else available))
        )
        gpu_ids = gpu_ids[:len(seg_work)]
        default_gpu = gpu_ids[0]
        logger.info("Encoding on %d GPU(s) %s — %d segments", len(gpu_ids), gpu_ids, len(seg_work))

        counter = mp.get_context("spawn").Value('i', 0)
        args = (
            len(gpu_ids), gpu_ids, dataset_root, model_path, dtype,
            seg_work, stride,
            in_progress_metadata.vae_temporal_downsample,
            in_progress_metadata.actual_fps,
            counter,
        )
        if len(gpu_ids) <= 1:
            _extraction_worker(0, *args)
        else:
            mp.spawn(_extraction_worker, args=args, nprocs=len(gpu_ids), join=True)
        logger.info("Latent extraction complete.")

    device_phase3 = torch.device(f"cuda:{default_gpu}")
    text_emb_dir = dataset_root / "text_emb"
    _save_text_embeddings(dataset_root, model_path, meta, device_phase3, dtype)
    _validate_completed_outputs(
        dataset_root, latent_base, text_emb_dir, cam_keys, segments,
        require_subtask_emb=meta.subtasks is not None and len(meta.subtasks) > 0,
    )
    write_frozen_latent_metadata(
        dataset_root, replace(in_progress_metadata, extraction_status=LATENT_METADATA_STATUS_COMPLETE),
    )
    logger.info("All done. Output at %s", dataset_root)


def main():
    parser = argparse.ArgumentParser(description="Extract latents & text embeddings for a v3.0 LeRobot dataset")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for single-GPU mode (ignored when --num-gpus > 1)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (0 = all available, 1 = single-GPU)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract_latents(
        dataset_root=Path(args.dataset_root), model_path=Path(args.model_path),
        device=args.device, num_gpus=args.num_gpus,
    )


if __name__ == "__main__":
    main()
