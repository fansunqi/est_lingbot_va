# LingBot-VA Latent Data Design

## Goal

Use LeRobot 0.5 metadata directly and keep LingBot-VA training on precomputed latents without decoding videos during training.

## Main Rule

Do not maintain a second segment metadata file.

Training segments should be built at runtime from LeRobot metadata, then filtered by latent-file existence.

This avoids duplicate sources of truth.

## Segment Construction

Build segments per dataset at initialization time:

- Read only the needed index columns from `hf_dataset` in bulk, not per-frame one by one.
- Prefer a lightweight view such as `episode_index`, `frame_index`, `task_index`, and optional `subtask_index`.
- If `subtask_index` exists and `meta.subtasks` is available:
  - split each episode by contiguous runs of the same `subtask_index`
  - use the corresponding subtask text
- Otherwise:
  - one episode becomes one segment
  - use the episode task text

The same segment-building logic should be used in both latent extraction and training-time loading.

Each runtime segment only needs:

- `episode_index`
- `start_frame`
- `end_frame`
- `global_from`
- `global_to`
- `task_index`
- `subtask_index` (optional)

`global_from` and `global_to` come from the episode's global index range in LeRobot metadata, combined with the segment's local frame range.

## Latent File Naming

Latent files follow this convention:

- `episode_{episode_index:06d}_{start_frame}_{end_frame}.pth`

The segment boundaries computed at runtime must match the boundaries used during latent extraction.

If task or subtask annotations change, previously extracted latent files will no longer match the runtime segments and must be regenerated.

The loader should warn on limited latent mismatches and fail fast when a large fraction of constructed segments have no matching latent files.

## Recommended Layout

```text
dataset_root/
├── data/                      # LeRobot native
├── videos/                    # LeRobot native, not used in training
├── meta/
│   ├── info.json
│   ├── tasks.parquet
│   ├── subtasks.parquet       # optional
│   ├── episodes/...
│   └── stats.json
├── latents/
│   ├── observation.images.top/
│   └── observation.images.wrist/
└── text_emb/
    ├── task_emb.pth
    ├── subtask_emb.pth        # optional
    └── empty_emb.pth
```

Latent paths do not need to mirror LeRobot chunk layout.

A flat per-camera layout is preferred, for example:

- `latents/observation.images.top/episode_000000_0_264.pth`
- `latents/observation.images.wrist/episode_000000_0_264.pth`

## Latent Files

Latent files should stay minimal.

Recommended contents:

- `latent`
- `frame_ids`
- `latent_num_frames`
- `latent_height`
- `latent_width`
- `fps`

Keep `frame_ids` because it encodes the temporal downsampling applied during VAE encoding.

This cannot be derived from LeRobot metadata alone.

Action alignment uses it to compute `frame_stride`, `act_shift`, and the action length required by the latent sequence.

`fps` records the target sampling rate used during latent extraction. It is mainly for load-time sanity checks against dataset fps and `frame_ids`, not a primary training input.

The main space saving comes from removing repeated `text_emb` tensors from every latent file.

Do not store repeated business metadata in every latent file.

## Text Embeddings

Cache text embeddings per dataset:

- `task_emb.pth`: stacked tensor indexed by local `task_index`
- `subtask_emb.pth`: stacked tensor indexed by local `subtask_index`
- `empty_emb.pth`: used for CFG dropout

This should remain dataset-local, not global across mixed datasets.

Prefer padding to a dataset-local fixed sequence length so stacked tensors remain usable.

If sequence lengths are not made uniform, use a dataset-local mapping instead of a stacked tensor.

At training time:

- use `subtask_index` to look up `subtask_emb.pth` when subtasks exist
- otherwise use `task_index` to look up `task_emb.pth`
- apply CFG dropout by replacing the selected embedding with `empty_emb`

## Training Path

Do not use `LeRobotDataset.__getitem__()` to build segments.

Use only:

- `dataset.meta`
- `dataset.hf_dataset`

Reason:

- `LeRobotDataset.__getitem__()` decodes video frames when video keys exist
- `meta` and `hf_dataset` provide the needed metadata and action tables without video decode

Initialization can follow either of these safe paths:

- construct `LeRobotDataset` with local data present and `download_videos=False`, then use only `meta` and `hf_dataset`
- or construct `LeRobotDatasetMetadata` directly with `root` pointing to the local dataset, then load parquet data separately

The second path is the lighter option when training only needs metadata and action tables.

It assumes the local `meta/` directory is complete. If local metadata is missing, LeRobot may try to fall back to repository download behavior.

So the training path remains:

1. build segments from LeRobot metadata
2. filter out segments without latent files
3. load latent tensors from `latents/`
4. load actions from LeRobot parquet data
   - during init, use a lightweight index view over only the needed columns
   - during `__getitem__`, keep a separate tensor-formatted action view
   - use `global_from` and `global_to`, derived from `episodes["dataset_from_index"]`, to read action ranges
5. never decode video during training

## Multi-Camera Layout

Latent loading stays per camera, and multi-camera concatenation remains a dataset-level rule.

The concatenation strategy should stay in per-dataset config, because different environments may use different layouts.

Frame alignment should assume the same `frame_ids` across cameras for the same segment and validate this when loading latents.

## Multi-Dataset Note

Each dataset should keep its own:

- segment construction
- text embedding cache
- camera configuration
- action dimension mapping to the unified training action space
- action normalization statistics and rules

Mixed training should combine datasets above this layer, not by sharing task or subtask indices globally.

Sampling weights across datasets should be handled at the multi-dataset mixing layer, not inside the per-dataset latent format.
