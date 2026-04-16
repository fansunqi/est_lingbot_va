import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DATASET_USER_CONFIG_FILENAME = Path("meta/wan_va_config.json")
LATENT_METADATA_FILENAME = Path("latents/metadata.json")
LATENT_METADATA_VERSION = 1
DEFAULT_VAE_TEMPORAL_DOWNSAMPLE = 4
DEFAULT_VAE_SPATIAL_DOWNSAMPLE = 16
LATENT_METADATA_STATUS_EXTRACTING = "extracting"
LATENT_METADATA_STATUS_COMPLETE = "complete"

_SUPPORTED_LATENT_LAYOUTS = {"horizontal_concat", "robotwin_tshape"}
_SUPPORTED_ACTION_TRANSFORMS = {"identity", "robotwin_relative_pose_bimanual"}
_SUPPORTED_ACTION_NORM_METHODS = {"quantiles"}
_SUPPORTED_METADATA_STATUSES = {
    LATENT_METADATA_STATUS_EXTRACTING,
    LATENT_METADATA_STATUS_COMPLETE,
}
_ROBOTWIN_RELATIVE_POSE_RAW_ACTION_DIM = 16
_ROBOTWIN_RELATIVE_POSE_TRANSFORMED_ACTION_DIM = 16


@dataclass(frozen=True)
class CameraPresetSpec:
    name: str
    camera_count: int
    expected_latent_layout: str
    resize_resolutions: tuple[tuple[int, int], ...]

    def __post_init__(self):
        """Validate geometric invariants required by the corresponding latent layout."""
        latent_h = [h // DEFAULT_VAE_SPATIAL_DOWNSAMPLE for h, _ in self.resize_resolutions]
        latent_w = [w // DEFAULT_VAE_SPATIAL_DOWNSAMPLE for _, w in self.resize_resolutions]
        if self.expected_latent_layout == "horizontal_concat":
            if len(set(latent_h)) != 1:
                raise ValueError(
                    f"Preset {self.name!r}: horizontal_concat requires equal latent heights, got {latent_h}"
                )
        elif self.expected_latent_layout == "robotwin_tshape":
            if len(self.resize_resolutions) < 2:
                raise ValueError(f"Preset {self.name!r}: robotwin_tshape requires at least 2 cameras")
            if len(set(latent_h[1:])) != 1:
                raise ValueError(
                    f"Preset {self.name!r}: robotwin_tshape requires equal wrist latent heights, got {latent_h[1:]}"
                )
            if latent_w[0] != sum(latent_w[1:]):
                raise ValueError(
                    f"Preset {self.name!r}: robotwin_tshape requires primary latent width "
                    f"({latent_w[0]}) == sum of wrist widths ({sum(latent_w[1:])})"
                )


# Recommended presets mirroring the historical training setups:
# - one_primary_one_wrist_256: demo-style 2-camera square inputs at 256x256
# - one_primary_one_wrist_128: libero-style 2-camera square inputs at 128x128
# - one_primary_two_wrist_224x320: franka-style 3-camera horizontal concat
# - one_primary_two_wrist_tshape_256x320: robotwin-style 1 high + 2 wrist T-shape
CAMERA_PRESET_SPECS: dict[str, CameraPresetSpec] = {
    "one_primary_one_wrist_256": CameraPresetSpec(
        name="one_primary_one_wrist_256",
        camera_count=2,
        expected_latent_layout="horizontal_concat",
        resize_resolutions=((256, 256), (256, 256)),
    ),
    "one_primary_one_wrist_128": CameraPresetSpec(
        name="one_primary_one_wrist_128",
        camera_count=2,
        expected_latent_layout="horizontal_concat",
        resize_resolutions=((128, 128), (128, 128)),
    ),
    "one_primary_two_wrist_224x320": CameraPresetSpec(
        name="one_primary_two_wrist_224x320",
        camera_count=3,
        expected_latent_layout="horizontal_concat",
        resize_resolutions=((224, 320), (224, 320), (224, 320)),
    ),
    "one_primary_two_wrist_tshape_256x320": CameraPresetSpec(
        name="one_primary_two_wrist_tshape_256x320",
        camera_count=3,
        expected_latent_layout="robotwin_tshape",
        resize_resolutions=((256, 320), (128, 160), (128, 160)),
    ),
}


def _require_keys(data: dict[str, Any], keys: list[str], label: str):
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"{label}: missing required keys: {missing}")


def _validate_positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label}: expected a positive integer, got {value!r}")
    return value


def _validate_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}: expected a non-empty string, got {value!r}")
    return value


def _validate_string_list(values: Any, label: str) -> list[str]:
    if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values):
        raise ValueError(f"{label}: expected a non-empty list of strings")
    return list(values)


def _build_inverse_ids(
    action_dim: int,
    action_key_channel_map: list[list[int | None]],
) -> list[int]:
    """Map each model-space channel to a column in the zero-padded raw action.

    Active channels → their raw-concat offset.  Unused channels → the
    sentinel column (``total_raw_dim``, i.e. the trailing zero-pad).
    """
    total_raw_dim = sum(len(sub) for sub in action_key_channel_map)
    inverse_ids = [total_raw_dim] * action_dim  # default: sentinel
    raw_offset = 0
    for sub_map in action_key_channel_map:
        for model_id in sub_map:
            if model_id is not None:
                inverse_ids[model_id] = raw_offset
            raw_offset += 1
    return inverse_ids


def build_norm_stat_from_raw_stats(
    per_key_q01: list[np.ndarray],
    per_key_q99: list[np.ndarray],
    inverse_used_action_channel_ids: list[int],
    action_key_channel_map: list[list[int | None]],
) -> dict[str, list[float]]:
    """Build model-space q01/q99 from per-key raw dataset statistics.

    Concatenates per-key stats in raw order (plus a sentinel zero), then
    indexes by ``inverse_used_action_channel_ids`` — mirroring the same
    remapping applied to action data in ``_action_post_process``.
    """
    total_raw_dim = sum(len(sub) for sub in action_key_channel_map)
    raw_q01 = np.zeros(total_raw_dim + 1, dtype=np.float64)  # +1 sentinel
    raw_q99 = np.zeros(total_raw_dim + 1, dtype=np.float64)
    offset = 0
    for ki, sub_map in enumerate(action_key_channel_map):
        n = len(sub_map)
        raw_q01[offset:offset + n] = per_key_q01[ki]
        raw_q99[offset:offset + n] = per_key_q99[ki]
        offset += n
    return {
        "q01": raw_q01[inverse_used_action_channel_ids].tolist(),
        "q99": raw_q99[inverse_used_action_channel_ids].tolist(),
    }


def _camera_preset_spec(camera_preset: str, *, label: str) -> CameraPresetSpec:
    try:
        return CAMERA_PRESET_SPECS[camera_preset]
    except KeyError as exc:
        raise ValueError(
            f"{label}: unsupported camera_preset {camera_preset!r}. "
            f"supported={sorted(CAMERA_PRESET_SPECS)}"
        ) from exc


def _latent_hw(height: int, width: int) -> tuple[int, int]:
    return height // DEFAULT_VAE_SPATIAL_DOWNSAMPLE, width // DEFAULT_VAE_SPATIAL_DOWNSAMPLE


def validate_train_config_against_preprocess(
    preprocess: "LatentPreprocessConfig",
    training: "LatentTrainConfig",
    *,
    label: str,
):
    """Check that training.latent_layout matches the preset.

    Geometric invariants (equal heights for horizontal concat, width sum for
    T-shape) are validated once at preset definition time via
    ``CameraPresetSpec.__post_init__``, so they need not be re-checked here.
    """
    spec = preprocess.camera_preset_spec()
    if training.latent_layout != spec.expected_latent_layout:
        raise ValueError(
            f"{label}.training.latent_layout={training.latent_layout!r} is incompatible with "
            f"preprocess.camera_preset={preprocess.camera_preset!r}; expected "
            f"{spec.expected_latent_layout!r}"
        )


@dataclass(frozen=True)
class LatentPreprocessConfig:
    obs_cam_keys: list[str]
    frame_stride: int
    camera_preset: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "LatentPreprocessConfig":
        _require_keys(payload, ["obs_cam_keys", "frame_stride", "camera_preset"], label)

        obs_cam_keys = _validate_string_list(payload["obs_cam_keys"], f"{label}.obs_cam_keys")
        if len(set(obs_cam_keys)) != len(obs_cam_keys):
            raise ValueError(f"{label}.obs_cam_keys: duplicate camera keys are not allowed")
        frame_stride = _validate_positive_int(payload["frame_stride"], f"{label}.frame_stride")
        camera_preset = _validate_nonempty_string(payload["camera_preset"], f"{label}.camera_preset")
        spec = _camera_preset_spec(camera_preset, label=f"{label}.camera_preset")
        if len(obs_cam_keys) != spec.camera_count:
            raise ValueError(
                f"{label}.obs_cam_keys: camera_preset={camera_preset!r} expects {spec.camera_count} camera keys, "
                f"got {len(obs_cam_keys)}"
            )
        camera_resolutions = dict(zip(obs_cam_keys, spec.resize_resolutions))
        for cam_key, (height, width) in camera_resolutions.items():
            if height % DEFAULT_VAE_SPATIAL_DOWNSAMPLE != 0 or width % DEFAULT_VAE_SPATIAL_DOWNSAMPLE != 0:
                raise ValueError(
                    f"{label}: resolved resize for {cam_key!r} is {(height, width)}, which is not divisible "
                    f"by the VAE spatial downsample {DEFAULT_VAE_SPATIAL_DOWNSAMPLE}"
                )
            if min(_latent_hw(height, width)) < 1:
                raise ValueError(
                    f"{label}: resolved latent size for {cam_key!r} would be zero, "
                    f"got resize={(height, width)}"
                )

        return cls(
            obs_cam_keys=obs_cam_keys,
            frame_stride=frame_stride,
            camera_preset=camera_preset,
        )

    def resolved_camera_resolutions(self) -> dict[str, tuple[int, int]]:
        spec = self.camera_preset_spec()
        return dict(zip(self.obs_cam_keys, spec.resize_resolutions))

    def camera_preset_spec(self) -> CameraPresetSpec:
        return _camera_preset_spec(self.camera_preset, label="LatentPreprocessConfig.camera_preset")

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs_cam_keys": list(self.obs_cam_keys),
            "frame_stride": self.frame_stride,
            "camera_preset": self.camera_preset,
        }

    def min_segment_frames(self, *, min_sampled_frames: int) -> int:
        """Minimum raw-frame segment length to produce at least *min_sampled_frames* after striding."""
        if min_sampled_frames < 1:
            raise ValueError(f"min_sampled_frames must be >= 1, got {min_sampled_frames}")
        return self.frame_stride * (min_sampled_frames - 1) + 1

    def resolution_for(self, cam_key: str) -> tuple[int, int]:
        try:
            return self.resolved_camera_resolutions()[cam_key]
        except KeyError as exc:
            raise KeyError(f"Unknown camera key {cam_key!r}") from exc


@dataclass(frozen=True)
class LatentTrainConfig:
    """Training-time action configuration.

    Fields
    ------
    action_keys:
        Dataset field names to read, e.g. ``["action"]`` or
        ``["actions.end.position", "actions.effector.position"]``.
    action_key_channel_map:
        Nested list parallel to *action_keys* (serialised as
        ``used_action_channel_ids`` in the JSON config).  Each sub-list has
        length equal to the field's flattened dimension.  Non-null entries
        are model-space target indices; ``None`` skips that raw dimension.
    used_action_channel_ids / inverse_used_action_channel_ids:
        Flattened active model-space IDs and their inverse mapping, derived
        automatically from *action_key_channel_map*.
    norm_stat:
        ``None`` when ``action_transform == "identity"`` — q01/q99 are then
        auto-read from ``meta/stats.json`` at training time.  Non-identity
        transforms require the user to supply *norm_stat* explicitly.
    """

    latent_layout: str
    action_transform: str
    action_dim: int
    action_keys: list[str]
    action_key_channel_map: list[list[int | None]]
    used_action_channel_ids: list[int]
    inverse_used_action_channel_ids: list[int]
    action_norm_method: str
    norm_stat: dict[str, list[float]] | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "LatentTrainConfig":
        _require_keys(
            payload,
            [
                "latent_layout",
                "action_transform",
                "action_dim",
                "action_keys",
                "used_action_channel_ids",
                "action_norm_method",
            ],
            label,
        )

        latent_layout = _validate_nonempty_string(payload["latent_layout"], f"{label}.latent_layout")
        if latent_layout not in _SUPPORTED_LATENT_LAYOUTS:
            raise ValueError(
                f"{label}.latent_layout: unsupported value {latent_layout!r}. "
                f"supported={sorted(_SUPPORTED_LATENT_LAYOUTS)}"
            )

        action_transform = _validate_nonempty_string(
            payload["action_transform"], f"{label}.action_transform"
        )
        if action_transform not in _SUPPORTED_ACTION_TRANSFORMS:
            raise ValueError(
                f"{label}.action_transform: unsupported value {action_transform!r}. "
                f"supported={sorted(_SUPPORTED_ACTION_TRANSFORMS)}"
            )

        action_dim = _validate_positive_int(payload["action_dim"], f"{label}.action_dim")
        action_keys = _validate_string_list(payload["action_keys"], f"{label}.action_keys")
        if len(set(action_keys)) != len(action_keys):
            raise ValueError(f"{label}.action_keys: duplicate keys are not allowed")

        if action_transform != "identity" and len(action_keys) > 1:
            raise ValueError(
                f"{label}: action_transform={action_transform!r} requires exactly one "
                f"action_key because it hard-codes the concat layout; got {len(action_keys)} keys. "
                f"Use 'identity' for multi-field action datasets."
            )

        raw_channel_ids = payload["used_action_channel_ids"]
        if not isinstance(raw_channel_ids, list) or not raw_channel_ids:
            raise ValueError(f"{label}.used_action_channel_ids: expected a non-empty list of lists")
        if len(raw_channel_ids) != len(action_keys):
            raise ValueError(
                f"{label}.used_action_channel_ids: expected {len(action_keys)} sub-lists "
                f"(one per action_key), got {len(raw_channel_ids)}"
            )

        action_key_channel_map: list[list[int | None]] = []
        flat_used_ids: list[int] = []
        seen_model_ids: set[int] = set()

        for ki, (key, sub_list) in enumerate(zip(action_keys, raw_channel_ids)):
            sub_label = f"{label}.used_action_channel_ids[{ki}] ({key!r})"
            if not isinstance(sub_list, list):
                raise ValueError(f"{sub_label}: expected a list, got {type(sub_list).__name__}")
            parsed_sub: list[int | None] = []
            for ji, elem in enumerate(sub_list):
                if elem is None:
                    parsed_sub.append(None)
                elif isinstance(elem, int) and not isinstance(elem, bool):
                    if elem < 0 or elem >= action_dim:
                        raise ValueError(
                            f"{sub_label}[{ji}]: model-space id {elem} out of range [0, {action_dim})"
                        )
                    if elem in seen_model_ids:
                        raise ValueError(
                            f"{sub_label}[{ji}]: duplicate model-space id {elem}"
                        )
                    seen_model_ids.add(elem)
                    flat_used_ids.append(elem)
                    parsed_sub.append(elem)
                else:
                    raise ValueError(
                        f"{sub_label}[{ji}]: expected int or null, got {elem!r}"
                    )
            action_key_channel_map.append(parsed_sub)

        action_norm_method = _validate_nonempty_string(
            payload["action_norm_method"], f"{label}.action_norm_method"
        )
        if action_norm_method not in _SUPPORTED_ACTION_NORM_METHODS:
            raise ValueError(
                f"{label}.action_norm_method: unsupported value {action_norm_method!r}. "
                f"supported={sorted(_SUPPORTED_ACTION_NORM_METHODS)}"
            )

        if action_transform == "identity":
            if "norm_stat" in payload:
                raise ValueError(
                    f"{label}.norm_stat: must not be specified when action_transform "
                    f"is 'identity' (statistics are read from meta/stats.json)."
                )
            norm_stat = None
        else:
            if "norm_stat" not in payload:
                raise ValueError(
                    f"{label}: norm_stat is required when action_transform is "
                    f"{action_transform!r} (non-identity transforms invalidate "
                    f"raw dataset statistics)."
                )
            raw_norm_stat = payload["norm_stat"]
            if not isinstance(raw_norm_stat, dict):
                raise ValueError(f"{label}.norm_stat: expected an object")
            _require_keys(raw_norm_stat, ["q01", "q99"], f"{label}.norm_stat")
            q01 = raw_norm_stat["q01"]
            q99 = raw_norm_stat["q99"]
            if not isinstance(q01, list) or not isinstance(q99, list):
                raise ValueError(f"{label}.norm_stat: q01 and q99 must both be lists")
            if len(q01) != action_dim or len(q99) != action_dim:
                raise ValueError(
                    f"{label}.norm_stat: q01/q99 must both have length {action_dim}"
                )
            norm_stat = {
                "q01": [float(v) for v in q01],
                "q99": [float(v) for v in q99],
            }

        inverse_ids = payload.get("inverse_used_action_channel_ids")
        expected_inverse_ids = _build_inverse_ids(action_dim, action_key_channel_map)
        if inverse_ids is None:
            inverse_ids = expected_inverse_ids
        elif inverse_ids != expected_inverse_ids:
            raise ValueError(
                f"{label}.inverse_used_action_channel_ids does not match used_action_channel_ids"
            )

        return cls(
            latent_layout=latent_layout,
            action_transform=action_transform,
            action_dim=action_dim,
            action_keys=list(action_keys),
            action_key_channel_map=[list(sub) for sub in action_key_channel_map],
            used_action_channel_ids=list(flat_used_ids),
            inverse_used_action_channel_ids=list(inverse_ids),
            action_norm_method=action_norm_method,
            norm_stat=norm_stat,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "latent_layout": self.latent_layout,
            "action_transform": self.action_transform,
            "action_dim": self.action_dim,
            "action_keys": list(self.action_keys),
            "used_action_channel_ids": [list(sub) for sub in self.action_key_channel_map],
            "inverse_used_action_channel_ids": list(self.inverse_used_action_channel_ids),
            "action_norm_method": self.action_norm_method,
        }
        if self.norm_stat is not None:
            d["norm_stat"] = {
                "q01": list(self.norm_stat["q01"]),
                "q99": list(self.norm_stat["q99"]),
            }
        return d

    def transformed_action_dim(self, raw_action_dim: int, *, label: str) -> int:
        """Action width after ``action_transform`` (identity preserves width)."""
        if self.action_transform == "identity":
            return raw_action_dim

        if raw_action_dim < _ROBOTWIN_RELATIVE_POSE_RAW_ACTION_DIM:
            raise ValueError(
                f"{label}: training.action_transform='robotwin_relative_pose_bimanual' requires "
                f"dataset action dimension >= {_ROBOTWIN_RELATIVE_POSE_RAW_ACTION_DIM}, got {raw_action_dim}"
            )

        return _ROBOTWIN_RELATIVE_POSE_TRANSFORMED_ACTION_DIM

    def validate_runtime_contract(self, raw_action_dims: list[int], *, label: str):
        """Check that channel maps match actual dataset field dimensions."""
        if len(raw_action_dims) != len(self.action_keys):
            raise ValueError(
                f"{label}: expected {len(self.action_keys)} raw_action_dims "
                f"(one per action_key), got {len(raw_action_dims)}"
            )
        for ki, (key, sub_map, rdim) in enumerate(
            zip(self.action_keys, self.action_key_channel_map, raw_action_dims)
        ):
            if len(sub_map) != rdim:
                raise ValueError(
                    f"{label}: action_key {key!r} has {rdim} dimensions in the dataset, "
                    f"but used_action_channel_ids[{ki}] has length {len(sub_map)}"
                )
        total_raw_dim = sum(raw_action_dims)
        transformed_dim = self.transformed_action_dim(total_raw_dim, label=label)
        max_raw_offset = max(
            (v for v in self.inverse_used_action_channel_ids if v < total_raw_dim),
            default=-1,
        )
        if max_raw_offset >= transformed_dim:
            raise ValueError(
                f"{label}: an active channel maps to raw offset {max_raw_offset}, but "
                f"action_transform={self.action_transform!r} produces only {transformed_dim} columns"
            )


@dataclass(frozen=True)
class DatasetUserConfig:
    preprocess: LatentPreprocessConfig
    training: LatentTrainConfig

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "DatasetUserConfig":
        _require_keys(payload, ["preprocess", "training"], label)
        preprocess = payload["preprocess"]
        training = payload["training"]
        if not isinstance(preprocess, dict):
            raise ValueError(f"{label}.preprocess: expected an object")
        if not isinstance(training, dict):
            raise ValueError(f"{label}.training: expected an object")
        preprocess_cfg = LatentPreprocessConfig.from_dict(preprocess, label=f"{label}.preprocess")
        training_cfg = LatentTrainConfig.from_dict(training, label=f"{label}.training")
        return cls(preprocess=preprocess_cfg, training=training_cfg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess": self.preprocess.to_dict(),
            "training": self.training.to_dict(),
        }


@dataclass(frozen=True)
class FrozenLatentMetadata:
    preprocess: LatentPreprocessConfig
    codebase_version: str
    actual_fps: float
    vae_temporal_downsample: int = DEFAULT_VAE_TEMPORAL_DOWNSAMPLE
    extraction_status: str = LATENT_METADATA_STATUS_COMPLETE
    model_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "FrozenLatentMetadata":
        _require_keys(
            payload,
            [
                "metadata_version",
                "preprocess",
                "codebase_version",
                "actual_fps",
                "vae_temporal_downsample",
            ],
            label,
        )
        version = payload["metadata_version"]
        if version != LATENT_METADATA_VERSION:
            raise ValueError(
                f"{label}.metadata_version: expected {LATENT_METADATA_VERSION}, got {version!r}"
            )
        preprocess = payload["preprocess"]
        if not isinstance(preprocess, dict):
            raise ValueError(f"{label}.preprocess: expected an object")
        return cls(
            preprocess=LatentPreprocessConfig.from_dict(preprocess, label=f"{label}.preprocess"),
            codebase_version=str(payload["codebase_version"]),
            actual_fps=float(payload["actual_fps"]),
            vae_temporal_downsample=_validate_positive_int(
                payload["vae_temporal_downsample"],
                f"{label}.vae_temporal_downsample",
            ),
            extraction_status=_validate_metadata_status(
                payload.get("extraction_status", LATENT_METADATA_STATUS_COMPLETE),
                f"{label}.extraction_status",
            ),
            model_path=payload.get("model_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_version": LATENT_METADATA_VERSION,
            "preprocess": self.preprocess.to_dict(),
            "codebase_version": self.codebase_version,
            "actual_fps": self.actual_fps,
            "vae_temporal_downsample": self.vae_temporal_downsample,
            "extraction_status": self.extraction_status,
            "model_path": self.model_path,
        }


def _validate_metadata_status(value: Any, label: str) -> str:
    value = _validate_nonempty_string(value, label)
    if value not in _SUPPORTED_METADATA_STATUSES:
        raise ValueError(
            f"{label}: unsupported value {value!r}. supported={sorted(_SUPPORTED_METADATA_STATUSES)}"
        )
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{label}: expected a JSON object at {path}")
    return data


def _load_json_file(path: Path, loader, missing_message: str):
    if not path.exists():
        raise FileNotFoundError(missing_message)
    label = str(path)
    return loader(_read_json(path, label=label), label=label)


def load_dataset_user_config(dataset_root: str | Path) -> DatasetUserConfig:
    dataset_root = Path(dataset_root)
    path = dataset_root / DATASET_USER_CONFIG_FILENAME
    return _load_json_file(
        path,
        DatasetUserConfig.from_dict,
        f"{dataset_root.name}: missing dataset config at {path}. "
        "Create meta/wan_va_config.json before latent extraction or training.",
    )


def load_frozen_latent_metadata(dataset_root: str | Path) -> FrozenLatentMetadata:
    dataset_root = Path(dataset_root)
    path = dataset_root / LATENT_METADATA_FILENAME
    return _load_json_file(
        path,
        FrozenLatentMetadata.from_dict,
        f"{dataset_root.name}: missing latent metadata at {path}. "
        "Run wan_va.dataset.extract_latents after preparing meta/wan_va_config.json.",
    )


def write_frozen_latent_metadata(dataset_root: str | Path, metadata: FrozenLatentMetadata):
    dataset_root = Path(dataset_root)
    path = dataset_root / LATENT_METADATA_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata.to_dict(), f, indent=2)
        f.write("\n")
    tmp_path.replace(path)
