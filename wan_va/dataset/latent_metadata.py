from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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


def _build_inverse_ids(action_dim: int, used_action_channel_ids: list[int]) -> list[int]:
    inverse_ids = [len(used_action_channel_ids)] * action_dim
    for i, channel_id in enumerate(used_action_channel_ids):
        inverse_ids[channel_id] = i
    return inverse_ids


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
    target_fps: int
    camera_preset: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "LatentPreprocessConfig":
        _require_keys(payload, ["obs_cam_keys", "target_fps", "camera_preset"], label)

        obs_cam_keys = _validate_string_list(payload["obs_cam_keys"], f"{label}.obs_cam_keys")
        if len(set(obs_cam_keys)) != len(obs_cam_keys):
            raise ValueError(f"{label}.obs_cam_keys: duplicate camera keys are not allowed")
        target_fps = _validate_positive_int(payload["target_fps"], f"{label}.target_fps")
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
            target_fps=target_fps,
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
            "target_fps": self.target_fps,
            "camera_preset": self.camera_preset,
        }

    def frame_stride(self, dataset_fps: float) -> int:
        return max(1, round(dataset_fps / self.target_fps))

    def min_segment_frames(self, dataset_fps: float, *, min_sampled_frames: int) -> int:
        """Minimum raw-frame segment length to produce at least *min_sampled_frames* after striding."""
        if min_sampled_frames < 1:
            raise ValueError(f"min_sampled_frames must be >= 1, got {min_sampled_frames}")
        return self.frame_stride(dataset_fps) * (min_sampled_frames - 1) + 1

    def resolution_for(self, cam_key: str) -> tuple[int, int]:
        try:
            return self.resolved_camera_resolutions()[cam_key]
        except KeyError as exc:
            raise KeyError(f"Unknown camera key {cam_key!r}") from exc


@dataclass(frozen=True)
class LatentTrainConfig:
    latent_layout: str
    action_transform: str
    action_dim: int
    used_action_channel_ids: list[int]
    inverse_used_action_channel_ids: list[int]
    action_norm_method: str
    norm_stat: dict[str, list[float]]

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, label: str) -> "LatentTrainConfig":
        _require_keys(
            payload,
            [
                "latent_layout",
                "action_transform",
                "action_dim",
                "used_action_channel_ids",
                "action_norm_method",
                "norm_stat",
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
        used_action_channel_ids = payload["used_action_channel_ids"]
        if (
            not isinstance(used_action_channel_ids, list)
            or any(not isinstance(v, int) or isinstance(v, bool) for v in used_action_channel_ids)
        ):
            raise ValueError(f"{label}.used_action_channel_ids: expected a list of integers")
        if len(set(used_action_channel_ids)) != len(used_action_channel_ids):
            raise ValueError(f"{label}.used_action_channel_ids: duplicate channel ids are not allowed")
        if any(v < 0 or v >= action_dim for v in used_action_channel_ids):
            raise ValueError(
                f"{label}.used_action_channel_ids: ids must be in [0, {action_dim})"
            )

        action_norm_method = _validate_nonempty_string(
            payload["action_norm_method"], f"{label}.action_norm_method"
        )
        if action_norm_method not in _SUPPORTED_ACTION_NORM_METHODS:
            raise ValueError(
                f"{label}.action_norm_method: unsupported value {action_norm_method!r}. "
                f"supported={sorted(_SUPPORTED_ACTION_NORM_METHODS)}"
            )

        norm_stat = payload["norm_stat"]
        if not isinstance(norm_stat, dict):
            raise ValueError(f"{label}.norm_stat: expected an object")
        _require_keys(norm_stat, ["q01", "q99"], f"{label}.norm_stat")
        q01 = norm_stat["q01"]
        q99 = norm_stat["q99"]
        if not isinstance(q01, list) or not isinstance(q99, list):
            raise ValueError(f"{label}.norm_stat: q01 and q99 must both be lists")
        if len(q01) != action_dim or len(q99) != action_dim:
            raise ValueError(
                f"{label}.norm_stat: q01/q99 must both have length {action_dim}"
            )

        inverse_ids = payload.get("inverse_used_action_channel_ids")
        expected_inverse_ids = _build_inverse_ids(action_dim, list(used_action_channel_ids))
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
            used_action_channel_ids=list(used_action_channel_ids),
            inverse_used_action_channel_ids=list(inverse_ids),
            action_norm_method=action_norm_method,
            norm_stat={
                "q01": [float(v) for v in q01],
                "q99": [float(v) for v in q99],
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latent_layout": self.latent_layout,
            "action_transform": self.action_transform,
            "action_dim": self.action_dim,
            "used_action_channel_ids": list(self.used_action_channel_ids),
            "inverse_used_action_channel_ids": list(self.inverse_used_action_channel_ids),
            "action_norm_method": self.action_norm_method,
            "norm_stat": {
                "q01": list(self.norm_stat["q01"]),
                "q99": list(self.norm_stat["q99"]),
            },
        }

    def transformed_action_dim(self, raw_action_dim: int, *, label: str) -> int:
        """Return the action width after applying action_transform to raw parquet actions.

        Note: ``action_dim`` is the *model-space* width (e.g. 30) used for
        channel remapping via ``inverse_used_action_channel_ids``.  It is
        intentionally **not** compared to ``raw_action_dim`` here — a compact
        dataset (e.g. 6-dim) is mapped into a wider model space by
        ``_action_post_process``.
        """
        if self.action_transform == "identity":
            return raw_action_dim

        if raw_action_dim < _ROBOTWIN_RELATIVE_POSE_RAW_ACTION_DIM:
            raise ValueError(
                f"{label}: training.action_transform='robotwin_relative_pose_bimanual' requires "
                f"dataset action dimension >= {_ROBOTWIN_RELATIVE_POSE_RAW_ACTION_DIM}, got {raw_action_dim}"
            )

        return _ROBOTWIN_RELATIVE_POSE_TRANSFORMED_ACTION_DIM

    def validate_runtime_contract(self, raw_action_dim: int, *, label: str):
        transformed_dim = self.transformed_action_dim(raw_action_dim, label=label)
        sentinel_id = len(self.used_action_channel_ids)
        if sentinel_id > transformed_dim:
            raise ValueError(
                f"{label}: used_action_channel_ids has length {sentinel_id}, but "
                f"training.action_transform={self.action_transform!r} exposes only {transformed_dim} channels"
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
