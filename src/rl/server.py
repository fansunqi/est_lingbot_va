"""Online action-only GRPO server for LingBot-VA on RoboTwin.

This entrypoint deliberately lives beside, rather than inside,
``src.inference.server``. It reuses the inference server's model loading,
VAE/text encoder placement, observation encoding, and KV-cache machinery, but
owns optimizer state, rollout storage, and GRPO updates.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
import yamlinclude
from easydict import EasyDict
from einops import rearrange
from tqdm import tqdm

from src.distributed.util import init_distributed
from src.inference.server import VA_Server, _show_denoise_progress, load_inference_config
from src.rl.grpo import (
    compute_group_advantages,
    flow_grpo_gaussian_logprob,
    flow_grpo_sde_transition,
    grpo_clipped_loss,
    scheduler_transition_mean,
)
from src.rl.lora import apply_lora
from src.rl.rollout_store import RolloutChunk, RolloutStore
from src.utils import data_seq_to_patch, init_logger, logger, run_async_server_mode


def _recursive_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _recursive_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _as_easydict(value):
    if isinstance(value, dict):
        return EasyDict({k: _as_easydict(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_as_easydict(v) for v in value]
    return value


def _json_safe(value):
    if isinstance(value, EasyDict):
        value = dict(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return str(tuple(value.shape))
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _episode_summary_stats(records: list[dict[str, Any]]) -> dict[str, float]:
    n = len(records)
    rewards = [float(item.get("reward", 0.0)) for item in records]
    successes = [1.0 if bool(item.get("success", False)) else 0.0 for item in records]
    steps = [float(item.get("step_count", 0.0)) for item in records]
    chunks = [float(item.get("chunks", 0.0)) for item in records]

    reward_sum = sum(rewards)
    success_count = sum(successes)
    step_sum = sum(steps)
    chunk_sum = sum(chunks)
    reward_mean = reward_sum / n if n else 0.0
    reward_var = sum((value - reward_mean) ** 2 for value in rewards) / n if n else 0.0
    return {
        "episode_count": float(n),
        "reward_sum": float(reward_sum),
        "reward_mean": float(reward_mean),
        "reward_std": float(reward_var ** 0.5),
        "success_count": float(success_count),
        "success_rate": float(success_count / n) if n else 0.0,
        "step_count_sum": float(step_sum),
        "step_count_mean": float(step_sum / n) if n else 0.0,
        "chunk_count_sum": float(chunk_sum),
        "chunk_count_mean": float(chunk_sum / n) if n else 0.0,
    }


def _resolve_grpo_minibatch_sizes(
    rl_cfg,
    *,
    group_size: int,
    rollout_groups_per_update: int,
    world_size: int,
    sharded_group_enabled: bool,
) -> tuple[int, int]:
    """Return (global_minibatch_size, local_minibatch_size).

    In replicated sharded-group mode each optimizer step averages gradients
    across ranks. The config's ``batch_size`` is therefore the logical/global
    episode count, while each rank should process ``batch_size / world_size``
    local episodes so the post-allreduce gradient is the mean over the same
    global minibatch size as single-GPU GRPO.
    """
    configured = rl_cfg.get("batch_size", None)
    if configured is None:
        global_minibatch_size = max(
            1,
            int(group_size) * max(int(rollout_groups_per_update), 1),
        )
    else:
        global_minibatch_size = max(1, int(configured))

    if not sharded_group_enabled:
        return global_minibatch_size, global_minibatch_size

    if global_minibatch_size % int(world_size) != 0:
        raise ValueError(
            "Sharded replicated GRPO requires rl.batch_size to be divisible "
            f"by world_size so every rank contributes the same local minibatch "
            f"size (got batch_size={global_minibatch_size}, world_size={world_size})."
        )
    local_minibatch_size = max(1, global_minibatch_size // int(world_size))
    return global_minibatch_size, local_minibatch_size


def load_rl_config(config_path: str) -> EasyDict:
    config_path = Path(config_path)
    config_root = config_path.parent.parent
    yaml.add_constructor(
        "!inc",
        yamlinclude.YamlIncludeConstructor(base_dir=str(config_root)),
        Loader=yaml.FullLoader,
    )
    with open(config_path, "r") as f:
        raw = yaml.full_load(f)

    base_config = raw.pop("base_inference_config")
    base = dict(load_inference_config(base_config))
    merged = _recursive_update(base, raw)
    cfg = _as_easydict(merged)
    cfg.rl = _as_easydict(cfg.get("rl", {}))
    cfg.mode = "server"
    cfg.infer_mode = "server"
    # GRPO v1 action sampling is defined without action CFG.
    cfg.action_guidance_scale = 1
    if not bool(cfg.rl.get("use_fsdp", False)):
        cfg.pop("fsdp", None)
        cfg.enable_offload = bool(cfg.get("enable_offload", False))
        cfg.vae_offload = bool(cfg.get("vae_offload", cfg.enable_offload))
    return cfg


class GRPOTrainingServer(VA_Server):
    def __init__(self, job_config):
        super().__init__(job_config)
        self.rl_cfg = job_config.rl
        self.disable_updates = bool(self.rl_cfg.get("disable_updates", False))
        self.lora_cfg = self.rl_cfg.get("lora", {}) or {}
        self.use_lora = bool(self.lora_cfg.get("enabled", False))
        if self.disable_updates:
            self.transformer.eval()
            self.transformer.requires_grad_(False)
        elif self.use_lora:
            if self.fsdp_enabled:
                raise NotImplementedError("GRPO LoRA is currently supported only without FSDP")
            self.transformer.train()
            target_modules = self.lora_cfg.get(
                "target_modules",
                ["to_q", "to_k", "to_v", "to_out.0", "action_embedder", "action_proj_out"],
            )
            stats = apply_lora(
                self.transformer,
                rank=int(self.lora_cfg.get("rank", 8)),
                alpha=float(self.lora_cfg.get("alpha", 16.0)),
                dropout=float(self.lora_cfg.get("dropout", 0.0)),
                target_modules=list(target_modules),
                freeze_base=True,
            )
            logger.info(
                "GRPO LoRA enabled: rank=%s lora_rank=%s alpha=%s dropout=%s wrapped=%s trainable_params=%s total_params=%s",
                self.rank,
                int(self.lora_cfg.get("rank", 8)),
                float(self.lora_cfg.get("alpha", 16.0)),
                float(self.lora_cfg.get("dropout", 0.0)),
                len(stats.wrapped_modules),
                stats.trainable_parameters,
                stats.total_parameters,
            )
        else:
            self.transformer.train()
            self.transformer.requires_grad_(True)
        if hasattr(self.transformer, "condition_embedder_action"):
            branch = self.transformer.condition_embedder_action
            if hasattr(branch, "text_embedder"):
                for p in branch.text_embedder.parameters():
                    p.requires_grad_(False)

        # Sharded-group GRPO: under replicated multi-GPU, a single logical
        # group of size ``group_size`` is split across ranks (each rank stores
        # ``local_group_size = group_size // world_size`` members) and
        # advantages are normalized over the FULL group via cross-rank reward
        # gather. This keeps effective batch identical to single-GPU runs and
        # halves rollout wall-clock with world_size=2. Single-GPU and FSDP runs
        # keep the legacy behavior (one rank owns the entire group).
        self.group_size = int(self.rl_cfg.get("group_size", 2))
        self.sharded_group_enabled = bool(getattr(self, "replicated_enabled", False))
        if self.sharded_group_enabled:
            if self.group_size % self.world_size != 0:
                raise ValueError(
                    "Sharded-group GRPO requires rl.group_size divisible by world_size "
                    f"(got group_size={self.group_size}, world_size={self.world_size})."
                )
            self.local_group_size = self.group_size // self.world_size
        else:
            self.local_group_size = self.group_size
        self.rollout_store = RolloutStore(group_size=self.local_group_size)
        self.rollout_groups_per_update = int(self.rl_cfg.get("rollout_groups_per_update", 1))
        self._pending_ready_groups = []
        self._rollout_generators: dict[str, torch.Generator] = {}
        self.global_update_step = 0
        self.global_rollout_episode = 0
        self.global_eval_episode = 0
        # Periodic eval pass. eval_every=N (in global_update_step units) arms
        # the eval phase after each GRPO update whose step is a multiple of N.
        # Client polls run_pending_updates / get_eval_phase, then drives one
        # deterministic pass (no per-step noise) over its assignment with
        # is_eval=True; episodes route through _eval_sessions / _eval_results
        # and never enter _pending_ready_groups.
        # eval_enabled is a hard master switch: when False, no eval ever runs
        # (no initial eval, no periodic eval) regardless of eval_every. Useful
        # for cheap debug runs where eval would dominate wall-clock.
        self.eval_enabled = bool(self.rl_cfg.get("eval_enabled", True))
        self.eval_every = int(self.rl_cfg.get("eval_every", 0)) if self.eval_enabled else 0
        self.eval_action_num_inference_steps = int(
            self.rl_cfg.get(
                "eval_action_num_inference_steps",
                getattr(job_config, "action_num_inference_steps", 0),
            )
        )
        if self.eval_action_num_inference_steps <= 0:
            raise ValueError(
                "rl.eval_action_num_inference_steps must be positive "
                f"(got {self.eval_action_num_inference_steps})"
            )
        # Arm an initial eval pass on fresh startup so we measure the policy's
        # success rate before any GRPO update runs. load_checkpoint clears this
        # when resuming from a non-zero step (that run already evaluated at its
        # boundary).
        self._eval_pending: bool = self.eval_enabled and self.eval_every > 0
        self._eval_sessions: set[str] = set()
        self._eval_results: list[dict[str, Any]] = []
        self.last_stats: dict[str, Any] = {}
        self._wandb_run = None
        # Background checkpoint writer. Disk IO on /root/autodl-tmp can stall
        # for minutes; doing torch.save() inside the websocket handler freezes
        # the asyncio event loop and rollout clients time out waiting for
        # run_pending_updates to return.
        self._checkpoint_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="grpo-ckpt"
        )
        self._pending_checkpoint_future: Future | None = None

        self.trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]
        self._sync_replicated_trainable_params_from_rank0()
        self.optimizer = None
        if not self.disable_updates:
            opt_cfg = self.rl_cfg.get("optimizer", {}) or {}
            lr = float(opt_cfg.get("lr", 1e-6))
            weight_decay = float(opt_cfg.get("weight_decay", 0.0))
            betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
            if not self.trainable_params:
                raise RuntimeError("GRPO updates are enabled but no trainable parameters were found")
            self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler_lr = None

        # Flow-GRPO ODE-to-SDE noise scale ``a`` in sigma_t = a * sqrt(t / (1 - t)).
        # Legacy ``action_noise_std`` no longer controls rollout noise; the
        # transition std now comes from sigma_t * sqrt(abs(dt)).
        self.flow_grpo_noise_level = float(
            self.rl_cfg.get("flow_grpo_noise_level", self.rl_cfg.get("noise_level", 0.7))
        )
        self.flow_t_eps = float(self.rl_cfg.get("flow_t_eps", 1e-5))
        # Optional clamp on the SDE diffusion coefficient sigma_t = a*sqrt(t/(1-t))
        # into a narrow band, applied consistently to drift + std (see
        # flow_grpo_sde_transition). None = unclamped raw schedule.
        _sigma_min = self.rl_cfg.get("flow_grpo_sigma_min", None)
        _sigma_max = self.rl_cfg.get("flow_grpo_sigma_max", None)
        self.flow_grpo_sigma_min = float(_sigma_min) if _sigma_min is not None else None
        self.flow_grpo_sigma_max = float(_sigma_max) if _sigma_max is not None else None
        self.clip_range = float(self.rl_cfg.get("clip_range", 0.2))
        # Optional asymmetric trust region (LaST-R1 clip_ratio_high/low style).
        # None = fall back to symmetric clip_range. A wider high side gives
        # positive-advantage samples more room to raise probability.
        _ch = self.rl_cfg.get("clip_range_high", None)
        _cl = self.rl_cfg.get("clip_range_low", None)
        self.clip_range_high = float(_ch) if _ch is not None else self.clip_range
        self.clip_range_low = float(_cl) if _cl is not None else self.clip_range
        self.log_ratio_clip = self.rl_cfg.get("log_ratio_clip", 20.0)
        self.log_ratio_clip = None if self.log_ratio_clip is None else float(self.log_ratio_clip)
        self.beta_kl = float(self.rl_cfg.get("beta_kl", 0.0))
        self.entropy_coef = float(self.rl_cfg.get("entropy_coef", 0.0))
        self.target_kl = self.rl_cfg.get("target_kl", None)
        self.target_kl = None if self.target_kl is None else float(self.target_kl)
        self.update_epochs = int(self.rl_cfg.get("update_epochs", 1))
        # Minibatch size in *episodes* used inside one GRPO update. In
        # replicated sharded-group mode, the config's batch_size is logical /
        # global; each rank processes batch_size/world_size local episodes and
        # the all-reduced gradient is the mean over the global minibatch.
        self.global_minibatch_size, self.minibatch_size = _resolve_grpo_minibatch_sizes(
            self.rl_cfg,
            group_size=self.group_size,
            rollout_groups_per_update=self.rollout_groups_per_update,
            world_size=self.world_size,
            sharded_group_enabled=self.sharded_group_enabled,
        )
        self.validate_logprob_consistency = bool(
            self.rl_cfg.get("validate_logprob_consistency", False)
        )
        self.logprob_consistency_atol = float(
            self.rl_cfg.get("logprob_consistency_atol", 5e-3)
        )
        self.diagnose_update_effect = bool(
            self.rl_cfg.get("diagnose_update_effect", False)
        )
        # Per-denoising-step log_ratio diagnostic. Answers: is the chunk-level
        # ratio dominated by the low-std (late, near-clean) transitions? For
        # each step index i we record (t_i, sigma_i, new_lp_i, old_lp_i) during
        # the detached forward, then aggregate |log_ratio_i| by i at update end.
        # If |log_ratio_i| grows as sigma_i shrinks -> the fixed sigma schedule
        # is ill-conditioned across steps; if it's flat -> the schedule is fine
        # and outliers come from elsewhere (forward drift, trajectory length).
        self.diagnose_per_step_logratio = bool(
            self.rl_cfg.get("diagnose_per_step_logratio", False)
        )
        self._per_step_diag_active = False
        self._per_step_diag_records: list[dict[str, float]] = []
        self.normalize_denoising_horizon = bool(
            self.rl_cfg.get("logprob", {}).get("normalize_denoising_horizon", True)
        )
        logprob_cfg = self.rl_cfg.get("logprob", {}) or {}
        if "reduce" in logprob_cfg:
            self.logprob_reduce = str(logprob_cfg.get("reduce", "mean")).lower()
            if self.logprob_reduce not in ("sum", "mean"):
                raise ValueError(
                    f"rl.logprob.reduce must be 'sum' or 'mean', got {self.logprob_reduce!r}"
                )
            self.normalize_action_dim = self.logprob_reduce == "mean"
        else:
            self.normalize_action_dim = bool(logprob_cfg.get("normalize_action_dim", True))
            self.logprob_reduce = "mean" if self.normalize_action_dim else "sum"
        # When True, trajectory logprob is mean-over-chunks instead of sum-over-chunks.
        # This removes length-vs-reward correlation bias: with sum, failed episodes
        # (long, ~13 chunks) dominate the per-trajectory logprob magnitude vs successful
        # ones (short, ~3-7 chunks), and the PPO ratio drifts systematically below 1.
        self.normalize_episode_length = bool(
            self.rl_cfg.get("logprob", {}).get("normalize_episode_length", False)
        )

        noise_schedule = str(self.rl_cfg.get("noise_schedule", "per_chunk")).lower()
        if noise_schedule not in ("per_chunk", "per_step"):
            raise ValueError(
                f"rl.noise_schedule must be 'per_chunk' or 'per_step', got {noise_schedule!r}"
            )
        self.noise_schedule = noise_schedule
        if self.noise_schedule == "per_chunk" and self.normalize_denoising_horizon:
            logger.warning(
                "rl.logprob.normalize_denoising_horizon ignored: rank=%s "
                "under noise_schedule=per_chunk "
                "(there is only one Gaussian transition per chunk); forcing to False.",
                self.rank,
            )
            self.normalize_denoising_horizon = False

        Path(self.job_config.save_root).mkdir(parents=True, exist_ok=True)
        self._init_wandb()
        logger.info(
            "GRPO server ready: rank=%s group_size=%s local_group_size=%s sharded=%s "
            "update_epochs=%s flow_grpo_noise_level=%s flow_t_eps=%s logprob_reduce=%s "
            "noise_schedule=%s "
            "disable_updates=%s train_action_steps=%s eval_action_steps=%s "
            "global_batch_size=%s local_batch_size=%s "
            "diagnose_update_effect=%s",
            self.rank,
            self.group_size,
            self.local_group_size,
            self.sharded_group_enabled,
            self.update_epochs,
            self.flow_grpo_noise_level,
            self.flow_t_eps,
            self.logprob_reduce,
            self.noise_schedule,
            self.disable_updates,
            int(self.job_config.action_num_inference_steps),
            self.eval_action_num_inference_steps,
            self.global_minibatch_size,
            self.minibatch_size,
            self.diagnose_update_effect,
        )
        if self._eval_pending:
            logger.info(
                "GRPO initial eval phase armed at startup: rank=%s eval_every=%s global_update_step=%s",
                self.rank,
                self.eval_every,
                self.global_update_step,
            )

    def _init_wandb(self) -> None:
        if not self._is_rank0():
            return
        wb_cfg = self.job_config.get("wandb", None)
        if not wb_cfg:
            return
        mode = wb_cfg.get("mode", "online")
        if mode == "disabled":
            return
        try:
            import wandb
        except Exception as exc:
            logger.warning("wandb logging requested but unavailable: rank=%s %s", self.rank, exc)
            return

        wandb_dir = Path(self.job_config.save_root) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        init_kwargs = {
            "project": wb_cfg.get("project", "lingbot-grpo"),
            "name": wb_cfg.get("name"),
            "entity": wb_cfg.get("entity"),
            "mode": mode,
            "dir": str(wandb_dir),
            "config": _json_safe(dict(self.job_config)),
        }
        for optional_key in ("group", "job_type", "tags", "notes", "id", "resume"):
            if wb_cfg.get(optional_key) is not None:
                init_kwargs[optional_key] = wb_cfg.get(optional_key)
        init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}
        try:
            self._wandb_run = wandb.init(**init_kwargs)
            self._define_wandb_metrics(wandb)
            logger.info(
                "wandb logging enabled: rank=%s project=%s name=%s mode=%s",
                self.rank,
                init_kwargs["project"],
                init_kwargs.get("name"),
                mode,
            )
        except Exception as exc:
            logger.warning("Failed to initialize wandb logging: rank=%s %s", self.rank, exc)
            self._wandb_run = None

    def _define_wandb_metrics(self, wandb_module) -> None:
        """Give each metric family its own x-axis in W&B.

        Without explicit step metrics, every eval episode advances W&B's
        internal `_step`. A 64-episode eval pass then stretches rollout charts
        and creates vertical-looking 0/1 bands even though no rollout happened.
        """
        definitions = [
            ("train/global_update_step", None),
            ("train/*", "train/global_update_step"),
            ("train_epoch/global_update_step", None),
            ("train_epoch/*", "train_epoch/global_update_step"),
            ("global_rollout_episode", None),
            ("rollout/global_rollout_episode", "global_rollout_episode"),
            ("rollout/*", "global_rollout_episode"),
            ("group/global_rollout_episode", "global_rollout_episode"),
            ("group/*", "global_rollout_episode"),
            ("eval/global_update_step", None),
            ("eval/success_rate", "eval/global_update_step"),
            ("eval/reward_mean", "eval/global_update_step"),
            ("eval/n_episodes", "eval/global_update_step"),
            ("eval/per_seed/*", "eval/global_update_step"),
            ("eval/global_eval_episode", None),
            ("eval/episode_success", "eval/global_eval_episode"),
            ("eval/episode_reward", "eval/global_eval_episode"),
            ("eval/episode_step_count", "eval/global_eval_episode"),
            ("eval/phase_episode", "eval/global_eval_episode"),
            ("checkpoint/global_update_step", None),
            ("checkpoint/*", "checkpoint/global_update_step"),
        ]
        for metric, step_metric in definitions:
            kwargs = {"step_metric": step_metric} if step_metric is not None else {}
            wandb_module.define_metric(metric, **kwargs)

    def _info0(self, msg: str, *args: Any) -> None:
        """logger.info emitted only on rank 0.

        Used for per-rank-local lifecycle / detail lines (minibatch progress,
        group-ready, episode-finished, ...) so a 4-rank run logs one stream
        instead of four interleaved copies. The authoritative cross-rank numbers
        live in the aggregated ``GRPO update epoch [global]`` line.
        """
        if self._is_rank0():
            logger.info(msg, *args)

    def _wandb_log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        if self._wandb_run is None or not self._is_rank0():
            return
        try:
            self._wandb_run.log(_json_safe(data), step=step)
        except Exception as exc:
            logger.warning("Failed to log wandb metrics: rank=%s %s", self.rank, exc)

    def _action_logprob_mask(self, frame_st_id: int, reference: torch.Tensor) -> torch.Tensor:
        mask = self.action_mask.to(reference.device).view(1, -1, 1, 1, 1)
        mask = mask.expand(reference.shape[0], -1, reference.shape[2], reference.shape[3], reference.shape[4]).clone()
        if frame_st_id == 0:
            mask[:, :, 0:1] = False
        return mask

    def _flow_time(self, timestep: torch.Tensor) -> torch.Tensor:
        return timestep.to(device=self.device, dtype=self.dtype) / float(self.action_scheduler.num_train_timesteps)

    def _flow_dt(self, timestep: torch.Tensor, next_timestep: torch.Tensor) -> torch.Tensor:
        return (
            next_timestep.to(device=self.device, dtype=self.dtype)
            - timestep.to(device=self.device, dtype=self.dtype)
        ) / float(self.action_scheduler.num_train_timesteps)

    def _flow_dt_for_stored_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        scheduler_timesteps = self.action_scheduler.timesteps.to(device=self.device, dtype=self.dtype)
        timestep_d = timestep.to(device=self.device, dtype=self.dtype)
        timestep_id = torch.argmin((scheduler_timesteps - timestep_d).abs())
        if int(timestep_id) + 1 >= scheduler_timesteps.numel():
            next_timestep = torch.zeros_like(timestep_d)
        else:
            next_timestep = scheduler_timesteps[int(timestep_id) + 1]
        return self._flow_dt(timestep_d, next_timestep)

    def _flow_grpo_action_transition(
        self,
        actions: torch.Tensor,
        timestep: torch.Tensor,
        dt: torch.Tensor,
        velocity: torch.Tensor,
        *,
        eps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        transition = flow_grpo_sde_transition(
            actions,
            self._flow_time(timestep),
            dt,
            velocity,
            noise_level=self.flow_grpo_noise_level,
            t_eps=self.flow_t_eps,
            sigma_min=getattr(self, "flow_grpo_sigma_min", None),
            sigma_max=getattr(self, "flow_grpo_sigma_max", None),
        )
        if eps is not None:
            transition["x_next"] = transition["mean"] + transition["std"] * eps
            transition["eps"] = eps
        return transition

    def _action_transformer_step(
        self,
        actions: torch.Tensor,
        t: torch.Tensor,
        frame_st_id: int,
        *,
        last_step: bool,
    ):
        """Run one action-mode transformer forward and return the RF velocity.

        Returns ``(v_theta, action_cond)``. ``v_theta`` is None when ``last_step`` is True — that
        call's sole purpose is to update the KV cache with the final denoised actions.
        """
        action_cond = torch.zeros(
            [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
            device=self.device,
            dtype=self.dtype,
        ) if frame_st_id == 0 else None
        input_dict = self._prepare_latent_input(
            None,
            actions,
            t,
            t,
            None,
            action_cond,
            frame_st_id=frame_st_id,
        )
        action_noise_pred = self.transformer(
            self._repeat_input_for_cfg(input_dict["action_res_lst"]),
            update_cache=1 if last_step else 0,
            cache_name=self.cache_name,
            action_mode=True,
        )
        if last_step:
            return None, action_cond
        action_noise_pred = rearrange(
            action_noise_pred, "b (f n) c -> b c f n 1", f=self.job_config.frame_chunk_size
        )
        action_noise_pred = action_noise_pred[:1]
        return action_noise_pred, action_cond

    def _run_video_prefix(self, obs, frame_st_id: int, latent_noise: torch.Tensor | None = None) -> torch.Tensor:
        frame_chunk_size = self.job_config.frame_chunk_size
        with torch.no_grad():
            if frame_st_id == 0:
                init_latent = self._encode_obs(obs)
                self.init_latent = init_latent
            else:
                init_latent = self.init_latent
        self._ensure_transformer_cache()

        latents = latent_noise
        if latents is None:
            latents = torch.randn(
                1,
                48,
                frame_chunk_size,
                self.latent_height,
                self.latent_width,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            latents = latents.to(self.device, dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        video_step = self.job_config.video_exec_step
        self.scheduler.set_timesteps(video_inference_step)
        timesteps = F.pad(self.scheduler.timesteps, (0, 1), mode="constant", value=0)
        if video_step != -1:
            timesteps = timesteps[:video_step]

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps, disable=not _show_denoise_progress())):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )
                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (
                            video_noise_pred[:1] - video_noise_pred[1:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred, t, latents)
                if frame_st_id == 0:
                    latents[:, :, 0:1] = latent_cond
        return latents

    @torch.no_grad()
    def _sample_action_chunk(
        self,
        obs,
        frame_st_id: int,
        generator: torch.Generator | None = None,
        *,
        eval_mode: bool = False,
    ):
        """Sample one action chunk.

        When ``eval_mode`` is True every denoising transition uses the deterministic
        scheduler mean (no Gaussian injection, no logprob), matching the plain
        inference server's path. The returned chunk has placeholder logprob /
        action_chain fields — eval episodes are never appended to
        ``_pending_ready_groups`` and are dropped immediately after
        ``finish_episode``.
        """
        infer_start = time.monotonic()
        frame_chunk_size = self.job_config.frame_chunk_size
        latent_noise = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self._run_video_prefix(obs, frame_st_id, latent_noise=latent_noise)

        actions = torch.randn(
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        action_num_inference_steps = (
            self.eval_action_num_inference_steps
            if eval_mode
            else int(self.job_config.action_num_inference_steps)
        )
        self.action_scheduler.set_timesteps(action_num_inference_steps)
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)
        action_dts = (
            action_timesteps[1:] - action_timesteps[:-1]
        ) / float(self.action_scheduler.num_train_timesteps)
        mask = self._action_logprob_mask(frame_st_id, actions)

        per_chunk = self.noise_schedule == "per_chunk"
        # Index of the last noise-bearing transition (= last real scheduler step before the t==0 padding).
        final_noise_idx = len(action_timesteps) - 2

        if eval_mode or per_chunk:
            # In eval / per-chunk mode we store the minimal chunk needed to
            # round-trip back to the client; eval skips noise entirely so its
            # chunk is a pure placeholder.
            action_chain: list[torch.Tensor] = []
            transition_logprobs: list[torch.Tensor] = []
            stored_timesteps: list[torch.Tensor] = []
            stored_dts: list[torch.Tensor] = []
        else:
            action_chain = [actions.detach().cpu()]
            transition_logprobs = []
            stored_timesteps = None  # fall back to action_timesteps[:-1] at the end
            stored_dts = None  # fall back to action_dts at the end

        with torch.no_grad():
            for i, t in enumerate(tqdm(action_timesteps, disable=not _show_denoise_progress())):
                last_step = i == len(action_timesteps) - 1
                v_theta, action_cond = self._action_transformer_step(
                    actions, t, frame_st_id, last_step=last_step,
                )
                if last_step:
                    if action_cond is not None:
                        actions[:, :, 0:1] = action_cond
                    continue

                if eval_mode or (per_chunk and i < final_noise_idx):
                    # Deterministic ODE step: actions = scheduler mean, no noise, no logprob.
                    # eval_mode applies this to every step including the final
                    # noise-bearing one, so the trajectory matches plain
                    # deterministic inference under the same LoRA weights.
                    next_actions = scheduler_transition_mean(self.action_scheduler, v_theta, t, actions)
                    next_actions[:, ~self.action_mask] *= 0
                    if action_cond is not None:
                        next_actions[:, :, 0:1] = action_cond
                    actions = next_actions
                    continue

                # Noise-bearing Flow-GRPO SDE transition.
                dt = action_dts[i].to(device=self.device, dtype=self.dtype)
                eps = torch.randn(
                    actions.shape,
                    device=actions.device,
                    dtype=actions.dtype,
                    generator=generator,
                )
                transition = self._flow_grpo_action_transition(
                    actions,
                    t,
                    dt,
                    v_theta,
                    eps=eps,
                )
                next_actions = transition["x_next"]
                next_actions[:, ~self.action_mask] *= 0
                if action_cond is not None:
                    next_actions[:, :, 0:1] = action_cond
                lp = flow_grpo_gaussian_logprob(
                    next_actions,
                    transition["mean"],
                    transition["std"],
                    mask=mask,
                    logprob_reduce=self.logprob_reduce,
                )

                if per_chunk:
                    # Record the (prev, final) pair and the single timestep used.
                    action_chain.append(actions.detach().cpu())
                    action_chain.append(next_actions.detach().cpu())
                    stored_timesteps.append(t.detach().cpu().reshape(1))
                    stored_dts.append(dt.detach().cpu().reshape(1))
                    transition_logprobs.append(lp.detach().cpu())
                else:
                    transition_logprobs.append(lp.detach().cpu())
                    action_chain.append(next_actions.detach().cpu())

                actions = next_actions

        actions[:, ~self.action_mask] *= 0
        if eval_mode:
            old_logprobs_tensor = torch.zeros(1)
            old_logprob_summary = torch.zeros(1)
            stored_timesteps_tensor = torch.empty(0, dtype=action_timesteps.dtype)
            stored_dts_tensor = torch.empty(0, dtype=action_dts.dtype)
        elif per_chunk:
            assert len(transition_logprobs) == 1, "per_chunk mode must produce exactly one logprob term"
            old_logprobs_tensor = torch.stack(transition_logprobs, dim=0)
            old_logprob_summary = transition_logprobs[0]
            stored_timesteps_tensor = torch.cat(stored_timesteps, dim=0)
            stored_dts_tensor = torch.cat(stored_dts, dim=0)
        else:
            old_logprobs_tensor = torch.stack(transition_logprobs, dim=0)
            old_logprob_summary = old_logprobs_tensor.sum(dim=0)
            if self.normalize_denoising_horizon and transition_logprobs:
                old_logprob_summary = old_logprob_summary / len(transition_logprobs)
            stored_timesteps_tensor = action_timesteps[:-1].detach().cpu()
            stored_dts_tensor = action_dts.detach().cpu()
        env_action = self.postprocess_action(actions)
        logger.debug(
            "GRPO sample_action done: rank=%s frame_st_id=%s elapsed_ms=%.1f old_logprob=%.6f "
            "noise_schedule=%s eval_mode=%s action_steps=%s",
            self.rank,
            frame_st_id,
            (time.monotonic() - infer_start) * 1000,
            float(old_logprob_summary.mean()),
            self.noise_schedule,
            eval_mode,
            action_num_inference_steps,
        )
        return env_action, RolloutChunk(
            obs=copy.deepcopy(obs),
            frame_st_id=frame_st_id,
            latent_noise=latent_noise.detach().cpu(),
            action_chain=action_chain,
            old_logprobs=old_logprobs_tensor.detach().cpu(),
            action_timesteps=stored_timesteps_tensor,
            action_mask=mask.detach().cpu(),
            env_action=env_action,
            action_dts=stored_dts_tensor,
        )

    def _record_per_step_diag(
        self,
        chunk: RolloutChunk,
        step_idx: int,
        timestep: torch.Tensor,
        std: torch.Tensor,
        new_lp: torch.Tensor,
    ) -> None:
        """Record one denoising-step's (t, sigma, new_lp, old_lp) for the
        per-step log_ratio diagnostic. ``old_lp`` is the stored rollout-time
        per-transition logprob, aligned by index with the recompute loop."""
        old_flat = chunk.old_logprobs.reshape(-1)
        old_lp = float(old_flat[step_idx]) if step_idx < old_flat.numel() else float("nan")
        self._per_step_diag_records.append(
            {
                "i": int(step_idx),
                "t": float(self._flow_time(timestep).detach().float().mean().cpu()),
                "std": float(std.detach().float().mean().cpu()),
                "new_lp": float(new_lp.detach().float().mean().cpu()),
                "old_lp": old_lp,
            }
        )

    def _recompute_chunk_logprob(self, chunk: RolloutChunk) -> torch.Tensor:
        self._run_video_prefix(chunk.obs, chunk.frame_st_id, latent_noise=chunk.latent_noise)
        action_timesteps = chunk.action_timesteps.to(self.device)
        action_dts = (
            chunk.action_dts.to(self.device)
            if getattr(chunk, "action_dts", None) is not None and chunk.action_dts.numel() > 0
            else None
        )
        action_chain = [x.to(self.device, dtype=self.dtype) for x in chunk.action_chain]
        mask = chunk.action_mask.to(self.device)
        self.action_scheduler.set_timesteps(self.job_config.action_num_inference_steps)

        per_chunk = action_timesteps.numel() == 1
        if per_chunk:
            assert len(action_chain) == 2, (
                f"per_chunk chunk expected action_chain of length 2, got {len(action_chain)}"
            )
            t = action_timesteps[0]
            dt = (
                action_dts[0].to(device=self.device, dtype=self.dtype)
                if action_dts is not None
                else self._flow_dt_for_stored_timestep(t)
            )
            actions_prev = action_chain[0]
            next_actions = action_chain[1]
            v_theta, _ = self._action_transformer_step(
                actions_prev, t, chunk.frame_st_id, last_step=False,
            )
            transition = self._flow_grpo_action_transition(
                actions_prev,
                t,
                dt,
                v_theta,
            )
            lp = flow_grpo_gaussian_logprob(
                next_actions,
                transition["mean"],
                transition["std"],
                mask=mask,
                logprob_reduce=self.logprob_reduce,
            )
            if self._per_step_diag_active:
                self._record_per_step_diag(chunk, 0, t, transition["std"], lp)
            # Mirror rollout's last_step cache write (see per_step branch below
            # for the full rationale): without it the per_chunk recompute would
            # have the same KV-cache drift as per_step.
            with torch.no_grad():
                final_t = torch.zeros_like(t)
                self._action_transformer_step(
                    next_actions, final_t, chunk.frame_st_id, last_step=True,
                )
            return lp

        transition_logprobs = []
        for i, t in enumerate(action_timesteps):
            actions = action_chain[i]
            next_actions = action_chain[i + 1]
            dt = (
                action_dts[i].to(device=self.device, dtype=self.dtype)
                if action_dts is not None
                else self._flow_dt_for_stored_timestep(t)
            )
            v_theta, _ = self._action_transformer_step(
                actions,
                t,
                chunk.frame_st_id,
                last_step=False,
            )
            transition = self._flow_grpo_action_transition(
                actions,
                t,
                dt,
                v_theta,
            )
            lp = flow_grpo_gaussian_logprob(
                next_actions,
                transition["mean"],
                transition["std"],
                mask=mask,
                logprob_reduce=self.logprob_reduce,
            )
            if self._per_step_diag_active:
                self._record_per_step_diag(chunk, i, t, transition["std"], lp)
            transition_logprobs.append(lp)

        # Mirror rollout's last_step KV-cache write. _sample_action_chunk runs an
        # extra _action_transformer_step(..., last_step=True) at t=0 with the
        # final denoised actions, which goes through update_cache=1 and is what
        # the next chunk's transformer forwards attend to. Without this write,
        # recompute's cache state diverges from rollout by ~1 bf16 ulp per chunk
        # and accumulates over the chain — surfacing as the 0.0625 max_abs_diff
        # in validate_logprob_consistency and the occasional -15 to -33 outlier
        # log_ratio on long, heterogeneous trajectories.
        with torch.no_grad():
            final_actions = action_chain[-1]
            final_t = torch.zeros_like(action_timesteps[0])
            self._action_transformer_step(
                final_actions,
                final_t,
                chunk.frame_st_id,
                last_step=True,
            )

        logprob = torch.stack(transition_logprobs).sum(dim=0)
        if self.normalize_denoising_horizon and transition_logprobs:
            logprob = logprob / len(transition_logprobs)
        return logprob

    def get_action_logprobs(self, episode) -> torch.Tensor:
        # _reset re-encodes the prompt via the (non-trainable) T5 text encoder. Without
        # no_grad, self.prompt_embeds retains a live autograd graph that gets baked into
        # every chunk's transformer forward through _prepare_latent_input's `.clone()` of
        # prompt_embeds. Backward on the first chunk frees that shared graph; the second
        # chunk then trips "backward through the graph a second time". Running the reset
        # under no_grad makes prompt_embeds a plain buffer, independent across chunks.
        with torch.no_grad():
            self._reset(prompt=episode.prompt)
        logprobs = []
        for chunk in episode.chunks:
            logprobs.append(self._recompute_chunk_logprob(chunk))
            if chunk.keyframes is not None and chunk.state is not None:
                with torch.no_grad():
                    self._compute_kv_cache({"obs": chunk.keyframes, "state": chunk.state})
        stacked = torch.stack(logprobs)
        if self.normalize_episode_length and stacked.shape[0] > 0:
            return stacked.sum(dim=0) / stacked.shape[0]
        return stacked.sum(dim=0)

    def _run_grpo_update(self, groups: list[list]) -> None:
        """Standard GRPO update over a batch of groups.

        Episodes from all groups are concatenated into a single training pool.
        Advantages are normalized **within each group** (GRPO's defining property),
        then concatenated. Each of ``update_epochs`` epochs shuffles the pool and
        steps the optimizer once per minibatch of size ``self.minibatch_size``
        episodes. Each minibatch recomputes ``new_logprobs`` so ratio != 1
        starting from the very first minibatch of epoch 2.
        """
        if self.optimizer is None:
            raise RuntimeError("Cannot run GRPO update when disable_updates=True")
        if not groups:
            return

        # Sharded mode: all ranks must enter the update collective in lockstep.
        # The dist.barrier() here pairs with the subsequent all_gather_object;
        # if one rank's clients lagged (e.g. crashed mid-rollout) we'd rather
        # hang here than silently mismatch advantages.
        if self.sharded_group_enabled:
            dist.barrier()
            global_rewards = self._gather_sharded_rewards(groups)
        else:
            global_rewards = None

        update_step = self.global_update_step + 1
        if getattr(self, "replicated_enabled", False):
            (
                replicated_rollout_records,
                rank_rollout_counts,
            ) = self._gather_replicated_rollout_records(groups)
            self._log_replicated_rollout_group_wandb(
                replicated_rollout_records,
                update_step=update_step,
                rank_rollout_counts=rank_rollout_counts,
            )

        # Rollout-level (pre-filter) behavior stats. These describe what the
        # policy actually did over the rolled-out episodes; they must NOT be
        # restricted to the kept-after-degenerate-filter subset, otherwise
        # an all-fail group gets dropped from success_rate / reward_mean and
        # the reported metric overstates policy performance.
        rollout_episodes = [ep for group in groups for ep in (group or [])]
        rollout_rewards = [float(ep.reward) for ep in rollout_episodes]
        rollout_successes = [1.0 if bool(ep.success) else 0.0 for ep in rollout_episodes]
        rollout_step_counts = [float(ep.step_count) for ep in rollout_episodes if ep.step_count is not None]
        rollout_chunk_counts = [float(len(ep.chunks)) for ep in rollout_episodes]
        rollout_reward_tensor = torch.tensor(rollout_rewards) if rollout_rewards else torch.zeros(0)

        episodes: list = []
        advantage_chunks: list[torch.Tensor] = []
        group_rewards_summary: list[float] = []
        skipped_groups = 0
        for group in groups:
            if not group:
                continue
            if self.sharded_group_enabled:
                # Use the merged cross-rank rewards for both the degenerate
                # filter and advantage normalization. We then slice out the
                # advantages corresponding to *this* rank's local members.
                gid = group[0].group_id
                global_bucket = global_rewards[gid]
                global_members_sorted = sorted(global_bucket.keys())
                g_rewards = [global_bucket[m] for m in global_members_sorted]
                if len(g_rewards) >= 2 and float(torch.tensor(g_rewards).std(unbiased=False)) <= 0.0:
                    skipped_groups += 1
                    continue
                global_adv = compute_group_advantages(g_rewards)
                # Map global member → its position in global_members_sorted so
                # we can fetch the right advantage for each local episode.
                member_to_pos = {m: i for i, m in enumerate(global_members_sorted)}
                local_adv = torch.stack([
                    global_adv[member_to_pos[int(ep.metadata["group_member"])]]
                    for ep in group
                ])
                episodes.extend(group)
                advantage_chunks.append(local_adv)
                group_rewards_summary.append(sum(g_rewards) / max(len(g_rewards), 1))
            else:
                g_rewards = [float(ep.reward) for ep in group]
                # Skip all-success or all-fail groups: compute_group_advantages returns
                # all zeros, so they contribute no gradient but still occupy minibatch
                # capacity. Filtering keeps the training pool concentrated on groups
                # that actually carry an advantage signal.
                if len(g_rewards) >= 2 and float(torch.tensor(g_rewards).std(unbiased=False)) <= 0.0:
                    skipped_groups += 1
                    continue
                episodes.extend(group)
                advantage_chunks.append(compute_group_advantages(g_rewards))
                group_rewards_summary.append(sum(g_rewards) / max(len(g_rewards), 1))
        if skipped_groups:
            self._info0(
                "GRPO degenerate groups filtered: rank=%s skipped=%s kept=%s total=%s",
                self.rank,
                skipped_groups,
                len(advantage_chunks),
                len(groups),
            )
        if not episodes:
            self._info0(
                "GRPO update skipped: rank=%s all_groups_degenerate skipped_groups=%s",
                self.rank,
                skipped_groups,
            )
            return

        advantages = torch.cat(advantage_chunks, dim=0).to(self.device)
        old_logprobs = self._episode_old_logprobs(episodes).to(self.device)
        group_ids = [grp[0].group_id for grp in groups if grp]
        advantage_cpu = advantages.detach().cpu()
        n_eps = len(episodes)
        n_rollout_eps = len(rollout_episodes)
        n_kept_groups = len(advantage_chunks)
        minibatch_size = max(1, min(self.minibatch_size, n_eps))
        global_minibatch_size = (
            minibatch_size * self.world_size
            if self.sharded_group_enabled
            else minibatch_size
        )
        # Behavior stats use ALL rolled-out episodes (pre-filter). Training
        # stats (advantage, n_eps, group_count) use post-filter only.
        self._info0(
            "GRPO update start [rank0-local]: rank=%s step=%s rollout_groups=%s kept_groups=%s rollout_episodes=%s "
            "trained_episodes=%s minibatch_size=%s global_minibatch_size=%s degenerate_groups_skipped=%s "
            "reward_mean=%.4f reward_std=%.4f success_rate=%.4f "
            "old_logprob_mean=%.4f old_logprob_std=%.4f advantage_mean=%.4f advantage_std=%.4f "
            "step_count_mean=%.1f chunk_count_mean=%.1f epochs=%s",
            self.rank,
            update_step,
            len(groups),
            n_kept_groups,
            n_rollout_eps,
            n_eps,
            minibatch_size,
            global_minibatch_size,
            skipped_groups,
            float(rollout_reward_tensor.mean()) if rollout_reward_tensor.numel() else 0.0,
            float(rollout_reward_tensor.std(unbiased=False)) if rollout_reward_tensor.numel() > 1 else 0.0,
            float(sum(rollout_successes) / max(len(rollout_successes), 1)),
            float(old_logprobs.mean().detach().cpu()),
            float(old_logprobs.std(unbiased=False).detach().cpu()) if old_logprobs.numel() > 1 else 0.0,
            float(advantage_cpu.mean()),
            float(advantage_cpu.std(unbiased=False)) if advantage_cpu.numel() > 1 else 0.0,
            float(torch.tensor(rollout_step_counts).mean()) if rollout_step_counts else 0.0,
            float(torch.tensor(rollout_chunk_counts).mean()) if rollout_chunk_counts else 0.0,
            self.update_epochs,
        )

        # Optimizer steps per epoch may vary if n_eps % minibatch_size != 0.
        stats: dict[str, float] = {}
        stop_outer = False
        for epoch_idx in range(self.update_epochs):
            if stop_outer:
                break
            perm = torch.randperm(n_eps).tolist()
            if self.diagnose_per_step_logratio:
                self._per_step_diag_records = []
            epoch_loss = 0.0
            epoch_ratio = 0.0
            epoch_ratio_min = float("inf")
            epoch_ratio_max = float("-inf")
            epoch_ratio_std = 0.0
            epoch_clipfrac = 0.0
            epoch_kl = 0.0
            epoch_new_logprob = 0.0
            mbs0_ratio_mean: float | None = None
            mbs0_ratio_min: float | None = None
            mbs0_ratio_max: float | None = None
            # Per-episode diagnostics: needed to attribute KL blow-ups to
            # specific outlier trajectories. Each entry is one episode's
            # (log_ratio, advantage, chunks, success, group_id, group_member,
            # ep_seed, ep_step_count) — populated incrementally as minibatches
            # run. log_ratio is the trajectory-level (new - old) diff in fp32.
            epoch_episode_diag: list[dict[str, Any]] = []
            epoch_grad_norms: list[float] = []
            epoch_effect_signed_adv_delta: list[float] = []
            epoch_effect_delta: list[float] = []
            epoch_effect_pos_delta: list[float] = []
            epoch_effect_neg_delta: list[float] = []
            epoch_effect_pos_delta_positive: list[float] = []
            epoch_effect_neg_delta_negative: list[float] = []
            epoch_effect_active_count = 0
            n_mbs = 0
            for mb_start in range(0, n_eps, minibatch_size):
                mb_idx = perm[mb_start : mb_start + minibatch_size]
                mb_episodes = [episodes[i] for i in mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Detached forward: one no_grad forward per episode to
                # compute the ratio and clip decisions. This forward NEVER
                # serves as the backward starting point — that would chain
                # the PPO gradient ``-ratio*adv`` through a *separate*
                # grad-forward inside ``_backward_episode_logprob``, and any
                # bf16 drift between the two forwards (KV cache state,
                # reduction order) means the gradient is applied to a
                # different surface than the one ``ratio`` was derived from.
                # In the LoRA-from-zero regime this manifests as ratio_max
                # pinned to ~1.0 even on positive-advantage episodes.
                self.optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    # Per-step diagnostic records only this clean detached
                    # forward (not the grad-forward or the update-effect
                    # recompute), so it reflects the pre-step policy on the
                    # exact surface the ratio/clip decisions are derived from.
                    self._per_step_diag_active = self.diagnose_per_step_logratio
                    mb_new_logprobs_detached = torch.cat(
                        [self.get_action_logprobs(ep).reshape(1) for ep in mb_episodes],
                        dim=0,
                    )
                    self._per_step_diag_active = False
                # Stats are computed entirely in detached space — used only
                # for logging. Loss is informational; we do NOT backward it.
                loss_stats = grpo_clipped_loss(
                    mb_new_logprobs_detached,
                    mb_old_logprobs,
                    mb_advantages,
                    clip_range=self.clip_range,
                    clip_range_high=self.clip_range_high,
                    clip_range_low=self.clip_range_low,
                    log_ratio_clip=self.log_ratio_clip,
                    beta_kl=self.beta_kl,
                    entropy_coef=self.entropy_coef,
                )
                # Per-episode PPO surrogate gradient wrt new_logprob,
                # computed manually from the detached ratio so the clip
                # boundary is exact (not subject to autograd's
                # ``min``/``clamp`` subgradient at the boundary):
                #   adv>0, ratio>1+c → on clipped branch, grad=0
                #   adv<0, ratio<1-c → on clipped branch, grad=0
                #   otherwise        → grad = -ratio * adv / B
                # The 1/B factor matches ``.mean()`` in grpo_clipped_loss so
                # gradient magnitudes are unchanged.
                log_ratio_for_grad = mb_new_logprobs_detached.float() - mb_old_logprobs.float()
                if self.log_ratio_clip is not None:
                    log_ratio_for_grad = log_ratio_for_grad.clamp(
                        -self.log_ratio_clip,
                        self.log_ratio_clip,
                    )
                ratio_for_grad = torch.exp(log_ratio_for_grad)
                adv_f32 = mb_advantages.float()
                clipped_high = (adv_f32 > 0) & (ratio_for_grad > 1.0 + self.clip_range_high)
                clipped_low = (adv_f32 < 0) & (ratio_for_grad < 1.0 - self.clip_range_low)
                clip_mask_episode = clipped_high | clipped_low
                grad_coef = torch.where(
                    clip_mask_episode,
                    torch.zeros_like(ratio_for_grad),
                    -(ratio_for_grad * adv_f32) / float(len(mb_episodes)),
                )

                # KL-to-old penalty (added directly to the surrogate gradient,
                # because grpo_clipped_loss's beta_kl term is computed for
                # logging only and never backwarded — see the no_grad/loss
                # comment above). Using the k3 estimator
                #   KL(new‖old) ≈ ratio - 1 - log_ratio   (per element)
                # whose gradient wrt new_logprob is ``ratio - 1``. We add
                # ``beta_kl * (ratio - 1) / B`` to grad_coef. This term is
                # applied UNCONDITIONALLY (not gated by clip_mask): its whole
                # purpose is to pull back exactly the large-ratio outliers the
                # PPO clip zeroes out, giving a continuous restoring force
                # toward ratio=1. ``/ B`` matches the surrogate term's mean
                # reduction. NOTE(update_epochs=1): within a single epoch
                # new≈old for most minibatches, so this mainly constrains the
                # in-step outliers (the high-clipfrac steps); it does not stop
                # cross-update-step drift (the anchor resets each step).
                if self.beta_kl:
                    grad_coef = grad_coef + (
                        float(self.beta_kl) * (ratio_for_grad - 1.0) / float(len(mb_episodes))
                    )

                # Grad-forward + per-chunk backward. Returned tensor is the
                # logprob the gradient actually saw (per-chunk grad-forward
                # path), used below for ``new_logprob_mean`` logging and for
                # the optional two-forward drift diagnostic.
                mb_new_logprobs_gradfwd_list: list[torch.Tensor] = []
                for episode, coef in zip(mb_episodes, grad_coef):
                    ep_lp = self._backward_episode_logprob(episode, coef.to(self.device))
                    mb_new_logprobs_gradfwd_list.append(ep_lp.reshape(1))
                mb_new_logprobs_gradfwd = torch.cat(mb_new_logprobs_gradfwd_list, dim=0)

                if self.validate_logprob_consistency:
                    drift = (mb_new_logprobs_gradfwd - mb_new_logprobs_detached.float()).abs()
                    drift_max = float(drift.max().detach().cpu())
                    if drift_max > self.logprob_consistency_atol:
                        logger.warning(
                            "GRPO two-forward drift exceeds atol: rank=%s max_abs_diff=%.3e "
                            "atol=%.3e mb_size=%s",
                            self.rank,
                            drift_max,
                            self.logprob_consistency_atol,
                            len(mb_episodes),
                        )
                # Synchronize gradients across replicated-mode ranks BEFORE
                # clipping — clip_grad_norm_ uses the global norm, so it must
                # see the synchronized grads to produce a consistent clip
                # factor on every rank.
                self._allreduce_gradients()
                grad_clip = self.rl_cfg.get("max_grad_norm", None)
                pre_clip_norm_f = 0.0
                if grad_clip is not None:
                    # clip_grad_norm_ returns the *pre-clip* total norm — capture it
                    # so we can see when a single outlier episode produced gradients
                    # that would otherwise blow up the policy without the clamp.
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, float(grad_clip)
                    )
                    pre_clip_norm_f = float(pre_clip_norm)
                    epoch_grad_norms.append(pre_clip_norm_f)
                mb_rewards = [float(ep.reward) for ep in mb_episodes]
                mb_successes = [1.0 if bool(ep.success) else 0.0 for ep in mb_episodes]
                mb_chunks = [float(len(ep.chunks)) for ep in mb_episodes]
                mb_steps = [float(ep.step_count) for ep in mb_episodes if ep.step_count is not None]
                self._info0(
                    "GRPO update minibatch [rank0-local]: rank=%s step=%s epoch=%s/%s mb=%s loss=%.6f approx_kl=%.6f "
                    "ratio=%.6f ratio_min=%.4f ratio_max=%.4f clipfrac=%.4f "
                    "episodes=%s reward_mean=%.4f success_rate=%.4f adv_mean=%.4f adv_min=%.4f adv_max=%.4f "
                    "old_logprob_mean=%.4f new_logprob_mean=%.4f logratio_abs_max=%.4f "
                    "chunks_mean=%.2f step_count_mean=%.1f grad_norm=%.4f lr=%.3e",
                    self.rank,
                    update_step,
                    epoch_idx + 1,
                    self.update_epochs,
                    n_mbs + 1,
                    float(loss_stats.loss.detach().cpu()),
                    float(loss_stats.approx_kl.detach().cpu()),
                    float(loss_stats.ratio_mean.detach().cpu()),
                    float(loss_stats.ratio_min.detach().cpu()),
                    float(loss_stats.ratio_max.detach().cpu()),
                    float(loss_stats.clipfrac.detach().cpu()),
                    len(mb_episodes),
                    float(torch.tensor(mb_rewards).mean()) if mb_rewards else 0.0,
                    float(sum(mb_successes) / max(len(mb_successes), 1)),
                    float(mb_advantages.detach().float().mean().cpu()),
                    float(mb_advantages.detach().float().min().cpu()),
                    float(mb_advantages.detach().float().max().cpu()),
                    float(mb_old_logprobs.detach().float().mean().cpu()),
                    float(mb_new_logprobs_gradfwd.detach().float().mean().cpu()),
                    float(loss_stats.log_ratio.detach().float().abs().max().cpu()),
                    float(torch.tensor(mb_chunks).mean()) if mb_chunks else 0.0,
                    float(torch.tensor(mb_steps).mean()) if mb_steps else 0.0,
                    pre_clip_norm_f,
                    float(self.optimizer.param_groups[0]["lr"]),
                )
                self.optimizer.step()
                if self.scheduler_lr is not None:
                    self.scheduler_lr.step()

                if self.diagnose_update_effect:
                    # Expensive but decisive: measure the actual parameter step's
                    # effect on the same minibatch logprobs. Positive advantages
                    # should usually get positive deltas; negative advantages
                    # should usually get negative deltas. This is intentionally
                    # post-step, unlike loss_stats.log_ratio above.
                    with torch.no_grad():
                        mb_post_logprobs = torch.cat(
                            [self.get_action_logprobs(ep).reshape(1) for ep in mb_episodes],
                            dim=0,
                        )
                    mb_effect_delta = (
                        mb_post_logprobs.detach().float().cpu()
                        - mb_new_logprobs_detached.detach().float().cpu()
                    )
                    mb_effect_adv = mb_advantages.detach().float().cpu()
                    mb_effect_active = grad_coef.detach().float().cpu().abs() > 0
                    if bool(mb_effect_active.any()):
                        active_delta = mb_effect_delta[mb_effect_active]
                        active_adv = mb_effect_adv[mb_effect_active]
                        epoch_effect_active_count += int(active_delta.numel())
                        for delta_val, adv_val in zip(active_delta.tolist(), active_adv.tolist()):
                            delta_f = float(delta_val)
                            adv_f = float(adv_val)
                            epoch_effect_delta.append(delta_f)
                            epoch_effect_signed_adv_delta.append(delta_f * adv_f)
                            if adv_f > 0:
                                epoch_effect_pos_delta.append(delta_f)
                                epoch_effect_pos_delta_positive.append(1.0 if delta_f > 0 else 0.0)
                            elif adv_f < 0:
                                epoch_effect_neg_delta.append(delta_f)
                                epoch_effect_neg_delta_negative.append(1.0 if delta_f < 0 else 0.0)

                # Record per-episode diagnostics for outlier attribution. Each
                # element of loss_stats.log_ratio corresponds to one episode in
                # mb_episodes (trajectory logprob is one scalar per episode).
                mb_log_ratio_cpu = loss_stats.log_ratio.cpu()
                mb_advantage_cpu = mb_advantages.detach().float().cpu()
                for ep_local_idx, episode in enumerate(mb_episodes):
                    epoch_episode_diag.append(
                        {
                            "log_ratio": float(mb_log_ratio_cpu[ep_local_idx]),
                            "advantage": float(mb_advantage_cpu[ep_local_idx]),
                            "chunks": int(len(episode.chunks)),
                            "step_count": int(episode.step_count or 0),
                            "success": bool(episode.success),
                            "reward": float(episode.reward),
                            "group_id": str(getattr(episode, "group_id", "")),
                            "group_member": int(episode.metadata.get("group_member", -1)) if hasattr(episode, "metadata") else -1,
                            "seed": int(getattr(episode, "seed", -1)),
                        }
                    )

                epoch_loss += float(loss_stats.loss.detach().cpu())
                epoch_ratio += float(loss_stats.ratio_mean.cpu())
                epoch_ratio_min = min(epoch_ratio_min, float(loss_stats.ratio_min.cpu()))
                epoch_ratio_max = max(epoch_ratio_max, float(loss_stats.ratio_max.cpu()))
                epoch_ratio_std += float(loss_stats.ratio_std.cpu())
                epoch_clipfrac += float(loss_stats.clipfrac.cpu())
                epoch_kl += float(loss_stats.approx_kl.cpu())
                # Use the grad-forward logprob (the surface the gradient
                # actually flowed through) so the logged ``new_logprob_mean``
                # matches the value referenced by the gradient update — not
                # the detached forward used to decide the clip mask.
                epoch_new_logprob += float(mb_new_logprobs_gradfwd.mean().detach().cpu())
                if n_mbs == 0:
                    mbs0_ratio_mean = float(loss_stats.ratio_mean.cpu())
                    mbs0_ratio_min = float(loss_stats.ratio_min.cpu())
                    mbs0_ratio_max = float(loss_stats.ratio_max.cpu())
                n_mbs += 1

            denom = max(n_mbs, 1)

            # Aggregate per-episode diagnostics (one entry per episode seen this
            # epoch). The point: when approx_kl blows up, attribute it to a
            # specific outlier (long episode, very negative advantage, ...) rather
            # than to "the lr is too high in general".
            if epoch_episode_diag:
                lr_arr = torch.tensor([d["log_ratio"] for d in epoch_episode_diag])
                adv_arr = torch.tensor([d["advantage"] for d in epoch_episode_diag])
                chunks_arr = torch.tensor([d["chunks"] for d in epoch_episode_diag], dtype=torch.float32)
                logratio_min = float(lr_arr.min())
                logratio_max = float(lr_arr.max())
                logratio_std = float(lr_arr.std(unbiased=False)) if lr_arr.numel() > 1 else 0.0
                logratio_abs_max = float(lr_arr.abs().max())
                outlier_threshold = 5.0
                outlier_mask = lr_arr.abs() > outlier_threshold
                outlier_count = int(outlier_mask.sum())
                advantage_min = float(adv_arr.min())
                advantage_max = float(adv_arr.max())
                advantage_abs_max = float(adv_arr.abs().max())
                chunks_min = float(chunks_arr.min())
                chunks_max = float(chunks_arr.max())
                chunks_mean = float(chunks_arr.mean())
                # Per-chunk normalized |log_ratio|: if outliers are driven purely
                # by trajectory length, this should be roughly constant across
                # episodes; if it varies a lot, something else is going on.
                logratio_per_chunk_abs_max = float((lr_arr.abs() / chunks_arr.clamp(min=1)).max())
                # Correlation between trajectory length (chunks) and |log_ratio|:
                # high positive = longer trajectories accumulate more logprob
                # drift (expected with per_step noise); near-zero = drift is
                # caused by something other than length.
                if lr_arr.numel() > 1:
                    lr_abs = lr_arr.abs().float()
                    ch = chunks_arr.float()
                    lr_centered = lr_abs - lr_abs.mean()
                    ch_centered = ch - ch.mean()
                    denom_corr = (lr_centered.std(unbiased=False) * ch_centered.std(unbiased=False))
                    chunks_logratio_corr = float((lr_centered * ch_centered).mean() / denom_corr) if denom_corr > 0 else 0.0
                else:
                    chunks_logratio_corr = 0.0
            else:
                logratio_min = logratio_max = logratio_std = logratio_abs_max = 0.0
                outlier_count = 0
                advantage_min = advantage_max = advantage_abs_max = 0.0
                chunks_min = chunks_max = chunks_mean = 0.0
                logratio_per_chunk_abs_max = 0.0
                chunks_logratio_corr = 0.0

            if epoch_grad_norms:
                gn = torch.tensor(epoch_grad_norms)
                grad_norm_mean = float(gn.mean())
                grad_norm_max = float(gn.max())
                grad_norm_min = float(gn.min())
            else:
                grad_norm_mean = grad_norm_max = grad_norm_min = 0.0

            def _mean_or_zero(values: list[float]) -> float:
                return float(torch.tensor(values, dtype=torch.float32).mean()) if values else 0.0

            update_effect_delta_mean = _mean_or_zero(epoch_effect_delta)
            update_effect_signed_adv_delta_mean = _mean_or_zero(epoch_effect_signed_adv_delta)
            update_effect_pos_adv_delta_mean = _mean_or_zero(epoch_effect_pos_delta)
            update_effect_neg_adv_delta_mean = _mean_or_zero(epoch_effect_neg_delta)
            update_effect_pos_adv_delta_positive_frac = _mean_or_zero(epoch_effect_pos_delta_positive)
            update_effect_neg_adv_delta_negative_frac = _mean_or_zero(epoch_effect_neg_delta_negative)
            update_effect_global_stats = None
            if self.diagnose_update_effect:
                update_effect_global_stats = self._allreduce_update_effect_stats(
                    delta_values=epoch_effect_delta,
                    signed_adv_delta_values=epoch_effect_signed_adv_delta,
                    pos_delta_values=epoch_effect_pos_delta,
                    neg_delta_values=epoch_effect_neg_delta,
                    pos_delta_positive_values=epoch_effect_pos_delta_positive,
                    neg_delta_negative_values=epoch_effect_neg_delta_negative,
                )

            # Identify the 3 worst offenders by |log_ratio| so we can see
            # exactly which (group_id, seed, group_member) trajectories are
            # responsible for the KL blow-up. Logged as a single info line.
            worst_offenders = sorted(
                epoch_episode_diag,
                key=lambda d: abs(d["log_ratio"]),
                reverse=True,
            )[:3]

            stats = {
                "loss": epoch_loss / denom,
                "ratio": epoch_ratio / denom,
                "ratio_min": epoch_ratio_min if epoch_ratio_min != float("inf") else 0.0,
                "ratio_max": epoch_ratio_max if epoch_ratio_max != float("-inf") else 0.0,
                "ratio_std": epoch_ratio_std / denom,
                "mbs0_ratio_mean": mbs0_ratio_mean if mbs0_ratio_mean is not None else 0.0,
                "mbs0_ratio_min": mbs0_ratio_min if mbs0_ratio_min is not None else 0.0,
                "mbs0_ratio_max": mbs0_ratio_max if mbs0_ratio_max is not None else 0.0,
                "clipfrac": epoch_clipfrac / denom,
                "approx_kl": epoch_kl / denom,
                "old_logprob_mean": float(old_logprobs.mean().detach().cpu()),
                "new_logprob_mean": epoch_new_logprob / denom,
                # Behavior stats: over ALL rolled-out episodes, NOT filtered.
                "reward_mean": float(rollout_reward_tensor.mean()) if rollout_reward_tensor.numel() else 0.0,
                "reward_std": float(rollout_reward_tensor.std(unbiased=False)) if rollout_reward_tensor.numel() > 1 else 0.0,
                "reward_min": float(min(rollout_rewards)) if rollout_rewards else 0.0,
                "reward_max": float(max(rollout_rewards)) if rollout_rewards else 0.0,
                "success_rate": float(sum(rollout_successes) / max(len(rollout_successes), 1)),
                "step_count_mean": float(torch.tensor(rollout_step_counts).mean()) if rollout_step_counts else 0.0,
                "chunk_count_mean": float(torch.tensor(rollout_chunk_counts).mean()) if rollout_chunk_counts else 0.0,
                "rollout_episode_count": n_rollout_eps,
                # Training stats: post-filter pool the optimizer actually saw.
                "rollout_group_count": len(groups),
                "kept_group_count": n_kept_groups,
                "degenerate_groups_skipped": skipped_groups,
                "trained_episode_count": n_eps,
                "trained_group_size": (n_eps // n_kept_groups) if n_kept_groups else 0,
                "minibatch_size": minibatch_size,
                "global_minibatch_size": global_minibatch_size,
                "minibatches_per_epoch": n_mbs,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                # Per-episode outlier diagnostics for KL attribution.
                "logratio_min": logratio_min,
                "logratio_max": logratio_max,
                "logratio_std": logratio_std,
                "logratio_abs_max": logratio_abs_max,
                "logratio_per_chunk_abs_max": logratio_per_chunk_abs_max,
                "logratio_outlier_count": outlier_count,
                "chunks_logratio_corr": chunks_logratio_corr,
                "advantage_min": advantage_min,
                "advantage_max": advantage_max,
                "advantage_abs_max": advantage_abs_max,
                "chunks_min": chunks_min,
                "chunks_max": chunks_max,
                "chunks_mean": chunks_mean,
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_max": grad_norm_max,
                "grad_norm_min": grad_norm_min,
            }
            if self.diagnose_update_effect:
                assert update_effect_global_stats is not None
                stats.update({
                    "update_effect_delta_mean": update_effect_global_stats["delta_mean"],
                    "update_effect_signed_adv_delta_mean": update_effect_global_stats["signed_adv_delta_mean"],
                    "update_effect_pos_adv_delta_mean": update_effect_global_stats["pos_adv_delta_mean"],
                    "update_effect_neg_adv_delta_mean": update_effect_global_stats["neg_adv_delta_mean"],
                    "update_effect_pos_adv_delta_positive_frac": update_effect_global_stats["pos_adv_delta_positive_frac"],
                    "update_effect_neg_adv_delta_negative_frac": update_effect_global_stats["neg_adv_delta_negative_frac"],
                    "update_effect_active_count": update_effect_global_stats["active_count"],
                    "update_effect_pos_count": update_effect_global_stats["pos_count"],
                    "update_effect_neg_count": update_effect_global_stats["neg_count"],
                })
            # Reduce per-rank epoch stats to a single global view. Every rank
            # executes the collective (lockstep at epoch end); only rank 0 emits
            # the consolidated line and pushes to wandb. The per-rank locals
            # remain for the rank-0-gated debug lines that follow.
            global_stats = self._allreduce_named_stats(stats)
            if self._is_rank0():
                logger.info(
                    "GRPO update epoch [global]: step=%s epoch=%s/%s ranks=%s mbs/rank=%s "
                    "loss=%.6f approx_kl=%.6f ratio=%.6f ratio_min=%.4f ratio_max=%.4f ratio_std=%.4f "
                    "mbs0_ratio=%.4f mbs0_ratio_min=%.4f mbs0_ratio_max=%.4f "
                    "logratio_min=%.4f logratio_max=%.4f logratio_std=%.4f logratio_abs_max=%.4f "
                    "logratio_per_chunk_abs_max=%.4f logratio_outlier_count=%s chunks_logratio_corr=%.3f "
                    "adv_min=%.4f adv_max=%.4f adv_abs_max=%.4f "
                    "chunks_min=%.0f chunks_max=%.0f chunks_mean=%.2f "
                    "grad_norm_mean=%.3f grad_norm_max=%.3f grad_norm_min=%.3f "
                    "clipfrac=%.6f old_logprob_mean=%.4f new_logprob_mean=%.4f lr=%.3e",
                    update_step,
                    epoch_idx + 1,
                    self.update_epochs,
                    self.world_size,
                    n_mbs,
                    global_stats["loss"],
                    global_stats["approx_kl"],
                    global_stats["ratio"],
                    global_stats["ratio_min"],
                    global_stats["ratio_max"],
                    global_stats["ratio_std"],
                    global_stats["mbs0_ratio_mean"],
                    global_stats["mbs0_ratio_min"],
                    global_stats["mbs0_ratio_max"],
                    global_stats["logratio_min"],
                    global_stats["logratio_max"],
                    global_stats["logratio_std"],
                    global_stats["logratio_abs_max"],
                    global_stats["logratio_per_chunk_abs_max"],
                    global_stats["logratio_outlier_count"],
                    global_stats["chunks_logratio_corr"],
                    global_stats["advantage_min"],
                    global_stats["advantage_max"],
                    global_stats["advantage_abs_max"],
                    global_stats["chunks_min"],
                    global_stats["chunks_max"],
                    global_stats["chunks_mean"],
                    global_stats["grad_norm_mean"],
                    global_stats["grad_norm_max"],
                    global_stats["grad_norm_min"],
                    global_stats["clipfrac"],
                    global_stats["old_logprob_mean"],
                    global_stats["new_logprob_mean"],
                    global_stats["lr"],
                )
                # Offenders are per-rank-local (worst |log_ratio| this rank saw);
                # rank 0's are a representative sample, not a global top-k.
                for rank_idx, off in enumerate(worst_offenders):
                    logger.info(
                        "GRPO update offender [rank0-local]: idx=%s log_ratio=%.4f advantage=%.4f chunks=%s "
                        "step_count=%s success=%s reward=%.3f seed=%s group_member=%s group_id=%s",
                        rank_idx,
                        off["log_ratio"],
                        off["advantage"],
                        off["chunks"],
                        off["step_count"],
                        off["success"],
                        off["reward"],
                        off["seed"],
                        off["group_member"],
                        off["group_id"],
                    )
            if self.diagnose_per_step_logratio and self._per_step_diag_records:
                # Group this epoch's per-transition records by denoising-step
                # index i and report, per i: the flow-time t, sigma, and the
                # distribution of per-step log_ratio = new_lp - old_lp. The
                # decisive read: scan i from 0 (high-t, high-sigma) to the last
                # (low-t, low-sigma) and watch |log_ratio|. If it rises as sigma
                # falls, the fixed sigma schedule is ill-conditioned across
                # steps (low-sigma steps dominate the chunk ratio); if it stays
                # flat, the schedule is fine and ratio outliers come from
                # elsewhere (two-forward drift, trajectory length).
                by_step: dict[int, list[dict[str, float]]] = {}
                for rec in self._per_step_diag_records:
                    by_step.setdefault(rec["i"], []).append(rec)
                per_step_summary: list[dict[str, float]] = []
                for i in sorted(by_step):
                    recs = by_step[i]
                    lr = torch.tensor(
                        [r["new_lp"] - r["old_lp"] for r in recs], dtype=torch.float32
                    )
                    t_mean = float(torch.tensor([r["t"] for r in recs]).mean())
                    std_mean = float(torch.tensor([r["std"] for r in recs]).mean())
                    per_step_summary.append(
                        {
                            "i": i,
                            "t": t_mean,
                            "sigma": std_mean,
                            "count": int(lr.numel()),
                            "logratio_mean": float(lr.mean()),
                            "logratio_std": float(lr.std(unbiased=False)) if lr.numel() > 1 else 0.0,
                            "logratio_abs_mean": float(lr.abs().mean()),
                            "logratio_abs_max": float(lr.abs().max()),
                        }
                    )
                # Per-step records are this rank's local transitions. The sigma
                # schedule is identical across ranks, so rank 0's per-step profile
                # is representative; emit only it to avoid a 5*world_size line dump.
                if self._is_rank0():
                    for s in per_step_summary:
                        logger.info(
                            "GRPO per-step logratio [rank0-local]: step=%s epoch=%s/%s i=%s t=%.4f sigma=%.4f "
                            "count=%s logratio_mean=%.5f logratio_std=%.5f logratio_abs_mean=%.5f logratio_abs_max=%.5f",
                            update_step,
                            epoch_idx + 1,
                            self.update_epochs,
                            s["i"],
                            s["t"],
                            s["sigma"],
                            s["count"],
                            s["logratio_mean"],
                            s["logratio_std"],
                            s["logratio_abs_mean"],
                            s["logratio_abs_max"],
                        )
                self._wandb_log(
                    {
                        f"debug/per_step_logratio/i{int(s['i'])}/{key}": s[key]
                        for s in per_step_summary
                        for key in ("sigma", "logratio_abs_mean", "logratio_std")
                    },
                )
                if self.rank == 0:
                    self._write_json(
                        "latest_per_step_logratio.json",
                        {"update_step": update_step, "epoch": epoch_idx + 1, "per_step": per_step_summary},
                    )
            if self.diagnose_update_effect and self._is_rank0():
                # These fields are already cross-rank-reduced (global_active = sum
                # over ranks); emit once from rank 0. Per-rank locals dropped.
                logger.info(
                    "GRPO update effect [global]: step=%s epoch=%s/%s active=%s "
                    "delta_mean=%.6e signed_adv_delta_mean=%.6e "
                    "pos_adv_delta_mean=%.6e pos_adv_delta_positive_frac=%.3f "
                    "neg_adv_delta_mean=%.6e neg_adv_delta_negative_frac=%.3f",
                    update_step,
                    epoch_idx + 1,
                    self.update_epochs,
                    stats["update_effect_active_count"],
                    stats["update_effect_delta_mean"],
                    stats["update_effect_signed_adv_delta_mean"],
                    stats["update_effect_pos_adv_delta_mean"],
                    stats["update_effect_pos_adv_delta_positive_frac"],
                    stats["update_effect_neg_adv_delta_mean"],
                    stats["update_effect_neg_adv_delta_negative_frac"],
                )
                self._wandb_log({
                    "train_epoch/global_update_step": update_step,
                    "train_epoch/update_effect_delta_mean": stats["update_effect_delta_mean"],
                    "train_epoch/update_effect_signed_adv_delta_mean": stats["update_effect_signed_adv_delta_mean"],
                    "train_epoch/update_effect_pos_adv_delta_mean": stats["update_effect_pos_adv_delta_mean"],
                    "train_epoch/update_effect_neg_adv_delta_mean": stats["update_effect_neg_adv_delta_mean"],
                    "train_epoch/update_effect_pos_adv_delta_positive_frac": stats["update_effect_pos_adv_delta_positive_frac"],
                    "train_epoch/update_effect_neg_adv_delta_negative_frac": stats["update_effect_neg_adv_delta_negative_frac"],
                    "train_epoch/update_effect_active_count": stats["update_effect_active_count"],
                    "train_epoch/update_effect_pos_count": stats["update_effect_pos_count"],
                    "train_epoch/update_effect_neg_count": stats["update_effect_neg_count"],
                })
            # Push the cross-rank-reduced epoch stats (rank-0-gated inside
            # _wandb_log). reward_mean/success_rate are now the global rollout
            # behavior, not rank 0's slice.
            self._wandb_log(
                {
                    "train_epoch/global_update_step": update_step,
                    "train_epoch/epoch": epoch_idx + 1,
                    "train_epoch/minibatches": n_mbs,
                    **{
                        f"train_epoch/{key}": global_stats[key]
                        for key in (
                            "loss", "approx_kl", "ratio", "ratio_min", "ratio_max",
                            "ratio_std", "mbs0_ratio_mean", "mbs0_ratio_min",
                            "mbs0_ratio_max", "clipfrac", "old_logprob_mean",
                            "new_logprob_mean", "lr", "logratio_min", "logratio_max",
                            "logratio_std", "logratio_abs_max",
                            "logratio_per_chunk_abs_max", "logratio_outlier_count",
                            "chunks_logratio_corr", "advantage_min", "advantage_max",
                            "advantage_abs_max", "chunks_min", "chunks_max",
                            "chunks_mean", "grad_norm_mean", "grad_norm_max",
                            "grad_norm_min", "reward_mean", "reward_std",
                            "success_rate", "step_count_mean",
                        )
                    },
                },
            )
            if self.target_kl is not None:
                # All ranks must reach the same early-stop decision or they
                # diverge on the next epoch. Use MAX across ranks: if any
                # rank's batch shows runaway KL, everyone stops.
                kl_for_stop = self._allreduce_scalar_max(stats["approx_kl"])
                if kl_for_stop > self.target_kl:
                    logger.warning(
                        "Stopping GRPO epochs early: rank=%s approx_kl(max) %.6f > target %.6f",
                        self.rank,
                        kl_for_stop,
                        self.target_kl,
                    )
                    stop_outer = True

        self.global_update_step += 1
        # global_stats holds the last epoch's cross-rank-reduced view (every rank
        # computed it in lockstep). Use it as the canonical record so update
        # complete / checkpoints / wandb all report the global numbers, not rank
        # 0's local slice. Falls back to local stats if no epoch ran.
        canonical_stats = locals().get("global_stats", stats)
        canonical_stats["global_update_step"] = self.global_update_step
        self.last_stats = canonical_stats
        self._write_json("latest_stats.json", canonical_stats)
        self._wandb_log(
            {f"train/{key}": value for key, value in canonical_stats.items()},
        )
        if self.global_update_step % int(self.rl_cfg.get("checkpoint_interval", 1)) == 0:
            self.save_checkpoint()
        if self.eval_every > 0 and self.global_update_step % self.eval_every == 0:
            # Arm the eval phase. Clients pick this up from the next
            # run_pending_updates / get_eval_phase response and drive one
            # deterministic pass through their assignment.
            self._eval_pending = True
            self._eval_results = []
            self._info0(
                "GRPO eval phase armed: rank=%s global_update_step=%s eval_every=%s eval_action_steps=%s",
                self.rank,
                self.global_update_step,
                self.eval_every,
                self.eval_action_num_inference_steps,
            )
        self._info0("GRPO update complete [global]: %s", self.last_stats)

    def _run_pending_updates(self) -> dict[str, Any]:
        if self.disable_updates:
            self._info0("GRPO update skipped: rank=%s updates_disabled pending_ready_groups=%s", self.rank, len(self._pending_ready_groups))
            return {
                "updated": False,
                "reason": "updates_disabled",
                "pending_ready_groups": len(self._pending_ready_groups),
                "global_update_step": self.global_update_step,
                "in_eval": self._eval_pending,
                "status": self.last_stats,
            }
        if len(self._pending_ready_groups) < self.rollout_groups_per_update:
            self._info0(
                "GRPO update skipped: rank=%s not_enough_ready_groups pending=%s required=%s",
                self.rank,
                len(self._pending_ready_groups),
                self.rollout_groups_per_update,
            )
            return {
                "updated": False,
                "reason": "not_enough_ready_groups",
                "pending_ready_groups": len(self._pending_ready_groups),
                "required_ready_groups": self.rollout_groups_per_update,
                "global_update_step": self.global_update_step,
                "in_eval": self._eval_pending,
                "status": self.last_stats,
            }

        ready_groups = list(self._pending_ready_groups)
        self._pending_ready_groups = []
        start_step = self.global_update_step
        self._info0(
            "GRPO pending update trigger: rank=%s ready_groups=%s required=%s group_ids=%s",
            self.rank,
            len(ready_groups),
            self.rollout_groups_per_update,
            [group[0].group_id if group else "<empty>" for group in ready_groups],
        )
        try:
            if self.validate_logprob_consistency:
                for group in ready_groups:
                    self._validate_group_logprob_consistency(group)
            self._run_grpo_update(ready_groups)
            for group in ready_groups:
                # Free the consumed episodes' chunks (each holds CPU obs / KV /
                # latent_noise / keyframes). Otherwise _episodes grows
                # unbounded across update steps and the server OOMs on CPU.
                self.rollout_store.drop_episodes(group)
        except Exception:
            # Re-queue everything so the next trigger retries.
            self._pending_ready_groups = ready_groups + self._pending_ready_groups
            raise
        self._info0(
            "GRPO pending update complete: rank=%s updated_groups=%s global_update_step=%s last_stats=%s",
            self.rank,
            len(ready_groups),
            self.global_update_step,
            self.last_stats,
        )
        return {
            "updated": self.global_update_step > start_step,
            "updated_groups": len(ready_groups),
            "global_update_step": self.global_update_step,
            "in_eval": self._eval_pending,
            "status": self.last_stats,
        }

    def _detach_kv_cache_pools(self) -> None:
        """Strip stale grad_fn from attention KV cache pool buffers.

        WanAttention.forward does in-place setitem of grad-tracked key/value
        into cache['k']/['v'] (then restore_cache only flips the mask, the
        tensor data + autograd state stays). After backward releases that
        graph's saved tensors, the pool buffers still carry a dangling
        grad_fn chain. The next chunk's setitem would chain back to those
        freed nodes and raise "backward through the graph a second time".
        Detaching in place makes the pool a fresh leaf again, breaking the
        cross-chunk autograd linkage that the cache buffer would otherwise
        smuggle in.
        """
        for module in self.transformer.modules():
            caches = getattr(module, "attn_caches", None)
            if caches is None:
                continue
            for cache in caches.values():
                if cache is None:
                    continue
                for field in ("k", "v"):
                    tensor = cache.get(field)
                    if tensor is not None and tensor.grad_fn is not None:
                        tensor.detach_()

    def _backward_episode_logprob(self, episode, grad: torch.Tensor) -> torch.Tensor:
        """Per-chunk grad-forward + backward of chunk_logprob * grad * normalizer.

        ``grad`` is the per-episode PPO surrogate gradient wrt the trajectory
        new_logprob (i.e. ``-ratio_detached * advantage / B`` when the
        unclipped surrogate is selected, 0 when clipped — see
        ``_run_grpo_update`` for derivation). The clip decision is made in
        detached space; the backward path here is the ONLY forward whose
        chunk_logprob participates in the autograd graph, so the gradient is
        always computed against a consistent surface.

        Per-chunk backward (vs whole-episode backward) bounds peak activation
        memory to a single chunk because chunks are autograd-independent once
        the kv cache pool buffers are detached between chunks (see
        ``_detach_kv_cache_pools``). ``_run_video_prefix`` and
        ``_compute_kv_cache`` both run under no_grad, and the per_step
        recompute loop calls the transformer with ``update_cache=0``.

        Returns the (detached fp32) trajectory-level logprob computed during
        this grad-forward pass — used by the caller for the actual
        ``new_logprob_mean`` log entry and for the two-forward drift
        diagnostic.
        """
        with torch.no_grad():
            self._reset(prompt=episode.prompt)
        n_chunks = len(episode.chunks)
        if n_chunks == 0:
            return torch.zeros((), device=self.device, dtype=torch.float32)
        scale = grad.reshape(()).to(self.device)
        normalizer = (1.0 / n_chunks) if self.normalize_episode_length else 1.0
        chunk_scale = scale * normalizer
        # Clipped episodes have coef==0: skip the autograd graph entirely so
        # we don't allocate activations for transformer forwards whose
        # gradient contribution is zero. The cache writes inside
        # _recompute_chunk_logprob still happen because subsequent code (and
        # the next minibatch's _reset) depends on a consistent cache state
        # being left behind regardless of whether we backward'd through it.
        skip_backward = bool(chunk_scale.detach().abs().item() == 0.0)
        total_lp = torch.zeros((), device=self.device, dtype=torch.float32)
        norm_f = float(normalizer)
        for chunk in episode.chunks:
            if skip_backward:
                with torch.no_grad():
                    chunk_logprob = self._recompute_chunk_logprob(chunk).reshape(())
            else:
                chunk_logprob = self._recompute_chunk_logprob(chunk).reshape(())
                (chunk_logprob * chunk_scale).backward()
            total_lp = total_lp + chunk_logprob.detach().float() * norm_f
            self._detach_kv_cache_pools()
            if chunk.keyframes is not None and chunk.state is not None:
                with torch.no_grad():
                    self._compute_kv_cache({"obs": chunk.keyframes, "state": chunk.state})
        return total_lp

    def _episode_old_logprobs(self, episodes) -> torch.Tensor:
        result = []
        for ep in episodes:
            chunk_lps = []
            for chunk in ep.chunks:
                transition_lps = chunk.old_logprobs.reshape(-1)
                if transition_lps.numel() == 0:
                    chunk_lps.append(torch.zeros((), dtype=torch.float32))
                    continue
                chunk_lp = transition_lps.sum()
                if self.normalize_denoising_horizon and transition_lps.numel() > 1:
                    chunk_lp = chunk_lp / transition_lps.numel()
                chunk_lps.append(chunk_lp)
            chunk_lps = torch.stack(chunk_lps)
            if self.normalize_episode_length and chunk_lps.numel() > 0:
                result.append(chunk_lps.mean())
            else:
                result.append(chunk_lps.sum())
        return torch.stack(result)

    def _validate_group_logprob_consistency(self, episodes) -> dict[str, Any]:
        old_logprobs = self._episode_old_logprobs(episodes).to(self.device)
        with torch.no_grad():
            new_logprobs = torch.cat([
                self.get_action_logprobs(ep).reshape(1) for ep in episodes
            ], dim=0)
        abs_diff = (new_logprobs - old_logprobs).abs()
        stats = {
            "old_logprob_mean": float(old_logprobs.mean().detach().cpu()),
            "new_logprob_mean": float(new_logprobs.mean().detach().cpu()),
            "max_abs_diff": float(abs_diff.max().detach().cpu()),
            "mean_abs_diff": float(abs_diff.mean().detach().cpu()),
            "passed": bool(abs_diff.max().detach().cpu() <= self.logprob_consistency_atol),
            "atol": self.logprob_consistency_atol,
        }
        self.last_stats["logprob_consistency"] = stats
        self._write_json("latest_logprob_consistency.json", stats)
        if stats["passed"]:
            logger.info("GRPO logprob consistency passed: rank=%s %s", self.rank, stats)
        else:
            logger.warning("GRPO logprob consistency failed: rank=%s %s", self.rank, stats)
        self._wandb_log(
            {f"debug/logprob_consistency/{key}": value for key, value in stats.items()},
        )
        return stats

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        # In replicated mode every rank reaches this with identical data; if
        # we let them all write the file races and partial content is observed
        # by tail-watchers. Only rank 0 writes.
        if not self._is_rank0():
            return
        path = Path(self.job_config.save_root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _sync_replicated_trainable_params_from_rank0(self) -> None:
        """Broadcast trainable params so replicated ranks start from one state."""
        if not getattr(self, "replicated_enabled", False):
            return
        if not getattr(self, "trainable_params", None):
            return
        for param in self.trainable_params:
            dist.broadcast(param.data, src=0)
        if self._is_rank0():
            n_tensors = len(self.trainable_params)
            n_params = sum(param.numel() for param in self.trainable_params)
            logger.info(
                "Replicated trainable parameter sync complete: rank=%s source_rank=0 tensors=%s params=%s",
                self.rank,
                n_tensors,
                n_params,
            )

    def _allreduce_gradients(self) -> None:
        """Average gradients across replicated-mode ranks before optimizer.step().

        DDP-style replicated GRPO: each rank rolls out and backwards on its own
        episodes, then we average the accumulated grads so every rank applies
        the same update. No-op outside replicated mode (single-GPU or FSDP).
        """
        if not getattr(self, "replicated_enabled", False):
            return
        ws = float(self.world_size)
        for param in self.trainable_params:
            if param.grad is None:
                # A rank can have an all-clipped minibatch and skip backward,
                # while peer ranks still have real gradients. All ranks must
                # still enter the exact same all-reduce sequence; this rank's
                # contribution is simply zero.
                param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(ws)

    def _gather_replicated_rollout_records(
        self, groups: list[list]
    ) -> tuple[list[dict[str, Any]], dict[int, int]]:
        """Collect lightweight rollout records across replicated ranks.

        Episode/group websocket callbacks are asynchronous across ranks, so they
        cannot safely use collectives. This helper is called only from the GRPO
        update path, where all replicated ranks are already in lockstep.
        """
        local_records: list[dict[str, Any]] = []
        for group in groups:
            for episode in group or []:
                metadata = getattr(episode, "metadata", {}) or {}
                local_records.append({
                    "server_rank": int(self.rank),
                    "group_id": str(getattr(episode, "group_id", "")),
                    "reward": float(getattr(episode, "reward", 0.0)),
                    "success": bool(getattr(episode, "success", False)),
                    "step_count": int(getattr(episode, "step_count", 0) or 0),
                    "chunks": int(len(getattr(episode, "chunks", []) or [])),
                    "episode_idx": metadata.get("episode_idx"),
                    "group_member": metadata.get("group_member"),
                })

        if not getattr(self, "replicated_enabled", False):
            return local_records, {int(self.rank): int(self.global_rollout_episode)}

        local_payload = {
            "records": local_records,
            "global_rollout_episode": int(self.global_rollout_episode),
        }
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_payload)

        merged_records: list[dict[str, Any]] = []
        rank_rollout_counts: dict[int, int] = {}
        for rank, payload in enumerate(gathered):
            payload = payload or {}
            rank_rollout_counts[int(rank)] = int(payload.get("global_rollout_episode", 0))
            for record in payload.get("records", []) or []:
                item = dict(record)
                item.setdefault("server_rank", int(rank))
                merged_records.append(item)
        return merged_records, rank_rollout_counts

    def _log_replicated_rollout_group_wandb(
        self,
        records: list[dict[str, Any]],
        *,
        update_step: int,
        rank_rollout_counts: dict[int, int],
    ) -> None:
        """Log DDP rollout/group W&B metrics as global aggregates on rank 0."""
        if not getattr(self, "replicated_enabled", False):
            return
        if not self._is_rank0() or not records:
            return

        global_rollout_episode = float(sum(rank_rollout_counts.values()))
        rollout_stats = _episode_summary_stats(records)
        rank_episode_counts: dict[int, int] = {}
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            rank = int(record.get("server_rank", -1))
            rank_episode_counts[rank] = rank_episode_counts.get(rank, 0) + 1
            group_id = str(record.get("group_id", ""))
            grouped.setdefault(group_id, []).append(record)

        def _optional_int(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def _rollout_record_sort_key(item: tuple[int, dict[str, Any]]) -> tuple:
            original_pos, record = item
            episode_idx = _optional_int(record.get("episode_idx"))
            group_member = _optional_int(record.get("group_member"))
            return (
                episode_idx is None,
                episode_idx if episode_idx is not None else original_pos,
                int(record.get("server_rank", -1)),
                str(record.get("group_id", "")),
                group_member if group_member is not None else -1,
                original_pos,
            )

        indexed_records = list(enumerate(records))
        ordered_records = [
            record for _, record in sorted(indexed_records, key=_rollout_record_sort_key)
        ]
        # Reconstruct a monotonically increasing per-episode x-axis for W&B.
        first_rollout_episode = max(int(global_rollout_episode) - len(ordered_records) + 1, 1)
        for offset, record in enumerate(ordered_records):
            episode_rollout_idx = float(first_rollout_episode + offset)
            self._wandb_log({
                "global_rollout_episode": episode_rollout_idx,
                "rollout/global_rollout_episode": episode_rollout_idx,
                "rollout/global_update_step": update_step,
                "rollout/episode_reward": float(record.get("reward", 0.0)),
                "rollout/episode_success": float(bool(record.get("success", False))),
                "rollout/episode_step_count": float(record.get("step_count", 0.0)),
                "rollout/episode_chunks": float(record.get("chunks", 0.0)),
            })

        self._wandb_log({
            "global_rollout_episode": global_rollout_episode,
            "rollout/global_rollout_episode": global_rollout_episode,
            "rollout/global_update_step": update_step,
            "rollout/reward_mean": rollout_stats["reward_mean"],
            "rollout/success_rate": rollout_stats["success_rate"],
            "rollout/step_count_mean": rollout_stats["step_count_mean"],
            "rollout/chunk_count_mean": rollout_stats["chunk_count_mean"],
            "rollout/episode_count": rollout_stats["episode_count"],
            "rollout/success_count": rollout_stats["success_count"],
            "rollout/reward_sum": rollout_stats["reward_sum"],
            "rollout/step_count_sum": rollout_stats["step_count_sum"],
            "rollout/chunk_count_sum": rollout_stats["chunk_count_sum"],
            "rollout/group_count": float(len(grouped)),
            "rollout/rank_count": float(len(rank_episode_counts)),
        })

        for group_records in grouped.values():
            group_stats = _episode_summary_stats(group_records)
            self._wandb_log({
                "global_rollout_episode": global_rollout_episode,
                "group/global_rollout_episode": global_rollout_episode,
                "group/global_update_step": update_step,
                "group/reward_mean": group_stats["reward_mean"],
                "group/reward_std": group_stats["reward_std"],
                "group/reward_sum": group_stats["reward_sum"],
                "group/success_rate": group_stats["success_rate"],
                "group/success_count": group_stats["success_count"],
                "group/size": group_stats["episode_count"],
                "group/step_count_mean": group_stats["step_count_mean"],
                "group/step_count_sum": group_stats["step_count_sum"],
                "group/chunk_count_mean": group_stats["chunk_count_mean"],
                "group/chunk_count_sum": group_stats["chunk_count_sum"],
                "group/pending_ready_groups": float(len(grouped)),
            })

        self._info0(
            "GRPO replicated rollout aggregate: rank=%s step=%s episodes=%s groups=%s "
            "success_rate=%.4f reward_mean=%.4f rank_episode_counts=%s",
            self.rank,
            update_step,
            int(rollout_stats["episode_count"]),
            len(grouped),
            rollout_stats["success_rate"],
            rollout_stats["reward_mean"],
            rank_episode_counts,
        )

    def _gather_sharded_rewards(
        self, groups: list[list]
    ) -> dict[str, dict[int, float]]:
        """Collect per-group rewards keyed by ``group_member`` across all ranks.

        Sharded-group GRPO splits one logical group of size ``group_size`` across
        ranks (each holds ``local_group_size`` members). To normalize advantages
        over the full group we exchange just the lightweight ``(group_id,
        member, reward)`` triples; the heavy episode tensors stay on the rank
        that owns them and are loaded only for backward on that rank's slice.

        Returns ``{group_id: {member: reward}}`` after merging all ranks' slices.
        Raises if a member appears on more than one rank or if a group is missing
        members after the merge — both signal a client-side slicing bug.
        """
        if not getattr(self, "sharded_group_enabled", False):
            raise RuntimeError("_gather_sharded_rewards called outside sharded mode")

        # Build local payload: list of (group_id, [(member, reward), ...]) so a
        # single all_gather covers every group this rank knows about.
        local_payload: list[tuple[str, list[tuple[int, float]]]] = []
        for group in groups:
            if not group:
                continue
            members = []
            for ep in group:
                meta = getattr(ep, "metadata", {}) or {}
                if "group_member" not in meta:
                    raise RuntimeError(
                        f"Sharded GRPO requires episode.metadata['group_member']; "
                        f"missing on episode group_id={ep.group_id!r}"
                    )
                members.append((int(meta["group_member"]), float(ep.reward)))
            local_payload.append((str(group[0].group_id), members))

        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_payload)

        merged: dict[str, dict[int, float]] = {}
        for rank_payload in gathered:
            if not rank_payload:
                continue
            for gid, members in rank_payload:
                bucket = merged.setdefault(gid, {})
                for member, reward in members:
                    if member in bucket:
                        raise RuntimeError(
                            f"Sharded GRPO: duplicate group_member={member} for "
                            f"group_id={gid!r} across ranks (clients must use "
                            "global slicing: global_id = rank*num_clients + client_id)"
                        )
                    bucket[member] = float(reward)

        for gid, bucket in merged.items():
            if len(bucket) != self.group_size:
                raise RuntimeError(
                    f"Sharded GRPO: incomplete group {gid!r} after cross-rank merge: "
                    f"got {len(bucket)} members, expected {self.group_size}. "
                    "Check client --world_size/--rank/--num_clients flags and that "
                    "every assignment item completes on every rank."
                )
        return merged

    def _gather_replicated_eval_results(self) -> list[dict[str, Any]]:
        """Barrier and merge eval results across replicated server ranks."""
        local_results = [dict(result) for result in self._eval_results]
        if not getattr(self, "replicated_enabled", False):
            return local_results

        # Every rank's client-0 calls end_eval_phase after its local clients have
        # finished. This barrier turns those local file barriers into one global
        # eval-phase boundary before any rank clears _eval_results.
        dist.barrier()
        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, local_results)

        merged: list[dict[str, Any]] = []
        for rank, rank_results in enumerate(gathered):
            for result in rank_results or []:
                item = dict(result)
                item.setdefault("server_rank", int(rank))
                merged.append(item)
        return merged

    def _allreduce_scalar_max(self, value: float) -> float:
        """All-reduce a scalar by MAX across replicated ranks (used for KL early-stop).

        Returns ``value`` unchanged outside replicated mode. Using MAX makes
        early-stop a worst-case decision so all ranks reach the same verdict
        and stop together.
        """
        if not getattr(self, "replicated_enabled", False):
            return value
        t = torch.tensor([float(value)], device=self.device, dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return float(t.item())

    def _allreduce_update_effect_stats(
        self,
        *,
        delta_values: list[float],
        signed_adv_delta_values: list[float],
        pos_delta_values: list[float],
        neg_delta_values: list[float],
        pos_delta_positive_values: list[float],
        neg_delta_negative_values: list[float],
    ) -> dict[str, float]:
        """Aggregate post-step update-effect diagnostics across replicated ranks."""

        def _sum(values: list[float]) -> float:
            return float(sum(values)) if values else 0.0

        payload = torch.tensor(
            [
                _sum(delta_values),
                float(len(delta_values)),
                _sum(signed_adv_delta_values),
                float(len(signed_adv_delta_values)),
                _sum(pos_delta_values),
                float(len(pos_delta_values)),
                _sum(neg_delta_values),
                float(len(neg_delta_values)),
                _sum(pos_delta_positive_values),
                float(len(pos_delta_positive_values)),
                _sum(neg_delta_negative_values),
                float(len(neg_delta_negative_values)),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if getattr(self, "replicated_enabled", False):
            dist.all_reduce(payload, op=dist.ReduceOp.SUM)

        values = payload.detach().cpu().tolist()

        def _mean(sum_value: float, count_value: float) -> float:
            return float(sum_value / count_value) if count_value > 0 else 0.0

        active_count = values[1]
        signed_count = values[3]
        pos_count = values[5]
        neg_count = values[7]
        return {
            "delta_mean": _mean(values[0], active_count),
            "signed_adv_delta_mean": _mean(values[2], signed_count),
            "pos_adv_delta_mean": _mean(values[4], pos_count),
            "neg_adv_delta_mean": _mean(values[6], neg_count),
            "pos_adv_delta_positive_frac": _mean(values[8], pos_count),
            "neg_adv_delta_negative_frac": _mean(values[10], neg_count),
            "active_count": int(active_count),
            "pos_count": int(pos_count),
            "neg_count": int(neg_count),
        }

    # Per-epoch stats reduction spec: which keys combine across ranks by which op.
    # Means assume each rank trained on an equal episode count (true unless
    # degenerate-group filtering dropped different counts per rank, which is rare
    # with shaped rewards); the small inaccuracy is acceptable for a log/wandb
    # aggregate. Keys absent from all four lists pass through from the local dict
    # (lr, *_minibatch_size, update_effect_* which are already global, etc.).
    _EPOCH_STATS_MEAN_KEYS = (
        "loss", "ratio", "ratio_std", "mbs0_ratio_mean", "clipfrac", "approx_kl",
        "old_logprob_mean", "new_logprob_mean", "reward_mean", "reward_std",
        "success_rate", "step_count_mean", "chunk_count_mean", "chunks_mean",
        "grad_norm_mean", "chunks_logratio_corr", "logratio_std",
    )
    _EPOCH_STATS_MIN_KEYS = (
        "ratio_min", "mbs0_ratio_min", "reward_min", "logratio_min",
        "advantage_min", "chunks_min", "grad_norm_min",
    )
    _EPOCH_STATS_MAX_KEYS = (
        "ratio_max", "mbs0_ratio_max", "reward_max", "logratio_max",
        "logratio_abs_max", "logratio_per_chunk_abs_max", "advantage_max",
        "advantage_abs_max", "chunks_max", "grad_norm_max",
    )
    _EPOCH_STATS_SUM_KEYS = (
        "rollout_episode_count", "rollout_group_count", "kept_group_count",
        "degenerate_groups_skipped", "trained_episode_count",
        "minibatches_per_epoch", "logratio_outlier_count",
    )

    def _allreduce_named_stats(self, stats: dict[str, Any]) -> dict[str, float | Any]:
        """Reduce a per-rank epoch ``stats`` dict to a global one across ranks.

        Mean keys are averaged, min/max keys reduced with the matching op, sum
        keys summed; everything else is passed through unchanged. No-op (returns
        a shallow copy) when running single-rank. Safe to call only at a
        synchronization point where every rank executes it in lockstep — the
        per-epoch logging site qualifies (all ranks finish the epoch together)."""
        out = dict(stats)
        if getattr(self, "world_size", 1) <= 1 or not dist.is_initialized():
            return out

        mean_keys = [k for k in self._EPOCH_STATS_MEAN_KEYS if k in stats]
        sum_keys = [k for k in self._EPOCH_STATS_SUM_KEYS if k in stats]
        min_keys = [k for k in self._EPOCH_STATS_MIN_KEYS if k in stats]
        max_keys = [k for k in self._EPOCH_STATS_MAX_KEYS if k in stats]

        def _reduce(keys: list[str], op) -> list[float]:
            if not keys:
                return []
            t = torch.tensor(
                [float(stats[k]) for k in keys], device=self.device, dtype=torch.float64
            )
            dist.all_reduce(t, op=op)
            return t.detach().cpu().tolist()

        # mean and sum share one SUM all_reduce; means are divided afterwards.
        sum_pack = mean_keys + sum_keys
        for k, v in zip(sum_pack, _reduce(sum_pack, dist.ReduceOp.SUM)):
            out[k] = v / float(self.world_size) if k in self._EPOCH_STATS_MEAN_KEYS else v
        for k, v in zip(min_keys, _reduce(min_keys, dist.ReduceOp.MIN)):
            out[k] = v
        for k, v in zip(max_keys, _reduce(max_keys, dist.ReduceOp.MAX)):
            out[k] = v
        return out

    def _trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict containing only trainable params (LoRA adapters when LoRA is on)."""
        trainable_names = {
            name for name, param in self.transformer.named_parameters() if param.requires_grad
        }
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.transformer.state_dict().items()
            if name in trainable_names
        }

    def save_checkpoint(self) -> str:
        """Snapshot current state on the main thread, write to disk in the background.

        Captures the (CPU) tensors here so the websocket event loop is not blocked
        on torch.save() — which can take minutes on /root/autodl-tmp under load
        and previously froze the server entirely (see notes on the 2026-05-13
        validation hang). The actual file write runs in a single-worker thread
        pool; if a previous save is still in flight we skip rather than queue.

        In replicated (DDP-style) mode all ranks hold identical weights after
        the gradient-synced optimizer step, so only rank 0 writes the file.
        """
        if not self._is_rank0():
            return ""
        ckpt_dir = Path(self.job_config.save_root) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        step = self.global_update_step
        path = ckpt_dir / f"grpo_step_{step:06d}.pt"

        if (
            self._pending_checkpoint_future is not None
            and not self._pending_checkpoint_future.done()
        ):
            logger.warning(
                "Skipping GRPO checkpoint: rank=%s step=%s previous save still in flight",
                self.rank,
                step,
            )
            return str(path)

        # Under LoRA the base model is frozen and known from model_name_or_path; only
        # the adapter deltas need to be persisted. save_full_transformer=true forces
        # the full state dict (useful for non-LoRA runs or full-finetune debugging).
        save_full = bool(self.rl_cfg.get("save_full_transformer", not self.use_lora))
        transformer_state = (
            {k: v.detach().cpu() for k, v in self.transformer.state_dict().items()}
            if save_full
            else self._trainable_state_dict()
        )
        payload: dict[str, Any] = {
            "transformer": transformer_state,
            "transformer_is_partial": not save_full,
            "global_update_step": step,
            "config": dict(self.job_config),
        }
        # rollout_store contains full episode chunks (obs / keyframes / state /
        # action_chain / old_logprobs); at 80 episodes it was already 9.4GB
        # serialized. Off by default — only needed for resume-with-replay.
        if bool(self.rl_cfg.get("save_rollout_store", False)):
            payload["rollout_store"] = self.rollout_store.state_dict()
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()

        export_inference = bool(self.rl_cfg.get("export_inference_checkpoint", True))
        adapter_snapshot = self._trainable_state_dict() if (export_inference and self.use_lora) else None

        self._pending_checkpoint_future = self._checkpoint_executor.submit(
            self._write_checkpoint_async,
            payload=payload,
            path=path,
            step=step,
            save_full=save_full,
            export_inference=export_inference,
            adapter_snapshot=adapter_snapshot,
        )
        return str(path)

    def _write_checkpoint_async(
        self,
        *,
        payload: dict[str, Any],
        path: Path,
        step: int,
        save_full: bool,
        export_inference: bool,
        adapter_snapshot: dict[str, torch.Tensor] | None,
    ) -> None:
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            torch.save(payload, tmp)
            os.replace(tmp, path)
            latest = path.parent / "latest.pt"
            latest_tmp = latest.with_suffix(latest.suffix + ".tmp")
            torch.save(payload, latest_tmp)
            os.replace(latest_tmp, latest)
            if export_inference:
                self._write_inference_export_async(step, adapter_snapshot)
            logger.info(
                "Saved GRPO checkpoint: rank=%s path=%s transformer_keys=%d full=%s",
                self.rank,
                path,
                len(payload.get("transformer", {})),
                save_full,
            )
            self._wandb_log(
                {
                    "checkpoint/global_update_step": step,
                    "checkpoint/transformer_keys": len(payload.get("transformer", {})),
                    "checkpoint/save_full_transformer": save_full,
                },
            )
        except Exception:
            logger.error(
                "Background GRPO checkpoint write failed: rank=%s step=%s\n%s",
                self.rank,
                step,
                traceback.format_exc(),
            )

    def _write_inference_export_async(
        self,
        step: int,
        adapter_snapshot: dict[str, torch.Tensor] | None,
    ) -> str | None:
        if not self._is_rank0():
            return None
        export_root = Path(self.job_config.save_root) / "inference_exports" / f"step_{step:06d}"
        export_root.mkdir(parents=True, exist_ok=True)
        base_model = Path(self.job_config.wan22_pretrained_model_name_or_path)
        for name in ("vae", "tokenizer", "text_encoder"):
            link_path = export_root / name
            target = base_model / name
            if link_path.exists() or link_path.is_symlink():
                continue
            try:
                os.symlink(target, link_path, target_is_directory=True)
            except OSError:
                logger.warning(
                    "Could not symlink: rank=%s %s -> %s; write access may be restricted",
                    self.rank,
                    link_path,
                    target,
                )
        try:
            if self.use_lora:
                adapter_dir = export_root / "transformer_lora"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                assert adapter_snapshot is not None, "adapter snapshot missing for LoRA export"
                torch.save(adapter_snapshot, adapter_dir / "adapter.pt")
                adapter_meta = {
                    "format": "lingbot_va_lora_v1",
                    "base_transformer_path": str(base_model / "transformer"),
                    "lora": {
                        "rank": int(self.lora_cfg.get("rank", 8)),
                        "alpha": float(self.lora_cfg.get("alpha", 16.0)),
                        "dropout": float(self.lora_cfg.get("dropout", 0.0)),
                        "target_modules": list(self.lora_cfg.get(
                            "target_modules",
                            ["to_q", "to_k", "to_v", "to_out.0", "action_embedder", "action_proj_out"],
                        )),
                    },
                    "global_update_step": step,
                }
                with open(adapter_dir / "adapter_config.json", "w") as f:
                    json.dump(adapter_meta, f, indent=2)
                base_link = export_root / "transformer_base"
                if not base_link.exists() and not base_link.is_symlink():
                    try:
                        os.symlink(base_model / "transformer", base_link, target_is_directory=True)
                    except OSError:
                        logger.warning(
                            "Could not symlink: rank=%s %s -> %s",
                            self.rank,
                            base_link,
                            base_model / "transformer",
                        )
            else:
                # Non-LoRA export still touches GPU weights; this path was not exercised
                # in the validation run that hit the hang. We keep the legacy behavior
                # but it remains synchronous against the transformer object — call sites
                # should set export_inference_checkpoint=false when not using LoRA.
                self.transformer.save_pretrained(
                    export_root / "transformer",
                    safe_serialization=True,
                )
        except Exception:
            logger.exception(
                "Failed to write inference-compatible transformer export: rank=%s",
                self.rank,
            )
            return None
        latest = Path(self.job_config.save_root) / "inference_exports" / "latest"
        try:
            if latest.is_symlink() or latest.exists():
                latest.unlink()
            os.symlink(export_root.name, latest, target_is_directory=True)
        except OSError:
            pass
        return str(export_root)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.transformer.load_state_dict(ckpt["transformer"], strict=False)
        if self.optimizer is not None and "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self._sync_replicated_trainable_params_from_rank0()
        self.global_update_step = int(ckpt.get("global_update_step", 0))
        if self.global_update_step > 0:
            self._eval_pending = False
            self._eval_results = []
        if "rollout_store" in ckpt:
            self.rollout_store.load_state_dict(ckpt["rollout_store"])
            # Resuming across a topology change (e.g. single-GPU checkpoint
            # loaded under sharded multi-GPU) would otherwise inherit the
            # serialized group_size and mismatch the current run's local
            # threshold. The store should be empty after drop_episodes anyway,
            # so re-asserting local_group_size is safe.
            self.rollout_store.group_size = self.local_group_size

    def infer(self, obs):
        session_id = obs.pop("_session_id", None)
        command = obs.get("command")
        if command is None:
            if obs.get("reset", False):
                command = "reset_episode"
            elif obs.get("compute_kv_cache", False):
                command = "commit_chunk"
            else:
                command = "sample_action"

        if command == "reset_episode":
            prompt = obs.get("prompt")
            task = obs.get("task") or obs.get("task_name") or "unknown_task"
            seed = int(obs.get("seed", 0))
            group_id = obs.get("group_id") or f"{task}:{seed}:{prompt}"
            is_eval = bool(obs.get("is_eval", False))
            if self._active_session_id is not None and self._active_session_id != session_id:
                self._swap_out(self._active_session_id)
            if session_id in self._session_store:
                del self._session_store[session_id]
            self._active_session_id = session_id
            with torch.no_grad():
                self._reset(prompt=prompt)
                rollout_seed = int(obs.get("rollout_seed", seed + 1000003 * int(obs.get("episode_idx", 0))))
                self._rollout_generators[session_id] = torch.Generator(device=self.device).manual_seed(rollout_seed)
            if is_eval:
                self._eval_sessions.add(session_id)
            else:
                self._eval_sessions.discard(session_id)
            episode = self.rollout_store.start_episode(
                session_id=session_id,
                prompt=prompt,
                task=task,
                seed=seed,
                group_id=group_id,
                metadata={k: v for k, v in obs.items() if k not in {"command", "prompt", "task", "task_name", "seed", "group_id"}},
            )
            return {"episode_id": episode.episode_id, "group_id": episode.group_id, "is_eval": is_eval}

        if command == "sample_action":
            eval_mode = session_id in self._eval_sessions
            with torch.no_grad():
                self._switch_to_session(session_id)
                action, chunk = self._sample_action_chunk(
                    obs,
                    frame_st_id=self.frame_st_id,
                    generator=self._rollout_generators.get(session_id),
                    eval_mode=eval_mode,
                )
            self.rollout_store.add_chunk(session_id, chunk)
            return {"action": action}

        if command == "commit_chunk":
            self._switch_to_session(session_id)
            self.rollout_store.attach_chunk_context(session_id, keyframes=obs.get("obs"), state=obs.get("state"))
            with torch.no_grad():
                self._compute_kv_cache(obs)
            return {}

        if command == "finish_episode":
            episode, ready_group = self.rollout_store.finish_episode(
                session_id,
                success=bool(obs.get("success", False)),
                step_count=obs.get("step_count"),
                reward=obs.get("reward"),
                metadata={k: v for k, v in obs.items() if k not in {"command", "success", "step_count", "reward"}},
            )
            if session_id in self._eval_sessions:
                # Eval episode: never feeds GRPO. Stash the per-episode result
                # for end_eval_phase to aggregate, log under eval/*, and drop
                # the chunk immediately so eval doesn't bloat CPU memory.
                self._eval_sessions.discard(session_id)
                result = {
                    "task": episode.task,
                    "seed": int(episode.seed),
                    "success": bool(episode.success),
                    "reward": float(episode.reward),
                    "step_count": int(episode.step_count or 0),
                    "group_id": episode.group_id,
                }
                self._eval_results.append(result)
                self.global_eval_episode += 1
                self._wandb_log(
                    {
                        "eval/global_eval_episode": self.global_eval_episode,
                        "eval/phase_episode": len(self._eval_results),
                        "eval/episode_success": float(result["success"]),
                        "eval/episode_reward": result["reward"],
                        "eval/episode_step_count": float(result["step_count"]),
                        "eval/global_update_step": self.global_update_step,
                    },
                )
                self._info0(
                    "GRPO eval episode: rank=%s task=%s seed=%s success=%s reward=%.3f "
                    "step_count=%s collected=%s action_steps=%s",
                    self.rank,
                    result["task"],
                    result["seed"],
                    result["success"],
                    result["reward"],
                    result["step_count"],
                    len(self._eval_results),
                    self.eval_action_num_inference_steps,
                )
                self.rollout_store.drop_episodes([episode])
                return {
                    "ready_for_update": False,
                    "pending_ready_groups": len(self._pending_ready_groups),
                    "in_eval": self._eval_pending,
                    "is_eval": True,
                    "status": self.last_stats,
                }
            self.global_rollout_episode += 1
            episode_idx = episode.metadata.get("episode_idx")
            group_member = episode.metadata.get("group_member")
            if not getattr(self, "replicated_enabled", False):
                self._wandb_log(
                    {
                        "rollout/episode_reward": float(episode.reward),
                        "rollout/episode_success": float(bool(episode.success)),
                        "rollout/episode_step_count": float(episode.step_count or 0),
                        "rollout/episode_chunks": len(episode.chunks),
                        "rollout/ready_group": float(ready_group is not None),
                        "rollout/active_sessions": len(self.rollout_store._active_by_session),
                        "rollout/global_update_step": self.global_update_step,
                        "rollout/global_rollout_episode": self.global_rollout_episode,
                        "global_rollout_episode": self.global_rollout_episode,
                    },
                )
            if ready_group is not None:
                group_rewards = [float(ep.reward) for ep in ready_group]
                group_successes = [1.0 if bool(ep.success) else 0.0 for ep in ready_group]
                group_steps = [float(ep.step_count) for ep in ready_group if ep.step_count is not None]
                group_chunks = [float(len(ep.chunks)) for ep in ready_group]
                pending_ready_groups_after = len(self._pending_ready_groups) + (0 if self.disable_updates else 1)
                if not getattr(self, "replicated_enabled", False):
                    self._wandb_log(
                        {
                            "group/reward_mean": float(torch.tensor(group_rewards).mean()) if group_rewards else 0.0,
                            "group/reward_std": float(torch.tensor(group_rewards).std(unbiased=False)) if group_rewards else 0.0,
                            "group/success_rate": float(sum(group_successes) / max(len(group_successes), 1)),
                            "group/size": len(ready_group),
                            "group/step_count_mean": float(torch.tensor(group_steps).mean()) if group_steps else 0.0,
                            "group/pending_ready_groups": pending_ready_groups_after,
                            "group/global_rollout_episode": self.global_rollout_episode,
                            "group/global_update_step": self.global_update_step,
                            "global_rollout_episode": self.global_rollout_episode,
                        },
                    )
                if not self.disable_updates:
                    self._pending_ready_groups.append(ready_group)
                self._info0(
                    "GRPO group ready [rank0-local]: rank=%s group_id=%s size=%s success_rate=%.4f reward_mean=%.4f "
                    "reward_std=%.4f step_count_mean=%.1f step_count_min=%.0f step_count_max=%.0f "
                    "chunk_count_mean=%.1f pending_ready_groups=%s/%s update_deferred=True",
                    self.rank,
                    episode.group_id,
                    len(ready_group),
                    float(sum(group_successes) / max(len(group_successes), 1)),
                    float(torch.tensor(group_rewards).mean()) if group_rewards else 0.0,
                    float(torch.tensor(group_rewards).std(unbiased=False)) if group_rewards else 0.0,
                    float(torch.tensor(group_steps).mean()) if group_steps else 0.0,
                    float(min(group_steps)) if group_steps else 0.0,
                    float(max(group_steps)) if group_steps else 0.0,
                    float(torch.tensor(group_chunks).mean()) if group_chunks else 0.0,
                    len(self._pending_ready_groups),
                    self.rollout_groups_per_update,
                )
            self._info0(
                "GRPO episode finished: rank=%s rollout_episode=%s task=%s seed=%s group_id=%s episode_idx=%s "
                "group_member=%s success=%s reward=%.3f step_count=%s chunks=%s ready_group=%s "
                "pending_ready_groups=%s active_sessions=%s",
                self.rank,
                self.global_rollout_episode,
                episode.task,
                episode.seed,
                episode.group_id,
                episode_idx,
                group_member,
                bool(episode.success),
                float(episode.reward),
                episode.step_count,
                len(episode.chunks),
                ready_group is not None,
                len(self._pending_ready_groups),
                len(self.rollout_store._active_by_session),
            )
            return {
                "ready_for_update": ready_group is not None,
                "pending_ready_groups": len(self._pending_ready_groups),
                "status": self.last_stats,
            }

        if command == "run_pending_updates":
            return self._run_pending_updates()

        if command == "save_checkpoint":
            return {"checkpoint": self.save_checkpoint()}

        if command == "get_status":
            return {
                "global_update_step": self.global_update_step,
                "active_sessions": len(self.rollout_store._active_by_session),
                "pending_ready_groups": len(self._pending_ready_groups),
                "in_eval": self._eval_pending,
                "eval_results_collected": len(self._eval_results),
                "last_stats": self.last_stats,
                "train_action_num_inference_steps": int(self.job_config.action_num_inference_steps),
                "eval_action_num_inference_steps": self.eval_action_num_inference_steps,
            }

        if command == "get_eval_phase":
            return {
                "in_eval": self._eval_pending,
                "global_update_step": self.global_update_step,
                "eval_results_collected": len(self._eval_results),
                "eval_action_num_inference_steps": self.eval_action_num_inference_steps,
            }

        if command == "end_eval_phase":
            local_n = len(self._eval_results)
            results_snapshot = self._gather_replicated_eval_results()
            n = len(results_snapshot)
            success_rate = 0.0
            reward_mean = 0.0
            rank_counts: dict[int, int] = {}
            for result in results_snapshot:
                if "server_rank" in result:
                    rank = int(result["server_rank"])
                    rank_counts[rank] = rank_counts.get(rank, 0) + 1
            if n > 0:
                successes = [1.0 if r["success"] else 0.0 for r in results_snapshot]
                rewards = [r["reward"] for r in results_snapshot]
                success_rate = sum(successes) / n
                reward_mean = sum(rewards) / n
                per_seed = {
                    f"eval/per_seed/{r['task']}/{r['seed']}/success": float(r["success"])
                    for r in results_snapshot
                }
                self._wandb_log(
                    {
                        "eval/success_rate": success_rate,
                        "eval/reward_mean": reward_mean,
                        "eval/n_episodes": float(n),
                        "eval/local_n_episodes": float(local_n),
                        "eval/global_update_step": self.global_update_step,
                        **per_seed,
                    },
                )
                if self._is_rank0():
                    logger.info(
                        "GRPO eval phase done: rank=%s n=%s success_rate=%.4f reward_mean=%.4f "
                        "global_update_step=%s local_n=%s rank_counts=%s",
                        self.rank,
                        n,
                        success_rate,
                        reward_mean,
                        self.global_update_step,
                        local_n,
                        rank_counts or None,
                    )
            elif self._is_rank0():
                logger.warning(
                    "GRPO end_eval_phase called with no merged eval results: "
                    "rank=%s clearing flag anyway.",
                    self.rank,
                )
            elif local_n:
                # Non-rank0 bookkeeping only; rank 0 emits the aggregate above.
                logger.debug(
                    "GRPO eval rank merged: rank=%s local_n=%s merged_n=%s global_update_step=%s",
                    self.rank,
                    local_n,
                    n,
                    self.global_update_step,
                )
            self._eval_pending = False
            self._eval_results = []
            return {
                "in_eval": False,
                "n_episodes": n,
                "local_n_episodes": local_n,
                "success_rate": success_rate,
                "reward_mean": reward_mean,
                "global_update_step": self.global_update_step,
                "rank_counts": {str(rank): count for rank, count in rank_counts.items()},
                "results": results_snapshot,
            }

        raise ValueError(f"Unknown GRPO websocket command: {command}")

    def on_session_closed(self, session_id):
        self.rollout_store.remove_session(session_id)
        self._rollout_generators.pop(session_id, None)
        super().on_session_closed(session_id)


def run(args):
    config = load_rl_config(args.config)
    if args.port is not None:
        config.port = args.port
    if args.save_root is not None:
        config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    use_fsdp = bool(config.rl.get("use_fsdp", False))
    if use_fsdp and world_size <= 1:
        raise RuntimeError("GRPO FSDP config requires torchrun with WORLD_SIZE > 1")
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    # Replicated (DDP-style) mode: torchrun-launched, each rank holds a full
    # copy of the model and binds its own websocket port (base_port + rank) so
    # rollouts run in parallel. Gradients are synchronized across ranks at
    # update time inside _run_grpo_update.
    replicated_mode = world_size > 1 and not use_fsdp
    rank_port = int(config.port) + rank if replicated_mode else int(config.port)

    model = GRPOTrainingServer(config)
    if args.resume_from:
        model.load_checkpoint(args.resume_from)
    run_async_server_mode(model, local_rank, config.host, rank_port)


def main():
    parser = argparse.ArgumentParser(description="LingBot-VA RoboTwin GRPO server")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--save-root", "--save_root", dest="save_root", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
