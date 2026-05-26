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
    gaussian_logprob,
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
                "GRPO LoRA enabled: rank=%s alpha=%s dropout=%s wrapped=%s trainable_params=%s total_params=%s",
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

        self.rollout_store = RolloutStore(group_size=int(self.rl_cfg.get("group_size", 2)))
        self.rollout_groups_per_update = int(self.rl_cfg.get("rollout_groups_per_update", 1))
        self._pending_ready_groups = []
        self._rollout_generators: dict[str, torch.Generator] = {}
        self.global_update_step = 0
        self.global_rollout_episode = 0
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

        self.action_noise_std = float(self.rl_cfg.get("action_noise_std", 0.05))
        self.clip_range = float(self.rl_cfg.get("clip_range", 0.2))
        self.entropy_coef = float(self.rl_cfg.get("entropy_coef", 0.0))
        self.target_kl = self.rl_cfg.get("target_kl", None)
        self.target_kl = None if self.target_kl is None else float(self.target_kl)
        self.update_epochs = int(self.rl_cfg.get("update_epochs", 1))
        # Minibatch size in *episodes* used inside one GRPO update. When the
        # update batches `rollout_groups_per_update` groups together, this
        # controls how many episodes are processed per optimizer step. Default
        # to processing the full batch in one step (one optimizer step / epoch).
        _batch_size = self.rl_cfg.get("batch_size", None)
        if _batch_size is None:
            self.minibatch_size = max(
                1,
                int(self.rl_cfg.get("group_size", 2))
                * max(int(self.rollout_groups_per_update), 1),
            )
        else:
            self.minibatch_size = max(1, int(_batch_size))
        self.validate_logprob_consistency = bool(
            self.rl_cfg.get("validate_logprob_consistency", False)
        )
        self.logprob_consistency_atol = float(
            self.rl_cfg.get("logprob_consistency_atol", 5e-3)
        )
        self.normalize_denoising_horizon = bool(
            self.rl_cfg.get("logprob", {}).get("normalize_denoising_horizon", True)
        )
        self.normalize_action_dim = bool(
            self.rl_cfg.get("logprob", {}).get("normalize_action_dim", True)
        )
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
                "rl.logprob.normalize_denoising_horizon ignored under noise_schedule=per_chunk "
                "(there is only one Gaussian transition per chunk); forcing to False."
            )
            self.normalize_denoising_horizon = False

        Path(self.job_config.save_root).mkdir(parents=True, exist_ok=True)
        self._init_wandb()
        logger.info(
            "GRPO server ready: group_size=%s update_epochs=%s action_noise_std=%s "
            "noise_schedule=%s disable_updates=%s train_action_steps=%s eval_action_steps=%s",
            self.rollout_store.group_size,
            self.update_epochs,
            self.action_noise_std,
            self.noise_schedule,
            self.disable_updates,
            int(self.job_config.action_num_inference_steps),
            self.eval_action_num_inference_steps,
        )
        if self._eval_pending:
            logger.info(
                "GRPO initial eval phase armed at startup: eval_every=%s global_update_step=%s",
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
            logger.warning("wandb logging requested but unavailable: %s", exc)
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
            logger.info("wandb logging enabled: project=%s name=%s mode=%s", init_kwargs["project"], init_kwargs.get("name"), mode)
        except Exception as exc:
            logger.warning("Failed to initialize wandb logging: %s", exc)
            self._wandb_run = None

    def _wandb_log(self, data: dict[str, Any], *, step: int | None = None) -> None:
        if self._wandb_run is None or not self._is_rank0():
            return
        try:
            self._wandb_run.log(_json_safe(data), step=step)
        except Exception as exc:
            logger.warning("Failed to log wandb metrics: %s", exc)

    def _action_logprob_mask(self, frame_st_id: int, reference: torch.Tensor) -> torch.Tensor:
        mask = self.action_mask.to(reference.device).view(1, -1, 1, 1, 1)
        mask = mask.expand(reference.shape[0], -1, reference.shape[2], reference.shape[3], reference.shape[4]).clone()
        if frame_st_id == 0:
            mask[:, :, 0:1] = False
        return mask

    def _sample_action_transition(self, mean: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + noise * self.action_noise_std

    def _action_transformer_step(
        self,
        actions: torch.Tensor,
        t: torch.Tensor,
        frame_st_id: int,
        *,
        last_step: bool,
    ):
        """Run one action-mode transformer forward and (if not last_step) the scheduler mean.

        Returns ``(mean, action_cond)``. ``mean`` is None when ``last_step`` is True — that
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
        mean = scheduler_transition_mean(self.action_scheduler, action_noise_pred, t, actions)
        return mean, action_cond

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
        else:
            action_chain = [actions.detach().cpu()]
            transition_logprobs = []
            stored_timesteps = None  # fall back to action_timesteps[:-1] at the end

        with torch.no_grad():
            for i, t in enumerate(tqdm(action_timesteps, disable=not _show_denoise_progress())):
                last_step = i == len(action_timesteps) - 1
                mean, action_cond = self._action_transformer_step(
                    actions, t, frame_st_id, last_step=last_step,
                )
                if last_step:
                    if action_cond is not None:
                        actions[:, :, 0:1] = action_cond
                    continue

                if eval_mode or (per_chunk and i < final_noise_idx):
                    # Deterministic step: actions = mean, no noise, no logprob.
                    # eval_mode applies this to every step including the final
                    # noise-bearing one, so the trajectory matches plain
                    # deterministic inference under the same LoRA weights.
                    next_actions = mean
                    next_actions[:, ~self.action_mask] *= 0
                    if action_cond is not None:
                        next_actions[:, :, 0:1] = action_cond
                    actions = next_actions
                    continue

                # Noise-bearing transition.
                next_actions = self._sample_action_transition(mean, generator)
                next_actions[:, ~self.action_mask] *= 0
                if action_cond is not None:
                    next_actions[:, :, 0:1] = action_cond
                lp = gaussian_logprob(
                    next_actions,
                    mean,
                    self.action_noise_std,
                    mask=mask,
                    normalize_action_dim=self.normalize_action_dim,
                )

                if per_chunk:
                    # Record the (prev, final) pair and the single timestep used.
                    action_chain.append(actions.detach().cpu())
                    action_chain.append(next_actions.detach().cpu())
                    stored_timesteps.append(t.detach().cpu().reshape(1))
                    transition_logprobs.append(lp.detach().cpu())
                else:
                    transition_logprobs.append(lp.detach().cpu())
                    action_chain.append(next_actions.detach().cpu())

                actions = next_actions

        actions[:, ~self.action_mask] *= 0
        if eval_mode:
            old_logprob = torch.zeros(1)
            stored_timesteps_tensor = torch.empty(0, dtype=action_timesteps.dtype)
        elif per_chunk:
            assert len(transition_logprobs) == 1, "per_chunk mode must produce exactly one logprob term"
            old_logprob = transition_logprobs[0]
            stored_timesteps_tensor = torch.cat(stored_timesteps, dim=0)
        else:
            old_logprob = torch.stack(transition_logprobs).sum(dim=0)
            if self.normalize_denoising_horizon and transition_logprobs:
                old_logprob = old_logprob / len(transition_logprobs)
            stored_timesteps_tensor = action_timesteps[:-1].detach().cpu()
        env_action = self.postprocess_action(actions)
        logger.debug(
            "GRPO sample_action done: frame_st_id=%s elapsed_ms=%.1f old_logprob=%.6f "
            "noise_schedule=%s eval_mode=%s action_steps=%s",
            frame_st_id,
            (time.monotonic() - infer_start) * 1000,
            float(old_logprob.mean()),
            self.noise_schedule,
            eval_mode,
            action_num_inference_steps,
        )
        return env_action, RolloutChunk(
            obs=copy.deepcopy(obs),
            frame_st_id=frame_st_id,
            latent_noise=latent_noise.detach().cpu(),
            action_chain=action_chain,
            old_logprobs=old_logprob.detach().cpu(),
            action_timesteps=stored_timesteps_tensor,
            action_mask=mask.detach().cpu(),
            env_action=env_action,
        )

    def _recompute_chunk_logprob(self, chunk: RolloutChunk) -> torch.Tensor:
        self._run_video_prefix(chunk.obs, chunk.frame_st_id, latent_noise=chunk.latent_noise)
        frame_chunk_size = self.job_config.frame_chunk_size
        action_timesteps = chunk.action_timesteps.to(self.device)
        action_chain = [x.to(self.device, dtype=self.dtype) for x in chunk.action_chain]
        mask = chunk.action_mask.to(self.device)

        per_chunk = action_timesteps.numel() == 1
        if per_chunk:
            assert len(action_chain) == 2, (
                f"per_chunk chunk expected action_chain of length 2, got {len(action_chain)}"
            )
            # Rebuild sigma schedule used by scheduler.step; FlowMatchScheduler is stateless beyond that.
            self.action_scheduler.set_timesteps(self.job_config.action_num_inference_steps)
            t = action_timesteps[0]
            actions_prev = action_chain[0]
            next_actions = action_chain[1]
            mean, _ = self._action_transformer_step(
                actions_prev, t, chunk.frame_st_id, last_step=False,
            )
            lp = gaussian_logprob(
                next_actions,
                mean,
                self.action_noise_std,
                mask=mask,
                normalize_action_dim=self.normalize_action_dim,
            )
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
            input_dict = self._prepare_latent_input(
                None,
                actions,
                t,
                t,
                None,
                torch.zeros(
                    [1, self.job_config.action_dim, 1, self.action_per_frame, 1],
                    device=self.device,
                    dtype=self.dtype,
                ) if chunk.frame_st_id == 0 else None,
                frame_st_id=chunk.frame_st_id,
            )
            action_noise_pred = self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=0,
                cache_name=self.cache_name,
                action_mode=True,
            )
            action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
            action_noise_pred = action_noise_pred[:1]
            mean = scheduler_transition_mean(self.action_scheduler, action_noise_pred, t, actions)
            lp = gaussian_logprob(
                next_actions,
                mean,
                self.action_noise_std,
                mask=mask,
                normalize_action_dim=self.normalize_action_dim,
            )
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
            logger.info(
                "GRPO degenerate groups filtered: skipped=%s kept=%s total=%s",
                skipped_groups,
                len(advantage_chunks),
                len(groups),
            )
        if not episodes:
            logger.info(
                "GRPO update skipped: all_groups_degenerate skipped_groups=%s",
                skipped_groups,
            )
            return

        advantages = torch.cat(advantage_chunks, dim=0).to(self.device)
        old_logprobs = self._episode_old_logprobs(episodes).to(self.device)
        update_step = self.global_update_step + 1
        group_ids = [grp[0].group_id for grp in groups if grp]
        advantage_cpu = advantages.detach().cpu()
        n_eps = len(episodes)
        n_rollout_eps = len(rollout_episodes)
        n_kept_groups = len(advantage_chunks)
        minibatch_size = max(1, min(self.minibatch_size, n_eps))
        # Behavior stats use ALL rolled-out episodes (pre-filter). Training
        # stats (advantage, n_eps, group_count) use post-filter only.
        logger.info(
            "GRPO update start: step=%s rollout_groups=%s kept_groups=%s rollout_episodes=%s "
            "trained_episodes=%s minibatch_size=%s degenerate_groups_skipped=%s "
            "reward_mean=%.4f reward_std=%.4f success_rate=%.4f "
            "old_logprob_mean=%.4f old_logprob_std=%.4f advantage_mean=%.4f advantage_std=%.4f "
            "step_count_mean=%.1f chunk_count_mean=%.1f epochs=%s",
            update_step,
            len(groups),
            n_kept_groups,
            n_rollout_eps,
            n_eps,
            minibatch_size,
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
                    mb_new_logprobs_detached = torch.cat(
                        [self.get_action_logprobs(ep).reshape(1) for ep in mb_episodes],
                        dim=0,
                    )
                # Stats are computed entirely in detached space — used only
                # for logging. Loss is informational; we do NOT backward it.
                loss_stats = grpo_clipped_loss(
                    mb_new_logprobs_detached,
                    mb_old_logprobs,
                    mb_advantages,
                    clip_range=self.clip_range,
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
                ratio_for_grad = torch.exp(
                    mb_new_logprobs_detached.float() - mb_old_logprobs.float()
                )
                adv_f32 = mb_advantages.float()
                clipped_high = (adv_f32 > 0) & (ratio_for_grad > 1.0 + self.clip_range)
                clipped_low = (adv_f32 < 0) & (ratio_for_grad < 1.0 - self.clip_range)
                clip_mask_episode = clipped_high | clipped_low
                grad_coef = torch.where(
                    clip_mask_episode,
                    torch.zeros_like(ratio_for_grad),
                    -(ratio_for_grad * adv_f32) / float(len(mb_episodes)),
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
                            "GRPO two-forward drift exceeds atol: max_abs_diff=%.3e "
                            "atol=%.3e mb_size=%s",
                            drift_max,
                            self.logprob_consistency_atol,
                            len(mb_episodes),
                        )
                grad_clip = self.rl_cfg.get("max_grad_norm", None)
                if grad_clip is not None:
                    # clip_grad_norm_ returns the *pre-clip* total norm — capture it
                    # so we can see when a single outlier episode produced gradients
                    # that would otherwise blow up the policy without the clamp.
                    pre_clip_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable_params, float(grad_clip)
                    )
                    epoch_grad_norms.append(float(pre_clip_norm))
                self.optimizer.step()
                if self.scheduler_lr is not None:
                    self.scheduler_lr.step()

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
            logger.info(
                "GRPO update epoch: step=%s epoch=%s/%s mbs=%s loss=%.6f approx_kl=%.6f ratio=%.6f "
                "ratio_min=%.4f ratio_max=%.4f ratio_std=%.4f "
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
                n_mbs,
                stats["loss"],
                stats["approx_kl"],
                stats["ratio"],
                stats["ratio_min"],
                stats["ratio_max"],
                stats["ratio_std"],
                stats["mbs0_ratio_mean"],
                stats["mbs0_ratio_min"],
                stats["mbs0_ratio_max"],
                stats["logratio_min"],
                stats["logratio_max"],
                stats["logratio_std"],
                stats["logratio_abs_max"],
                stats["logratio_per_chunk_abs_max"],
                stats["logratio_outlier_count"],
                stats["chunks_logratio_corr"],
                stats["advantage_min"],
                stats["advantage_max"],
                stats["advantage_abs_max"],
                stats["chunks_min"],
                stats["chunks_max"],
                stats["chunks_mean"],
                stats["grad_norm_mean"],
                stats["grad_norm_max"],
                stats["grad_norm_min"],
                stats["clipfrac"],
                stats["old_logprob_mean"],
                stats["new_logprob_mean"],
                stats["lr"],
            )
            for rank_idx, off in enumerate(worst_offenders):
                logger.info(
                    "GRPO update offender rank=%s log_ratio=%.4f advantage=%.4f chunks=%s "
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
            self._wandb_log(
                {
                    "train_epoch/global_update_step": update_step,
                    "train_epoch/epoch": epoch_idx + 1,
                    "train_epoch/loss": stats["loss"],
                    "train_epoch/approx_kl": stats["approx_kl"],
                    "train_epoch/ratio": stats["ratio"],
                    "train_epoch/ratio_min": stats["ratio_min"],
                    "train_epoch/ratio_max": stats["ratio_max"],
                    "train_epoch/ratio_std": stats["ratio_std"],
                    "train_epoch/mbs0_ratio_mean": stats["mbs0_ratio_mean"],
                    "train_epoch/mbs0_ratio_min": stats["mbs0_ratio_min"],
                    "train_epoch/mbs0_ratio_max": stats["mbs0_ratio_max"],
                    "train_epoch/clipfrac": stats["clipfrac"],
                    "train_epoch/old_logprob_mean": stats["old_logprob_mean"],
                    "train_epoch/new_logprob_mean": stats["new_logprob_mean"],
                    "train_epoch/lr": stats["lr"],
                    "train_epoch/minibatches": n_mbs,
                    "train_epoch/logratio_min": stats["logratio_min"],
                    "train_epoch/logratio_max": stats["logratio_max"],
                    "train_epoch/logratio_std": stats["logratio_std"],
                    "train_epoch/logratio_abs_max": stats["logratio_abs_max"],
                    "train_epoch/logratio_per_chunk_abs_max": stats["logratio_per_chunk_abs_max"],
                    "train_epoch/logratio_outlier_count": stats["logratio_outlier_count"],
                    "train_epoch/chunks_logratio_corr": stats["chunks_logratio_corr"],
                    "train_epoch/advantage_min": stats["advantage_min"],
                    "train_epoch/advantage_max": stats["advantage_max"],
                    "train_epoch/advantage_abs_max": stats["advantage_abs_max"],
                    "train_epoch/chunks_min": stats["chunks_min"],
                    "train_epoch/chunks_max": stats["chunks_max"],
                    "train_epoch/chunks_mean": stats["chunks_mean"],
                    "train_epoch/grad_norm_mean": stats["grad_norm_mean"],
                    "train_epoch/grad_norm_max": stats["grad_norm_max"],
                    "train_epoch/grad_norm_min": stats["grad_norm_min"],
                },
            )
            if self.target_kl is not None and stats["approx_kl"] > self.target_kl:
                logger.warning(
                    "Stopping GRPO epochs early: approx_kl %.6f > target %.6f",
                    stats["approx_kl"],
                    self.target_kl,
                )
                stop_outer = True

        self.global_update_step += 1
        stats["global_update_step"] = self.global_update_step
        self.last_stats = stats
        self._write_json("latest_stats.json", stats)
        self._wandb_log(
            {f"train/{key}": value for key, value in stats.items()},
        )
        if self.global_update_step % int(self.rl_cfg.get("checkpoint_interval", 1)) == 0:
            self.save_checkpoint()
        if self.eval_every > 0 and self.global_update_step % self.eval_every == 0:
            # Arm the eval phase. Clients pick this up from the next
            # run_pending_updates / get_eval_phase response and drive one
            # deterministic pass through their assignment.
            self._eval_pending = True
            self._eval_results = []
            logger.info(
                "GRPO eval phase armed: global_update_step=%s eval_every=%s eval_action_steps=%s",
                self.global_update_step,
                self.eval_every,
                self.eval_action_num_inference_steps,
            )
        logger.info("GRPO update complete: %s", stats)

    def _run_pending_updates(self) -> dict[str, Any]:
        if self.disable_updates:
            logger.info("GRPO update skipped: updates_disabled pending_ready_groups=%s", len(self._pending_ready_groups))
            return {
                "updated": False,
                "reason": "updates_disabled",
                "pending_ready_groups": len(self._pending_ready_groups),
                "global_update_step": self.global_update_step,
                "in_eval": self._eval_pending,
                "status": self.last_stats,
            }
        if len(self._pending_ready_groups) < self.rollout_groups_per_update:
            logger.info(
                "GRPO update skipped: not_enough_ready_groups pending=%s required=%s",
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
        logger.info(
            "GRPO pending update trigger: ready_groups=%s required=%s group_ids=%s",
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
        logger.info(
            "GRPO pending update complete: updated_groups=%s global_update_step=%s last_stats=%s",
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
            chunk_lps = torch.stack([chunk.old_logprobs.reshape(-1)[0] for chunk in ep.chunks])
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
            logger.info("GRPO logprob consistency passed: %s", stats)
        else:
            logger.warning("GRPO logprob consistency failed: %s", stats)
        self._wandb_log(
            {f"debug/logprob_consistency/{key}": value for key, value in stats.items()},
        )
        return stats

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        path = Path(self.job_config.save_root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

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
        """
        ckpt_dir = Path(self.job_config.save_root) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        step = self.global_update_step
        path = ckpt_dir / f"grpo_step_{step:06d}.pt"

        if (
            self._pending_checkpoint_future is not None
            and not self._pending_checkpoint_future.done()
        ):
            logger.warning(
                "Skipping GRPO checkpoint at step %s: previous save still in flight",
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
                "Saved GRPO checkpoint: %s (transformer_keys=%d full=%s)",
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
                "Background GRPO checkpoint write failed at step %s:\n%s",
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
                logger.warning("Could not symlink %s -> %s; write access may be restricted", link_path, target)
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
                        logger.warning("Could not symlink %s -> %s", base_link, base_model / "transformer")
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
            logger.exception("Failed to write inference-compatible transformer export")
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
        self.global_update_step = int(ckpt.get("global_update_step", 0))
        if self.global_update_step > 0:
            self._eval_pending = False
            self._eval_results = []
        if "rollout_store" in ckpt:
            self.rollout_store.load_state_dict(ckpt["rollout_store"])

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
                self._wandb_log(
                    {
                        "eval/episode_success": float(result["success"]),
                        "eval/episode_reward": result["reward"],
                        "eval/episode_step_count": float(result["step_count"]),
                        "eval/episodes_collected": len(self._eval_results),
                        "eval/global_update_step": self.global_update_step,
                    },
                )
                logger.info(
                    "GRPO eval episode: task=%s seed=%s success=%s reward=%.3f "
                    "step_count=%s collected=%s action_steps=%s",
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
                },
            )
            if ready_group is not None:
                group_rewards = [float(ep.reward) for ep in ready_group]
                group_successes = [1.0 if bool(ep.success) else 0.0 for ep in ready_group]
                group_steps = [float(ep.step_count) for ep in ready_group if ep.step_count is not None]
                group_chunks = [float(len(ep.chunks)) for ep in ready_group]
                pending_ready_groups_after = len(self._pending_ready_groups) + (0 if self.disable_updates else 1)
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
                    },
                )
                if not self.disable_updates:
                    self._pending_ready_groups.append(ready_group)
                logger.info(
                    "GRPO group ready: group_id=%s size=%s success_rate=%.4f reward_mean=%.4f "
                    "reward_std=%.4f step_count_mean=%.1f step_count_min=%.0f step_count_max=%.0f "
                    "chunk_count_mean=%.1f pending_ready_groups=%s/%s update_deferred=True",
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
            logger.info(
                "GRPO episode finished: rollout_episode=%s task=%s seed=%s group_id=%s episode_idx=%s "
                "group_member=%s success=%s reward=%.3f step_count=%s chunks=%s ready_group=%s "
                "pending_ready_groups=%s active_sessions=%s",
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
            n = len(self._eval_results)
            success_rate = 0.0
            reward_mean = 0.0
            if n > 0:
                successes = [1.0 if r["success"] else 0.0 for r in self._eval_results]
                rewards = [r["reward"] for r in self._eval_results]
                success_rate = sum(successes) / n
                reward_mean = sum(rewards) / n
                per_seed = {
                    f"eval/per_seed/{r['task']}/{r['seed']}/success": float(r["success"])
                    for r in self._eval_results
                }
                self._wandb_log(
                    {
                        "eval/success_rate": success_rate,
                        "eval/reward_mean": reward_mean,
                        "eval/n_episodes": float(n),
                        "eval/global_update_step": self.global_update_step,
                        **per_seed,
                    },
                )
                logger.info(
                    "GRPO eval phase done: n=%s success_rate=%.4f reward_mean=%.4f global_update_step=%s",
                    n,
                    success_rate,
                    reward_mean,
                    self.global_update_step,
                )
            else:
                logger.warning(
                    "GRPO end_eval_phase called with no eval results; clearing flag anyway."
                )
            results_snapshot = list(self._eval_results)
            self._eval_pending = False
            self._eval_results = []
            return {
                "in_eval": False,
                "n_episodes": n,
                "success_rate": success_rate,
                "reward_mean": reward_mean,
                "global_update_step": self.global_update_step,
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
    if bool(config.rl.get("use_fsdp", False)) and world_size <= 1:
        raise RuntimeError("GRPO FSDP config requires torchrun with WORLD_SIZE > 1")
    if not bool(config.rl.get("use_fsdp", False)) and world_size > 1:
        raise RuntimeError("Distributed GRPO launch requires rl.use_fsdp=true")
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    model = GRPOTrainingServer(config)
    if args.resume_from:
        model.load_checkpoint(args.resume_from)
    run_async_server_mode(model, local_rank, config.host, config.port)


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
