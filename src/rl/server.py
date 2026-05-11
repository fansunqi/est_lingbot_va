"""Online action-only GRPO server for LingBot-VA on RoboTwin.

This entrypoint deliberately lives beside, rather than inside,
``src.inference.server``. It reuses the inference server's model loading,
offloaded VAE/text encoder, observation encoding, and KV-cache machinery, but
owns optimizer state, rollout storage, and GRPO updates.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
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
        cfg.enable_offload = True
        cfg.vae_offload = True
    return cfg


class GRPOTrainingServer(VA_Server):
    def __init__(self, job_config):
        super().__init__(job_config)
        self.rl_cfg = job_config.rl
        self.disable_updates = bool(self.rl_cfg.get("disable_updates", False))
        if self.disable_updates:
            self.transformer.eval()
            self.transformer.requires_grad_(False)
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
        self.last_stats: dict[str, Any] = {}

        self.optimizer = None
        if not self.disable_updates:
            opt_cfg = self.rl_cfg.get("optimizer", {}) or {}
            lr = float(opt_cfg.get("lr", 1e-6))
            weight_decay = float(opt_cfg.get("weight_decay", 0.0))
            betas = tuple(opt_cfg.get("betas", (0.9, 0.95)))
            params = [p for p in self.transformer.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        self.scheduler_lr = None

        self.action_noise_std = float(self.rl_cfg.get("action_noise_std", 0.05))
        self.clip_range = float(self.rl_cfg.get("clip_range", 0.2))
        self.entropy_coef = float(self.rl_cfg.get("entropy_coef", 0.0))
        self.target_kl = self.rl_cfg.get("target_kl", None)
        self.target_kl = None if self.target_kl is None else float(self.target_kl)
        self.update_epochs = int(self.rl_cfg.get("update_epochs", 1))
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

        Path(self.job_config.save_root).mkdir(parents=True, exist_ok=True)
        logger.info(
            "GRPO server ready: group_size=%s update_epochs=%s action_noise_std=%s disable_updates=%s",
            self.rollout_store.group_size,
            self.update_epochs,
            self.action_noise_std,
            self.disable_updates,
        )

    def _action_logprob_mask(self, frame_st_id: int, reference: torch.Tensor) -> torch.Tensor:
        mask = self.action_mask.to(reference.device).view(1, -1, 1, 1, 1)
        mask = mask.expand(reference.shape[0], -1, reference.shape[2], reference.shape[3], reference.shape[4]).clone()
        if frame_st_id == 0:
            mask[:, :, 0:1] = False
        return mask

    def _sample_action_transition(self, mean: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        noise = torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        return mean + noise * self.action_noise_std

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
    def _sample_action_chunk(self, obs, frame_st_id: int, generator: torch.Generator | None = None):
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
        self.action_scheduler.set_timesteps(self.job_config.action_num_inference_steps)
        action_timesteps = F.pad(self.action_scheduler.timesteps, (0, 1), mode="constant", value=0)
        action_chain = [actions.detach().cpu()]
        transition_logprobs = []
        mask = self._action_logprob_mask(frame_st_id, actions)

        with torch.no_grad():
            for i, t in enumerate(tqdm(action_timesteps, disable=not _show_denoise_progress())):
                last_step = i == len(action_timesteps) - 1
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
                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred, "b (f n) c -> b c f n 1", f=frame_chunk_size)
                    action_noise_pred = action_noise_pred[:1]
                    mean = scheduler_transition_mean(self.action_scheduler, action_noise_pred, t, actions)
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
                    transition_logprobs.append(lp.detach().cpu())
                    actions = next_actions
                    action_chain.append(actions.detach().cpu())
                elif action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~self.action_mask] *= 0
        old_logprob = torch.stack(transition_logprobs).sum(dim=0)
        if self.normalize_denoising_horizon and transition_logprobs:
            old_logprob = old_logprob / len(transition_logprobs)
        env_action = self.postprocess_action(actions)
        logger.info(
            "GRPO sample_action done: frame_st_id=%s elapsed_ms=%.1f old_logprob=%.6f",
            frame_st_id,
            (time.monotonic() - infer_start) * 1000,
            float(old_logprob.mean()),
        )
        return env_action, RolloutChunk(
            obs=copy.deepcopy(obs),
            frame_st_id=frame_st_id,
            latent_noise=latent_noise.detach().cpu(),
            action_chain=action_chain,
            old_logprobs=old_logprob.detach().cpu(),
            action_timesteps=action_timesteps[:-1].detach().cpu(),
            action_mask=mask.detach().cpu(),
            env_action=env_action,
        )

    def _recompute_chunk_logprob(self, chunk: RolloutChunk) -> torch.Tensor:
        self._run_video_prefix(chunk.obs, chunk.frame_st_id, latent_noise=chunk.latent_noise)
        frame_chunk_size = self.job_config.frame_chunk_size
        action_timesteps = chunk.action_timesteps.to(self.device)
        action_chain = [x.to(self.device, dtype=self.dtype) for x in chunk.action_chain]
        mask = chunk.action_mask.to(self.device)
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

        logprob = torch.stack(transition_logprobs).sum(dim=0)
        if self.normalize_denoising_horizon and transition_logprobs:
            logprob = logprob / len(transition_logprobs)
        return logprob

    def get_action_logprobs(self, episode) -> torch.Tensor:
        self._reset(prompt=episode.prompt)
        logprobs = []
        for chunk in episode.chunks:
            logprobs.append(self._recompute_chunk_logprob(chunk))
            if chunk.keyframes is not None and chunk.state is not None:
                with torch.no_grad():
                    self._compute_kv_cache({"obs": chunk.keyframes, "state": chunk.state})
        return torch.stack(logprobs).sum(dim=0)

    def _run_grpo_update(self, episodes) -> None:
        if self.optimizer is None:
            raise RuntimeError("Cannot run GRPO update when disable_updates=True")
        rewards = [float(ep.reward) for ep in episodes]
        advantages = compute_group_advantages(rewards).to(self.device)
        old_logprobs = self._episode_old_logprobs(episodes).to(self.device)

        stats = {}
        for _ in range(self.update_epochs):
            self.optimizer.zero_grad(set_to_none=True)
            new_logprobs = torch.cat([self.get_action_logprobs(ep).reshape(1) for ep in episodes], dim=0)
            loss_stats = grpo_clipped_loss(
                new_logprobs,
                old_logprobs,
                advantages,
                clip_range=self.clip_range,
                entropy_coef=self.entropy_coef,
            )
            loss_stats.loss.backward()
            grad_clip = self.rl_cfg.get("max_grad_norm", None)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), float(grad_clip))
            self.optimizer.step()
            if self.scheduler_lr is not None:
                self.scheduler_lr.step()
            stats = {
                "loss": float(loss_stats.loss.detach().cpu()),
                "ratio": float(loss_stats.ratio_mean.cpu()),
                "clipfrac": float(loss_stats.clipfrac.cpu()),
                "approx_kl": float(loss_stats.approx_kl.cpu()),
                "old_logprob_mean": float(old_logprobs.mean().detach().cpu()),
                "new_logprob_mean": float(new_logprobs.mean().detach().cpu()),
                "reward_mean": float(torch.tensor(rewards).mean()),
                "reward_std": float(torch.tensor(rewards).std(unbiased=False)),
                "success_rate": float(sum(rewards) / max(len(rewards), 1)),
            }
            if self.target_kl is not None and stats["approx_kl"] > self.target_kl:
                logger.warning("Stopping GRPO epochs early: approx_kl %.6f > target %.6f", stats["approx_kl"], self.target_kl)
                break

        self.global_update_step += 1
        stats["global_update_step"] = self.global_update_step
        self.last_stats = stats
        self._write_json("latest_stats.json", stats)
        if self.global_update_step % int(self.rl_cfg.get("checkpoint_interval", 1)) == 0:
            self.save_checkpoint()
        logger.info("GRPO update complete: %s", stats)

    def _episode_old_logprobs(self, episodes) -> torch.Tensor:
        return torch.stack([
            torch.stack([chunk.old_logprobs.reshape(-1)[0] for chunk in ep.chunks]).sum()
            for ep in episodes
        ])

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
        return stats

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        path = Path(self.job_config.save_root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def save_checkpoint(self) -> str:
        ckpt_dir = Path(self.job_config.save_root) / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"grpo_step_{self.global_update_step:06d}.pt"
        payload = {
            "transformer": self.transformer.state_dict(),
            "global_update_step": self.global_update_step,
            "rollout_store": self.rollout_store.state_dict(),
            "config": dict(self.job_config),
        }
        if self.optimizer is not None:
            payload["optimizer"] = self.optimizer.state_dict()
        torch.save(payload, path)
        latest = ckpt_dir / "latest.pt"
        torch.save(payload, latest)
        if bool(self.rl_cfg.get("export_inference_checkpoint", True)):
            self.save_inference_export()
        return str(path)

    def save_inference_export(self) -> str | None:
        if not self._is_rank0():
            return None
        export_root = Path(self.job_config.save_root) / "inference_exports" / f"step_{self.global_update_step:06d}"
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
            if self._active_session_id is not None and self._active_session_id != session_id:
                self._swap_out(self._active_session_id)
            if session_id in self._session_store:
                del self._session_store[session_id]
            self._active_session_id = session_id
            with torch.no_grad():
                self._reset(prompt=prompt)
                rollout_seed = int(obs.get("rollout_seed", seed + 1000003 * int(obs.get("episode_idx", 0))))
                self._rollout_generators[session_id] = torch.Generator(device=self.device).manual_seed(rollout_seed)
            episode = self.rollout_store.start_episode(
                session_id=session_id,
                prompt=prompt,
                task=task,
                seed=seed,
                group_id=group_id,
                metadata={k: v for k, v in obs.items() if k not in {"command", "prompt", "task", "task_name", "seed", "group_id"}},
            )
            return {"episode_id": episode.episode_id, "group_id": episode.group_id}

        if command == "sample_action":
            with torch.no_grad():
                self._switch_to_session(session_id)
                action, chunk = self._sample_action_chunk(
                    obs,
                    frame_st_id=self.frame_st_id,
                    generator=self._rollout_generators.get(session_id),
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
            _, ready_group = self.rollout_store.finish_episode(
                session_id,
                success=bool(obs.get("success", False)),
                step_count=obs.get("step_count"),
                reward=obs.get("reward"),
                metadata={k: v for k, v in obs.items() if k not in {"command", "success", "step_count", "reward"}},
            )
            if ready_group is not None:
                if self.validate_logprob_consistency:
                    self._validate_group_logprob_consistency(ready_group)
                if not self.disable_updates:
                    self._pending_ready_groups.append(ready_group)
                    if len(self._pending_ready_groups) >= self.rollout_groups_per_update:
                        ready_groups = self._pending_ready_groups
                        self._pending_ready_groups = []
                        for group in ready_groups:
                            self._run_grpo_update(group)
            return {"ready_for_update": ready_group is not None, "status": self.last_stats}

        if command == "save_checkpoint":
            return {"checkpoint": self.save_checkpoint()}

        if command == "get_status":
            return {
                "global_update_step": self.global_update_step,
                "active_sessions": len(self.rollout_store._active_by_session),
                "last_stats": self.last_stats,
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
