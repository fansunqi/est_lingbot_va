# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import time
from functools import partial
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

import yaml
import yamlinclude
from easydict import EasyDict

from src.distributed.fsdp import apply_ac, shard_model
from src.distributed.util import _configure_model, init_distributed
from src.model.loaders import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from src.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    dtype_from_str,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


def _show_denoise_progress():
    return os.environ.get("WAN_VA_SHOW_DENOISE_PROGRESS", "").lower() in (
        "1", "true", "yes", "on"
    )


def _dist_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _dist_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


# --------------------------------------------------------------------------- #
# YAML config loading (replaces the old EasyDict-based VA_CONFIGS registry)
# --------------------------------------------------------------------------- #

def load_inference_config(config_path):
    """Load a YAML inference config and return an EasyDict.

    Resolves ``!inc`` relative to ``configs/`` and flattens the optional
    ``_base`` key (i2va overlays) so consumer code keeps doing attribute
    access on a single namespace.
    """
    from pathlib import Path

    config_path = Path(config_path)
    config_root = config_path.parent.parent  # configs/inference/foo.yaml -> configs/
    yaml.add_constructor(
        '!inc',
        yamlinclude.YamlIncludeConstructor(base_dir=str(config_root)),
        Loader=yaml.FullLoader,
    )
    with open(config_path, 'r') as f:
        raw = yaml.full_load(f)

    # Merge ``_base`` (if present) under top-level overrides — top-level wins.
    base = raw.pop('_base', None) or {}
    merged = {**base, **raw}

    cfg = EasyDict(merged)
    cfg.param_dtype = dtype_from_str(cfg.get('param_dtype', 'bfloat16'))
    if 'patch_size' in cfg:
        cfg.patch_size = tuple(cfg.patch_size)

    # Server code reads ``wan22_pretrained_model_name_or_path`` (matches the
    # training-side name passed through ``_inject_data_dependencies`` in
    # ``src.run``), but inference YAMLs only declare ``model_name_or_path``.
    # Alias them so either key works without touching every callsite.
    if 'model_name_or_path' in cfg and 'wan22_pretrained_model_name_or_path' not in cfg:
        cfg.wan22_pretrained_model_name_or_path = cfg.model_name_or_path

    # ``inverse_used_action_channel_ids`` is a Python-computed field that
    # used to live in the EasyDict configs (va_*_cfg.py:36-41). Recompute it
    # at load time so YAML stays declarative.
    if 'used_action_channel_ids' in cfg and 'action_dim' in cfg:
        used = list(cfg.used_action_channel_ids)
        action_dim = int(cfg.action_dim)
        inv = [len(used)] * action_dim
        for i, j in enumerate(used):
            inv[j] = i
        cfg.inverse_used_action_channel_ids = inv

    return cfg


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        # Optional per-step dumps of latents/actions/obs into <save_root>/real/.
        # Off by default — during GRPO training this would generate three .pt
        # files per inference call and quickly fill the disk. Nothing in the
        # codebase reads them back; set to True only for manual debugging.
        self.save_debug_dumps = bool(getattr(job_config, 'save_debug_dumps', False))
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.rank = int(getattr(job_config, 'rank', _dist_rank()))
        self.world_size = int(getattr(job_config, 'world_size', _dist_world_size()))
        self.fsdp_enabled = dist.is_initialized() and self.world_size > 1
        fsdp_cfg = getattr(job_config, 'fsdp', {}) or {}
        self.fsdp_rank0_aux_only = bool(fsdp_cfg.get('rank0_aux_only', self.fsdp_enabled))
        self._has_aux_models = (not self.fsdp_rank0_aux_only) or self.rank == 0
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram
        # When vae_offload=False, the VAE stays on GPU so we skip the
        # transformer<->vae swap in every _encode_obs call. This saves a few
        # hundred ms per chunk at the cost of ~2.7 GB extra VRAM. Defaults to
        # the same value as enable_offload to preserve previous behavior.
        self.vae_offload = getattr(job_config, 'vae_offload', self.enable_offload)

        # ---- Multi-session state ----
        # CPU-side storage for inactive sessions' KV caches + metadata
        self._session_store = {}   # session_id -> {kv_cache: [...], metadata: {...}}
        self._active_session_id = None  # which session currently occupies GPU cache
        self._swap_stream = None  # lazily initialized CUDA stream for async copies

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        if self.fsdp_enabled and self.enable_offload:
            logger.warning(
                "FSDP server is running with enable_offload=True. The transformer "
                "will stay sharded on GPU; only auxiliary VAE/text_encoder moves."
            )

        if self._has_aux_models:
            self.vae = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.vae_offload else self.device,
            )
            self.streaming_vae = WanVAEStreamingWrapper(self.vae)
        else:
            self.vae = None
            self.streaming_vae = None

        if self._has_aux_models:
            self.tokenizer = load_tokenizer(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'tokenizer'), )
        else:
            self.tokenizer = None

        if self._has_aux_models:
            self.text_encoder = load_text_encoder(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'text_encoder'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
        else:
            self.text_encoder = None

        transformer_load_device = self.device
        if self.fsdp_enabled and bool(fsdp_cfg.get('load_transformer_on_cpu', True)):
            transformer_load_device = 'cpu'
        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=transformer_load_device,
            attn_mode="torch"
        )
        if self.fsdp_enabled:
            ac_granularity = fsdp_cfg.get('ac_granularity', 'none')
            self.transformer = apply_ac(self.transformer, granularity=ac_granularity)
        reduce_dtype = dtype_from_str(fsdp_cfg.get('reduce_dtype', 'float32'))
        shard_fn = partial(
            shard_model,
            param_dtype=self.dtype,
            reduce_dtype=reduce_dtype,
            reshard_after_forward=bool(fsdp_cfg.get('reshard_after_forward', True)),
            block_only=bool(fsdp_cfg.get('block_only', False)),
        )
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape' and self._has_aux_models:
            # Share the same AutoencoderKLWan module between the two streaming
            # wrappers. Each wrapper still owns its own feat_cache (see
            # WanVAEStreamingWrapper.__init__), so encoding high-res and
            # left+right-res branches remains independent. This avoids
            # loading a duplicate 2.7 GB copy of the VAE weights.
            self.streaming_vae_half = WanVAEStreamingWrapper(self.vae)

        if self.fsdp_enabled:
            logger.info(
                "FSDP server initialized: rank=%s world_size=%s "
                "rank0_aux_only=%s transformer_load_device=%s",
                self.rank,
                self.world_size,
                self.fsdp_rank0_aux_only,
                transformer_load_device,
            )

    def _dist_active(self):
        return dist.is_available() and dist.is_initialized() and self.world_size > 1

    def _is_rank0(self):
        return self.rank == 0

    def _broadcast_optional_tensor_from_rank0(self, tensor):
        if not self._dist_active():
            return tensor

        has_tensor = torch.tensor(
            [1 if tensor is not None else 0],
            dtype=torch.int64,
            device=self.device,
        )
        dist.broadcast(has_tensor, src=0)
        if has_tensor.item() == 0:
            return None

        meta = [None]
        if self._is_rank0():
            tensor = tensor.to(self.device)
            meta[0] = (tuple(tensor.shape), str(tensor.dtype).replace('torch.', ''))
        dist.broadcast_object_list(meta, src=0)
        shape, dtype_name = meta[0]
        if not self._is_rank0():
            tensor = torch.empty(shape,
                                 dtype=getattr(torch, dtype_name),
                                 device=self.device)
        dist.broadcast(tensor, src=0)
        return tensor

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        if self.fsdp_rank0_aux_only and not self._is_rank0():
            return self._broadcast_optional_tensor_from_rank0(None)

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        if self.enable_offload and not self.fsdp_enabled:
            self.transformer.to('cpu')
            torch.cuda.empty_cache()
        text_encoder_on_cpu = (
            self.enable_offload
            and self.fsdp_enabled
            and bool(getattr(self.job_config, "text_encoder_cpu_offload", True))
        )
        keep_text_encoder_loaded = bool(getattr(self, "_keep_text_encoder_loaded", False))
        if self.enable_offload and not text_encoder_on_cpu:
            self.text_encoder.to(self.device)
        elif text_encoder_on_cpu:
            self.text_encoder.to("cpu")
        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        if self.enable_offload and not text_encoder_on_cpu and not keep_text_encoder_loaded:
            self.text_encoder.to('cpu')
            torch.cuda.empty_cache()
        if self.enable_offload and not self.fsdp_enabled:
            self.transformer.to(self.device)
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        prompt_embeds = prompt_embeds.to(device)
        return self._broadcast_optional_tensor_from_rank0(prompt_embeds)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        keep_text_encoder_loaded = (
            self.enable_offload
            and not (
                self.fsdp_enabled
                and bool(getattr(self.job_config, "text_encoder_cpu_offload", True))
            )
            and prompt_embeds is None
            and do_classifier_free_guidance
            and negative_prompt_embeds is None
            and self.text_encoder is not None
        )
        if keep_text_encoder_loaded:
            self._keep_text_encoder_loaded = True
        try:
            if prompt_embeds is None:
                prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

            if do_classifier_free_guidance and negative_prompt_embeds is None:
                negative_prompt = negative_prompt or ""
                negative_prompt = batch_size * [negative_prompt] if isinstance(
                    negative_prompt, str) else negative_prompt

                if prompt is not None and type(prompt) is not type(
                        negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}.")
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`.")

                negative_prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=negative_prompt,
                    num_videos_per_prompt=num_videos_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )
        finally:
            if keep_text_encoder_loaded:
                self._keep_text_encoder_loaded = False
                if self.text_encoder is not None:
                    self.text_encoder.to('cpu')
                torch.cuda.empty_cache()
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        if self._is_rank0():
            logger.debug("FRAME START ID: %s", frame_st_id)
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        if self.fsdp_rank0_aux_only and not self._is_rank0():
            return self._broadcast_optional_tensor_from_rank0(None)

        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return self._broadcast_optional_tensor_from_rank0(None)
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.vae_offload:
            if self.enable_offload and not self.fsdp_enabled:
                self.transformer.to('cpu')
                torch.cuda.empty_cache()
            self.vae.to(self.device)
            if self.streaming_vae_half is not None and self.streaming_vae_half.vae is not self.vae:
                self.streaming_vae_half.vae.to(self.device)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        if self.vae_offload:
            self.vae.to('cpu')
            if self.streaming_vae_half is not None and self.streaming_vae_half.vae is not self.vae:
                self.streaming_vae_half.vae.to('cpu')
            torch.cuda.empty_cache()
            if self.enable_offload and not self.fsdp_enabled:
                self.transformer.to(self.device)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        video_latent = video_latent.to(self.device)
        return self._broadcast_optional_tensor_from_rank0(video_latent)

    def _reset(self, prompt=None):
        logger.debug("Reset inference session state.")
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self._transformer_cache_ready = False
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        if self.streaming_vae is not None:
            self.streaming_vae.clear_cache()
        torch.cuda.empty_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            if self.streaming_vae_half is not None:
                self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=int(getattr(self.job_config, "prompt_max_sequence_length", 512)),
                device=self.device,
                dtype=self.dtype,
            )

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self._cache_latent_token_per_chunk = latent_token_per_chunk
        self._cache_action_token_per_chunk = action_token_per_chunk

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        if self.save_debug_dumps:
            os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _ensure_transformer_cache(self):
        if getattr(self, "_transformer_cache_ready", False):
            return
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            self._cache_latent_token_per_chunk,
                                            self._cache_action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size=2 if self.use_cfg else 1)
        self._transformer_cache_ready = True

    def _infer(self, obs, frame_st_id=0):
        infer_start_time = time.monotonic()
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent
        self._ensure_transformer_cache()

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)
        if self._is_rank0():
            logger.info(
                "Infer loop start: frame_st_id=%s video_steps=%s action_steps=%s",
                frame_st_id,
                len(timesteps),
                len(action_timesteps),
            )

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(
                    timesteps,
                    disable=not _show_denoise_progress())):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(
                    action_timesteps,
                    disable=not _show_denoise_progress())):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        if self._is_rank0() and self.save_debug_dumps:
            save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
            save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        if self._is_rank0():
            logger.info(
                "Infer loop done: frame_st_id=%s elapsed_ms=%.1f",
                frame_st_id,
                (time.monotonic() - infer_start_time) * 1000,
            )
        return actions, latents

    def _compute_kv_cache(self, obs):
        ### optional async save obs for debug
        if self._is_rank0() and self.save_debug_dumps:
            save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.debug(
            "get KV cache obs: %s %s",
            latent_model_input.shape,
            action_model_input.shape,
        )
        self._ensure_transformer_cache()
        self.transformer.clear_pred_cache(self.cache_name)
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]

    # ==================== Multi-Session KV Cache Swap ====================

    def _get_swap_stream(self):
        if self._swap_stream is None:
            self._swap_stream = torch.cuda.Stream(device=self.device)
        return self._swap_stream

    def _swap_out(self, session_id):
        """Move current GPU KV cache + VAE feat_cache + metadata to CPU."""
        if session_id is None:
            return
        stream = self._get_swap_stream()
        # Collect KV cache tensors from all transformer blocks
        cpu_kv = []
        with torch.cuda.stream(stream):
            for block in self.transformer.blocks:
                cache = block.attn1.attn_caches.get(self.cache_name)
                if cache is not None:
                    cpu_cache = {}
                    for key, val in cache.items():
                        if isinstance(val, torch.Tensor):
                            cpu_cache[key] = val.to('cpu', non_blocking=True).pin_memory()
                        else:
                            cpu_cache[key] = val
                    cpu_kv.append(cpu_cache)
                else:
                    cpu_kv.append(None)
        stream.synchronize()

        # Save VAE streaming feat_cache (list of tensors or Nones)
        def _save_feat_cache(feat_cache):
            saved = []
            for item in feat_cache:
                if item is not None and isinstance(item, torch.Tensor):
                    saved.append(item.cpu().pin_memory())
                else:
                    saved.append(item)
            return saved

        vae_feat_cache = (
            _save_feat_cache(self.streaming_vae.feat_cache)
            if self.streaming_vae is not None else None
        )
        vae_half_feat_cache = None
        if self.streaming_vae_half is not None:
            vae_half_feat_cache = _save_feat_cache(self.streaming_vae_half.feat_cache)

        # Save per-session metadata
        self._session_store[session_id] = {
            'kv_cache': cpu_kv,
            'vae_feat_cache': vae_feat_cache,
            'vae_half_feat_cache': vae_half_feat_cache,
            'frame_st_id': self.frame_st_id,
            'init_latent': self.init_latent.cpu().pin_memory() if self.init_latent is not None else None,
            'prompt_embeds': self.prompt_embeds.cpu().pin_memory() if self.prompt_embeds is not None else None,
            'negative_prompt_embeds': self.negative_prompt_embeds.cpu().pin_memory() if self.negative_prompt_embeds is not None else None,
            'use_cfg': self.use_cfg,
            'exp_name': getattr(self, 'exp_name', None),
            'exp_save_root': getattr(self, 'exp_save_root', None),
        }
        logger.debug("Swapped out session '%s' to CPU", session_id)

    def _swap_in(self, session_id):
        """Restore a session's KV cache + VAE feat_cache from CPU to GPU."""
        state = self._session_store.get(session_id)
        if state is None:
            return False

        stream = self._get_swap_stream()
        restored_cache_count = 0
        missing_cache_count = 0
        with torch.cuda.stream(stream):
            for block, cpu_cache in zip(self.transformer.blocks, state['kv_cache']):
                if cpu_cache is not None:
                    gpu_cache = {}
                    for key, val in cpu_cache.items():
                        if isinstance(val, torch.Tensor):
                            gpu_cache[key] = val.to(self.device, non_blocking=True)
                        else:
                            gpu_cache[key] = val
                    block.attn1.attn_caches[self.cache_name] = gpu_cache
                    restored_cache_count += 1
                else:
                    block.attn1.attn_caches[self.cache_name] = None
                    missing_cache_count += 1
        stream.synchronize()

        # Restore VAE streaming feat_cache
        def _restore_feat_cache(saved):
            restored = []
            for item in saved:
                if item is not None and isinstance(item, torch.Tensor):
                    restored.append(item.to(self.device))
                else:
                    restored.append(item)
            return restored

        if self.streaming_vae is not None and state['vae_feat_cache'] is not None:
            self.streaming_vae.feat_cache = _restore_feat_cache(state['vae_feat_cache'])
        if self.streaming_vae_half is not None and state['vae_half_feat_cache'] is not None:
            self.streaming_vae_half.feat_cache = _restore_feat_cache(state['vae_half_feat_cache'])

        # Restore metadata
        self.frame_st_id = state['frame_st_id']
        self.init_latent = state['init_latent'].to(self.device) if state['init_latent'] is not None else None
        self.prompt_embeds = state['prompt_embeds'].to(self.device) if state['prompt_embeds'] is not None else None
        self.negative_prompt_embeds = state['negative_prompt_embeds'].to(self.device) if state['negative_prompt_embeds'] is not None else None
        self.use_cfg = state['use_cfg']
        # A session can be swapped out after reset but before its first inference,
        # in which case every per-block KV cache is still None. Treat that as
        # "not ready" so the next _ensure_transformer_cache() allocates caches
        # instead of letting clear_pred_cache/update_cache index into None.
        self._transformer_cache_ready = restored_cache_count > 0 and missing_cache_count == 0
        if state['exp_name'] is not None:
            self.exp_name = state['exp_name']
        if state['exp_save_root'] is not None:
            self.exp_save_root = state['exp_save_root']

        logger.debug(
            "Swapped in session '%s' to GPU (frame_st_id=%s, cache_ready=%s, restored_blocks=%s, missing_blocks=%s)",
            session_id,
            self.frame_st_id,
            self._transformer_cache_ready,
            restored_cache_count,
            missing_cache_count,
        )
        return True

    def _switch_to_session(self, session_id):
        """Ensure the given session's KV cache is on GPU. Swap if necessary."""
        if session_id == self._active_session_id:
            return  # already active, no swap needed
        # Swap out current
        self._swap_out(self._active_session_id)
        # Swap in target
        if session_id in self._session_store:
            self._swap_in(session_id)
        self._active_session_id = session_id

    def on_session_closed(self, session_id):
        """Free resources when a client disconnects."""
        if session_id in self._session_store:
            del self._session_store[session_id]
            logger.info(f"Freed session '{session_id}' from CPU store")
        if self._active_session_id == session_id:
            self._active_session_id = None
            # Clear active-session GPU caches.
            self.transformer.clear_cache(self.cache_name)
            self._transformer_cache_ready = False
            if self.streaming_vae is not None:
                self.streaming_vae.clear_cache()
            if self.streaming_vae_half is not None:
                self.streaming_vae_half.clear_cache()
            torch.cuda.empty_cache()

    # ===================================================================

    @torch.no_grad()
    def infer(self, obs):
        session_id = obs.pop('_session_id', None)
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            # New episode for this session: swap out current, clear, reset
            if self._active_session_id is not None and self._active_session_id != session_id:
                self._swap_out(self._active_session_id)
            # Discard old cache for this session
            if session_id in self._session_store:
                del self._session_store[session_id]
            self._active_session_id = session_id
            logger.info(f"******************* Reset server (session='{session_id}') ******************")
            self._reset(prompt=prompt)
            return dict()
        elif compute_kv_cache:
            self._switch_to_session(session_id)
            logger.info(
                f"##### Compute KV Cache (session='{session_id}') #####")
            self._compute_kv_cache(obs)
            return dict()
        else:
            self._switch_to_session(session_id)
            logger.info(f"##### Infer One Chunk (session='{session_id}') #####")
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            return dict(action=action)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()

        if not self._is_rank0():
            return
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):
    config = load_inference_config(args.config)
    # ``mode`` is the YAML field name; legacy code reads ``infer_mode``.
    if 'mode' in config and 'infer_mode' not in config:
        config.infer_mode = config.mode

    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info("****************************** USE I2AV mode ******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info("****************************** USE Server mode ******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")


def run_i2va(config_path, save_root=None):
    """Programmatic entry used by sample.py (no argparse)."""
    args = argparse.Namespace(config=config_path, port=None, save_root=save_root)
    run(args)


def main():
    parser = argparse.ArgumentParser(description="LingBot-VA inference server / i2va entry")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML inference config")
    parser.add_argument("--port", type=int, default=None, help="Override server port")
    parser.add_argument("--save-root", "--save_root", dest="save_root", type=str, default=None,
                        help="Override config.save_root")
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
