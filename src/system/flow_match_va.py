# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Flow-matching training System for the WAN video-action model.

Owns:

- Two ``FlowMatchScheduler`` instances (latent + action) with independent
  ``snr_shift``.
- Per-step noise injection on packed latents and packed actions, including
  the noisy-condition CFG dropout for the latent stream.
- The all-reduce of attention sequence length so every rank's
  FlexAttention call uses the same shape (required for FA4 cute-dsl, where
  per-shape compilation diverges across ranks otherwise).
- The packed loss with per-frame validity masking.
- Persistence of the ``PackingDataset`` epoch/bin position via
  ``on_save_checkpoint`` / ``on_train_start``.

Activation checkpointing + FSDP2 sharding are applied in ``configure_model``
so the granularity is driven by ``trainer.fsdp.*`` in the YAML and Lightning
materializes the model on the strategy's device mesh.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

from src.distributed.fsdp import apply_ac, shard_model
from src.model.modules import FlexAttnFunc
from src.model.wan_va_wrapper import WanVATransformerWrapper
from src.system.base import BaseVASystem
from src.utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    dtype_from_str,
    get_mesh_id_packed,
    logger,
    sample_timestep_id,
)


class FlowMatchVASystem(BaseVASystem):

    def __init__(self, config):
        super().__init__(config)
        net_cfg = config['components']['model']['network']

        # Configure FlexAttention backend BEFORE instantiating the model.
        # set_backend probes for FA4 support and falls back to Triton with a
        # warning if unavailable. Holding a stale backend on a FlexAttnFunc
        # instance after it's been reused triggers cute-dsl recompiles, so
        # this must happen pre-load.
        requested = net_cfg.get('flex_attn_backend', 'triton')
        flex_mode = net_cfg.get('flex_attn_compile_mode', 'default')
        active = FlexAttnFunc.set_backend(requested, compile_mode=flex_mode)
        logger.info("FlexAttention backend: requested=%s active=%s compile_mode=%s",
                    requested, active, flex_mode)

        self.transformer_wrapper = WanVATransformerWrapper(
            model_name_or_path=config['model_name_or_path'],
            resume_from=config.get('resume_from'),
            compile_model=net_cfg.get('compile_model', False),
            compile_mode=net_cfg.get('compile_mode', 'default'),
        )
        self._fsdp_applied = False

        self.patch_size = tuple(net_cfg['patch_size'])
        self.attn_seq_align = int(net_cfg.get('attn_seq_align', 128))

        self.train_scheduler_latent = FlowMatchScheduler(
            shift=net_cfg['snr_shift'], sigma_min=0.0, extra_one_step=True,
        )
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(
            shift=net_cfg['action_snr_shift'], sigma_min=0.0, extra_one_step=True,
        )
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.loss_weights = config.get('loss', {'latent_loss': 1.0, 'action_loss': 1.0})
        self.noisy_cond_prob = float(config.get('noisy_cond_prob', 0.5))

        self._pending_packing_state = None

    # ------------------------------------------------------------------ #
    # configure_model: AC + FSDP2 sharding hook
    # ------------------------------------------------------------------ #
    # ``ModelParallelStrategy`` requires this hook to be overridden — it sets
    # ``self._device_mesh`` in setup_environment, then Lightning's
    # ``_call_configure_model`` invokes us. We apply activation checkpointing
    # and ``fully_shard`` here so weights are materialized as DTensors on the
    # strategy's data-parallel mesh. Lightning's strategy.setup() then handles
    # any remaining materialization and moves whatever's still on CPU to the
    # device.

    def configure_model(self):
        if self._fsdp_applied:
            return  # idempotent — Lightning may call us multiple times across stages
        fsdp_cfg = self.config.get('trainer', {}).get('fsdp', {}) or {}

        ac_granularity = fsdp_cfg.get('ac_granularity', 'every_2')
        block_only = bool(fsdp_cfg.get('block_only', True))
        reshard_after_forward = bool(fsdp_cfg.get('reshard_after_forward', True))
        param_dtype = dtype_from_str(fsdp_cfg.get('param_dtype', 'bfloat16'))
        reduce_dtype = dtype_from_str(fsdp_cfg.get('reduce_dtype', 'float32'))

        mesh = getattr(self, '_device_mesh', None)
        dp_mesh = mesh['data_parallel'] if mesh is not None else None

        net = self.transformer_wrapper.net
        apply_ac(net, granularity=ac_granularity)
        shard_model(
            net,
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            mesh=dp_mesh,
            reshard_after_forward=reshard_after_forward,
            block_only=block_only,
        )
        net.train()
        net.requires_grad_(True)
        # condition_embedder_action is deepcopy'd from condition_embedder, but
        # WanTimeTextImageEmbedding.forward only invokes time_embedder/time_proj.
        # The text_embedder submodule on the action branch is registered yet
        # never sees gradients, so AdamW's lazy state init skips it and the
        # saved optimizer state ends up missing those keys -- which then breaks
        # DCP resume with "Missing key in checkpoint state_dict". Freeze the
        # dead branch so the optimizer never tracks it in the first place.
        for p in net.condition_embedder_action.text_embedder.parameters():
            p.requires_grad_(False)
        self._fsdp_applied = True

        if self.global_rank == 0:
            logger.info(
                "FSDP2 configured: ac=%s reshard_after_forward=%s block_only=%s "
                "param_dtype=%s reduce_dtype=%s dp_mesh_size=%s",
                ac_granularity, reshard_after_forward, block_only,
                param_dtype, reduce_dtype,
                dp_mesh.size() if dp_mesh is not None else 'none',
            )

    # ------------------------------------------------------------------ #
    # Training step
    # ------------------------------------------------------------------ #

    def training_step(self, batch, batch_idx):
        batch = {k: v.to(self.device) for k, v in batch.items()}
        input_dict = self._prepare_input_dict(batch)
        latent_pred, action_pred = self.transformer_wrapper(input_dict, train_mode=True)
        latent_loss, action_loss = self._compute_loss(input_dict, (latent_pred, action_pred))
        loss = (
            self.loss_weights['latent_loss'] * latent_loss
            + self.loss_weights['action_loss'] * action_loss
        )

        # Lightning sync_dist handles cross-rank averaging for logged scalars,
        # so we don't need the manual all_reduce(AVG) the legacy loop did.
        self.log_dict(
            {
                'train/global_avg_video_loss': latent_loss.detach(),
                'train/global_avg_action_loss': action_loss.detach(),
                'train/loss': loss.detach(),
                'train/lr': self.lr_schedulers().get_last_lr()[0],
            },
            prog_bar=True, on_step=True, on_epoch=False, sync_dist=True,
        )
        self.log_dict(
            {
                'train/global_max_video_loss': latent_loss.detach(),
                'train/global_max_action_loss': action_loss.detach(),
            },
            prog_bar=False, on_step=True, on_epoch=False,
            sync_dist=True, reduce_fx='max',
        )
        return loss

    # ------------------------------------------------------------------ #
    # Input prep — ported verbatim from wan_va/train.py
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, episode_boundaries,
                   action_mask=None, action_mode=False, noisy_cond_prob=0.0,
                   frame_offsets=None):
        B, _, F_dim, H, W = latent.shape
        ep_lens = [int(x) for x in episode_boundaries.tolist()]
        assert sum(ep_lens) == F_dim

        timestep_ids = sample_timestep_id(
            batch_size=F_dim,
            num_train_timesteps=train_scheduler.num_train_timesteps,
        )
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noise = torch.zeros_like(latent).normal_()
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        # Episode-global RoPE grid_id.
        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        fo = [o // patch_f for o in frame_offsets] if frame_offsets else None
        latent_grid_id = get_mesh_id_packed(
            [L // patch_f for L in ep_lens], H // patch_h, W // patch_w,
            t=1 if action_mode else 0, action=action_mode,
            frame_offsets=fo,
        ).to(self.device)[None].repeat(B, 1, 1)

        cond_timesteps = torch.zeros_like(timesteps)
        if noisy_cond_prob > 0.0:
            drop = torch.rand(len(ep_lens)) < noisy_cond_prob
            if drop.any():
                ep_lens_t = torch.as_tensor(ep_lens, device=self.device)
                drop_per_frame = torch.repeat_interleave(
                    drop.to(self.device), ep_lens_t,
                )
                cond_ids = sample_timestep_id(
                    batch_size=F_dim,
                    min_timestep_bd=0.5, max_timestep_bd=1.0,
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
                cond_ts = train_scheduler.timesteps[cond_ids].to(device=self.device)
                noised = train_scheduler.add_noise(
                    latent, torch.zeros_like(latent).normal_(), cond_ts, t_dim=2,
                )
                latent = torch.where(drop_per_frame.view(1, 1, F_dim, 1, 1), noised, latent)
                cond_timesteps = torch.where(drop_per_frame, cond_ts, cond_timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _compute_attn_target_length(self, batch_dict):
        """Pick a self-attention sequence length that all ranks agree on.

        ``forward_train`` concatenates ``[latent | cond_latent | action |
        cond_action]`` before flex_attention. Each rank's local sum is rounded
        up to a multiple of ``attn_seq_align``, then all-reduced (MAX) so
        every rank pads to the same target. Bounds distinct cute-dsl kernel
        shapes to ``~max_tokens / align`` and keeps compile cost in lockstep
        — without this, FA4's per-shape compile on a slow rank trips Gloo's
        monitoredBarrier and the run dies.
        """
        align = int(self.attn_seq_align)
        if align <= 0:
            align = 128

        p_h, p_w = self.patch_size[1], self.patch_size[2]
        F_lat, H_lat, W_lat = batch_dict['latents'].shape[-3:]
        F_act, H_act, W_act = batch_dict['actions'].shape[-3:]
        lat_seq = int(F_lat) * (int(H_lat) // p_h) * (int(W_lat) // p_w)
        act_seq = int(F_act) * int(H_act) * int(W_act)
        local_total = 2 * lat_seq + 2 * act_seq
        local_aligned = ((local_total + align - 1) // align) * align

        if dist.is_initialized():
            t = torch.tensor(local_aligned, device=self.device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return int(t.item())
        return local_aligned

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        ep_bounds = batch_dict['episode_boundaries']
        ep_lens = [int(x) for x in ep_bounds.tolist()]

        # Per-episode RoPE offset from frame_ids_episode.
        fids = batch_dict['frame_ids_episode'].squeeze(0)
        cum = 0
        frame_offsets = []
        for L in ep_lens:
            frame_offsets.append(int(fids[cum].item()))
            cum += L

        latent_dict = self._add_noise(
            latent=batch_dict['latents'],
            train_scheduler=self.train_scheduler_latent,
            episode_boundaries=ep_bounds,
            noisy_cond_prob=self.noisy_cond_prob,
            frame_offsets=frame_offsets,
        )
        action_dict = self._add_noise(
            latent=batch_dict['actions'],
            train_scheduler=self.train_scheduler_action,
            episode_boundaries=ep_bounds,
            action_mask=batch_dict['actions_mask'],
            action_mode=True,
            frame_offsets=frame_offsets,
        )

        latent_dict['text_emb'] = batch_dict['text_emb']
        latent_dict['latents_mask'] = batch_dict['latents_mask']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        return {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
            'seq_ids_per_frame': batch_dict['seq_ids'],
            'frame_ids_per_frame': batch_dict['frame_ids_episode'],
            'n_episodes': int(ep_bounds.shape[0]),
            'attn_target_length': self._compute_attn_target_length(batch_dict),
        }

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #

    def _compute_loss(self, input_dict, pred):
        latent_pred, action_pred = pred
        action_pred = rearrange(
            action_pred, 'b (f n) c -> b c f n 1',
            f=input_dict['action_dict']['targets'].shape[-3],
        )
        latent_pred = data_seq_to_patch(
            self.patch_size, latent_pred,
            input_dict['latent_dict']['targets'].shape[-3],
            input_dict['latent_dict']['targets'].shape[-2],
            input_dict['latent_dict']['targets'].shape[-1],
            batch_size=latent_pred.shape[0],
        )
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(
            input_dict['latent_dict']['timesteps'].flatten()
        ).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(
            input_dict['action_dict']['timesteps'].flatten()
        ).reshape(Bn, Fn)

        frame_valid = input_dict['latent_dict']['latents_mask'].flatten().float()

        latent_loss = F.mse_loss(
            latent_pred.float(),
            input_dict['latent_dict']['targets'].float().detach(),
            reduction='none',
        )
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        latent_loss_per_frame = latent_loss.sum(dim=1) / latent_loss.shape[1]
        latent_loss = (latent_loss_per_frame * frame_valid).sum() / (frame_valid.sum() + 1e-6)

        action_loss = F.mse_loss(
            action_pred.float(),
            input_dict['action_dict']['targets'].float().detach(),
            reduction='none',
        )
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        action_mask = input_dict['action_dict']['actions_mask'].float() \
            .permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss_per_frame = action_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-6)
        action_loss = (action_loss_per_frame * frame_valid).sum() / (frame_valid.sum() + 1e-6)

        return latent_loss, action_loss

    # ------------------------------------------------------------------ #
    # Checkpoint hooks for packing dataset state
    # ------------------------------------------------------------------ #

    def on_save_checkpoint(self, checkpoint):
        dm = self.trainer.datamodule
        checkpoint['packing_state'] = {
            'epoch': int(getattr(dm, '_packing_epoch', 0)),
            'bin_in_epoch': int(getattr(dm, '_bin_in_epoch', 0)),
        }

    def on_load_checkpoint(self, checkpoint):
        if 'packing_state' in checkpoint:
            self._pending_packing_state = checkpoint['packing_state']

    def on_train_start(self):
        if self._pending_packing_state is not None:
            ps = self._pending_packing_state
            self.trainer.datamodule.fast_forward(
                epoch=ps['epoch'], bin_in_epoch=ps['bin_in_epoch'],
            )
            self._pending_packing_state = None
