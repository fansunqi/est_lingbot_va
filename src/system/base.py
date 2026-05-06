# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Base LightningModule for VA training.

Owns the optimizer + LR schedule construction. Concrete training logic
lives in subclasses such as ``FlowMatchVASystem``.
"""
import torch
import lightning.pytorch as pl

from src.utils import warmup_constant_lambda, warmup_linear_lambda


class BaseVASystem(pl.LightningModule):
    """Configures optimizer + LR schedule from ``config['optimizer']``.

    Subclasses must define their own ``training_step`` and instantiate the
    underlying network (typically a ``WanVATransformerWrapper``).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # ``save_hyperparameters`` lets Lightning persist the config dict in
        # the checkpoint so resume gets the same hyperparams without needing
        # the YAML file. Wrap in a dict so PL doesn't try to introspect attrs.
        self.save_hyperparameters({'config': config})

    def configure_optimizers(self):
        opt_cfg = self.config['optimizer']
        target = opt_cfg.get('__target__', 'AdamW')
        params = [p for p in self.parameters() if p.requires_grad]

        if target == 'AdamW':
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg['learning_rate'],
                betas=tuple(opt_cfg.get('betas', (0.9, 0.95))),
                eps=opt_cfg.get('eps', 1e-8),
                weight_decay=opt_cfg.get('weight_decay', 0.0),
                fused=opt_cfg.get('fused', True),
                foreach=False,
            )
        elif target == 'Muon':
            # Muon for ndim>=2 weights, AdamW (built into the same Muon
            # optimizer via use_muon=False) for everything else.
            # NS iteration in zeropower_via_newtonschulz5 requires ndim>=2,
            # so any 1-D param (norm scales, biases) goes to the AdamW path.
            from muon_fsdp2 import Muon

            muon_params = [p for p in params if p.ndim >= 2]
            adam_params = [p for p in params if p.ndim < 2]

            betas = tuple(opt_cfg.get('betas', (0.9, 0.95)))
            wd = float(opt_cfg.get('weight_decay', 0.0))
            adamw_lr = float(opt_cfg['learning_rate'])
            muon_lr = float(opt_cfg.get('muon_lr', adamw_lr * 10.0))

            param_groups = []
            if muon_params:
                param_groups.append(dict(
                    params=muon_params,
                    lr=muon_lr,
                    momentum=float(opt_cfg.get('muon_momentum', 0.95)),
                    weight_decay=wd,
                    rms_scale=bool(opt_cfg.get('muon_rms_scale', True)),
                    nesterov=bool(opt_cfg.get('muon_nesterov', True)),
                    ns_steps=int(opt_cfg.get('muon_ns_steps', 5)),
                    use_muon=True,
                ))
            if adam_params:
                param_groups.append(dict(
                    params=adam_params,
                    lr=adamw_lr,
                    betas=betas,
                    eps=opt_cfg.get('eps', 1e-10),
                    weight_decay=wd,
                    use_muon=False,
                ))
            optimizer = Muon(param_groups)
        else:
            raise ValueError(f"unknown optimizer __target__={target!r}")

        schedule = opt_cfg.get('lr_schedule', 'warmup_constant')
        warmup = int(opt_cfg.get('warmup_steps', 0))
        if schedule == 'warmup_constant':
            # LambdaLR with a single lambda is broadcast across all param
            # groups; per-group base_lr is preserved (Muon vs AdamW lr
            # ratio survives the warmup ramp).
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda s: warmup_constant_lambda(s, warmup_steps=warmup),
            )
        elif schedule == 'warmup_linear':
            # Linear warmup, then linear decay to ``min_lr_ratio * base_lr``
            # over ``trainer.max_steps`` (override with ``total_steps``).
            trainer_cfg = self.config.get('trainer', {}) or {}
            total_steps = int(opt_cfg.get('total_steps',
                                          trainer_cfg.get('max_steps', 0)))
            min_lr_ratio = float(opt_cfg.get('min_lr_ratio', 0.1))
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda s: warmup_linear_lambda(
                    s, warmup_steps=warmup,
                    total_steps=total_steps, min_lr_ratio=min_lr_ratio),
            )
        else:
            raise ValueError(f"unknown lr_schedule={schedule!r}")

        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    # ------------------------------------------------------------------ #
    # Gradient clipping + grad-norm logging
    # ------------------------------------------------------------------ #
    def configure_gradient_clipping(self, optimizer,
                                    gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """Run norm-based gradient clipping and log the (pre-clip) total norm.

        We replace Lightning's default clipping with a direct call to
        ``torch.nn.utils.clip_grad_norm_`` so we can capture its return value
        (the global L2 norm, DTensor-aware under FSDP2/ModelParallelStrategy).
        This costs nothing extra — the same op runs anyway.
        """
        if gradient_clip_val is None:
            return
        params = [p for p in self.parameters() if p.grad is not None]
        if not params:
            return
        total_norm = torch.nn.utils.clip_grad_norm_(
            params, max_norm=float(gradient_clip_val))
        self.log('train/grad_norm', total_norm.detach(),
                 on_step=True, on_epoch=False, prog_bar=False, sync_dist=False)
