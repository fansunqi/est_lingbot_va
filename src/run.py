# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Training entry point.

Usage::

    # Single GPU
    python -m src.run --task configs/tasks/train_test.yaml --devices 1

    # 8 GPUs (Lightning auto-spawns subprocesses; no torchrun needed)
    python -m src.run --task configs/tasks/train_test.yaml --devices 8

    # Disable wandb
    python -m src.run --task configs/tasks/train_test.yaml --wandb-mode disabled

Loads a YAML task config (with ``!inc`` includes for model/data sub-configs),
builds DataModule + System, and hands off to ``lightning.pytorch.Trainer``.
Lightning's ``ModelParallelStrategy`` (FSDP2-native) is wired in automatically;
sharding granularity / AC / mixed precision live under ``trainer.fsdp:`` in the
YAML and are applied in ``FlowMatchVASystem.configure_model``.

CLI flags override YAML; YAML is the source of truth for everything else.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

import yaml
import yamlinclude
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.distributed.fsdp_strategy import build_strategy
from src.system.callbacks import (
    FSDPMetricsCallback,
    FullStateDictCheckpoint,
    PeriodicGCCallback,
)
from src.utils import init_logger, logger


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

def load_config(config_path: str | Path) -> dict:
    """Load a YAML config, resolving ``!inc`` relative to ``configs/``."""
    config_path = Path(config_path)
    config_root = config_path.parent.parent  # configs/tasks/foo.yaml -> configs/
    yaml.add_constructor(
        '!inc',
        yamlinclude.YamlIncludeConstructor(base_dir=str(config_root)),
        Loader=yaml.FullLoader,
    )
    with open(config_path, 'r') as f:
        return yaml.full_load(f)


def import_class(class_path: str):
    module_path, class_name = class_path.rsplit('.', 1)
    return getattr(importlib.import_module(module_path), class_name)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #

def _inject_data_dependencies(config: dict) -> dict:
    """Push fields the dataset needs (which live on the model/task level) down
    into the data sub-config so the DataModule can stay stand-alone."""
    data_cfg = dict(config['components']['data'])
    net_cfg = config['components']['model']['network']

    data_cfg.setdefault('patch_size', net_cfg['patch_size'])
    data_cfg.setdefault('wan22_pretrained_model_name_or_path', config['model_name_or_path'])
    # cfg_prob is already in data YAML; defensive default.
    data_cfg.setdefault('cfg_prob', 0.0)
    return data_cfg


def build_callbacks(config: dict, experiment_dir: Path, has_logger: bool) -> list:
    callbacks: list = []

    ckpt_cfg = dict(config.get('checkpoint', {}))
    full_ckpt_every = int(ckpt_cfg.pop('full_ckpt_every_n_steps', 0) or 0)
    ckpt_cfg.setdefault('dirpath', str(experiment_dir / 'checkpoints'))
    ckpt_cfg.setdefault('save_last', True)
    callbacks.append(ModelCheckpoint(**ckpt_cfg))

    if full_ckpt_every > 0:
        callbacks.append(FullStateDictCheckpoint(
            dirpath=str(experiment_dir / 'checkpoints'),
            every_n_train_steps=full_ckpt_every,
        ))

    cb_cfg = config.get('callbacks', {}) or {}
    if cb_cfg.get('gc_interval'):
        callbacks.append(PeriodicGCCallback(every_n_steps=int(cb_cfg['gc_interval'])))
    if cb_cfg.get('fsdp_metrics'):
        callbacks.append(FSDPMetricsCallback(
            results_path=str(Path(config['save_root']) / 'sweep_results.csv'),
            tag=str(cb_cfg.get('fsdp_metrics_tag', config['experiment_name'])),
            warmup_steps=int(cb_cfg.get('fsdp_metrics_warmup', 3)),
        ))

    # LearningRateMonitor refuses to attach when there's no logger; the LR
    # is still trivially recoverable from the optimizer schedule, so just skip.
    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval='step'))
    return callbacks


def build_logger(config: dict):
    wb_cfg = config.get('wandb')
    if not wb_cfg:
        return False
    mode = wb_cfg.get('mode', 'online')
    if mode == 'disabled':
        return False
    return WandbLogger(
        project=wb_cfg['project'],
        name=wb_cfg.get('name'),
        entity=wb_cfg.get('entity'),
        mode=mode,
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run(args):
    config = load_config(args.task)
    if args.save_root:
        config['save_root'] = args.save_root
    if args.resume_from:
        config['resume_from'] = args.resume_from
    if args.devices is not None:
        config.setdefault('trainer', {})['devices'] = args.devices
    if args.num_nodes is not None:
        config.setdefault('trainer', {})['num_nodes'] = args.num_nodes
    if args.max_steps is not None:
        config.setdefault('trainer', {})['max_steps'] = args.max_steps
    if args.wandb_mode is not None:
        config.setdefault('wandb', {})['mode'] = args.wandb_mode

    # FSDP overrides for the sweep harness — keeps configs/ untouched per cell.
    fsdp_overrides = {
        'ac_granularity': args.fsdp_ac,
        'reshard_after_forward': args.fsdp_reshard,
        'block_only': args.fsdp_block_only,
    }
    if any(v is not None for v in fsdp_overrides.values()):
        trainer_cfg = config.setdefault('trainer', {})
        fsdp_cfg = trainer_cfg.setdefault('fsdp', {})
        for k, v in fsdp_overrides.items():
            if v is not None:
                fsdp_cfg[k] = v
    if args.fsdp_metrics_tag is not None:
        config.setdefault('callbacks', {})['fsdp_metrics_tag'] = args.fsdp_metrics_tag
    if args.compile_model is not None:
        config['components']['model']['network']['compile_model'] = args.compile_model
    if args.compile_mode is not None:
        config['components']['model']['network']['compile_mode'] = args.compile_mode
    if args.flex_attn_compile_mode is not None:
        config['components']['model']['network']['flex_attn_compile_mode'] = args.flex_attn_compile_mode
    if args.fused_optim is not None:
        config.setdefault('optimizer', {})['fused'] = args.fused_optim

    # Default save_root keeps the run self-contained when the YAML's value
    # points at a missing cluster path.
    config.setdefault('save_root', './experiments')
    experiment_dir = Path(config['save_root']) / config['experiment_name']
    experiment_dir.mkdir(parents=True, exist_ok=True)
    config['experiment_dir'] = str(experiment_dir)

    # DataModule — instantiate first so we can pass its config through.
    data_cfg = _inject_data_dependencies(config)
    DataModuleClass = import_class(config['components']['data']['module'])
    datamodule = DataModuleClass(data_cfg)

    # System
    SystemClass = import_class(config['system'])
    system = SystemClass(config)

    # Strategy / callbacks / logger
    strategy = build_strategy(config)
    config.setdefault('callbacks', {})  # ensure dict exists for callbacks lookup
    wb_logger = build_logger(config)
    callbacks = build_callbacks(config, experiment_dir, has_logger=bool(wb_logger))

    # ``fsdp`` lives under ``trainer:`` in the YAML for grouping but isn't a
    # ``pl.Trainer`` kwarg — it's read by ``build_strategy`` and
    # ``configure_model``, so drop it before forwarding.
    trainer_kwargs = {
        k: v for k, v in config['trainer'].items() if k not in ('strategy', 'fsdp')
    }
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=wb_logger,
        strategy=strategy,
        use_distributed_sampler=False,  # PackingDataset already shards by rank internally
        **trainer_kwargs,
    )

    logger.info("Starting Trainer.fit() — task=%s experiment=%s",
                args.task, config['experiment_name'])
    trainer.fit(system, datamodule=datamodule, ckpt_path=config.get('resume_from'))


def main():
    parser = argparse.ArgumentParser(description="LingBot-VA Lightning training entry")
    parser.add_argument('--task', required=True, help="Path to a YAML task config")
    parser.add_argument('--save-root', default=None, help="Override config['save_root']")
    parser.add_argument('--resume-from', default=None, help="Lightning .ckpt to resume")
    parser.add_argument('--devices', default=None, type=lambda s: int(s) if s.isdigit() else s,
                        help="Override trainer.devices (int N or 'auto'); Lightning spawns this many subprocs")
    parser.add_argument('--num-nodes', default=None, type=int, help="Override trainer.num_nodes")
    parser.add_argument('--max-steps', default=None, type=int, help="Override trainer.max_steps")
    parser.add_argument('--wandb-mode', default=None, choices=['online', 'offline', 'disabled'],
                        help="Override wandb.mode")
    parser.add_argument('--fsdp-ac', default=None,
                        choices=['none', 'every_2', 'every_4', 'all'],
                        help="Override trainer.fsdp.ac_granularity")
    parser.add_argument('--fsdp-reshard', default=None,
                        type=lambda s: s.lower() in ('1', 'true', 'yes'),
                        help="Override trainer.fsdp.reshard_after_forward (true/false)")
    parser.add_argument('--fsdp-block-only', default=None,
                        type=lambda s: s.lower() in ('1', 'true', 'yes'),
                        help="Override trainer.fsdp.block_only (true/false)")
    parser.add_argument('--fsdp-metrics-tag', default=None,
                        help="Tag column written to sweep_results.csv by FSDPMetricsCallback")
    parser.add_argument('--compile-model', default=None,
                        type=lambda s: s.lower() in ('1', 'true', 'yes'),
                        help="Override components.model.network.compile_model (true/false)")
    parser.add_argument('--compile-mode', default=None,
                        choices=['default', 'reduce-overhead',
                                 'max-autotune', 'max-autotune-no-cudagraphs'],
                        help="Override components.model.network.compile_mode")
    parser.add_argument('--flex-attn-compile-mode', default=None,
                        choices=['default', 'reduce-overhead',
                                 'max-autotune', 'max-autotune-no-cudagraphs'],
                        help="Override components.model.network.flex_attn_compile_mode")
    parser.add_argument('--fused-optim', default=None,
                        type=lambda s: s.lower() in ('1', 'true', 'yes'),
                        help="Override optimizer.fused (true/false) for AdamW")
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    init_logger()
    main()
