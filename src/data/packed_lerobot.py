# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LightningDataModule for packed LeRobot latent training.

Wraps ``MultiLatentLeRobotDataset`` with sequence packing and exposes the
``fast_forward`` hook used by ``FlowMatchVASystem.on_train_start`` to
restore the (epoch, bin) position from a Lightning checkpoint.
"""
from __future__ import annotations

from easydict import EasyDict

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from src.data.lerobot_latent_dataset import MultiLatentLeRobotDataset
from src.data.packing import PackingDataset, packed_collate


class PackedLatentLeRobotDataModule(pl.LightningDataModule):

    def __init__(self, config):
        super().__init__()
        # Cache as plain dict + EasyDict view for downstream code that does
        # attribute access (MultiLatentLeRobotDataset uses ``config.cfg_prob``,
        # ``config.patch_size``, ``config.dataset_path`` etc).
        self.config = dict(config)
        self.dataloader_cfg = dict(self.config.get('train_dataloader', {}))

        self._packing_epoch = 0
        self._bin_in_epoch = 0
        self._dataset_built = False

        self.multi_ds: MultiLatentLeRobotDataset | None = None
        self.packing_ds: PackingDataset | None = None

    # ------------------------------------------------------------------ #
    # Lightning lifecycle
    # ------------------------------------------------------------------ #

    def setup(self, stage=None):
        if self._dataset_built:
            return

        ds_config = EasyDict(self.config)
        ds_config.world_size = self.trainer.world_size if self.trainer else 1
        ds_config.rank = self.trainer.global_rank if self.trainer else 0

        self.multi_ds = MultiLatentLeRobotDataset(config=ds_config)
        self.packing_ds = PackingDataset(
            multi_ds=self.multi_ds,
            max_tokens=int(self.config['max_tokens']),
            max_episodes_per_bin=int(self.config['max_episodes_per_bin']),
            world_size=ds_config.world_size,
            rank=ds_config.rank,
            seed=int(self.config.get('seed', 42)),
            epoch=self._packing_epoch,
        )
        self._dataset_built = True

    def train_dataloader(self):
        assert self.packing_ds is not None, "setup() must run before train_dataloader()"
        return DataLoader(
            self.packing_ds,
            batch_size=int(self.dataloader_cfg.get('batch_size', 1)),
            shuffle=False,  # PackingDataset already permutes per epoch
            num_workers=int(self.config.get('load_worker', 0)),
            collate_fn=packed_collate,
            persistent_workers=bool(self.dataloader_cfg.get('persistent_workers', False)),
            pin_memory=bool(self.dataloader_cfg.get('pin_memory', False)),
        )

    # ------------------------------------------------------------------ #
    # Resume support
    # ------------------------------------------------------------------ #

    def fast_forward(self, epoch: int, bin_in_epoch: int):
        """Restore the (epoch, bin) position saved in the checkpoint.

        Mirrors the legacy ``Trainer._load_training_state`` recovery path:
        if the saved bin index is past the current per-rank epoch length,
        roll forward into the next epoch; otherwise resume mid-epoch.
        """
        assert self.packing_ds is not None, "setup() must run before fast_forward()"
        self._packing_epoch = int(epoch)
        self._bin_in_epoch = int(bin_in_epoch)

        self.packing_ds.set_epoch(self._packing_epoch, start_bin=0)
        per_rank_bins = len(self.packing_ds)
        if self._bin_in_epoch >= per_rank_bins:
            self._packing_epoch += 1
            self._bin_in_epoch = 0
            self.packing_ds.set_epoch(self._packing_epoch, start_bin=0)
        elif self._bin_in_epoch > 0:
            self.packing_ds.set_epoch(self._packing_epoch, start_bin=self._bin_in_epoch)
