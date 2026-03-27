"""
DetectionLightningDataModule — single Lightning DataModule for all label modes.

Delegates all dataset and sampler construction to training_setup so that
the Lightning path is byte-for-byte equivalent to legacy train.py.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from softlabel_od.object_detection.training_setup import (
    build_datasets,
    build_train_loader,
    build_val_loader,
    build_overfit_subset,
    build_overfit_train_loader,
)
from softlabel_od.object_detection.utils.training_helpers import collate_fn


class DetectionLightningDataModule(pl.LightningDataModule):
    """
    Label-mode agnostic DataModule.

    Constructor args:
        cfg:  frozen YACS config node.

    Overfit mode:
        When cfg.TRAIN.OVERFIT_BATCHES > 0 a deterministic fixed-size Subset
        is drawn once in setup() and reused every epoch for both train and val,
        matching the OverfitDataLoader behaviour in the legacy path.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None
        self._train_loader = None
        self._val_loader = None
        self._is_overfit = False

    def setup(self, stage=None):
        cfg = self.cfg
        self.train_dataset, self.val_dataset = build_datasets(cfg)

        overfit_batches = getattr(cfg.TRAIN, "OVERFIT_BATCHES", 0)
        if overfit_batches and overfit_batches > 0:
            self._is_overfit = True
            subset = build_overfit_subset(cfg, self.train_dataset)
            overfit_loader = build_overfit_train_loader(cfg, subset)
            self._train_loader = overfit_loader
            # Legacy: val_loader == train_loader in overfit mode
            self._val_loader = overfit_loader
        else:
            self._is_overfit = False
            self._train_loader = build_train_loader(cfg, self.train_dataset)
            self._val_loader = build_val_loader(cfg, self.val_dataset)

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader