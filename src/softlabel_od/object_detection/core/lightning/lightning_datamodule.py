"""
DetectionLightningDataModule — single Lightning DataModule for all label modes.

Delegates all dataset and sampler construction to training_setup so that
the Lightning path is byte-for-byte equivalent to legacy train.py.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset
from softlabel_od.object_detection.adapters.datasets import build_detection_dataset
from softlabel_od.object_detection.utils.training_helpers import (
    collate_fn,
    create_balanced_sampler,
)
from softlabel_od.object_detection.utils.augmentations import (
    ClassConditionalAugmentation,
    MinimalAugmentation,
)


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
        self.train_dataset, self.val_dataset = self.build_datasets(cfg)

        overfit_batches = getattr(cfg.TRAIN, "OVERFIT_BATCHES", 0)
        if overfit_batches and overfit_batches > 0:
            self._is_overfit = True
            subset = self.build_overfit_subset(cfg, self.train_dataset)
            overfit_loader = self.build_overfit_loader(cfg, subset)
            self._train_loader = overfit_loader
            self._val_loader = overfit_loader
        else:
            self._is_overfit = False
            self._train_loader = self.build_train_loader(cfg, self.train_dataset)
            self._val_loader = self.build_val_loader(cfg, self.val_dataset)

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader
    
    def build_transforms(self, cfg):
        train_transform = ClassConditionalAugmentation(
            rare_class_indices=cfg.AUGMENTATION.RARE_CLASS_INDICES,
            rare_augment_prob=cfg.AUGMENTATION.RARE_CLASS_AUGMENT_PROB,
            common_augment_prob=cfg.AUGMENTATION.COMMON_CLASS_AUGMENT_PROB,
            rotation_degrees=cfg.AUGMENTATION.ROTATION_DEGREES,
            brightness_range=cfg.AUGMENTATION.BRIGHTNESS_RANGE,
            contrast_range=cfg.AUGMENTATION.CONTRAST_RANGE,
            noise_std=cfg.AUGMENTATION.NOISE_STD,
        )
        val_transform = MinimalAugmentation()
        return train_transform, val_transform
    
    def build_datasets(self, cfg):
        train_transform, val_transform = self.build_transforms(cfg)
        print("Initialising training dataset...")
        train_dataset = build_detection_dataset(cfg, train=True, transforms=train_transform)
        print("Initialising validation dataset...")
        val_dataset = build_detection_dataset(cfg, train=False, transforms=val_transform)
        return train_dataset, val_dataset
    
    # TRAIN LOADER
    def build_train_loader(self, cfg, train_dataset):
        # Step 1: Filter out pure no-finding images if requested
        dataset_to_sample = train_dataset
        if cfg.DATA.IGNORE_NOFINDING_IMAGES:
            print("Filtering pure No-Finding images (healthy CXR) from training set...")
            valid_indices = [
                idx for idx in range(len(train_dataset))
                if train_dataset.has_pathology_clusters(train_dataset.image_ids[idx])
            ]
            kept = len(valid_indices)
            total = len(train_dataset)
            print(f"  Kept {kept}/{total} images ({kept/total*100:.1f}% have pathology).")
            dataset_to_sample = Subset(train_dataset, valid_indices)
        
        # Step 2: Create sampler
        if cfg.DATA.BALANCED_SAMPLER:
            print("Using balanced (RFS) sampling...")
            sampler = create_balanced_sampler(
                train_dataset,
                # dataset_to_sample,
                rfs_threshold=cfg.DATA.OVERSAMPLE_FACTOR,
                downweight_nofinding=cfg.DATA.DOWNWEIGHT_NOFINDING_IMAGES,
                indices=valid_indices if cfg.DATA.IGNORE_NOFINDING_IMAGES else None,  # NEW parameter
            )
            return DataLoader(
                train_dataset if not cfg.DATA.IGNORE_NOFINDING_IMAGES else dataset_to_sample,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                sampler=sampler,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
        else:
            # No balanced sampler
            if cfg.DATA.IGNORE_NOFINDING_IMAGES:
                # Already filtered, just need deterministic sampling
                sampler = torch.utils.data.SubsetRandomSampler(valid_indices)
                return DataLoader(
                    train_dataset,
                    batch_size=cfg.TRAIN.BATCH_SIZE,
                    sampler=sampler,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2,
                )
        return DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    
    # VAL LOADER
    def build_val_loader(self, cfg, val_dataset):
        return DataLoader(
            val_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

    # Overfit Subset Builder
    def build_overfit_subset(self, cfg, train_dataset):
        # Filter out pure no-finding images if requested
        dataset_to_sample = train_dataset
        if cfg.DATA.IGNORE_NOFINDING_IMAGES:
            print("[Overfit] Filtering pure No-Finding images...")
            valid_indices = [
                idx for idx in range(len(train_dataset))
                if train_dataset.has_pathology_clusters(train_dataset.image_ids[idx])
            ]
            dataset_to_sample = Subset(train_dataset, valid_indices)
            print(f"[Overfit] Working with {len(valid_indices)}/{len(train_dataset)} pathology images.")
        
        num_needed = int(cfg.TRAIN.OVERFIT_BATCHES) * int(cfg.TRAIN.BATCH_SIZE)
        sampler = create_balanced_sampler(
            dataset_to_sample,
            rfs_threshold=cfg.DATA.OVERSAMPLE_FACTOR,
            downweight_nofinding=cfg.DATA.DOWNWEIGHT_NOFINDING_IMAGES,
        )
        g = torch.Generator()
        g.manual_seed(42)
        try:
            sampler.generator = g
        except Exception:
            pass
        it = iter(sampler)
        fixed_indices = [int(next(it)) for _ in range(num_needed)]
        
        # Map back to original dataset indices if we used a Subset
        if isinstance(dataset_to_sample, Subset):
            fixed_indices = [dataset_to_sample.indices[i] for i in fixed_indices]
        
        subset = Subset(train_dataset, fixed_indices)
        print(
            f"[Overfit] Fixed balanced subset: {len(subset)} samples "
            f"({cfg.TRAIN.OVERFIT_BATCHES} batches × {cfg.TRAIN.BATCH_SIZE})"
        )
        return subset
    
    def build_overfit_loader(self, cfg, subset):
        return DataLoader(
            subset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )