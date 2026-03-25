import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset

from softlabel_od.object_detection.adapters.datasets import build_detection_dataset
from softlabel_od.object_detection.utils.training_helpers import collate_fn, create_balanced_sampler
from softlabel_od.object_detection.utils.augmentations import ClassConditionalAugmentation, MinimalAugmentation


class VinDr1DDataModule(pl.LightningDataModule):
    """
    1D soft-label DataModule built on the dataset registry.

    The class name is kept for backward compatibility with the existing
    Lightning entry point, but the dataset itself is selected from config.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        cfg = self.cfg

        # Match train.py augmentation construction
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

        self.train_dataset = build_detection_dataset(cfg, train=True, transforms=train_transform)
        # train without No Finding instances       
        if cfg.DATA.IGNORE_NOFINDING==True:
            kept = []
            for img_id in self.train_dataset.image_ids:
                dominant = self.train_dataset.get_dominant_class(image_id=img_id)
                if dominant != 0:  # 0 == background / No Finding in your dataset logic
                    kept.append(img_id)
            self.train_dataset.image_ids = kept
            print(f"[DataModule] Filtered NoFinding: train images now {len(kept)}")
        
        if getattr(cfg.TRAIN, "OVERFIT_BATCHES", 0) and cfg.TRAIN.OVERFIT_BATCHES != 0:
            num_needed = int(cfg.TRAIN.OVERFIT_BATCHES) * int(cfg.TRAIN.BATCH_SIZE)
            sampler = create_balanced_sampler(
                self.train_dataset,
                oversample_factor=cfg.DATA.OVERSAMPLE_FACTOR,
            )
            g = torch.Generator()
            g.manual_seed(42)
            try:
                sampler.generator = g
            except Exception:
                pass

            fixed_indices = []
            it = iter(sampler)
            for _ in range(num_needed):
                fixed_indices.append(int(next(it)))

            self.train_dataset = Subset(self.train_dataset, fixed_indices)
            print(f"[Overfit] Using fixed balanced subset: {len(self.train_dataset)} samples "
                f"({cfg.TRAIN.OVERFIT_BATCHES} batches × {cfg.TRAIN.BATCH_SIZE})")
        self.val_dataset = build_detection_dataset(cfg, train=False, transforms=val_transform)

    def train_dataloader(self):
        cfg = self.cfg
        if isinstance(self.train_dataset, Subset):
            return DataLoader(
                self.train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )
        
        if cfg.DATA.BALANCED_SAMPLER:
            sampler = create_balanced_sampler(
                self.train_dataset,
                oversample_factor=cfg.DATA.OVERSAMPLE_FACTOR,
            )
            return DataLoader(
                self.train_dataset,
                batch_size=cfg.TRAIN.BATCH_SIZE,
                sampler=sampler,
                shuffle=False,  # sampler + shuffle should not both be set
                collate_fn=collate_fn,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
            )

        return DataLoader(
            self.train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    def val_dataloader(self):
        cfg = self.cfg
        return DataLoader(
            self.val_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )