import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from softlabel_od.adapters.dlcv.dataset import Vin_Dr_CXR_SoftLabel_Dataset
from softlabel_od.adapters.dlcv.utils import collate_fn, create_balanced_sampler
from softlabel_od.adapters.dlcv.augmentations import ClassConditionalAugmentation, MinimalAugmentation


class VinDrHardDataModule(pl.LightningDataModule):
    """
    Hard-label DataModule.

    Preserves:
    - Dataset's internal stratified split (Vin_Dr_CXR_SoftLabel_Dataset.create_stratified_split)
    - Balanced sampling via create_balanced_sampler
    - Same augmentations as train.py
    - Optional IGNORE_NOFINDING filtering
    - Optional deterministic overfit subset when TRAIN.OVERFIT_BATCHES != 0
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        cfg = self.cfg

        label_file = cfg.DATA.HARD_LABEL_FILE
        dataset_type = "hard"
        gmm_type = None

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

        self.train_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
            root=cfg.DATA.ROOT,
            label_file=label_file,
            transforms=train_transform,
            train=True,
            test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
            gmm_type=gmm_type,
            dataset_type=dataset_type,
        )

        # Optional: remove "No Finding" images (dominant class == 0) from training
        if getattr(cfg.DATA, "IGNORE_NOFINDING", False):
            kept = []
            for img_id in self.train_dataset.image_ids:
                labels = self.train_dataset.image_data[img_id]
                dominant = self.train_dataset.get_dominant_class(labels)
                if dominant != 0:
                    kept.append(img_id)
            self.train_dataset.image_ids = kept
            print(f"[DataModule] Filtered NoFinding: train images now {len(kept)}")

        # Deterministic overfit subset (still chosen via balanced sampler once)
        if getattr(cfg.TRAIN, "OVERFIT_BATCHES", 0) and cfg.TRAIN.OVERFIT_BATCHES != 0:
            num_needed = int(cfg.TRAIN.OVERFIT_BATCHES) * int(cfg.TRAIN.BATCH_SIZE)

            sampler = create_balanced_sampler(
                self.train_dataset,
                oversample_factor=cfg.DATA.OVERSAMPLE_FACTOR,
            )

            # Best-effort reproducibility of the sampled subset
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
            print(
                f"[Overfit] Using fixed balanced subset: {len(self.train_dataset)} samples "
                f"({cfg.TRAIN.OVERFIT_BATCHES} batches × {cfg.TRAIN.BATCH_SIZE})"
            )

        self.val_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
            root=cfg.DATA.ROOT,
            label_file=label_file,
            transforms=val_transform,
            train=False,
            test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
            gmm_type=gmm_type,
            dataset_type=dataset_type,
        )

    def train_dataloader(self):
        cfg = self.cfg

        # If we froze an overfit subset, do NOT apply a sampler again
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
                ignore_nofinding=cfg.DATA.IGNORE_NOFINDING
            )
            return DataLoader(
                self.train_dataset,
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