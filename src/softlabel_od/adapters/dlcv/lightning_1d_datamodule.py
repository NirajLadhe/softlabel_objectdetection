import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Subset

from softlabel_od.adapters.dlcv.dataset import Vin_Dr_CXR_SoftLabel_Dataset
from softlabel_od.adapters.dlcv.utils import collate_fn, create_balanced_sampler
from softlabel_od.adapters.dlcv.augmentations import ClassConditionalAugmentation, MinimalAugmentation


class VinDr1DDataModule(pl.LightningDataModule):
    """
    DataModule that preserves your current dataset behavior:
    - Dataset performs stratified split internally (create_stratified_split)
    - Train/val datasets are instantiated separately, just like train.py
    - Balanced sampling uses create_balanced_sampler(dataset)
    - Augmentations are exactly your existing classes
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        cfg = self.cfg

        # 1D soft labels only
        label_file = cfg.DATA.SOFT_LABEL_FILE_1D
        gmm_type = "1D"
        dataset_type = "soft"

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

        # IMPORTANT: instantiate separately, like in train.py,
        # so dataset’s internal stratified split logic is unchanged.
        self.train_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
            root=cfg.DATA.ROOT,
            label_file=label_file,
            transforms=train_transform,
            train=True,
            test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
            gmm_type=gmm_type,
            dataset_type=dataset_type,
        )
        # train without No Finding instances       
        if cfg.DATA.IGNORE_NOFINDING==True:
            kept = []
            for img_id in self.train_dataset.image_ids:
                labels = self.train_dataset.image_data[img_id]
                dominant = self.train_dataset.get_dominant_class(labels)
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