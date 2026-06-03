"""
training_setup.py — shared training assembly logic.

Both the vanilla PyTorch orchestrator (train.py) and the Lightning path
(train_lightning.py) call into this module so that all dataset, sampler,
model, optimizer, and scheduler construction is expressed once.
"""

from __future__ import annotations

import os
import torch
from torch.utils.data import DataLoader

from softlabel_od.object_detection.adapters.datasets import build_detection_dataset
from softlabel_od.object_detection.utils.config import resolve_label_mode, resolve_label_file
from softlabel_od.object_detection.utils.training_helpers import (
    collate_fn,
    create_balanced_sampler,
    load_pretrained_weights,
    freeze_layers,
)
from softlabel_od.object_detection.utils.augmentations import (
    ClassConditionalAugmentation,
    MinimalAugmentation,
)
from softlabel_od.object_detection.core.model.factory import get_model


# ---------------------------------------------------------------------------
# Label mode
# ---------------------------------------------------------------------------

def resolve_model_type(cfg) -> str:
    """Return the internal model_type string from config."""
    label_mode = resolve_label_mode(cfg)
    mapping = {
        "hard":    "hard",
        "soft_1d": "1d",
        "soft_2d": "2d",
    }
    if label_mode not in mapping:
        raise ValueError(f"Unsupported label mode: {label_mode}")
    return mapping[label_mode]


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

def build_transforms(cfg):
    """Return (train_transform, val_transform) exactly as legacy train.py does."""
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


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def build_datasets(cfg):
    """
    Build train and val datasets from config.
    Applies cluster-level ignore_nofinding_clusters via DATA.OPTIONS if set.
    """
    train_transform, val_transform = build_transforms(cfg)
    print("Initialising training dataset...")
    train_dataset = build_detection_dataset(cfg, split="train", transforms=train_transform)
    print("Initialising validation dataset...")
    train_dataset = build_detection_dataset(cfg, split="train", transforms=train_transform)
    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Sampler — this is the single source of truth for sampling logic
# ---------------------------------------------------------------------------

def build_train_sampler(cfg, train_dataset):
    """
    Build training sampler with unified no-finding handling.
    
    Three strategies for handling pure no-finding (healthy CXR) images:
        1. IGNORE_NOFINDING_IMAGES=True: Skip ALL pure no-finding images
        2. NOFINDING_INCLUDE_FRACTION=x: Include only x% of pure no-finding images (randomly sampled)
        3. DOWNWEIGHT_NOFINDING_IMAGES=True: Include all but downweight in balanced sampler
    
    Returns (sampler, shuffle).
    """
    import numpy as np
    from torch.utils.data import Subset, SubsetRandomSampler
    
    # Step 1: Determine which images to include based on no-finding strategy
    all_indices = list(range(len(train_dataset)))
    valid_indices = all_indices
    
    # Strategy 1: Completely ignore all pure no-finding images
    if cfg.DATA.IGNORE_NOFINDING_IMAGES:
        print("Filtering pure No-Finding images (healthy CXR) from training set...")
        valid_indices = [
            idx for idx in all_indices
            if train_dataset.has_pathology_clusters(train_dataset.image_ids[idx])
        ]
        kept = len(valid_indices)
        total = len(all_indices)
        print(f"  Kept {kept}/{total} images ({kept/total*100:.1f}% have pathology).")
    
    # Strategy 2: Include only a fraction of pure no-finding images
    elif cfg.DATA.NOFINDING_INCLUDE_FRACTION < 1.0:
        fraction = cfg.DATA.NOFINDING_INCLUDE_FRACTION
        print(f"Sampling {fraction*100:.1f}% of pure No-Finding images...")
        
        # Separate pathology and pure no-finding images
        pathology_indices = []
        nofinding_indices = []
        
        for idx in all_indices:
            if train_dataset.has_pathology_clusters(train_dataset.image_ids[idx]):
                pathology_indices.append(idx)
            else:
                nofinding_indices.append(idx)
        
        # Randomly sample the specified fraction of no-finding images
        if nofinding_indices:
            # Set seed for reproducibility (can be made configurable)
            np.random.seed(42)
            n_nofinding_to_keep = int(len(nofinding_indices) * fraction)
            sampled_nofinding = np.random.choice(
                nofinding_indices, 
                size=n_nofinding_to_keep, 
                replace=False
            ).tolist()
        else:
            sampled_nofinding = []
        
        # Combine: ALL pathology + sampled no-finding
        valid_indices = pathology_indices + sampled_nofinding
        valid_indices.sort()  # Maintain original order
        
        print(f"  Pathology images: {len(pathology_indices)} (all included)")
        print(f"  Pure no-finding images: {len(nofinding_indices)} total, kept {len(sampled_nofinding)} ({fraction*100:.1f}%)")
        print(f"  Final training set: {len(valid_indices)} images")
    
    # Create dataset to sample from (Subset if filtering was applied)
    dataset_to_sample = train_dataset
    if len(valid_indices) < len(all_indices):
        dataset_to_sample = Subset(train_dataset, valid_indices)
    
    # Step 2: Create sampler
    if cfg.DATA.BALANCED_SAMPLER:
        print("Using balanced (RFS) sampling...")
        sampler = create_balanced_sampler(
            train_dataset,  # Pass original dataset
            rfs_threshold=cfg.DATA.RFS_THRESHOLD,
            downweight_nofinding=cfg.DATA.DOWNWEIGHT_NOFINDING_IMAGES,
            indices=valid_indices if len(valid_indices) < len(all_indices) else None,
        )
        return sampler, False
    else:
        # No balanced sampler - use SubsetRandomSampler if we filtered
        if len(valid_indices) < len(all_indices):
            sampler = SubsetRandomSampler(valid_indices)
            return sampler, False
        else:
            return None, True


# ---------------------------------------------------------------------------
# DataLoaders
# ---------------------------------------------------------------------------

def _loader_kwargs(cfg):
    """DataLoader kwargs shared between train and val loaders."""
    return dict(
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


def build_train_loader(cfg, train_dataset):
    """Build the training DataLoader, including sampler logic."""
    sampler, shuffle = build_train_sampler(cfg, train_dataset)
    kwargs = _loader_kwargs(cfg)
    if sampler is not None:
        # sampler and shuffle are mutually exclusive in PyTorch
        kwargs.pop("prefetch_factor", None)
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
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


def build_val_loader(cfg, val_dataset):
    """Build the validation DataLoader."""
    return DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )


def build_dataloaders(cfg, train_dataset, val_dataset):
    """Convenience wrapper that returns (train_loader, val_loader)."""
    return build_train_loader(cfg, train_dataset), build_val_loader(cfg, val_dataset)


# ---------------------------------------------------------------------------
# Overfit (deterministic cached-batch mode)
# ---------------------------------------------------------------------------

def build_overfit_subset(cfg, train_dataset):
    """
    Return a fixed Subset of train_dataset for overfit runs.
    Lightning uses this instead of OverfitDataLoader because the trainer
    drives the loop; there is no place to wrap the loader.  The subset is
    drawn deterministically via create_balanced_sampler with a fixed seed.
    The resulting Subset is used with a plain shuffle=False DataLoader.
    
    Honors IGNORE_NOFINDING_IMAGES to exclude healthy CXR from overfit batches.
    """
    from torch.utils.data import Subset
    
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
        rfs_threshold=cfg.DATA.RFS_THRESHOLD,
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


def build_overfit_train_loader(cfg, subset):
    """DataLoader for an overfit subset — no sampler, no shuffle."""
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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(cfg, model_type: str, device=None):
    """Build and optionally move the model to device."""
    model = get_model(
        num_classes=cfg.MODEL.NUM_CLASSES,
        model_type=model_type,
        backbone_name=cfg.MODEL.BACKBONE,
        pretrained=True,
        probabilistic=cfg.MODEL.PROBABILISTIC,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS
    )
    if device is not None:
        model.to(device)
    return model


def configure_model_for_training(cfg, model, train_dataset, device=None):
    """
    Apply all post-construction model configuration:
      1. Load pretrained weights (if configured)
      2. Register the RoI-head class-weight buffer
      3. Freeze layers (if configured)

    This must be called after build_model() and after the train dataset
    is ready (class weights are computed from the train split).
    """
    if cfg.MISC.PRETRAINED_WEIGHTS:
        d = device or torch.device("cpu")
        model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, d)
        print(f"Loaded pretrained weights from: {cfg.MISC.PRETRAINED_WEIGHTS}")

    class_weights = train_dataset.compute_roi_head_class_weights_per_cluster(
        num_classes=cfg.MODEL.NUM_CLASSES,
        presence_prob_thresh=0.0,
        normalize=False,
        verbose=True,
    )
    target_device = device or next(model.parameters(), torch.empty(0)).device
    model.roi_heads.register_buffer("class_weights", class_weights.to(target_device))
    print(f"Registered class_weights buffer on roi_heads.")

    if cfg.MISC.FROZEN_LAYERS:
        freeze_layers(model, cfg.MISC.FROZEN_LAYERS)
        print(f"Frozen layers: {cfg.MISC.FROZEN_LAYERS}")

    return model


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def build_optimizer(cfg, model):
    """
    SGD with three parameter groups matching legacy train.py exactly:
      - bbox head  (box_predictor.bbox_pred / bbox_pred_gmm) → BASE_LR_BBOX
      - class head (box_predictor.cls_score)                 → BASE_LR_CLASS
      - everything else                                       → BASE_LR
    """
    bbox_params, class_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "box_predictor.bbox_pred" in name or "box_predictor.bbox_pred_gmm" in name:
            bbox_params.append(param)
        elif "box_predictor.cls_score" in name:
            class_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": bbox_params,  "lr": cfg.TRAIN.BASE_LR_BBOX},
            {"params": class_params, "lr": cfg.TRAIN.BASE_LR_CLASS},
            {"params": other_params, "lr": cfg.TRAIN.BASE_LR},
        ],
        momentum=0.9,
        weight_decay=0.0005,
    )
    return optimizer


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(cfg, optimizer):
    """
    Return the LR scheduler selected by cfg.TRAIN.SCHEDULER.
      'main_scheduler'      → MultiStepLR
      'secondary_scheduler' → CosineAnnealingLR
    """
    from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

    name = cfg.TRAIN.SCHEDULER
    if name == "main_scheduler":
        return MultiStepLR(
            optimizer,
            milestones=list(cfg.TRAIN.MILESTONES),
            gamma=cfg.TRAIN.GAMMA,
        )
    if name == "secondary_scheduler":
        return CosineAnnealingLR(
            optimizer,
            T_max=cfg.TRAIN.NUM_EPOCHS,
            eta_min=1e-5,
        )
    raise ValueError(f"Unsupported scheduler: {name!r}")