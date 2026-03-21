"""
Object detection training orchestrator (vanilla PyTorch).

Loads config → builds dataset → builds model → trains → saves.
This is the single authoritative entry point for vanilla PyTorch training.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from softlabel_od.object_detection.utils.config import get_cfg_defaults, get_cfg_from_file
from softlabel_od.object_detection.adapters.datasets.vindrcxr import Vin_Dr_CXR_SoftLabel_Dataset
from softlabel_od.object_detection.utils.training_helpers import (
    collate_fn,
    create_balanced_sampler,
    save_model,
    load_pretrained_weights,
    freeze_layers,
    OverfitDataLoader,
)
from softlabel_od.object_detection.utils.augmentations import ClassConditionalAugmentation, MinimalAugmentation
from softlabel_od.object_detection.utils.schedulers import warmup_lr_scheduler
from softlabel_od.object_detection.core.model.factory import get_model


def main(cfg):
    # ── Determine label file and dataset type ──────────────────────────────
    if cfg.DATA.GMM_TYPE == "HARD_LABEL":
        dataset_type = "hard"
        label_file = cfg.DATA.HARD_LABEL_FILE
        gmm_type = None
        model_type = "hard"
    else:
        dataset_type = "soft"
        gmm_type = cfg.DATA.GMM_TYPE
        model_type = cfg.DATA.GMM_TYPE.lower()
        if cfg.DATA.GMM_TYPE == "1D":
            label_file = cfg.DATA.SOFT_LABEL_FILE_1D
        elif cfg.DATA.GMM_TYPE == "2D":
            label_file = cfg.DATA.SOFT_LABEL_FILE_2D
        else:
            raise ValueError(f"Unsupported GMM type: {cfg.DATA.GMM_TYPE}")

    if not os.path.exists(cfg.DATA.ROOT):
        print(f"Warning: Dataset root not found: {cfg.DATA.ROOT}")
    if not os.path.exists(label_file):
        print(f"Warning: Label file not found: {label_file}")

    print(f"Config: {getattr(cfg, 'CONFIG_FILE_PATH', 'N/A')}")
    print(f"GMM type: {cfg.DATA.GMM_TYPE}")
    print(cfg.dump())

    device = torch.device("cuda" if torch.cuda.is_available() and not cfg.MISC.NO_CUDA else "cpu")
    print(f"Device: {device}")

    # ── Augmentations ──────────────────────────────────────────────────────
    train_transform = ClassConditionalAugmentation(
        rare_class_indices=cfg.AUGMENTATION.RARE_CLASS_INDICES,
        rare_augment_prob=cfg.AUGMENTATION.RARE_CLASS_AUGMENT_PROB,
        common_augment_prob=cfg.AUGMENTATION.COMMON_CLASS_AUGMENT_PROB,
        rotation_degrees=cfg.AUGMENTATION.ROTATION_DEGREES,
        brightness_range=cfg.AUGMENTATION.BRIGHTNESS_RANGE,
        contrast_range=cfg.AUGMENTATION.CONTRAST_RANGE,
        noise_std=cfg.AUGMENTATION.NOISE_STD,
    )
    test_transform = MinimalAugmentation()

    # ── Datasets ───────────────────────────────────────────────────────────
    print("Initialising training dataset...")
    train_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
        root=cfg.DATA.ROOT,
        label_file=label_file,
        transforms=train_transform,
        train=True,
        test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
        gmm_type=gmm_type,
        dataset_type=dataset_type,
    )

    print("Initialising validation dataset...")
    val_dataset = Vin_Dr_CXR_SoftLabel_Dataset(
        root=cfg.DATA.ROOT,
        label_file=label_file,
        transforms=test_transform,
        train=False,
        test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
        gmm_type=gmm_type,
        dataset_type=dataset_type,
    )

    # ── DataLoaders ────────────────────────────────────────────────────────
    if cfg.DATA.BALANCED_SAMPLER:
        print("Using balanced (RFS) sampling...")
        sampler = create_balanced_sampler(
            train_dataset,
            rfs_threshold=cfg.DATA.OVERSAMPLE_FACTOR,
            ignore_nofinding=cfg.DATA.NOFINDING_WEIGHT_GIVEN,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        if cfg.DATA.IGNORE_NOFINDING:
            print("Filtering No-Finding images...")
            valid_indices = [
                idx for idx in range(len(train_dataset))
                if train_dataset.get_dominant_class(train_dataset.image_data[train_dataset.image_ids[idx]]) != 0
            ]
            print(f"Kept {len(valid_indices)}/{len(train_dataset)} images.")
            sampler = torch.utils.data.SubsetRandomSampler(valid_indices)
            shuffle = False
        else:
            sampler = None
            shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # ── Overfit mode ───────────────────────────────────────────────────────
    overfit_batches = getattr(cfg.TRAIN, "OVERFIT_BATCHES", 0)
    if overfit_batches > 0:
        print(f"OVERFIT MODE: {overfit_batches} batches")
        train_loader = OverfitDataLoader(train_loader, num_batches=overfit_batches, move_to_device=device)
        val_loader = train_loader

    print(f"Train: {len(train_dataset)} images  |  Val: {len(val_dataset)} images")

    # ── Model ──────────────────────────────────────────────────────────────
    model = get_model(
        num_classes=cfg.MODEL.NUM_CLASSES,
        model_type=model_type,
        backbone_name=cfg.MODEL.BACKBONE,
        pretrained=True,
        probabilistic=cfg.MODEL.PROBABILISTIC,
    )
    model.to(device)

    if cfg.MISC.PRETRAINED_WEIGHTS:
        model = load_pretrained_weights(model, cfg.MISC.PRETRAINED_WEIGHTS, device)

    # Class weights for RoI head
    class_weights = train_dataset.compute_roi_head_class_weights_per_cluster(
        num_classes=cfg.MODEL.NUM_CLASSES,
        presence_prob_thresh=0.0,
        normalize=False,
        verbose=True,
    )
    model.roi_heads.register_buffer("class_weights", class_weights.to(device))

    if cfg.MISC.FROZEN_LAYERS:
        freeze_layers(model, cfg.MISC.FROZEN_LAYERS)
        print(f"Frozen layers: {cfg.MISC.FROZEN_LAYERS}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    bbox_params, class_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "box_predictor.bbox_pred" in name:
            bbox_params.append(param)
        elif "box_predictor.cls_score" in name:
            class_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": bbox_params, "lr": cfg.TRAIN.BASE_LR_BBOX},
            {"params": class_params, "lr": cfg.TRAIN.BASE_LR_CLASS},
            {"params": other_params, "lr": cfg.TRAIN.BASE_LR},
        ],
        momentum=0.9,
        weight_decay=0.001,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    main_scheduler = MultiStepLR(optimizer, milestones=cfg.TRAIN.MILESTONES, gamma=cfg.TRAIN.GAMMA)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.TRAIN.NUM_EPOCHS, eta_min=1e-5
    )
    if cfg.TRAIN.SCHEDULER == "main_scheduler":
        scheduler = main_scheduler
    elif cfg.TRAIN.SCHEDULER == "secondary_scheduler":
        scheduler = cosine_scheduler
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.TRAIN.SCHEDULER}")

    # ── Output directories ─────────────────────────────────────────────────
    os.makedirs(cfg.MISC.SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(cfg.MISC.RESULTS_CSV, exist_ok=True)

    # ── Training function dispatch ─────────────────────────────────────────
    if model_type == "1d":
        from softlabel_od.object_detection.core.training.train_1d import train_and_evaluate_model_1d as train_fn
    elif model_type == "2d":
        from softlabel_od.object_detection.core.training.train_2d import train_and_evaluate_model_2d as train_fn
    elif model_type == "hard":
        from softlabel_od.object_detection.core.training.train_hard import train_and_evaluate_model_hard as train_fn
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    train_fn(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=cfg.TRAIN.NUM_EPOCHS,
        device=device,
        early_stopping=cfg.TRAIN.EARLY_STOPPING,
        scheduler=scheduler,
        config_name=cfg.MISC.RUN_NAME,
    )

    # ── Save model + config ────────────────────────────────────────────────
    if cfg.MISC.SAVE_MODEL_PATH:
        save_model(model, os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME))
        print(f"Model saved to: {cfg.MISC.SAVE_MODEL_PATH}/{cfg.MISC.RUN_NAME}.pth")

    config_save_path = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + "_runConfig.yaml")
    with open(config_save_path, "w") as f:
        f.write(cfg.dump())
    print(f"Config saved to: {config_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VinDr-CXR object detection model")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    cfg = get_cfg_from_file(args.config)
    cfg.defrost()
    cfg.CONFIG_FILE_PATH = args.config
    cfg.freeze()
    main(cfg)
