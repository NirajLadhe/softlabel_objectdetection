"""
Object detection training orchestrator (vanilla PyTorch).

Loads config → builds dataset → builds model → trains → saves.
This is the single authoritative entry point for vanilla PyTorch training.
"""

import os
import torch

from softlabel_od.object_detection.utils.config import get_cfg_from_file
from softlabel_od.object_detection.utils.training_helpers import save_model, OverfitDataLoader
from softlabel_od.object_detection.training_setup import (
    resolve_model_type,
    build_datasets,
    build_dataloaders,
    build_model,
    configure_model_for_training,
    build_optimizer,
    build_scheduler,
)


def main(cfg):
    model_type = resolve_model_type(cfg)

    if not os.path.exists(cfg.DATA.ROOT):
        print(f"Warning: Dataset root not found: {cfg.DATA.ROOT}")

    print(f"Config: {getattr(cfg, 'CONFIG_FILE_PATH', 'N/A')}")
    print(f"Dataset: {cfg.DATA.DATASET}")
    print(cfg.dump())

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not cfg.MISC.NO_CUDA else "cpu"
    )
    print(f"Device: {device}")

    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_dataset, val_dataset)

    # ── Overfit mode ──────────────────────────────────────────────────────────
    overfit_batches = getattr(cfg.TRAIN, "OVERFIT_BATCHES", 0)
    if overfit_batches > 0:
        train_loader = OverfitDataLoader(
            train_loader, num_batches=overfit_batches, move_to_device=device
        )
        val_loader = train_loader

    print(f"Train: {len(train_dataset)} images  |  Val: {len(val_dataset)} images")

    model = build_model(cfg, model_type, device)
    model = configure_model_for_training(cfg, model, train_dataset, device)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # ── Output directories ────────────────────────────────────────────────────
    os.makedirs(cfg.MISC.SAVE_MODEL_PATH, exist_ok=True)
    os.makedirs(cfg.MISC.RESULTS_CSV, exist_ok=True)

    # ── Training function dispatch ────────────────────────────────────────────
    if model_type == "1d":
        from softlabel_od.object_detection.core.training.train_1d import (
            train_and_evaluate_model_1d as train_fn,
        )
    elif model_type == "2d":
        from softlabel_od.object_detection.core.training.train_2d import (
            train_and_evaluate_model_2d as train_fn,
        )
    elif model_type == "hard":
        from softlabel_od.object_detection.core.training.train_hard import (
            train_and_evaluate_model_hard as train_fn,
        )
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

    # ── Save model + config ───────────────────────────────────────────────────
    if cfg.MISC.SAVE_MODEL_PATH:
        save_model(model, os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME))
        print(f"Model saved to: {cfg.MISC.SAVE_MODEL_PATH}/{cfg.MISC.RUN_NAME}.pth")

    config_save_path = os.path.join(
        cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME + "_runConfig.yaml"
    )
    with open(config_save_path, "w") as f:
        f.write(cfg.dump())
    print(f"Config saved to: {config_save_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = get_cfg_from_file(args.config)
    cfg.defrost()
    cfg.CONFIG_FILE_PATH = args.config
    cfg.freeze()
    main(cfg)