"""
CLI entry point: softlabel-train-lightning

Parallel alternative to softlabel-train. Uses the Lightning training path
on top of the same training_setup assembly as the legacy PyTorch path.
"""

import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from softlabel_od.object_detection.utils.config import get_cfg_from_file, resolve_label_mode
from softlabel_od.object_detection.core.lightning.lightning_datamodule import (
    DetectionLightningDataModule,
)
from softlabel_od.object_detection.core.lightning.lightning_callbacks import (
    ExportBestPthCallback,
)
from softlabel_od.object_detection.core.lightning.lightning_theta_module import (
    SoftLabel2DThetaLightningModule,
)

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT  = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "1d_softlabel_config_albers.yaml")

def _get_module_class(label_mode: str):
    if label_mode == "soft_1d":
        from softlabel_od.object_detection.core.lightning.lightning_1d_module import (
            SoftLabel1DLightningModule,
        )
        return SoftLabel1DLightningModule
    if label_mode == "soft_2d":
        from softlabel_od.object_detection.core.lightning.lightning_2d_module import (
            SoftLabel2DLightningModule,
        )
        return SoftLabel2DLightningModule
    if label_mode == "soft_2d_theta":
        from softlabel_od.object_detection.core.lightning.lightning_theta_module import (
            SoftLabel2DThetaLightningModule,
        )
        return SoftLabel2DThetaLightningModule
    if label_mode == "hard":
        from softlabel_od.object_detection.core.lightning.lightning_hard_module import (
            HardLabelLightningModule,
        )
        return HardLabelLightningModule
    raise ValueError(f"Unsupported label mode for Lightning path: {label_mode!r}")


def _get_history_logger(label_mode: str, cfg):
    if label_mode == "soft_1d":
        from softlabel_od.object_detection.core.lightning.lightning_metrics_logger import (
            Phase1JSONHistoryLogger,
        )
        return Phase1JSONHistoryLogger(cfg, print_summary=True)
    if label_mode == "hard":
        from softlabel_od.object_detection.core.lightning.lightning_metric_logger_hard import (
            HardJSONHistoryLogger,
        )
        return HardJSONHistoryLogger(cfg, print_summary=True)
    # 2D: fall back to a minimal logger until a dedicated one is added
    from softlabel_od.object_detection.core.lightning.lightning_metric_logger_hard import (
        HardJSONHistoryLogger,
    )
    return HardJSONHistoryLogger(cfg, print_summary=True)


def main():
    parser = argparse.ArgumentParser(description="Train model (PyTorch Lightning)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    cfg = get_cfg_from_file(args.config)
    cfg.defrost()
    cfg.CONFIG_FILE_PATH = args.config
    cfg.freeze()

    label_mode  = resolve_label_mode(cfg)
    module_cls  = _get_module_class(label_mode)

    pl.seed_everything(42, workers=True)

    dm     = DetectionLightningDataModule(cfg)
    model  = module_cls(cfg)

    # -- Loggers ---------------------------------------------------------
    tb_logger = TensorBoardLogger(
        save_dir=cfg.MISC.RESULTS_CSV,
        name=cfg.MISC.RUN_NAME,
        default_hp_metric=False,
    )

    # -- Callbacks -------------------------------------------------------
    ckpt_dir = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(cfg.MISC.RESULTS_CSV, exist_ok=True)

    callbacks = [
        _get_history_logger(label_mode, cfg),
        ExportBestPthCallback(cfg),
    ]

    if cfg.TRAIN.EARLY_STOPPING:
        if label_mode == "soft_1d":
            # Use the composite early stopping matching train_1d.py exactly
            from softlabel_od.object_detection.core.lightning.lightning_callbacks import (
                CompositeEarlyStopping1D,
            )
            callbacks.append(CompositeEarlyStopping1D())
        else:
            callbacks.append(
                EarlyStopping(monitor="val/loss", mode="min", patience=5, min_delta=0.0)
            )

    # -- Trainer ---------------------------------------------------------
    trainer = pl.Trainer(
        accelerator=cfg.LIGHTNING.ACCELERATOR,
        devices=cfg.LIGHTNING.DEVICES,
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        accumulate_grad_batches=cfg.LIGHTNING.ACCUMULATE_GRAD_BATCHES,
        logger=tb_logger,
        callbacks=callbacks,
        strategy=DDPStrategy(find_unused_parameters=True),
        enable_checkpointing=cfg.LIGHTNING.ENABLE_CHECKPOINTING,    
        enable_progress_bar=cfg.LIGHTNING.ENABLE_PROGRESS_BAR,
        deterministic=cfg.LIGHTNING.DETERMINISTIC,
        precision=cfg.LIGHTNING.PRECISION,
        gradient_clip_val=3.0, # From 5 to 3 for nucls
        gradient_clip_algorithm="norm",
        # Overfit is handled at the DataModule level, not here.
        # Leave overfit_batches=0 so Lightning doesn't override the dataloader.
        overfit_batches=0,
        num_sanity_val_steps=cfg.LIGHTNING.NUM_SANITY_VAL_STEPS,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()