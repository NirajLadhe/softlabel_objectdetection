"""
CLI entry point: softlabel-train-lightning

PyTorch Lightning training path (experimental — parallel to vanilla PyTorch).
Dispatches to 1D or hard-label Lightning module based on config GMM_TYPE.
"""

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from softlabel_od.object_detection.utils.config import get_cfg_from_file

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "vindr_cxr_1d_config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Train VinDr-CXR model (PyTorch Lightning)")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to config YAML")
    args = parser.parse_args()

    cfg = get_cfg_from_file(args.config)

    gmm_type = cfg.DATA.GMM_TYPE
    if gmm_type == "HARD_LABEL":
        from softlabel_od.object_detection.adapters.lightning.lightning_hard_datamodule import VinDrHardDataModule
        from softlabel_od.object_detection.adapters.lightning.lightning_hard_module import HardLabelLightningModule
        from softlabel_od.object_detection.adapters.lightning.lightning_metric_logger_hard import HardJSONHistoryLogger
        dm_cls, model_cls, logger_cls = VinDrHardDataModule, HardLabelLightningModule, HardJSONHistoryLogger
    else:
        from softlabel_od.object_detection.adapters.lightning.lightning_1d_datamodule import VinDr1DDataModule
        from softlabel_od.object_detection.adapters.lightning.lightning_1d_module import SoftLabel1DLightningModule
        from softlabel_od.object_detection.adapters.lightning.lightning_metrics_logger import Phase1JSONHistoryLogger
        dm_cls, model_cls, logger_cls = VinDr1DDataModule, SoftLabel1DLightningModule, Phase1JSONHistoryLogger

    pl.seed_everything(42, workers=True)
    dm = dm_cls(cfg)
    model = model_cls(cfg)

    logger = TensorBoardLogger(
        save_dir=cfg.MISC.RESULTS_CSV, name=cfg.MISC.RUN_NAME, default_hp_metric=False
    )
    ckpt_dir = os.path.join(cfg.MISC.SAVE_MODEL_PATH, cfg.MISC.RUN_NAME)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="epoch{epoch:03d}-valloss{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
        logger_cls(cfg, print_summary=True),
    ]
    if cfg.TRAIN.EARLY_STOPPING:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=5, min_delta=0.0))

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        deterministic=False,
        precision=32,
        overfit_batches=cfg.TRAIN.OVERFIT_BATCHES,
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
