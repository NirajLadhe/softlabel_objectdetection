import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from softlabel_od.adapters.dlcv.config import get_cfg_from_file
from softlabel_od.adapters.dlcv.lightning_hard_datamodule import VinDrHardDataModule
from softlabel_od.adapters.dlcv.lightning_hard_module import HardLabelLightningModule
from softlabel_od.adapters.dlcv.lightning_metric_logger_hard import HardJSONHistoryLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/dlcv/configs/vindr_cxr_hard_label_linux.yaml",
    )
    args = parser.parse_args()

    cfg = get_cfg_from_file(args.config)
    pl.seed_everything(42, workers=True)

    dm = VinDrHardDataModule(cfg)
    model = HardLabelLightningModule(cfg)

    logger = TensorBoardLogger(save_dir=cfg.MISC.RESULTS_CSV, name=cfg.MISC.RUN_NAME)

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
        HardJSONHistoryLogger(cfg, print_summary=True),
    ]

    if cfg.TRAIN.EARLY_STOPPING:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=5,
                min_delta=0.0,
            )
        )

    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=cfg.TRAIN.NUM_EPOCHS,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
        precision=32,
        overfit_batches=cfg.TRAIN.OVERFIT_BATCHES,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()