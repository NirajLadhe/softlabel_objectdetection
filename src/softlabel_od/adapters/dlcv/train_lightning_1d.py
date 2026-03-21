import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from softlabel_od.adapters.dlcv.config import get_cfg_from_file
from softlabel_od.adapters.dlcv.lightning_1d_datamodule import VinDr1DDataModule
from softlabel_od.adapters.dlcv.lightning_1d_module import SoftLabel1DLightningModule
from softlabel_od.adapters.dlcv.lightning_metrics_logger import Phase1JSONHistoryLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="src/dlcv/configs/vindr_cxr_1d_config.yaml",
    )
    args = parser.parse_args()

    cfg = get_cfg_from_file(args.config)

    # Optional reproducibility (won't change dataset logic; only random seeds)
    pl.seed_everything(42, workers=True)

    dm = VinDr1DDataModule(cfg)
    model = SoftLabel1DLightningModule(cfg)

    # Logging & checkpoints (replaces manual torch.save / json dumping)
    logger = TensorBoardLogger(
        save_dir=cfg.MISC.RESULTS_CSV, 
        name=cfg.MISC.RUN_NAME,
        default_hp_metric=False  # Optional: disable default hparams metric
    )
    # tensorboard --logdir="/home/gagu9230@in.uni-weimar.de/ObjectDetection_SoftLabel/src/dlcv/results"

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
        Phase1JSONHistoryLogger(cfg, print_summary=True),
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
        deterministic=False,
        precision=32,
        overfit_batches=cfg.TRAIN.OVERFIT_BATCHES,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()