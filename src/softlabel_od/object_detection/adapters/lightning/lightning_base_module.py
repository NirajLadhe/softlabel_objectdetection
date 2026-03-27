"""
BaseDetectionLightningModule — shared model setup for all label modes.

Handles:
  - model construction (via training_setup.build_model)
  - pretrained loading + frozen layers + RoI class-weight buffer
    (via training_setup.configure_model_for_training)
  - optimizer / scheduler construction (via training_setup)
  - epoch-0 step warmup matching train_epoch.py exactly
  - gradient clipping matching train_epoch.py max_norm=5.0

Subclasses implement training_step / validation_step / on_validation_epoch_end.
"""

import torch
import pytorch_lightning as pl

from softlabel_od.object_detection.training_setup import (
    build_model,
    configure_model_for_training,
    build_optimizer,
    build_scheduler,
)


class BaseDetectionLightningModule(pl.LightningModule):
    """
    Args:
        cfg:        Frozen YACS config node.
        model_type: '1d', '2d', or 'hard'.
    """

    def __init__(self, cfg, model_type: str):
        super().__init__()
        self.cfg = cfg
        self.model_type = model_type

        # Build model WITHOUT moving to device; Lightning handles device placement.
        self.model = build_model(cfg, model_type, device=None)

        # configure_model_for_training needs the train dataset for class weights.
        # We defer that to on_fit_start() when the datamodule is already set up.
        self._model_configured = False

        # Warmup state (epoch-0 only, step-based, matching train_epoch.py)
        self._base_lrs = None
        self._warmup_factor = 1.0 / 1000.0

    # ------------------------------------------------------------------
    # Model configuration — called once after fit() starts
    # ------------------------------------------------------------------

    def on_fit_start(self):
        """
        Configure model after Lightning has placed it on the target device.
        Requires the datamodule to already have called setup() so that
        self.trainer.datamodule.train_dataset is available.
        """
        if self._model_configured:
            return

        train_dataset = self._get_train_dataset()
        device = self.device

        # This applies pretrained loading, class_weights buffer, frozen layers.
        configure_model_for_training(self.cfg, self.model, train_dataset, device)
        self._model_configured = True

    def _get_train_dataset(self):
        """Retrieve the training dataset from the Lightning datamodule."""
        dm = self.trainer.datamodule
        if dm is None or dm.train_dataset is None:
            raise RuntimeError(
                "DataModule.train_dataset is None when on_fit_start() is called. "
                "Make sure DetectionLightningDataModule.setup() is called first."
            )
        return dm.train_dataset

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, images, targets=None):
        if targets is None:
            return self.model(images)
        return self.model(images, targets)

    # ------------------------------------------------------------------
    # Loss dict unpacking
    # ------------------------------------------------------------------

    @staticmethod
    def _unwrap_loss_dict(out):
        """
        torchvision detection models return a dict in training mode.
        Some variants wrap it in a (result, loss_dict) tuple.
        """
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out[1]
        if isinstance(out, dict):
            return out
        raise TypeError(f"Unexpected model output type: {type(out)}")

    # ------------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        scheduler = build_scheduler(self.cfg, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ------------------------------------------------------------------
    # Epoch-0 warmup — matches train_epoch.py behaviour exactly
    #
    # train_epoch.py:
    #   warmup_factor = 1/1000
    #   warmup_iters  = min(1000, len(data_loader) - 1)
    #   step-wise; resets after epoch 0
    # ------------------------------------------------------------------

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, **kwargs):
        if self._base_lrs is None:
            self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        if self.current_epoch == 0 and self.trainer is not None:
            warmup_iters = min(
                1000, max(int(self.trainer.num_training_batches) - 1, 1)
            )
            x = min(self.global_step, warmup_iters)
            alpha = float(x) / float(max(warmup_iters, 1))
            scale = self._warmup_factor * (1.0 - alpha) + alpha
            for pg, base_lr in zip(optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    # ------------------------------------------------------------------
    # Gradient clipping — matches train_epoch.py max_norm=5.0
    #
    # Trainer must be created with gradient_clip_val=None so Lightning
    # does not clip independently; we apply clipping here manually just
    # before the step so the order matches legacy exactly.
    # ------------------------------------------------------------------

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)