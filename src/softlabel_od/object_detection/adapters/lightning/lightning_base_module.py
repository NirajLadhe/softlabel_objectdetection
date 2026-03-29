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
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, CosineAnnealingLR
from softlabel_od.object_detection.core.model.factory import get_model
from softlabel_od.object_detection.utils.training_helpers import (
    load_pretrained_weights,
    freeze_layers,
)
from softlabel_od.object_detection.utils.config import resolve_label_mode


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
        self._model_configured = False
        self._warmup_iters = None
        self.model = get_model(
            num_classes=cfg.MODEL.NUM_CLASSES,
            model_type=model_type,
            backbone_name=cfg.MODEL.BACKBONE,
            pretrained=True,
            probabilistic=cfg.MODEL.PROBABILISTIC,
        )

    # ------------------------------------------------------------------
    # Model configuration — called once after fit() starts
    # ------------------------------------------------------------------

    def on_fit_start(self):
        if self._model_configured:
            return
        dm = self.trainer.datamodule
        if dm is None or dm.train_dataset is None:
            raise RuntimeError("DataModule.train_dataset must exist before on_fit_start().")
        train_dataset = dm.train_dataset
        device = self.device
        if self.cfg.MISC.PRETRAINED_WEIGHTS:
            self.model = load_pretrained_weights(
                self.model,
                self.cfg.MISC.PRETRAINED_WEIGHTS,
                device,
            )
            print(f"Loaded pretrained weights from: {self.cfg.MISC.PRETRAINED_WEIGHTS}")
        class_weights = train_dataset.compute_roi_head_class_weights_per_cluster(
            num_classes=self.cfg.MODEL.NUM_CLASSES,
            presence_prob_thresh=0.0,
            normalize=False,
            verbose=True,
        )

        if "class_weights" in self.model.roi_heads._buffers:
            self.model.roi_heads._buffers["class_weights"] = class_weights.to(device)
        else:
            self.model.roi_heads.register_buffer("class_weights", class_weights.to(device))
        print("Registered class_weights buffer on roi_heads.")

        if self.cfg.MISC.FROZEN_LAYERS:
            freeze_layers(self.model, self.cfg.MISC.FROZEN_LAYERS)
            print(f"Frozen layers: {self.cfg.MISC.FROZEN_LAYERS}")

        self._warmup_iters = 1000 #min(1000, max(int(self.trainer.num_training_batches) - 1, 1))
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
        bbox_params, class_params, other_params = [], [], []
        for name, param in self.model.named_parameters():
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
                {"params": bbox_params, "lr": self.cfg.TRAIN.BASE_LR_BBOX},
                {"params": class_params, "lr": self.cfg.TRAIN.BASE_LR_CLASS},
                {"params": other_params, "lr": self.cfg.TRAIN.BASE_LR},
            ],
            momentum=0.9,
            weight_decay=0.001,
        )

        def warmup_lambda(step: int):
            warmup_iters = self._warmup_iters if self._warmup_iters is not None else 1
            if step >= warmup_iters:
                return 1.0
            alpha = float(step) / float(max(warmup_iters, 1))
            return (1.0 / 1000.0) * (1.0 - alpha) + alpha
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda)
        main_scheduler = MultiStepLR(
            optimizer,
            milestones=list(self.cfg.TRAIN.MILESTONES),
            gamma=self.cfg.TRAIN.GAMMA,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.TRAIN.NUM_EPOCHS,
            eta_min=1e-5,
        )
        if self.cfg.TRAIN.SCHEDULER == "main_scheduler":
            epoch_scheduler = main_scheduler
        elif self.cfg.TRAIN.SCHEDULER == "secondary_scheduler":
            epoch_scheduler = cosine_scheduler
        else:
            raise ValueError(f"Unsupported scheduler: {self.cfg.TRAIN.SCHEDULER}")
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": [
        #         {
        #             "scheduler": warmup_scheduler,
        #             "interval": "step",
        #             "frequency": 1,
        #             "name": "warmup",
        #         },
        #         {
        #             "scheduler": epoch_scheduler,
        #             "interval": "epoch",
        #             "frequency": 1,
        #             "name": "epoch_lr",
        #         },
        #     ],
        # }
        return (
            [optimizer],
            [
                {"scheduler": warmup_scheduler, "interval": "step", "frequency": 1, "name": "warmup"},
                {"scheduler": epoch_scheduler, "interval": "epoch", "frequency": 1, "name": "epoch_lr"},
            ],
        )