"""
SoftLabel2DLightningModule — 2D GMM vertex soft-label training.

Inherits all model setup from BaseDetectionLightningModule.
Validation matches validate_soft_labels_2d() in train_2d.py.
"""

import torch

from softlabel_od.object_detection.adapters.lightning.lightning_base_module import (
    BaseDetectionLightningModule,
)


class SoftLabel2DLightningModule(BaseDetectionLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg, model_type="2d")

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images  = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_out  = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)
        total_loss = sum(loss_dict.values())

        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images  = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        was_training = self.model.training
        self.model.train()
        loss_out  = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        device    = images[0].device
        val_total = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=device)
        val_cls   = loss_dict.get(
            "loss_probabilistic_classifier", torch.tensor(0.0, device=device)
        )
        val_vtx   = loss_dict.get(
            "loss_probabilistic_vertex_reg", torch.tensor(0.0, device=device)
        )

        self.log("val/loss",           val_total, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/class_loss",     val_cls,   on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/vertex_reg_loss",val_vtx,   on_step=False, on_epoch=True, prog_bar=False)

        self.model.train(was_training)