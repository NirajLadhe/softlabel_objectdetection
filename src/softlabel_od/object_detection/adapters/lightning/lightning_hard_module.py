"""
HardLabelLightningModule — hard-label Faster R-CNN training.

Inherits all model setup from BaseDetectionLightningModule.
Validation matches validate_hard_labels() in train_hard.py.
"""

import numpy as np
import torch

from softlabel_od.object_detection.adapters.lightning.lightning_base_module import (
    BaseDetectionLightningModule,
)
from softlabel_od.object_detection.core.evaluation.metrics import match_predictions_to_gt


class HardLabelLightningModule(BaseDetectionLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg, model_type="hard")

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Validation step — matches validate_hard_labels() in train_hard.py
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images  = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        # Pass 1: losses
        was_training = self.model.training
        self.model.train()
        loss_out  = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        device = images[0].device
        val_total = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=device)
        val_cls   = loss_dict.get("loss_classifier",  torch.tensor(0.0, device=device))
        val_reg   = loss_dict.get("loss_box_reg",     torch.tensor(0.0, device=device))
        val_rpn_obj = loss_dict.get("loss_objectness",    torch.tensor(0.0, device=device))
        val_rpn_reg = loss_dict.get("loss_rpn_box_reg",   torch.tensor(0.0, device=device))

        self.log("val/loss",          val_total,  on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/class_loss",    val_cls,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/reg_loss",      val_reg,    on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rpn_objectness",val_rpn_obj,on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rpn_box_reg",   val_rpn_reg,on_step=False, on_epoch=True, prog_bar=False)

        # Pass 2: inference for IoU and class accuracy
        self.model.train(was_training)
        self.model.eval()
        preds = self(images)

        all_ious = []
        correct = 0
        total   = 0

        for pred, target in zip(preds, targets):
            pred_boxes  = pred["boxes"].cpu().numpy()
            pred_scores = pred["scores"].cpu().numpy()
            pred_labels = pred["labels"].cpu().numpy()
            gt_boxes    = target["boxes"].cpu().numpy()
            gt_labels   = target["labels"].cpu().numpy()

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                matches, _, _ = match_predictions_to_gt(
                    pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
                )
                for m in matches:
                    all_ious.append(m[2])

                for p_box, p_label in zip(pred_boxes, pred_labels):
                    ious = [
                        self._iou(p_box, g_box) for g_box in gt_boxes
                    ]
                    best_idx = int(np.argmax(ious))
                    if ious[best_idx] > 0.5:
                        total += 1
                        if int(p_label) == int(gt_labels[best_idx]):
                            correct += 1

        avg_iou  = float(np.mean(all_ious)) if all_ious else 0.0
        accuracy = float(correct / total)   if total > 0  else 0.0

        self.log("val/avg_iou",        avg_iou,  on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/class_accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=False)

    @staticmethod
    def _iou(box1, box2):
        x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
        a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0