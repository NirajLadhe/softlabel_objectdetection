"""
SoftLabel1DLightningModule — 1D probabilistic soft-label training.

Inherits all model setup, optimizer, scheduler, warmup, and grad clipping
from BaseDetectionLightningModule.  Only 1D-specific step logic lives here.
"""

import numpy as np
import torch
import pytorch_lightning as pl

from softlabel_od.object_detection.adapters.lightning.lightning_base_module import (
    BaseDetectionLightningModule,
)
from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_calibration_error,
    compute_uncertainty_weighted_map,
    compute_soft_detection_metrics,
    compute_class_distribution_metrics,
    compute_expected_iou,
    compute_bbox_uncertainty_calibration,
    match_predictions_to_gt,
)


class SoftLabel1DLightningModule(BaseDetectionLightningModule):
    """
    LightningModule for 1D soft-label probabilistic Faster R-CNN.

    Validation follows the same two-pass pattern as validate_soft_labels_1d_quick():
      pass 1 — model.train() + no_grad  → loss dict
      pass 2 — model.eval()             → inference for metrics
    """

    def __init__(self, cfg):
        super().__init__(cfg, model_type="1d")
        self._val_predictions = []
        self._val_ground_truths = []

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        total_loss = sum(loss_dict.values())

        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss

    # ------------------------------------------------------------------
    # Validation step  — identical two-pass pattern to legacy train_1d.py
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        # Pass 1: losses (model must be in train mode for torchvision loss API)
        was_training = self.model.training
        self.model.train()
        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        # Named summaries matching legacy metric names
        device = images[0].device
        val_class_loss = loss_dict.get(
            "loss_probabilistic_classifier", torch.tensor(0.0, device=device)
        )
        val_reg_loss   = loss_dict.get("loss_box_mean",  torch.tensor(0.0, device=device))
        val_std_loss   = loss_dict.get("loss_box_var",   torch.tensor(0.0, device=device))
        val_rpn_obj    = loss_dict.get("loss_objectness",    torch.tensor(0.0, device=device))
        val_rpn_box    = loss_dict.get("loss_rpn_box_reg",   torch.tensor(0.0, device=device))

        val_total_loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=device)

        self.log("val/loss",        val_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/class_loss",  val_class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/reg_loss",    val_reg_loss,   on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/std_loss",    val_std_loss,   on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/rpn_objectness", val_rpn_obj, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/rpn_box_reg",    val_rpn_box, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Pass 2: inference for epoch-end metrics
        self.model.train(was_training)
        self.model.eval()
        preds = self(images)

        for pred, target in zip(preds, targets):
            pred_entry = {
                "boxes":  pred["boxes"].detach().cpu().numpy(),
                "labels": pred["labels"].detach().cpu().numpy(),
                "scores": pred["scores"].detach().cpu().numpy(),
            }
            if "bbox_stds" in pred:
                pred_entry["bbox_stds"]  = pred["bbox_stds"].detach().cpu().numpy()
                pred_entry["bbox_means"] = pred["boxes"].detach().cpu().numpy()
            if "class_probs" in pred:
                pred_entry["class_probs"] = pred["class_probs"].detach().cpu().numpy()

            gt_entry = {
                "boxes":       target["boxes"].detach().cpu().numpy(),
                "labels":      target["labels"].detach().cpu().numpy(),
                "image_size":  (images[0].shape[1], images[0].shape[2]),
            }
            if "gt_class_probs" in target:
                gt_entry["gt_class_probs"] = target["gt_class_probs"].detach().cpu().numpy()
            if "gt_stds" in target:
                gt_entry["gt_stds"] = target["gt_stds"].detach().cpu().numpy()

            self._val_predictions.append(pred_entry)
            self._val_ground_truths.append(gt_entry)

    # ------------------------------------------------------------------
    # Epoch-end metrics — port of the metric block in train_1d.py
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self):
        preds = self._val_predictions
        gts   = self._val_ground_truths
        self._val_predictions  = []
        self._val_ground_truths = []

        if not preds:
            return

        ece         = compute_calibration_error(preds, gts)
        uw_map      = compute_uncertainty_weighted_map(preds, gts)
        dist_metric = compute_class_distribution_metrics(preds, gts)
        soft_metric = compute_soft_detection_metrics(preds, gts)
        expected_iou = compute_expected_iou(preds, gts, num_samples=50)
        bbox_calib  = compute_bbox_uncertainty_calibration(preds, gts)

        total_ious = []
        for pred, gt in zip(preds, gts):
            if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
                continue
            matches, _, _ = match_predictions_to_gt(
                pred["boxes"], pred["scores"], gt["boxes"], iou_threshold=0.5
            )
            for m in matches:
                total_ious.append(m[2])
        avg_iou = float(np.mean(total_ious)) if total_ious else 0.0

        self.log("val/avg_iou",          avg_iou,                               prog_bar=False, sync_dist=True)
        self.log("val/expected_iou",     float(expected_iou),                   prog_bar=False, sync_dist=True)
        self.log("val/ece",              float(ece),                            prog_bar=False, sync_dist=True)
        self.log("val/uw_map",           float(uw_map),                         prog_bar=False, sync_dist=True)
        self.log("val/top1_accuracy",    float(soft_metric["top1_accuracy"]),   prog_bar=False, sync_dist=True)
        self.log("val/top3_accuracy",    float(soft_metric["top3_accuracy"]),   prog_bar=False, sync_dist=True)
        self.log("val/weighted_accuracy",float(soft_metric["weighted_accuracy"]),prog_bar=False, sync_dist=True)
        self.log("val/kl_divergence",    float(dist_metric["kl_divergence"]),   prog_bar=False, sync_dist=True)
        self.log("val/js_divergence",    float(dist_metric["js_divergence"]),   prog_bar=False, sync_dist=True)
        self.log("val/cosine_similarity",float(dist_metric["cosine_similarity"]),prog_bar=False, sync_dist=True)