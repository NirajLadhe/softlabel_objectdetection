"""
SoftLabel2DThetaLightningModule — 4x4 Sigma_theta soft-label training.

Rewritten to mirror the REAL, confirmed lightning_1d_module.py's
validation_step / on_validation_epoch_end structure exactly (pasted
directly by the user, not inferred) -- flat list buffering, numpy
(.detach().cpu().numpy()) prediction/GT entries, conditional field
inclusion via `if "field" in pred`.

Field-name correspondence to the 1D module, confirmed:
    1D:    "bbox_stds"  [N,4]    <-> theta: "bbox_covs"  [N,4,4]  (NEW field,
                                            not reusing "bbox_stds" since
                                            the shape differs)
    1D:    "bbox_means" = pred["boxes"] (NOT a separate model output --
                                          reused directly, confirmed)
                                          -- same reuse pattern kept here
    1D:    "gt_stds"    [N,4]    <-> theta: "gt_theta_cov" [N,4,4]
                                          (matches the field name already
                                          used in base.py's
                                          _process_theta_2d_labels target
                                          dict -- passed straight through)

compute_calibration_error, compute_class_distribution_metrics,
compute_soft_detection_metrics, match_predictions_to_gt are reused
UNMODIFIED -- confirmed dimension/representation-agnostic (operate on
class_probs and/or boxes+scores only, no dependency on 1D vs theta
spatial representation).

compute_uncertainty_weighted_map is also called unmodified for now --
internals not yet seen; flagged below as needing verification, same as
match_predictions_to_gt, before this is trusted in a real training run.

compute_expected_iou / compute_bbox_uncertainty_calibration are replaced
with compute_expected_iou_theta / compute_nees_theta (see
new_metrics_functions.py), since those two are confirmed to hardcode
independent-axis (bbox_stds [N,4]) sampling/comparison logic incompatible
with a [N,4,4] covariance.
"""

import numpy as np
import torch

from softlabel_od.object_detection.core.lightning.lightning_base_module import (
    BaseDetectionLightningModule,
)
from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_calibration_error,
    compute_uncertainty_weighted_map,
    compute_class_distribution_metrics,
    compute_soft_detection_metrics,
    compute_expected_iou_theta,
    compute_nees_theta,
    match_predictions_to_gt,
)


class SoftLabel2DThetaLightningModule(BaseDetectionLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg, model_type="theta_2d")
        self._val_predictions = []
        self._val_ground_truths = []

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

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        # ── Pass 1: losses (model must be in train mode for torchvision loss API) ──
        was_training = self.model.training
        self.model.train()
        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        # Named summaries -- theta-equivalent of the 1D module's legacy names.
        # RPN keys (loss_objectness, loss_rpn_box_reg) are unchanged: the RPN
        # is identical regardless of head type, confirmed from faster_rcnn_1d.py.
        device = images[0].device
        val_class_loss = loss_dict.get(
            "loss_probabilistic_classifier", torch.tensor(0.0, device=device)
        )
        val_theta_mean_loss = loss_dict.get("loss_theta_mean", torch.tensor(0.0, device=device))
        val_theta_cov_loss = loss_dict.get("loss_theta_cov", torch.tensor(0.0, device=device))
        val_rpn_obj = loss_dict.get("loss_objectness", torch.tensor(0.0, device=device))
        val_rpn_box = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0, device=device))

        val_total_loss = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=device)

        self.log("val/loss", val_total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/class_loss", val_class_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/theta_mean_loss", val_theta_mean_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/theta_cov_loss", val_theta_cov_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/rpn_objectness", val_rpn_obj, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/rpn_box_reg", val_rpn_box, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # ── Pass 2: inference for epoch-end metrics ──
        self.model.train(was_training)
        self.model.eval()
        preds = self(images)

        for pred, target in zip(preds, targets):
            pred_entry = {
                "boxes": pred["boxes"].detach().cpu().numpy(),
                "labels": pred["labels"].detach().cpu().numpy(),
                "scores": pred["scores"].detach().cpu().numpy(),
            }
            if "bbox_covs" in pred:
                pred_entry["bbox_covs"] = pred["bbox_covs"].detach().cpu().numpy()
                pred_entry["bbox_means"] = pred["boxes"].detach().cpu().numpy()
            if "class_probs" in pred:
                pred_entry["class_probs"] = pred["class_probs"].detach().cpu().numpy()

            gt_entry = {
                "boxes": target["boxes"].detach().cpu().numpy(),
                "labels": target["labels"].detach().cpu().numpy(),
                "image_size": (images[0].shape[1], images[0].shape[2]),
            }
            if "gt_class_probs" in target:
                gt_entry["gt_class_probs"] = target["gt_class_probs"].detach().cpu().numpy()
            if "gt_theta_cov" in target:
                gt_entry["gt_theta_cov"] = target["gt_theta_cov"].detach().cpu().numpy()

            self._val_predictions.append(pred_entry)
            self._val_ground_truths.append(gt_entry)

    def on_validation_epoch_end(self):
        preds = self._val_predictions
        gts = self._val_ground_truths
        self._val_predictions = []
        self._val_ground_truths = []

        if not preds:
            return

        # ── Class-distribution + generic metrics -- REUSED UNMODIFIED ──
        # Confirmed dimension-agnostic (class_probs only, or boxes+scores only).
        ece = compute_calibration_error(preds, gts)
        uw_map = compute_uncertainty_weighted_map(preds, gts)   # NOT YET independently verified -- see note below
        dist_metric = compute_class_distribution_metrics(preds, gts)
        soft_metric = compute_soft_detection_metrics(preds, gts)

        # ── Spatial/uncertainty metrics -- theta-specific versions ──
        expected_iou = compute_expected_iou_theta(preds, gts, num_samples=50)
        nees_result = compute_nees_theta(preds, gts)

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

        self.log("val/avg_iou", avg_iou, prog_bar=False, sync_dist=True)
        self.log("val/expected_iou", float(expected_iou), prog_bar=False, sync_dist=True)
        self.log("val/nees", float(nees_result["mean_nees"]), prog_bar=True, sync_dist=True)
        self.log("val/ece", float(ece), prog_bar=False, sync_dist=True)
        self.log("val/uw_map", float(uw_map), prog_bar=False, sync_dist=True)
        self.log("val/top1_accuracy", float(soft_metric["top1_accuracy"]), prog_bar=False, sync_dist=True)
        self.log("val/top3_accuracy", float(soft_metric["top3_accuracy"]), prog_bar=False, sync_dist=True)
        self.log("val/weighted_accuracy", float(soft_metric["weighted_accuracy"]), prog_bar=False, sync_dist=True)
        self.log("val/kl_divergence", float(dist_metric["kl_divergence"]), prog_bar=False, sync_dist=True)
        self.log("val/js_divergence", float(dist_metric["js_divergence"]), prog_bar=False, sync_dist=True)
        self.log("val/cosine_similarity", float(dist_metric["cosine_similarity"]), prog_bar=False, sync_dist=True)

        # NOTE: val/nees given prog_bar=True (deviating from the 1D module's
        # all-False convention in this method) specifically so NEES's target
        # value of 4.0 -- not 0 -- stays visible every epoch without digging
        # into logs, given how easy this specific metric is to misread.
