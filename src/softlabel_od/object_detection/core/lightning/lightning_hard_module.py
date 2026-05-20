"""
HardLabelLightningModule — hard-label Faster R-CNN training.

Validation:
  - losses: logged each batch, mean-reduced by Lightning (with sync_dist)
  - IoU / matched class accuracy: global epoch-end from Hungarian matches
  - mAP / ECE: every cfg.TRAIN.MAP_EVAL_FREQ epochs (+ final epoch)
"""

import numpy as np
import torch

from softlabel_od.object_detection.core.lightning.lightning_base_module import (
    BaseDetectionLightningModule,
)
from softlabel_od.object_detection.core.evaluation.metrics import (
    match_predictions_to_gt,
    compute_map,
    compute_calibration_error,
)


class HardLabelLightningModule(BaseDetectionLightningModule):

    def __init__(self, cfg):
        super().__init__(cfg, model_type="hard")
        self._val_predictions = []
        self._val_ground_truths = []
        self._val_match_ious = []
        self._val_class_correct = 0
        self._val_class_total = 0

    def on_validation_epoch_start(self):
        self._val_predictions = []
        self._val_ground_truths = []
        self._val_match_ious = []
        self._val_class_correct = 0
        self._val_class_total = 0

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

        device = images[0].device
        cls_loss = loss_dict.get("loss_classifier", torch.tensor(0.0, device=device))
        reg_loss = loss_dict.get("loss_box_reg", torch.tensor(0.0, device=device))
        rpn_obj = loss_dict.get("loss_objectness", torch.tensor(0.0, device=device))
        rpn_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0, device=device))

        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/class_loss", cls_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/reg_loss", reg_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/rpn_objectness", rpn_obj, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/rpn_box_reg", rpn_reg, on_step=False, on_epoch=True, sync_dist=True)

        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]
        device = images[0].device

        # Pass 1: losses (train mode required by torchvision)
        was_training = self.model.training
        self.model.train()
        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        val_total = sum(loss_dict.values()) if loss_dict else torch.tensor(0.0, device=device)
        val_cls = loss_dict.get("loss_classifier", torch.tensor(0.0, device=device))
        val_reg = loss_dict.get("loss_box_reg", torch.tensor(0.0, device=device))
        val_rpn_obj = loss_dict.get("loss_objectness", torch.tensor(0.0, device=device))
        val_rpn_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0, device=device))

        self.log("val/loss", val_total, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/class_loss", val_cls, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/reg_loss", val_reg, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/rpn_objectness", val_rpn_obj, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/rpn_box_reg", val_rpn_reg, on_step=False, on_epoch=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, sync_dist=True)

        # Pass 2: inference
        self.model.train(was_training)
        self.model.eval()
        preds = self(images)

        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"].detach().cpu().numpy()
            pred_scores = pred["scores"].detach().cpu().numpy()
            pred_labels = pred["labels"].detach().cpu().numpy()
            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy()

            self._val_predictions.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            })
            self._val_ground_truths.append({
                "boxes": gt_boxes,
                "labels": gt_labels,
            })

            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue

            matches, _, _ = match_predictions_to_gt(
                pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
            )
            for pred_idx, gt_idx, iou in matches:
                self._val_match_ious.append(float(iou))
                self._val_class_total += 1
                if int(pred_labels[pred_idx]) == int(gt_labels[gt_idx]):
                    self._val_class_correct += 1

    # ------------------------------------------------------------------
    # Epoch-end metrics
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return

        world_size = self.trainer.world_size

        # --- Global matched-pair class accuracy (all ranks) ---
        correct = torch.tensor(float(self._val_class_correct), device=self.device)
        total = torch.tensor(float(self._val_class_total), device=self.device)
        if world_size > 1:
            torch.distributed.all_reduce(correct, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
        accuracy = float(correct.item() / max(total.item(), 1.0))

        # --- Gather preds / GTs / IoUs from every rank (all ranks must call this) ---
        all_preds, all_gts, all_ious = self._gather_val_lists(
            self._val_predictions,
            self._val_ground_truths,
            self._val_match_ious,
            world_size,
        )
        avg_iou = float(np.mean(all_ious)) if all_ious else 0.0

        # Log quick metrics on all ranks (same values after gather / all_reduce)
        self.log("val/avg_iou", avg_iou, prog_bar=False, sync_dist=False)
        self.log("val/class_accuracy", accuracy, prog_bar=False, sync_dist=False)

        # --- mAP / ECE: compute once on rank 0 over the full val set ---
        epoch = int(self.current_epoch) + 1
        max_epochs = int(self.trainer.max_epochs)
        map_freq = int(getattr(self.cfg.TRAIN, "MAP_EVAL_FREQ", 10))
        run_map = (epoch % map_freq == 0) or (epoch == max_epochs)
        nan = float("nan")

        if self.trainer.global_rank != 0:
            return

        if run_map and all_preds:
            try:
                map_results = compute_map(
                    all_preds,
                    all_gts,
                    iou_thresholds=[0.4, 0.5, 0.75],
                )
                ece = compute_calibration_error(all_preds, all_gts)

                self.log("val/mAP", float(map_results["mAP"]), sync_dist=False)
                self.log("val/mAP_50", float(map_results["mAP_50"]), sync_dist=False)
                self.log("val/mAP_75", float(map_results["mAP_75"]), sync_dist=False)
                self.log("val/ece", float(ece), sync_dist=False)
                if "AP_NoFinding" in map_results:
                    self.log(
                        "val/AP_NoFinding",
                        float(map_results["AP_NoFinding"]),
                        sync_dist=False,
                    )

                print(
                    f"\n  [Comprehensive @ epoch {epoch}] "
                    f"mAP={map_results['mAP']:.4f}, "
                    f"mAP@50={map_results['mAP_50']:.4f}, "
                    f"mAP@75={map_results['mAP_75']:.4f}, "
                    f"ECE={ece:.4f}"
                )
                if "AP_NoFinding" in map_results:
                    print(f"  AP_NoFinding={map_results['AP_NoFinding']:.4f}")
            except Exception as exc:
                print(f"  Warning: comprehensive val metrics failed: {exc}")
                self.log("val/mAP", nan, sync_dist=False)
                self.log("val/mAP_50", nan, sync_dist=False)
                self.log("val/mAP_75", nan, sync_dist=False)
                self.log("val/ece", nan, sync_dist=False)
                self.log("val/AP_NoFinding", nan, sync_dist=False)
        else:
            self.log("val/mAP", nan, sync_dist=False)
            self.log("val/mAP_50", nan, sync_dist=False)
            self.log("val/mAP_75", nan, sync_dist=False)
            self.log("val/ece", nan, sync_dist=False)
            self.log("val/AP_NoFinding", nan, sync_dist=False)
    
    @staticmethod
    def _gather_val_lists(local_preds, local_gts, local_ious, world_size):
        """Concatenate validation preds/GTs/IoUs from all DDP ranks."""
        if world_size <= 1:
            return local_preds, local_gts, local_ious

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return local_preds, local_gts, local_ious

        payload = (local_preds, local_gts, local_ious)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, payload)

        all_preds, all_gts, all_ious = [], [], []
        for rank_preds, rank_gts, rank_ious in gathered:
            all_preds.extend(rank_preds)
            all_gts.extend(rank_gts)
            all_ious.extend(rank_ious)
        return all_preds, all_gts, all_ious