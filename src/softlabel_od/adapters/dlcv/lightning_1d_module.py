import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from softlabel_od.adapters.dlcv.model import get_model
from softlabel_od.adapters.dlcv.config import get_cfg_defaults
cfg = get_cfg_defaults()
from softlabel_od.adapters.dlcv.evaluation_metrics import (
    compute_calibration_error,
    compute_uncertainty_weighted_map,
    compute_soft_detection_metrics,
    compute_soft_class_metrics,
    compute_class_distribution_metrics,
    compute_expected_iou,
    compute_bbox_uncertainty_calibration,
    match_predictions_to_gt
)


class SoftLabel1DLightningModule(pl.LightningModule):
    """
    LightningModule for 1D soft-label probabilistic Faster R-CNN training.

    Key behaviors preserved from your current code:
    - Uses torchvision detection model losses via model(images, targets)
    - Applies your loss weighting (box reg * 10.0)
    - Validation computes losses by temporarily switching model.train() under no_grad()
      (same trick used in training_1d.py)
    - Computes your probabilistic/soft metrics epoch-wise from accumulated predictions
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Build the exact same model as before (model_type='1d')
        self.model = get_model(
            num_classes=cfg.MODEL.NUM_CLASSES,
            model_type="1d",
            backbone_name=cfg.MODEL.BACKBONE,
            pretrained=True,
            probabilistic=cfg.MODEL.PROBABILISTIC,
        )

        # Buffers to accumulate val predictions/GTs across batches
        self._val_predictions = []
        self._val_ground_truths = []

        # Warmup params matching training.py behavior (epoch 0 only)
        self._base_lrs = None  # captured on first optimizer_step
        self._warmup_factor = 1.0 / 1000.0

    def forward(self, images, targets=None):
        # Keep a thin wrapper; torchvision detection model API
        if targets is None:
            return self.model(images)
        return self.model(images, targets)
    def _summ(self, name, t):
        t = t.detach().cpu()
        if t.numel() == 0:
            return f"{name}: empty {tuple(t.shape)}"
        return (
            f"{name}: shape={tuple(t.shape)} "
            f"min={t.min().item():.3f} max={t.max().item():.3f} mean={t.float().mean().item():.3f}"
        )

    def debug_targets_once(self, images, targets, max_images=2):
        # debug, and then put the model back to train mode
        # images: list[Tensor(C,H,W)], targets: list[dict]
        print("\n=== DEBUG: raw targets before model transform ===")
        for i in range(min(len(images), max_images)):
            img = images[i]
            t = targets[i]
            H, W = img.shape[-2], img.shape[-1]
            print(f"\n[img {i}] image tensor size: HxW={H}x{W}")
            print(self._summ("boxes", t["boxes"]))
            print(self._summ("gt_means", t.get("gt_means", torch.empty(0))))
            print(self._summ("gt_stds", t.get("gt_stds", torch.empty(0))))

            if t["boxes"].numel() > 0:
                b = t["boxes"]
                oob = (b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > W) | (b[:, 3] > H) | (b[:, 2] <= b[:, 0]) | (b[:, 3] <= b[:, 1])
                print(f"boxes out-of-bounds or invalid: {int(oob.sum().item())}/{b.shape[0]}")

        # Now run the internal transform used by FasterRCNN (resizes images + boxes)
        if hasattr(self.model, "transform"):
            print("\n=== DEBUG: after model.transform(images, targets) ===")
            
            was_training = self.model.training
            self.model.eval()
            with torch.no_grad():
                imgs2, tgts2 = self.model.transform(images, targets)
            self.model.train(was_training)
            for i in range(min(len(imgs2.tensors), max_images)):
                H2, W2 = imgs2.tensors[i].shape[-2], imgs2.tensors[i].shape[-1]
                t2 = tgts2[i]
                print(f"\n[img {i}] transformed image size: HxW={H2}x{W2}")
                print(self._summ("boxes (transformed)", t2["boxes"]))
                # These two are unchanged unless YOU explicitly rescale them somewhere:
                if "gt_means" in t2:
                    print(self._summ("gt_means (still same object)", t2["gt_means"]))
                if "gt_stds" in t2:
                    print(self._summ("gt_stds (still same object)", t2["gt_stds"]))
    @staticmethod
    def _unwrap_loss_dict(loss_out):
        """
        Your code sometimes treats outputs as tuple vs dict.
        For torchvision detection models, training typically returns dict,
        but keep this robust to tuple (result, loss_dict).
        """
        if isinstance(loss_out, tuple) and len(loss_out) == 2 and isinstance(loss_out[1], dict):
            return loss_out[1]
        if isinstance(loss_out, dict):
            return loss_out
        raise TypeError(f"Unexpected loss output type: {type(loss_out)}")

    @staticmethod
    def _apply_loss_weighting(loss_dict):
        # Preserve your current behavior in training.py:
        # multiply 'loss_box_reg' by 10.0 when present.
        # Also handle your probabilistic naming if you ever emit that key.
        if "loss_box_mean" in loss_dict:
            loss_dict["loss_box_mean"] = loss_dict["loss_box_mean"] * cfg.TRAIN.BOX_REG_LOSS_WEIGHT
        if "loss_probabilistic_box_reg" in loss_dict:
            loss_dict["loss_probabilistic_box_mean"] = loss_dict["loss_probabilistic_box_mean"] * cfg.TRAIN.BOX_REG_LOSS_WEIGHT
        return loss_dict

    def training_step(self, batch, batch_idx):
        images, targets = batch  # collate_fn returns tuple(zip(*batch))
        # Lightning will usually move tensors to device recursively, but we keep it explicit-safe:
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]
        if self.global_step == 0 and (self.trainer is None or self.trainer.is_global_zero):
            self.debug_targets_once(images, targets)

        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)
        # loss_dict = self._apply_loss_weighting(loss_dict)

        total_loss = sum(loss for loss in loss_dict.values())

        # Log total + components
        self.log("train/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        # Ensure no grad leakage
        assert not torch.is_grad_enabled()

        # Save/restore mode (important)
        was_training = self.model.training
        self.model.train()
        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)
        # (Optional) apply weighting here if you still do any
        # loss_dict = self._apply_loss_weighting(loss_dict)

        # Log every component explicitly
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        # Derive the 3 requested summary terms (adjust names to your current keys)
        val_class_loss = loss_dict.get("loss_probabilistic_classifier", torch.tensor(0.0, device=images[0].device))
        val_reg_loss   = loss_dict.get("loss_box_mean", torch.tensor(0.0, device=images[0].device))
        val_std_loss   = loss_dict.get("loss_box_std", torch.tensor(0.0, device=images[0].device))

        # Total validation loss (sum everything)
        val_total_loss = sum(loss_dict.values()) if len(loss_dict) else torch.tensor(0.0, device=images[0].device)

        self.log("val/loss", val_total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/class_loss", val_class_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/reg_loss", val_reg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/std_loss", val_std_loss, on_step=False, on_epoch=True, prog_bar=False)

        # Restore mode and run inference for metrics
        self.model.train(was_training)
        self.model.eval()
        preds = self(images)

        # Accumulate for epoch-end metrics (match your dict formats)
        for pred, target in zip(preds, targets):
            pred_entry = {
                "boxes": pred["boxes"].detach().cpu().numpy(),
                "labels": pred["labels"].detach().cpu().numpy(),
                "scores": pred["scores"].detach().cpu().numpy(),
            }
            # Your probabilistic inference returns bbox_stds in 1D model
            if "bbox_stds" in pred:
                pred_entry["bbox_stds"] = pred["bbox_stds"].detach().cpu().numpy()
                pred_entry["bbox_means"] = pred["boxes"].detach().cpu().numpy()

            gt_entry = {
                "boxes": target["boxes"].detach().cpu().numpy(),
                "labels": target["labels"].detach().cpu().numpy(),
            }
            if "gt_class_probs" in target:
                gt_entry["gt_class_probs"] = target["gt_class_probs"].detach().cpu().numpy()

            self._val_predictions.append(pred_entry)
            self._val_ground_truths.append(gt_entry)

    def on_validation_epoch_end(self):
        # Compute your existing metrics once per epoch
        preds = self._val_predictions
        gts = self._val_ground_truths

        if len(preds) == 0:
            return

        ece = compute_calibration_error(preds, gts)
        uw_map = compute_uncertainty_weighted_map(preds, gts)
        soft_metrics = compute_class_distribution_metrics(preds, gts)
        dist_metric = compute_soft_detection_metrics(preds, gts)
        expected_iou = compute_expected_iou(preds, gts, num_samples=50)
        bbox_calib = compute_bbox_uncertainty_calibration(preds, gts)
        total_ious = []
        for pred, gt in zip(preds, gts):
            if len(pred["boxes"]) == 0 or len(gt["boxes"]) == 0:
                continue
            matches, _, _ = match_predictions_to_gt(pred["boxes"], pred["scores"], gt["boxes"], iou_threshold=0.5)
            for m in matches:
                total_ious.append(m[2])  # IoU is the third element in the match tuple
        avg_ios = float(np.mean(total_ious)) if len(total_ious) > 0 else 0.0
        self.log("val/avg_iou_raw", avg_ios, prog_bar=False)
            

        # Log (names align with what you print today in training_1d.py)
        self.log("val/ece", float(ece), prog_bar=False)
        self.log("val/uw_map", float(uw_map), prog_bar=False)

        self.log("val/top1_accuracy", float(dist_metric["top1_accuracy"]), prog_bar=False)
        self.log("val/top3_accuracy", float(dist_metric["top3_accuracy"]), prog_bar=False)
        self.log("val/weighted_accuracy", float(dist_metric["weighted_accuracy"]), prog_bar=False)

        self.log("val/kl_divergence", float(soft_metrics["kl_divergence"]), prog_bar=False)
        self.log("val/js_divergence", float(soft_metrics["js_divergence"]), prog_bar=False)
        self.log("val/cosine_similarity", float(soft_metrics["cosine_similarity"]), prog_bar=False)

        # self.log("val/avg_iou", float(avg_iou), prog_bar=False)
        self.log("val/expected_iou", float(expected_iou), prog_bar=False)
        self.log("val/bbox_std_error_corr", float(bbox_calib["std_error_correlation"]), prog_bar=False)
        self.log("val/bbox_calib_ratio", float(bbox_calib["calibration_ratio"]), prog_bar=False)

        # Clear buffers
        self._val_predictions = []
        self._val_ground_truths = []

    def configure_optimizers(self):
        # Move your optimizer param-group logic from train.py here,
        # but include probabilistic bbox head name too.
        bbox_params = []
        class_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if ("box_predictor.bbox_pred" in name) or ("box_predictor.bbox_pred_gmm" in name):
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
            weight_decay=0.0005,
        )

        scheduler = MultiStepLR(
            optimizer,
            milestones=list(self.cfg.TRAIN.MILESTONES),
            gamma=self.cfg.TRAIN.GAMMA,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_closure=None,
        **kwargs,
    ):
        """
        Optional: replicate your warmup_lr_scheduler behavior from training.py (epoch 0 only).
        training.py warmup uses:
          warmup_factor=1/1000
          warmup_iters=min(1000, len(data_loader)-1)

        Lightning gives num_training_batches via trainer.
        """
        if self._base_lrs is None:
            self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

        if self.current_epoch == 0 and self.trainer is not None:
            warmup_iters = min(1000, max(int(self.trainer.num_training_batches) - 1, 1))
            # global_step starts at 0; mimic step-based warmup
            x = min(self.global_step, warmup_iters)
            alpha = float(x) / float(warmup_iters)
            scale = self._warmup_factor * (1.0 - alpha) + alpha

            for pg, base_lr in zip(optimizer.param_groups, self._base_lrs):
                pg["lr"] = base_lr * scale

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()