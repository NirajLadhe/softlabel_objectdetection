import json
import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl

from softlabel_od.object_detection.utils.training_helpers import convert_np


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        # torch scalar tensor
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _pick(metrics: Dict[str, Any], *keys: str) -> Optional[float]:
    # Try several possible Lightning key variants across versions
    for k in keys:
        if k in metrics:
            v = _to_float(metrics[k])
            if v is not None:
                return v
    return None


class Phase1JSONHistoryLogger(pl.Callback):
    """
    Writes a metrics history JSON compatible with your training_1d.py style.

    Output path:
      <cfg.MISC.RESULTS_CSV>/<cfg.MISC.RUN_NAME>/training_metrics_phase_1.json
    """

    def __init__(self, cfg, filename: str = "training_metrics_phase_1.json", print_summary: bool = True):
        super().__init__()
        self.cfg = cfg
        self.print_summary = print_summary

        self.results_dir = os.path.join(cfg.MISC.RESULTS_CSV, cfg.MISC.RUN_NAME)
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(self.results_dir, filename)

        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "val_class_loss": [],
            "val_box_reg_loss": [],
            "val_box_std_loss": [],
            "val_rpn_obj": [],
            "val_rpn_reg": [],
            "val_class_accuracy": [],
            "val_avg_iou": [],
            "top1_accuracy": [],
            "top3_accuracy": [],
            "weighted_accuracy": [],
            "kl_divergence": [],
            "js_divergence": [],
            "cosine_similarity": [],
            "calibration_error": [],
            "uncertainty_weighted_map": [],
            "expected_iou": [],
            "bbox_calibration_error": [],
            "learning_rate": [],
        }

        self.best_val_loss = float("inf")

    def _current_lr(self, trainer: pl.Trainer) -> float:
        opt = trainer.optimizers[0] if trainer.optimizers else None
        if opt is None or not opt.param_groups:
            return float("nan")
        return float(opt.param_groups[0].get("lr", float("nan")))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = dict(trainer.callback_metrics)

        train_loss = _pick(m, "train/loss_epoch", "train/loss", "train_loss", "loss/train")
        val_loss = _pick(m, "val/loss", "val/loss_epoch", "val_loss")
        val_rpn_obj = _pick(m, "val/rpn_objectness", "val/rpn_objectness_epoch")
        val_rpn_reg = _pick(m, "val/rpn_box_reg", "val/rpn_box_reg_epoch")
        val_acc = _pick(m, "val/class_accuracy", "val/class_accuracy_epoch")
        val_class_loss = _pick(m, "val/class_loss", "val/class_loss_epoch")
        val_reg_loss = _pick(m, "val/reg_loss", "val/reg_loss_epoch")
        val_std_loss = _pick(m, "val/std_loss", "val/std_loss_epoch")
        val_avg_iou = _pick(m, "val/avg_iou", "val/avg_iou_epoch") 

        top1 = _pick(m, "val/top1_accuracy", "val/top1_accuracy_epoch")
        top3 = _pick(m, "val/top3_accuracy", "val/top3_accuracy_epoch")
        weighted = _pick(m, "val/weighted_accuracy", "val/weighted_accuracy_epoch")

        kl = _pick(m, "val/kl_divergence", "val/kl_divergence_epoch")
        js = _pick(m, "val/js_divergence", "val/js_divergence_epoch")
        cos = _pick(m, "val/cosine_similarity", "val/cosine_similarity_epoch")

        ece = _pick(m, "val/ece", "val/ece_epoch")
        uw_map = _pick(m, "val/uw_map", "val/uw_map_epoch")

        expected_iou = _pick(m, "val/expected_iou", "val/expected_iou_epoch")

        bbox_corr = _pick(m, "val/bbox_std_error_corr", "val/bbox_std_error_corr_epoch")

        lr = self._current_lr(trainer)

        # Fill with NaN if missing
        def _nan_if_none(x: Optional[float]) -> float:
            return float("nan") if x is None else float(x)

        self.metrics_history["train_loss"].append(_nan_if_none(train_loss))
        self.metrics_history["val_loss"].append(_nan_if_none(val_loss))
        self.metrics_history["val_class_loss"].append(_nan_if_none(val_class_loss))
        self.metrics_history["val_box_reg_loss"].append(_nan_if_none(val_reg_loss))
        self.metrics_history["val_box_std_loss"].append(_nan_if_none(val_std_loss))
        self.metrics_history["val_avg_iou"].append(_nan_if_none(val_avg_iou))
        self.metrics_history["val_rpn_obj"].append(_nan_if_none(val_rpn_obj))
        self.metrics_history["val_rpn_reg"].append(_nan_if_none(val_rpn_reg))
        self.metrics_history["val_class_accuracy"].append(_nan_if_none(val_acc))

        self.metrics_history["top1_accuracy"].append(_nan_if_none(top1))
        self.metrics_history["top3_accuracy"].append(_nan_if_none(top3))
        self.metrics_history["weighted_accuracy"].append(_nan_if_none(weighted))

        self.metrics_history["kl_divergence"].append(_nan_if_none(kl))
        self.metrics_history["js_divergence"].append(_nan_if_none(js))
        self.metrics_history["cosine_similarity"].append(_nan_if_none(cos))

        self.metrics_history["calibration_error"].append(_nan_if_none(ece))
        self.metrics_history["uncertainty_weighted_map"].append(_nan_if_none(uw_map))

        self.metrics_history["expected_iou"].append(_nan_if_none(expected_iou))
        self.metrics_history["bbox_calibration_error"].append(_nan_if_none(bbox_corr))

        self.metrics_history["learning_rate"].append(float(lr))

        # Track best val loss (like your best_val_loss print)
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        # Write JSON every epoch (like training_1d.py)
        with open(self.metrics_file_path, "w") as f:
            json.dump(convert_np(self.metrics_history), f, indent=2)

        # Console summary similar to training_1d.py
        if self.print_summary:
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
            print("\nEpoch Summary:")
            print(f"  Epoch: {epoch}/{max_epochs}")
            print(f"  Train Loss: {_nan_if_none(train_loss):.4f}")
            print(
                f"  Val Loss: {_nan_if_none(val_loss):.4f} "
                f" (Class: {_nan_if_none(val_class_loss):.4f}, Reg: {_nan_if_none(val_reg_loss):.4f}, Std: {_nan_if_none(val_std_loss):.4f})"
            )
            if val_avg_iou is not None:
                print(f"  Detection: Avg IoU={val_avg_iou:.3f}")
            print(
                "  Metrics: "
                f"Expected_IoU={_nan_if_none(expected_iou):.3f}, "
                f"Top1={_nan_if_none(top1):.2%}, Top3={_nan_if_none(top3):.2%}, Weighted={_nan_if_none(weighted):.2%}, "
                f"KL={_nan_if_none(kl):.3f}, JS={_nan_if_none(js):.3f}, Cos={_nan_if_none(cos):.3f}, "
                f"ECE={_nan_if_none(ece):.3f}, UW-mAP={_nan_if_none(uw_map):.3f}, "
                f"BboxCorr={_nan_if_none(bbox_corr):.3f}"
            )
            print(f"  LR: {lr:.6f}")
            print(f"  Best Val Loss so far: {self.best_val_loss:.4f}")
            print(f"  Metrics JSON: {self.metrics_file_path}")