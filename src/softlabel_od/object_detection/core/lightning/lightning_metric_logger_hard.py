import json
import os
from typing import Any, Dict, Optional
import numpy as np
import pytorch_lightning as pl


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None


def _pick(metrics: Dict[str, Any], *keys: str) -> Optional[float]:
    for k in keys:
        if k in metrics:
            v = _to_float(metrics[k])
            if v is not None:
                return v
    return None


class HardJSONHistoryLogger(pl.Callback):
    """
    Writes a minimal hard-label training history JSON:
    - train_loss, train_class_loss, train_reg_loss
    - val_loss, val_class_loss, val_reg_loss
    - val_avg_iou
    """

    def __init__(self, cfg, filename: str = "training_metrics_hard.json", print_summary: bool = True):
        super().__init__()
        self.cfg = cfg
        self.print_summary = print_summary

        self.results_dir = os.path.join(cfg.MISC.RESULTS_CSV, cfg.MISC.RUN_NAME)
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(self.results_dir, filename)

        self.metrics_history = {
            "train_loss": [],
            "train_class_loss": [],
            "train_reg_loss": [],
            "val_loss": [],
            "val_class_loss": [],
            "val_reg_loss": [],
            "val_avg_iou": [],
            "val_class_accuracy": [],
            "train_rpn_objectness": [],
            "train_rpn_box_reg": [],
            "val_rpn_objectness": [],
            "val_rpn_box_reg": [],
            "val_mAP": [],
            "val_mAP_50": [],
            "val_mAP_75": [],
            "val_AP_NoFinding": [],
            "val_ece": [],
            "learning_rate": [],
        }

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        m = dict(trainer.callback_metrics)

        train_loss = _pick(m, "train/loss_epoch", "train/loss")
        train_cls  = _pick(m, "train/class_loss_epoch", "train/class_loss", "train/loss_classifier_epoch", "train/loss_classifier")
        train_reg  = _pick(m, "train/reg_loss_epoch", "train/reg_loss", "train/loss_box_reg_epoch", "train/loss_box_reg")
        train_rpn_obj = _pick(m, "train/rpn_objectness_epoch", "train/rpn_objectness", "train/loss_objectness_epoch", "train/loss_objectness")
        train_rpn_reg = _pick(m, "train/rpn_box_reg_epoch", "train/rpn_box_reg", "train/loss_rpn_box_reg_epoch", "train/loss_rpn_box_reg")
        val_loss = _pick(m, "val/loss", "val/loss_epoch")
        val_cls  = _pick(m, "val/class_loss", "val/class_loss_epoch", "val/loss_classifier")
        val_reg  = _pick(m, "val/reg_loss", "val/reg_loss_epoch", "val/loss_box_reg")
        val_rpn_obj = _pick(m, "val/rpn_objectness", "val/rpn_objectness_epoch", "val/loss_objectness")
        val_rpn_reg = _pick(m, "val/rpn_box_reg", "val/rpn_box_reg_epoch", "val/loss_rpn_box_reg")
        val_iou  = _pick(m, "val/avg_iou", "val/avg_iou_epoch")
        val_acc  = _pick(m, "val/class_accuracy", "val/class_accuracy_epoch")
        val_map  = _pick(m, "val/mAP", "val/mAP_epoch")
        val_map50 = _pick(m, "val/mAP_50", "val/mAP_50_epoch")
        val_map75 = _pick(m, "val/mAP_75", "val/mAP_75_epoch")
        val_ap_nf = _pick(m, "val/AP_NoFinding", "val/AP_NoFinding_epoch")
        val_ece  = _pick(m, "val/ece", "val/ece_epoch")

        lr = float("nan")
        opt = trainer.optimizers[0] if trainer.optimizers else None
        if opt and opt.param_groups:
            lr = float(opt.param_groups[0].get("lr", float("nan")))

        def _nan_if_none(x: Optional[float]) -> float:
            return float("nan") if x is None else float(x)

        self.metrics_history["train_loss"].append(_nan_if_none(train_loss))
        self.metrics_history["train_class_loss"].append(_nan_if_none(train_cls))
        self.metrics_history["train_reg_loss"].append(_nan_if_none(train_reg))
        self.metrics_history["train_rpn_objectness"].append(_nan_if_none(train_rpn_obj))
        self.metrics_history["train_rpn_box_reg"].append(_nan_if_none(train_rpn_reg))
        self.metrics_history["val_loss"].append(_nan_if_none(val_loss))
        self.metrics_history["val_class_loss"].append(_nan_if_none(val_cls))
        self.metrics_history["val_reg_loss"].append(_nan_if_none(val_reg))
        self.metrics_history["val_rpn_objectness"].append(_nan_if_none(val_rpn_obj))
        self.metrics_history["val_rpn_box_reg"].append(_nan_if_none(val_rpn_reg))
        self.metrics_history["val_avg_iou"].append(_nan_if_none(val_iou))
        self.metrics_history["val_class_accuracy"].append(_nan_if_none(val_acc))
        self.metrics_history["val_mAP"].append(_nan_if_none(val_map))
        self.metrics_history["val_mAP_50"].append(_nan_if_none(val_map50))
        self.metrics_history["val_mAP_75"].append(_nan_if_none(val_map75))
        self.metrics_history["val_AP_NoFinding"].append(_nan_if_none(val_ap_nf))
        self.metrics_history["val_ece"].append(_nan_if_none(val_ece))
        self.metrics_history["learning_rate"].append(lr)

        with open(self.metrics_file_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        if self.print_summary:
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
            print("\nEpoch Summary (Hard labels):")
            print(f"  Epoch: {epoch}/{max_epochs}")
            print(f"  Train Loss: {_nan_if_none(train_loss):.4f} (Class: {_nan_if_none(train_cls):.4f}, Reg: {_nan_if_none(train_reg):.4f})")
            print(f"  Val Loss:   {_nan_if_none(val_loss):.4f} (Class: {_nan_if_none(val_cls):.4f}, Reg: {_nan_if_none(val_reg):.4f})")
            print(f"  Avg IoU:    {_nan_if_none(val_iou):.4f}")
            print(f"  Class Accuracy: {_nan_if_none(val_acc):.4f}")
            print(f"  mAP: {_nan_if_none(val_map):.4f} (50: {_nan_if_none(val_map50):.4f}, 75: {_nan_if_none(val_map75):.4f})")
            print(f"  AP NoFinding: {_nan_if_none(val_ap_nf):.4f}")
            print(f"  ECE: {_nan_if_none(val_ece):.4f}")
            print(f"  Learning Rate: {lr:.6f}")
            print(f"  Metrics JSON: {self.metrics_file_path}")