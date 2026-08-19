import json
import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl

from softlabel_od.object_detection.utils.training_helpers import convert_np


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


class ThetaJSONHistoryLogger(pl.Callback):
    """
    Theta-space equivalent of Phase1JSONHistoryLogger. Same structural
    pattern (pl.Callback, on_train_epoch_end hook, reads trainer.
    callback_metrics via _pick with multi-variant key fallback, writes
    JSON every epoch, NaN for any key not present) -- but reading the
    REAL keys SoftLabel2DThetaLightningModule actually logs, not the
    hard-label keys HardJSONHistoryLogger was previously (incorrectly)
    reused for.

    Confirmed root cause of the earlier "Class: nan, Reg: nan, mAP: nan"
    issue: train_lightning.py's _get_history_logger() had no branch for
    "soft_2d_theta" and fell through to HardJSONHistoryLogger, which
    looks for hard-label loss/metric key names that theta's loss dict
    never produces. This class is the actual fix -- not a workaround.

    Critically includes val_nees, the primary calibration diagnostic
    this whole investigation has been centred on, which has no 1D
    equivalent and was entirely absent from any history file before this.

    Output path:
      <cfg.MISC.RESULTS_CSV>/<cfg.MISC.RUN_NAME>/training_metrics_theta.json
    """

    def __init__(self, cfg, filename: str = "training_metrics_theta.json", print_summary: bool = True):
        super().__init__()
        self.cfg = cfg
        self.print_summary = print_summary

        self.results_dir = os.path.join(cfg.MISC.RESULTS_CSV, cfg.MISC.RUN_NAME)
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_file_path = os.path.join(self.results_dir, filename)

        self.metrics_history = {
            "train_loss": [],
            "train_class_loss": [],
            "train_theta_mean_loss": [],
            "train_theta_mean_loss_raw": [],
            "train_theta_cov_loss": [],
            "train_theta_cov_loss_raw": [],
            "train_theta_kl_loss": [],
            "train_theta_kl_loss_raw": [],
            "train_rpn_obj": [],
            "train_rpn_reg": [],
            "train_nees_delta": [],              
            "train_diag_raw_max": [],            
            "train_diag_clamp_saturation_frac": [],

            "train_nees_delta_x": [],
            "train_nees_delta_y": [],
            "train_nees_delta_w": [],
            "train_nees_delta_h": [],

            "val_loss": [],
            "val_class_loss": [],
            "val_theta_mean_loss": [],
            "val_theta_mean_loss_raw": [],
            "val_theta_cov_loss": [],
            "val_theta_cov_loss_raw": [],
            "val_theta_kl_loss": [],
            "val_theta_kl_loss_raw": [],
            "val_rpn_obj": [],
            "val_rpn_reg": [],
            "val_avg_iou": [],
            "val_expected_iou": [],
            "val_nees": [],
            "val_nees_delta": [],                
            "val_diag_raw_max": [],              
            "val_diag_clamp_saturation_frac": [],   

            "val_nees_delta_x": [],
            "val_nees_delta_y": [],
            "val_nees_delta_w": [],
            "val_nees_delta_h": [],

            "val_ece": [],
            "val_uw_map": [],
            "top1_accuracy": [],
            "top3_accuracy": [],
            "weighted_accuracy": [],
            "kl_divergence": [],
            "js_divergence": [],
            "cosine_similarity": [],
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

        # --- Training-side losses (confirmed keys, from training_step's
        #     self.log(f"train/{k}", v) loop over theta_spatial_loss's
        #     returned dict, plus the total and RPN terms) ---
        train_loss = _pick(m, "train/loss_epoch", "train/loss")
        train_class_loss = _pick(m, "train/loss_probabilistic_classifier_epoch", "train/loss_probabilistic_classifier")
        train_theta_mean = _pick(m, "train/loss_theta_mean_epoch", "train/loss_theta_mean")
        train_theta_mean_raw = _pick(m, "train/loss_theta_mean_raw_epoch", "train/loss_theta_mean_raw")
        train_theta_cov = _pick(m, "train/loss_theta_cov_epoch", "train/loss_theta_cov")
        train_theta_cov_raw = _pick(m, "train/loss_theta_cov_raw_epoch", "train/loss_theta_cov_raw")
        train_theta_kl = _pick(m, "train/loss_theta_kl_epoch", "train/loss_theta_kl")
        train_theta_kl_raw = _pick(m, "train/loss_theta_kl_raw_epoch", "train/loss_theta_kl_raw")
        train_rpn_obj = _pick(m, "train/loss_objectness_epoch", "train/loss_objectness")
        train_rpn_reg = _pick(m, "train/loss_rpn_box_reg_epoch", "train/loss_rpn_box_reg")
        train_nees_delta = _pick(m, "train/nees_delta_epoch", "train/nees_delta")
        train_diag_raw_max = _pick(m, "train/diag_raw_max_epoch", "train/diag_raw_max")
        train_diag_sat = _pick(m, "train/diag_clamp_saturation_frac_epoch", "train/diag_clamp_saturation_frac")
        
        train_nees_delta_x = _pick(m, "train/nees_delta_x_epoch", "train/nees_delta_x")
        train_nees_delta_y = _pick(m, "train/nees_delta_y_epoch", "train/nees_delta_y")
        train_nees_delta_w = _pick(m, "train/nees_delta_w_epoch", "train/nees_delta_w")
        train_nees_delta_h = _pick(m, "train/nees_delta_h_epoch", "train/nees_delta_h")

        # --- Validation-side losses (confirmed keys, from validation_step's
        #     named-summary self.log calls) ---
        val_loss = _pick(m, "val/loss", "val/loss_epoch")
        val_class_loss = _pick(m, "val/class_loss", "val/class_loss_epoch")
        val_theta_mean = _pick(m, "val/theta_mean_loss", "val/theta_mean_loss_epoch", "val/loss_theta_mean")
        val_theta_mean_raw = _pick(m, "val/loss_theta_mean_raw", "val/loss_theta_mean_raw_epoch")
        val_theta_cov = _pick(m, "val/theta_cov_loss", "val/theta_cov_loss_epoch", "val/loss_theta_cov")
        val_theta_cov_raw = _pick(m, "val/loss_theta_cov_raw", "val/loss_theta_cov_raw_epoch")
        val_theta_kl = _pick(m, "val/loss_theta_kl", "val/loss_theta_kl_epoch")
        val_theta_kl_raw = _pick(m, "val/loss_theta_kl_raw", "val/loss_theta_kl_raw_epoch")
        val_rpn_obj = _pick(m, "val/rpn_objectness", "val/rpn_objectness_epoch")
        val_rpn_reg = _pick(m, "val/rpn_box_reg", "val/rpn_box_reg_epoch")

        val_nees_delta_x = _pick(m, "val/nees_delta_x", "val/nees_delta_x_epoch")
        val_nees_delta_y = _pick(m, "val/nees_delta_y", "val/nees_delta_y_epoch")
        val_nees_delta_w = _pick(m, "val/nees_delta_w", "val/nees_delta_w_epoch")
        val_nees_delta_h = _pick(m, "val/nees_delta_h", "val/nees_delta_h_epoch")

        # --- Epoch-end metrics (from on_validation_epoch_end) ---
        val_avg_iou = _pick(m, "val/avg_iou", "val/avg_iou_epoch")
        val_expected_iou = _pick(m, "val/expected_iou", "val/expected_iou_epoch")
        val_nees = _pick(m, "val/nees", "val/nees_epoch")

        val_nees_delta = _pick(m, "val/nees_delta", "val/nees_delta_epoch")
        val_diag_raw_max = _pick(m, "val/diag_raw_max", "val/diag_raw_max_epoch")
        val_diag_sat = _pick(m, "val/diag_clamp_saturation_frac", "val/diag_clamp_saturation_frac_epoch")

        val_ece = _pick(m, "val/ece", "val/ece_epoch")
        val_uw_map = _pick(m, "val/uw_map", "val/uw_map_epoch")
        top1 = _pick(m, "val/top1_accuracy", "val/top1_accuracy_epoch")
        top3 = _pick(m, "val/top3_accuracy", "val/top3_accuracy_epoch")
        weighted = _pick(m, "val/weighted_accuracy", "val/weighted_accuracy_epoch")
        kl = _pick(m, "val/kl_divergence", "val/kl_divergence_epoch")
        js = _pick(m, "val/js_divergence", "val/js_divergence_epoch")
        cos = _pick(m, "val/cosine_similarity", "val/cosine_similarity_epoch")

        lr = self._current_lr(trainer)

        def _nan_if_none(x: Optional[float]) -> float:
            return float("nan") if x is None else float(x)

        self.metrics_history["train_loss"].append(_nan_if_none(train_loss))
        self.metrics_history["train_class_loss"].append(_nan_if_none(train_class_loss))
        self.metrics_history["train_theta_mean_loss"].append(_nan_if_none(train_theta_mean))
        self.metrics_history["train_theta_mean_loss_raw"].append(_nan_if_none(train_theta_mean_raw))
        self.metrics_history["train_theta_cov_loss"].append(_nan_if_none(train_theta_cov))
        self.metrics_history["train_theta_cov_loss_raw"].append(_nan_if_none(train_theta_cov_raw))
        self.metrics_history["train_theta_kl_loss"].append(_nan_if_none(train_theta_kl))
        self.metrics_history["train_theta_kl_loss_raw"].append(_nan_if_none(train_theta_kl_raw))
        self.metrics_history["train_rpn_obj"].append(_nan_if_none(train_rpn_obj))
        self.metrics_history["train_rpn_reg"].append(_nan_if_none(train_rpn_reg))

        self.metrics_history["train_nees_delta"].append(_nan_if_none(train_nees_delta))
        self.metrics_history["train_diag_raw_max"].append(_nan_if_none(train_diag_raw_max))
        self.metrics_history["train_diag_clamp_saturation_frac"].append(_nan_if_none(train_diag_sat))


        self.metrics_history["val_loss"].append(_nan_if_none(val_loss))
        self.metrics_history["val_class_loss"].append(_nan_if_none(val_class_loss))
        self.metrics_history["val_theta_mean_loss"].append(_nan_if_none(val_theta_mean))
        self.metrics_history["val_theta_mean_loss_raw"].append(_nan_if_none(val_theta_mean_raw))
        self.metrics_history["val_theta_cov_loss"].append(_nan_if_none(val_theta_cov))
        self.metrics_history["val_theta_cov_loss_raw"].append(_nan_if_none(val_theta_cov_raw))
        self.metrics_history["val_theta_kl_loss"].append(_nan_if_none(val_theta_kl))
        self.metrics_history["val_theta_kl_loss_raw"].append(_nan_if_none(val_theta_kl_raw))
        self.metrics_history["val_rpn_obj"].append(_nan_if_none(val_rpn_obj))
        self.metrics_history["val_rpn_reg"].append(_nan_if_none(val_rpn_reg))

        self.metrics_history["val_avg_iou"].append(_nan_if_none(val_avg_iou))
        self.metrics_history["val_expected_iou"].append(_nan_if_none(val_expected_iou))
        self.metrics_history["val_nees"].append(_nan_if_none(val_nees))
        
        self.metrics_history["val_nees_delta"].append(_nan_if_none(val_nees_delta))
        self.metrics_history["val_diag_raw_max"].append(_nan_if_none(val_diag_raw_max))
        self.metrics_history["val_diag_clamp_saturation_frac"].append(_nan_if_none(val_diag_sat))

        self.metrics_history["val_ece"].append(_nan_if_none(val_ece))
        self.metrics_history["val_uw_map"].append(_nan_if_none(val_uw_map))

        self.metrics_history["top1_accuracy"].append(_nan_if_none(top1))
        self.metrics_history["top3_accuracy"].append(_nan_if_none(top3))
        self.metrics_history["weighted_accuracy"].append(_nan_if_none(weighted))
        self.metrics_history["kl_divergence"].append(_nan_if_none(kl))
        self.metrics_history["js_divergence"].append(_nan_if_none(js))
        self.metrics_history["cosine_similarity"].append(_nan_if_none(cos))

        self.metrics_history["learning_rate"].append(float(lr))

        self.metrics_history["train_nees_delta_x"].append(_nan_if_none(train_nees_delta_x))
        self.metrics_history["train_nees_delta_y"].append(_nan_if_none(train_nees_delta_y))
        self.metrics_history["train_nees_delta_w"].append(_nan_if_none(train_nees_delta_w))
        self.metrics_history["train_nees_delta_h"].append(_nan_if_none(train_nees_delta_h))
        self.metrics_history["val_nees_delta_x"].append(_nan_if_none(val_nees_delta_x))
        self.metrics_history["val_nees_delta_y"].append(_nan_if_none(val_nees_delta_y))
        self.metrics_history["val_nees_delta_w"].append(_nan_if_none(val_nees_delta_w))
        self.metrics_history["val_nees_delta_h"].append(_nan_if_none(val_nees_delta_h))

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss

        with open(self.metrics_file_path, "w") as f:
            json.dump(convert_np(self.metrics_history), f, indent=2)

        if self.print_summary:
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
            print("\nEpoch Summary (Theta 2D):")
            print(f"  Epoch: {epoch}/{max_epochs}")
            print(f"  Train Loss: {_nan_if_none(train_loss):.4f}")
            print(
                f"  Val Loss: {_nan_if_none(val_loss):.4f} "
                f" (Class: {_nan_if_none(val_class_loss):.4f}, "
                f"ThetaMean: {_nan_if_none(val_theta_mean):.4f}, "
                f"ThetaCov: {_nan_if_none(val_theta_cov):.4f}, "
                f"ThetaKL: {_nan_if_none(val_theta_kl):.4f})"
            )
            print(
                f"  Raw (unweighted): ThetaMean={_nan_if_none(val_theta_mean_raw):.4f}, "
                f"ThetaCov={_nan_if_none(val_theta_cov_raw):.4f}, "
                f"ThetaKL={_nan_if_none(val_theta_kl_raw):.4f}"
            )
            if val_avg_iou is not None:
                print(f"  Detection: Avg IoU={val_avg_iou:.3f}")
            print(
                "  Calibration: "
                f"NEES={_nan_if_none(val_nees):.1f} (target: 4.0), "
                f"ECE={_nan_if_none(val_ece):.3f}, "
                f"Expected_IoU={_nan_if_none(val_expected_iou):.3f}, "
                f"UW-mAP={_nan_if_none(val_uw_map):.3f}"
            )
            print(
                "  Class metrics: "
                f"Top1={_nan_if_none(top1):.2%}, Top3={_nan_if_none(top3):.2%}, Weighted={_nan_if_none(weighted):.2%}, "
                f"KL={_nan_if_none(kl):.3f}, JS={_nan_if_none(js):.3f}, Cos={_nan_if_none(cos):.3f}"
            )
            print(f"  LR: {lr:.6f}")
            print(f"  Best Val Loss so far: {self.best_val_loss:.4f}")
            print(f"  Metrics JSON: {self.metrics_file_path}")
            print(
                f"  Delta-space diag: NEES_delta={_nan_if_none(val_nees_delta):.1f} (target: 4.0), "
                f"DiagRawMax={_nan_if_none(val_diag_raw_max):.3f}, "
                f"ClampSatFrac={_nan_if_none(val_diag_sat):.3f}"
            )
            print(f"  Per-axis (delta): x={_nan_if_none(val_nees_delta_x):.2f}, y={_nan_if_none(val_nees_delta_y):.2f}, w={_nan_if_none(val_nees_delta_w):.2f}, h={_nan_if_none(val_nees_delta_h):.2f}  (each target: 1.0)")
