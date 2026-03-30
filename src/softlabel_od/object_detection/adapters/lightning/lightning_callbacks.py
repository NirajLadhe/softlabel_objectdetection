"""
lightning_callbacks.py — artifact-export callbacks.

ExportBestPthCallback
    Mirrors what legacy train.py / train_1d.py do when a new best is found:
      • saves model.state_dict() to <SAVE_MODEL_PATH>/<RUN_NAME>_best_model.pth
      • saves the final state dict to    <RUN_NAME>.pth  at the end of training
      • saves the config YAML to         <RUN_NAME>_runConfig.yaml

CompositeEarlyStopping1D
    Replicates the composite early-stopping logic in train_and_evaluate_model_1d().
    Triggered by the 'val/loss' and several per-metric stagnation counters.
"""

from __future__ import annotations

import os
import torch
import pytorch_lightning as pl


class ExportBestPthCallback(pl.Callback):
    """
    Exports legacy-compatible .pth checkpoints alongside Lightning .ckpt files.

    Best model:   <save_dir>/<run_name>_best_model.pth   (updated when val/loss improves)
    Final model:  <save_dir>/<run_name>.pth               (written on training end)
    Config YAML:  <save_dir>/<run_name>_runConfig.yaml    (written on training end)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg      = cfg
        self.save_dir = cfg.MISC.SAVE_MODEL_PATH
        self.run_name = cfg.MISC.RUN_NAME
        self._best_val_loss = float("inf")
        os.makedirs(self.save_dir, exist_ok=True)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        metrics = dict(trainer.callback_metrics)
        val_loss = metrics.get("val/loss")
        if val_loss is None:
            return
        val_loss = float(val_loss)
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            path = os.path.join(self.save_dir, f"{self.run_name}_best_model.pth")
            state = pl_module.model.to(torch.device("cpu")).state_dict()
            torch.save(state, path)
            # Move model back to training device
            pl_module.model.to(pl_module.device)
            print(f"[ExportBestPth] New best val/loss={val_loss:.4f} → {path}")

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # Final state dict
        final_path = os.path.join(self.save_dir, f"{self.run_name}.pth")
        state = pl_module.model.to(torch.device("cpu")).state_dict()
        torch.save(state, final_path)
        pl_module.model.to(pl_module.device)
        print(f"[ExportBestPth] Final model saved → {final_path}")

        # Config YAML
        cfg_path = os.path.join(self.save_dir, f"{self.run_name}_runConfig.yaml")
        with open(cfg_path, "w") as f:
            f.write(self.cfg.dump())
        print(f"[ExportBestPth] Config saved → {cfg_path}")


class CompositeEarlyStopping1D(pl.Callback):
    """
    Port of the composite early-stopping logic in train_and_evaluate_model_1d().

    Stops when:
      - composite score (weighted combination of iou, top1, js, var) fails to
        improve by MIN_DELTA['composite'] for COMPOSITE_PATIENCE epochs, or
      - any of {avg_iou, js_divergence, top1_accuracy} stagnates for
        STAGNATION_PATIENCE epochs,
      - only after MIN_EPOCHS_BEFORE_STOPPING epochs have passed.

    Enable by adding this callback instead of (or in addition to) Lightning's
    built-in EarlyStopping.
    """

    MIN_EPOCHS_BEFORE_STOPPING = 15
    STAGNATION_PATIENCE        = 12
    COMPOSITE_PATIENCE         =  7
    EMA_ALPHA                  = 0.3
    MIN_DELTA = {
        "composite":     0.001,
        "avg_iou":       0.003,
        "js_divergence": 0.003,
        "top1_accuracy": 0.005,
    }

    def __init__(self):
        super().__init__()
        self._best_composite = -float("inf")
        self._no_improve     = 0
        self._stagnation     = {"avg_iou": 0, "js_divergence": 0, "top1_accuracy": 0}
        self._stag_best      = {
            "avg_iou":       -float("inf"),
            "js_divergence":  float("inf"),
            "top1_accuracy": -float("inf"),
        }
        self._ema = {}

    @staticmethod
    def _composite(m):
        iou   = m.get("val/avg_iou",       0.0)
        js    = m.get("val/js_divergence",  0.5)
        top1  = m.get("val/top1_accuracy",  0.0)
        var_l = m.get("val/std_loss",       0.1)
        return (
            0.40 * top1
          + 0.35 * iou
          + 0.15 * (1.0 - float(js))
          + 0.10 * (1.0 - min(float(var_l), 1.0))
        )

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        epoch = trainer.current_epoch + 1
        m = {k: float(v) for k, v in trainer.callback_metrics.items()}

        composite = self._composite(m)

        # EMA smoothing for stagnation checks
        for key in ("avg_iou", "js_divergence", "top1_accuracy"):
            raw = m.get(f"val/{key}", 0.0)
            self._ema[key] = (
                raw if key not in self._ema
                else self.EMA_ALPHA * raw + (1 - self.EMA_ALPHA) * self._ema[key]
            )

        # Composite improvement
        if composite > self._best_composite + self.MIN_DELTA["composite"]:
            self._best_composite = composite
            self._no_improve = 0
        else:
            self._no_improve += 1

        # Per-metric stagnation
        any_stagnated = False
        dirs = {"avg_iou": ">", "js_divergence": "<", "top1_accuracy": ">"}
        for key, direction in dirs.items():
            value = self._ema.get(key, 0.0)
            best  = self._stag_best[key]
            delta = self.MIN_DELTA[key]
            improved = (value > best + delta) if direction == ">" else (value < best - delta)
            if improved:
                self._stag_best[key]    = value
                self._stagnation[key]   = 0
            else:
                self._stagnation[key]  += 1
                if self._stagnation[key] >= self.STAGNATION_PATIENCE:
                    any_stagnated = True
                    print(
                        f"[CompositeES] {key} stagnated for "
                        f"{self.STAGNATION_PATIENCE} epochs"
                    )

        if epoch < self.MIN_EPOCHS_BEFORE_STOPPING:
            return

        if self._no_improve >= self.COMPOSITE_PATIENCE:
            print(
                f"[CompositeES] Composite score stagnated for "
                f"{self.COMPOSITE_PATIENCE} epochs — stopping."
            )
            trainer.should_stop = True

        if any_stagnated:
            print("[CompositeES] Per-metric stagnation detected — stopping.")
            trainer.should_stop = True