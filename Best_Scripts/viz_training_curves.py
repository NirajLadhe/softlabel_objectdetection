"""
viz_training_curves.py

Saves one PNG per plot group to the output directory.

Usage:
    python viz_training_curves.py \
        --metrics-json  path/to/training_metrics_phase_1.json \
        --mode          soft \
        --output-dir    outputs/visualizations/training_curves_soft

    python viz_training_curves.py \
        --metrics-json  path/to/hard_metrics.json \
        --mode          hard \
        --output-dir    outputs/visualizations/training_curves_hard

SOFT outputs (7 PNGs):
    01_overall_loss.png
    02_val_loss_breakdown.png     ← soft label specific
    03_rpn.png
    04_class_accuracy.png         ← soft label specific
    05_calibration.png            ← soft label specific
    06_iou_quality.png            ← soft label specific
    07_learning_rate.png

HARD outputs (8 PNGs):
    01_overall_loss.png
    02_cls_loss.png
    03_reg_loss.png
    04_rpn.png
    05_iou_accuracy.png
    06_map.png
    07_ece.png
    08_learning_rate.png
"""

import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Colours ────────────────────────────────────────────────────────────────
NAVY   = "#336B87"
ORANGE = "#F59927"
SAGE   = "#2E8B57"
TEAL   = "#028090"
PURPLE = "#8E44AD"
AUTUMN = "#763626"
BEIGE  = "#B5854A"
GREY   = "#95A5A6"
DARK   = "#3E464B"
GRID   = "#F0F0F0"


# ── Helpers ────────────────────────────────────────────────────────────────
def load(data, key):
    """Return numpy array for key, NaN-safe. Returns None if key missing."""
    if key not in data:
        return None
    v = np.array(data[key], dtype=float)
    return v if not np.all(np.isnan(v)) else None


def epochs(v):
    return np.arange(1, len(v) + 1)


def style_ax(ax, title, xlabel="Epoch", ylabel="Value", note=None):
    ax.set_title(title, fontsize=13, color=DARK, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10, color=DARK)
    ax.set_ylabel(ylabel, fontsize=10, color=DARK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(colors=DARK, labelsize=9)
    if note:
        ax.text(0.99, 0.97, note, transform=ax.transAxes,
                ha="right", va="top", fontsize=8,
                color=ORANGE, style="italic")


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


def single_fig():
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.13)
    return fig, ax


def double_fig():
    """Two stacked panels sharing x-axis."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7),
                                    facecolor="white", sharex=True)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.92,
                        bottom=0.08, hspace=0.35)
    return fig, ax1, ax2


# ══════════════════════════════════════════════════════════════════════════
# SOFT LABEL PLOTS
# ══════════════════════════════════════════════════════════════════════════
def soft_plots(data, out):

    # 01 — Overall loss: train + val
    tv = load(data, "train_loss")
    vv = load(data, "val_loss")
    fig, ax = single_fig()
    if tv is not None: ax.plot(epochs(tv), tv, color=NAVY,   lw=1.8, label="train loss")
    if vv is not None: ax.plot(epochs(vv), vv, color=ORANGE, lw=1.8, label="val loss")
    style_ax(ax, "Overall Loss — Train vs Val", ylabel="Loss")
    ax.legend(fontsize=9)
    save(fig, f"{out}/01_overall_loss.png")

    # 02 — Val loss breakdown (class / box mean / box std)
    vcl = load(data, "val_class_loss")
    vbm = load(data, "val_box_reg_loss")
    vbs = load(data, "val_box_std_loss")
    fig, ax = single_fig()
    if vcl is not None: ax.plot(epochs(vcl), vcl, color=NAVY,   lw=1.8, label="class loss")
    if vbm is not None: ax.plot(epochs(vbm), vbm, color=SAGE,   lw=1.8, label="box mean loss")
    if vbs is not None: ax.plot(epochs(vbs), vbs, color=PURPLE, lw=1.8, label="box std loss  ★")
    style_ax(ax, "Val Loss Breakdown", ylabel="Loss",
             note="★ box std loss = Gaussian σ learning (soft only)")
    ax.legend(fontsize=9)
    save(fig, f"{out}/02_val_loss_breakdown.png")

    # 03 — RPN: val obj + val reg (two panels)
    vrpno = load(data, "val_rpn_obj")
    vrpnr = load(data, "val_rpn_reg")
    fig, ax1, ax2 = double_fig()
    if vrpno is not None:
        ax1.plot(epochs(vrpno), vrpno, color=GREY, lw=1.8, label="val RPN obj")
    style_ax(ax1, "RPN Objectiveness Loss (val)", ylabel="Loss")
    ax1.legend(fontsize=9)
    if vrpnr is not None:
        ax2.plot(epochs(vrpnr), vrpnr, color=TEAL, lw=1.8, label="val RPN box reg")
    style_ax(ax2, "RPN Box Regression Loss (val)", ylabel="Loss")
    ax2.legend(fontsize=9)
    save(fig, f"{out}/03_rpn.png")

    # 04 — Class accuracy: top1 / top3 / weighted
    t1  = load(data, "top1_accuracy")
    t3  = load(data, "top3_accuracy")
    wac = load(data, "weighted_accuracy")
    fig, ax = single_fig()
    if t1  is not None: ax.plot(epochs(t1),  t1,  color=NAVY,   lw=1.8, label="Top-1 accuracy")
    if t3  is not None: ax.plot(epochs(t3),  t3,  color=TEAL,   lw=1.8, label="Top-3 accuracy", ls="--")
    if wac is not None: ax.plot(epochs(wac), wac, color=SAGE,   lw=1.8, label="Weighted accuracy", ls=":")
    style_ax(ax, "Class Accuracy (val — soft label only)", ylabel="Accuracy",
             note="★ soft label only")
    ax.legend(fontsize=9)
    save(fig, f"{out}/04_class_accuracy.png")

    # 05 — Calibration: ECE + JS divergence (two panels)
    ece = load(data, "calibration_error")
    jsd = load(data, "js_divergence")
    fig, ax1, ax2 = double_fig()
    if ece is not None:
        ax1.plot(epochs(ece), ece, color=AUTUMN, lw=1.8, label="ECE (calibration error)")
    style_ax(ax1, "Expected Calibration Error (val)", ylabel="ECE",
             note="★ soft label only  |  lower = better")
    ax1.legend(fontsize=9)
    if jsd is not None:
        ax2.plot(epochs(jsd), jsd, color=BEIGE, lw=1.8, label="JS Divergence")
    style_ax(ax2, "JS Divergence — Predicted vs GT Probs (val)", ylabel="JS Div",
             note="Lower = predicted class probs closer to soft label GT")
    ax2.legend(fontsize=9)
    save(fig, f"{out}/05_calibration.png")

    # 06 — IoU quality: avg IoU + expected IoU
    aiou = load(data, "val_avg_iou")
    eiou = load(data, "expected_iou")
    fig, ax = single_fig()
    if aiou is not None: ax.plot(epochs(aiou), aiou, color=SAGE,  lw=1.8, label="Avg IoU (matched)")
    if eiou is not None: ax.plot(epochs(eiou), eiou, color=TEAL,  lw=1.8, label="Expected IoU (from σ)", ls="--")
    style_ax(ax, "IoU Quality (val)", ylabel="IoU",
             note="★ Expected IoU from predicted σ = soft label only")
    ax.legend(fontsize=9)
    save(fig, f"{out}/06_iou_quality.png")

    # 07 — Learning rate
    lr = load(data, "learning_rate")
    fig, ax = single_fig()
    if lr is not None: ax.plot(epochs(lr), lr, color=DARK, lw=1.8, label="learning rate")
    style_ax(ax, "Learning Rate Schedule", ylabel="LR")
    ax.legend(fontsize=9)
    save(fig, f"{out}/07_learning_rate.png")


# ══════════════════════════════════════════════════════════════════════════
# HARD LABEL PLOTS
# ══════════════════════════════════════════════════════════════════════════
def hard_plots(data, out):

    # 01 — Overall loss: train + val
    tv = load(data, "train_loss")
    vv = load(data, "val_loss")
    fig, ax = single_fig()
    if tv is not None: ax.plot(epochs(tv), tv, color=NAVY,   lw=1.8, label="train loss")
    if vv is not None: ax.plot(epochs(vv), vv, color=ORANGE, lw=1.8, label="val loss")
    style_ax(ax, "Overall Loss — Train vs Val", ylabel="Loss")
    ax.legend(fontsize=9)
    save(fig, f"{out}/01_overall_loss.png")

    # 02 — Classification loss: train + val
    tcl = load(data, "train_class_loss")
    vcl = load(data, "val_class_loss")
    fig, ax = single_fig()
    if tcl is not None: ax.plot(epochs(tcl), tcl, color=NAVY,   lw=1.8, label="train cls loss")
    if vcl is not None: ax.plot(epochs(vcl), vcl, color=ORANGE, lw=1.8, label="val cls loss")
    style_ax(ax, "Classification Loss — Train vs Val", ylabel="Loss")
    ax.legend(fontsize=9)
    save(fig, f"{out}/02_cls_loss.png")

    # 03 — Regression loss: train + val
    trl = load(data, "train_reg_loss")
    vrl = load(data, "val_reg_loss")
    fig, ax = single_fig()
    if trl is not None: ax.plot(epochs(trl), trl, color=SAGE,   lw=1.8, label="train reg loss")
    if vrl is not None: ax.plot(epochs(vrl), vrl, color=ORANGE, lw=1.8, label="val reg loss")
    style_ax(ax, "Box Regression Loss — Train vs Val", ylabel="Loss")
    ax.legend(fontsize=9)
    save(fig, f"{out}/03_reg_loss.png")

    # 04 — RPN: objectiveness + box reg (two panels)
    trpno = load(data, "train_rpn_objectness")
    vrpno = load(data, "val_rpn_objectness")
    trpnr = load(data, "train_rpn_box_reg")
    vrpnr = load(data, "val_rpn_box_reg")
    fig, ax1, ax2 = double_fig()
    if trpno is not None: ax1.plot(epochs(trpno), trpno, color=GREY,   lw=1.8, label="train")
    if vrpno is not None: ax1.plot(epochs(vrpno), vrpno, color=ORANGE, lw=1.8, label="val")
    style_ax(ax1, "RPN Objectiveness Loss", ylabel="Loss")
    ax1.legend(fontsize=9)
    if trpnr is not None: ax2.plot(epochs(trpnr), trpnr, color=TEAL,   lw=1.8, label="train")
    if vrpnr is not None: ax2.plot(epochs(vrpnr), vrpnr, color=ORANGE, lw=1.8, label="val")
    style_ax(ax2, "RPN Box Regression Loss", ylabel="Loss")
    ax2.legend(fontsize=9)
    save(fig, f"{out}/04_rpn.png")

    # 05 — Val IoU + Val class accuracy
    viou = load(data, "val_avg_iou")
    vacc = load(data, "val_class_accuracy")
    fig, ax = single_fig()
    if viou is not None: ax.plot(epochs(viou), viou, color=SAGE,  lw=1.8, label="Avg IoU")
    if vacc is not None: ax.plot(epochs(vacc), vacc, color=NAVY,  lw=1.8, label="Class accuracy", ls="--")
    style_ax(ax, "IoU & Class Accuracy (val)", ylabel="Value",
             note="Both in [0, 1]")
    ax.legend(fontsize=9)
    save(fig, f"{out}/05_iou_accuracy.png")

    # 06 — mAP@50 + mAP@75
    m50 = load(data, "val_mAP_50")
    m75 = load(data, "val_mAP_75")
    fig, ax = single_fig()
    if m50 is not None: ax.plot(epochs(m50), m50, color=NAVY,   lw=1.8, label="mAP@50")
    if m75 is not None: ax.plot(epochs(m75), m75, color=AUTUMN, lw=1.8, label="mAP@75", ls="--")
    style_ax(ax, "mAP — Validation", ylabel="mAP",
             note="NaN at early epochs = no detections above threshold yet")
    ax.legend(fontsize=9)
    save(fig, f"{out}/06_map.png")

    # 07 — ECE
    ece = load(data, "val_ece")
    fig, ax = single_fig()
    if ece is not None: ax.plot(epochs(ece), ece, color=AUTUMN, lw=1.8, label="ECE")
    style_ax(ax, "Expected Calibration Error (val)", ylabel="ECE",
             note="Lower = better calibration")
    ax.legend(fontsize=9)
    save(fig, f"{out}/07_ece.png")

    # 08 — Learning rate
    lr = load(data, "learning_rate")
    fig, ax = single_fig()
    if lr is not None: ax.plot(epochs(lr), lr, color=DARK, lw=1.8, label="learning rate")
    style_ax(ax, "Learning Rate Schedule", ylabel="LR")
    ax.legend(fontsize=9)
    save(fig, f"{out}/08_learning_rate.png")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-json", required=True)
    ap.add_argument("--mode",         required=True, choices=["soft","hard"])
    ap.add_argument("--output-dir",   required=True,
                    help="Directory where individual PNGs are saved")
    args = ap.parse_args()

    with open(args.metrics_json) as f:
        data = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nMode : {args.mode}")
    print(f"Keys : {len(data)}")
    print(f"Out  : {args.output_dir}\n")

    if args.mode == "soft":
        soft_plots(data, args.output_dir)
    else:
        hard_plots(data, args.output_dir)

    print(f"\nDone — all plots saved to: {args.output_dir}")

if __name__ == "__main__":
    main()