"""
viz_eval_metrics.py  — clean simple plots, one PNG per group

Usage:
    python viz_eval_metrics.py --output-dir outputs/visualizations/eval_metrics

Outputs (8 PNGs):
    01_detection.png
    02_calibration_quality.png
    03_class_accuracy.png
    04_uncertainty.png
    05_pdq.png
    06_lrp.png
    07_tide.png
    08_per_class_ap.png
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Colours ────────────────────────────────────────────────────────────────
NAVY   = "#336B87"
TEAL   = "#028090"
SAGE   = "#2E8B57"
ORANGE = "#F59927"
AUTUMN = "#763626"
BEIGE  = "#B5854A"
PURPLE = "#8E44AD"
GREY   = "#95A5A6"
DARK   = "#3E464B"
BG     = "#FFFFFF"
GRID   = "#F0F0F0"

# ── V13 confirmed values ───────────────────────────────────────────────────
V = dict(
    map50          = 0.207,
    map75          = 0.100,
    avg_iou        = 0.768,
    ece            = 0.123,
    js_div         = 0.240,
    cosine_sim     = 0.683,
    kl_div         = 0.950,
    top1           = 0.543,
    top3           = 0.727,
    weighted_acc   = 0.505,
    std_err_corr   = 0.475,
    avg_pred_std   = 4.467,
    avg_actual_err = 2.695,
    calib_ratio    = 1.658,
    pdq            = 0.139,
    pdq_spatial    = 0.371,
    pdq_label      = 0.362,
    molrp          = 0.895,
    molrp_iou      = 0.514,
    molrp_fp       = 0.647,
    molrp_fn       = 0.583,
)

# ── Averages V8–V13 (main metrics only) ───────────────────────────────────
AVG = dict(
    map50   = 0.165,
    map75   = 0.055,
    avg_iou = 0.728,
    ece     = 0.210,
    pdq     = 0.119,
    molrp   = 0.923,
)

# ── Per-class data ─────────────────────────────────────────────────────────
CLASSES    = ["Tumor","Fibroblast","Macrophage","Lymphocyte",
              "Apoptotic","Plasma cell","Vasc. end.","Unlabeled"]
AP50       = [0.425, 0.262, 0.301, 0.265, 0.067, 0.039, 0.010, 0.029]

LRP_CLASSES = ["Tumor","Fibroblast","Macrophage","Lymphocyte",
               "Plasma cell","Apoptotic","Vasc. end.","Unlabeled",
               "Mitotic fig.","Myoepi."]
OLRP     = [0.749, 0.863, 0.822, 0.824, 0.956, 0.992, 0.988, 0.967, 1.000, 1.000]
OLRP_FP  = [0.270, 0.526, 0.571, 0.292, 0.922, 0.900, 0.833, 0.857, 1.000, 1.000]
OLRP_FN  = [0.441, 0.573, 0.571, 0.646, 0.600, 0.000, 0.974, 0.857, 0.000, 0.000]
OLRP_IOU = [0.458, 0.529, 0.348, 0.431, 0.373, 0.915, 0.479, 0.577, 0.0,   0.0  ]

TIDE_LABELS  = ["True Pos","Cls Error","Loc Error",
                "Both","Duplicate","Background","Missed GT"]
TIDE_COUNTS  = [302, 254, 42, 65, 635, 138, 425]
TIDE_DELTA   = [None, +0.158, +0.016, +0.003, -0.040, +0.007, +0.409]
TIDE_COLORS  = [SAGE, ORANGE, NAVY, PURPLE, AUTUMN, BEIGE, GREY]


# ══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════
def base_style(ax):
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=DARK, labelsize=10)
    ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)


def hbar(ax, labels, values, colors,
         averages=None, xlim=None, xlabel="Value",
         lower_better_indices=None):
    """
    Clean horizontal bar chart.
    averages: dict {index: avg_value} for metrics that have an average.
    lower_better_indices: list of bar indices where lower = better.
    """
    n = len(labels)
    y = np.arange(n)
    bars = ax.barh(y, values, color=colors,
                   height=0.50, zorder=3,
                   edgecolor="white", linewidth=0.8)

    # Average marker — thin vertical line through the bar
    if averages:
        for idx, avg in averages.items():
            ax.plot([avg, avg],
                    [idx - 0.30, idx + 0.30],
                    color=DARK, lw=2.0, zorder=5,
                    solid_capstyle="round")

    # Value labels
    xmax = xlim[1] if xlim else max(values) * 1.25
    for i, val in enumerate(values):
        ax.text(val + xmax * 0.012, i, f"{val:.3f}",
                va="center", ha="left",
                fontsize=10, color=DARK, fontweight="bold")

    # Y-axis labels — append direction hint where lower is better
    if lower_better_indices:
        new_labels = []
        for i, lbl in enumerate(labels):
            if i in lower_better_indices:
                new_labels.append(lbl + "  ↓")
            else:
                new_labels.append(lbl)
        ax.set_yticks(y)
        ax.set_yticklabels(new_labels, fontsize=10.5, color=DARK)
    else:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10.5, color=DARK)

    ax.set_xlabel(xlabel, fontsize=10, color=DARK)
    if xlim:
        ax.set_xlim(xlim)

    # Legend for average marker
    if averages:
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[Line2D([0],[0], color=DARK, lw=2, label="avg V8–V13")],
            fontsize=9, loc="lower right",
            framealpha=0.85, edgecolor="#CCCCCC")


def save_fig(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 01 — Detection: mAP@50, mAP@75, Avg IoU
# ══════════════════════════════════════════════════════════════════════════
def plot_detection(out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    base_style(ax)
    ax.set_title("Detection Quality — mAP@50 / mAP@75 / Avg IoU",
                 fontsize=13, color=DARK, fontweight="bold", pad=10)

    labels = ["mAP@50", "mAP@75", "Avg IoU"]
    values = [V["map50"], V["map75"], V["avg_iou"]]
    colors = [NAVY, TEAL, SAGE]
    avgs   = {0: AVG["map50"], 1: AVG["map75"], 2: AVG["avg_iou"]}

    hbar(ax, labels, values, colors,
         averages=avgs, xlim=(0, 1.08), xlabel="Value  (higher is better)")
    plt.tight_layout()
    save_fig(fig, "01_detection.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 02 — Calibration & distribution quality: ECE, JS Div, Cosine Sim
# ══════════════════════════════════════════════════════════════════════════
def plot_calibration_quality(out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    base_style(ax)
    ax.set_title("Calibration & Class Distribution Quality",
                 fontsize=13, color=DARK, fontweight="bold", pad=10)

    labels = ["ECE", "JS Divergence", "Cosine Similarity"]
    values = [V["ece"], V["js_div"], V["cosine_sim"]]
    colors = [AUTUMN, BEIGE, TEAL]
    avgs   = {0: AVG["ece"]}

    hbar(ax, labels, values, colors,
         averages=avgs, xlim=(0, 1.08), xlabel="Value",
         lower_better_indices=[0, 1])
    plt.tight_layout()
    save_fig(fig, "02_calibration_quality.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 03 — Class accuracy: Top-1, Top-3, Weighted Accuracy
# ══════════════════════════════════════════════════════════════════════════
def plot_class_accuracy(out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    base_style(ax)
    ax.set_title("Class Accuracy — Top-1 / Top-3 / Weighted",
                 fontsize=13, color=DARK, fontweight="bold", pad=10)

    labels = ["Top-1 Accuracy", "Top-3 Accuracy", "Weighted Accuracy"]
    values = [V["top1"], V["top3"], V["weighted_acc"]]
    colors = [NAVY, TEAL, SAGE]

    hbar(ax, labels, values, colors,
         xlim=(0, 1.08), xlabel="Accuracy  (higher is better)")
    plt.tight_layout()
    save_fig(fig, "03_class_accuracy.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 04 — Uncertainty quantification (3 sub-panels)
# ══════════════════════════════════════════════════════════════════════════
def plot_uncertainty(out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), facecolor=BG)
    fig.suptitle("Spatial Uncertainty Quantification",
                 fontsize=13, color=DARK, fontweight="bold", y=1.01)

    # Panel 1: Std-Error Correlation (0-1)
    ax = axes[0]
    base_style(ax)
    ax.set_title("Std-Error Correlation", fontsize=11,
                 color=DARK, fontweight="bold", pad=8)
    ax.barh([0], [V["std_err_corr"]], color=PURPLE,
            height=0.40, zorder=3, edgecolor="white")
    ax.barh([0], [1.0], color="#EEEEEE",
            height=0.40, zorder=2, edgecolor="white")
    ax.barh([0], [V["std_err_corr"]], color=PURPLE,
            height=0.40, zorder=3, edgecolor="white")
    ax.set_xlim(0, 1.2)
    ax.set_yticks([0])
    ax.set_yticklabels(["Corr"], fontsize=10, color=DARK)
    ax.set_xlabel("Correlation (higher is better)", fontsize=9, color=DARK)
    ax.text(V["std_err_corr"] + 0.02, 0,
            f"{V['std_err_corr']:.3f}",
            va="center", fontsize=11, fontweight="bold", color=DARK)

    # Panel 2: Pred σ vs Actual Error (pixel values)
    ax = axes[1]
    base_style(ax)
    ax.set_title("Predicted σ vs Actual Error", fontsize=11,
                 color=DARK, fontweight="bold", pad=8)
    bars = ax.barh([1, 0],
                   [V["avg_pred_std"], V["avg_actual_err"]],
                   color=[PURPLE, SAGE],
                   height=0.40, zorder=3, edgecolor="white")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Error", "Predicted σ"],
                       fontsize=10, color=DARK)
    ax.set_xlabel("Pixels", fontsize=9, color=DARK)
    ax.set_xlim(0, 6.5)
    for bar, val in zip(bars, [V["avg_pred_std"], V["avg_actual_err"]]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}px", va="center",
                fontsize=11, fontweight="bold", color=DARK)

    # Panel 3: Calibration Ratio
    ax = axes[2]
    base_style(ax)
    ax.set_title("Calibration Ratio  (σ / actual error)", fontsize=11,
                 color=DARK, fontweight="bold", pad=8)
    color_cr = AUTUMN if V["calib_ratio"] > 1.2 else SAGE
    ax.barh([0], [V["calib_ratio"]], color=color_cr,
            height=0.40, zorder=3, edgecolor="white")
    # Reference line at 1.0
    ax.axvline(1.0, color=DARK, lw=1.5,
               linestyle="--", zorder=5)
    ax.text(1.02, 0.55, "ideal = 1.0",
            fontsize=8.5, color=DARK, va="center")
    ax.set_xlim(0, 2.5)
    ax.set_yticks([0])
    ax.set_yticklabels(["Ratio"], fontsize=10, color=DARK)
    ax.set_xlabel("Ratio  (1.0 = perfectly calibrated)", fontsize=9, color=DARK)
    ax.text(V["calib_ratio"] + 0.05, 0,
            f"{V['calib_ratio']:.3f}",
            va="center", fontsize=11, fontweight="bold", color=DARK)

    plt.tight_layout()
    save_fig(fig, "04_uncertainty.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 05 — PDQ: PDQ Score, Spatial Q, Label Q
# ══════════════════════════════════════════════════════════════════════════
def plot_pdq(out_dir):
    fig, ax = plt.subplots(figsize=(8, 3.8), facecolor=BG)
    base_style(ax)
    ax.set_title("Probabilistic Detection Quality — PDQ",
                 fontsize=13, color=DARK, fontweight="bold", pad=10)

    labels = ["PDQ Score", "Spatial Quality", "Label Quality"]
    values = [V["pdq"], V["pdq_spatial"], V["pdq_label"]]
    colors = [NAVY, SAGE, TEAL]
    avgs   = {0: AVG["pdq"]}

    hbar(ax, labels, values, colors,
         averages=avgs, xlim=(0, 0.65),
         xlabel="Score  (higher is better)")
    plt.tight_layout()
    save_fig(fig, "05_pdq.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 06 — LRP (overview + per-class)
# ══════════════════════════════════════════════════════════════════════════
def plot_lrp(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle("Localisation Recall Precision — LRP  (lower is better)",
                 fontsize=13, color=DARK, fontweight="bold", y=1.01)

    # Left: moLRP + sub-metrics
    ax = axes[0]
    base_style(ax)
    ax.set_title("moLRP and Sub-components", fontsize=11,
                 color=DARK, fontweight="bold", pad=8)

    labels = ["moLRP", "moLRP_IoU", "moLRP_FP", "moLRP_FN"]
    values = [V["molrp"], V["molrp_iou"], V["molrp_fp"], V["molrp_fn"]]
    colors = [NAVY, SAGE, ORANGE, AUTUMN]
    y = np.arange(4)
    ax.barh(y, values, color=colors, height=0.50,
            zorder=3, edgecolor="white")

    # Average for moLRP
    ax.plot([AVG["molrp"], AVG["molrp"]],
            [-0.30, 0.30], color=DARK, lw=2.0, zorder=5)

    for i, val in enumerate(values):
        ax.text(val + 0.008, i, f"{val:.3f}",
                va="center", fontsize=10,
                fontweight="bold", color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, color=DARK)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Value  (lower is better)", fontsize=10, color=DARK)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], color=DARK, lw=2,
                               label="avg V8–V13 (moLRP)")],
              fontsize=9, loc="lower right",
              framealpha=0.85, edgecolor="#CCCCCC")

    # Right: per-class oLRP — sorted, single consistent colormap
    ax = axes[1]
    base_style(ax)
    ax.set_title("Per-class oLRP", fontsize=11,
                 color=DARK, fontweight="bold", pad=8)

    # Sort by oLRP ascending (best first)
    pairs = sorted(zip(OLRP, LRP_CLASSES), reverse=False)
    sorted_olrp, sorted_cls = zip(*pairs)

    n = len(sorted_cls)
    y2 = np.arange(n)

    # Single color gradient: good (low oLRP) = SAGE, bad (high) = AUTUMN
    cmap = plt.cm.RdYlGn_r
    norm_vals = [(v - min(OLRP)) / (max(OLRP) - min(OLRP) + 1e-8)
                 for v in sorted_olrp]
    bar_colors = [cmap(nv) for nv in norm_vals]

    bars = ax.barh(y2, sorted_olrp, color=bar_colors,
                   height=0.55, zorder=3, edgecolor="white")
    ax.set_yticks(y2)
    ax.set_yticklabels(sorted_cls, fontsize=10, color=DARK)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("oLRP  (lower is better)", fontsize=10, color=DARK)

    for bar, val in zip(bars, sorted_olrp):
        ax.text(val + 0.008, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center",
                fontsize=9.5, fontweight="bold", color=DARK)

    # Reference line at 1.0 (no detection)
    ax.axvline(1.0, color=GREY, lw=1.2,
               linestyle="--", zorder=4)
    ax.text(1.01, n - 0.5, "oLRP=1\n(no detect.)",
            fontsize=7.5, color=GREY, va="top")

    plt.tight_layout()
    save_fig(fig, "06_lrp.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 07 — TIDE
# ══════════════════════════════════════════════════════════════════════════
def plot_tide(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=BG)
    fig.suptitle("TIDE — Error Type Decomposition",
                 fontsize=13, color=DARK, fontweight="bold", y=1.01)

    # Left: error counts
    ax = axes[0]
    base_style(ax)
    ax.set_title("Error Counts", fontsize=11,
                 color=DARK, fontweight="bold", pad=8)

    x = np.arange(len(TIDE_LABELS))
    bars = ax.bar(x, TIDE_COUNTS, color=TIDE_COLORS,
                  zorder=3, edgecolor="white", linewidth=0.8,
                  width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(TIDE_LABELS, fontsize=10, color=DARK)
    ax.set_ylabel("Count", fontsize=10, color=DARK)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=DARK, labelsize=9)

    for bar, val in zip(bars, TIDE_COUNTS):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 8,
                str(val), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=DARK)

    # Right: ΔAP oracle gains
    ax = axes[1]
    base_style(ax)
    ax.set_title("Oracle ΔAP@50  (gain if error type eliminated)",
                 fontsize=11, color=DARK, fontweight="bold", pad=8)

    d_labels = [l for l, d in zip(TIDE_LABELS, TIDE_DELTA) if d is not None]
    d_vals   = [d for d in TIDE_DELTA if d is not None]
    d_colors = [c for c, d in zip(TIDE_COLORS, TIDE_DELTA) if d is not None]

    x2 = np.arange(len(d_labels))
    bars2 = ax.bar(x2, d_vals, color=d_colors,
                   zorder=3, edgecolor="white", linewidth=0.8,
                   width=0.6)
    ax.axhline(0, color=DARK, lw=1.0, zorder=4)
    ax.set_xticks(x2)
    ax.set_xticklabels(d_labels, fontsize=10, color=DARK)
    ax.set_ylabel("ΔAP@50", fontsize=10, color=DARK)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=DARK, labelsize=9)

    for bar, val in zip(bars2, d_vals):
        offset = 0.005 if val >= 0 else -0.012
        ax.text(bar.get_x() + bar.get_width()/2,
                val + offset,
                f"{val:+.3f}", ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold", color=DARK)

    plt.tight_layout()
    save_fig(fig, "07_tide.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 08 — Per-class AP@50
# ══════════════════════════════════════════════════════════════════════════
def plot_per_class_ap(out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.0), facecolor=BG)
    base_style(ax)
    ax.set_title("Per-class Average Precision @ IoU 0.50",
                 fontsize=13, color=DARK, fontweight="bold", pad=10)

    # Sort descending
    pairs = sorted(zip(AP50, CLASSES), reverse=True)
    sorted_ap, sorted_cls = zip(*pairs)

    y = np.arange(len(sorted_cls))
    # Single color per bar — gradient from good to poor
    cmap = plt.cm.Blues
    norm_vals = [(v - min(sorted_ap)) / (max(sorted_ap) - min(sorted_ap) + 1e-8)
                 for v in sorted_ap]
    bar_colors = [cmap(0.35 + 0.55 * nv) for nv in norm_vals]

    ax.barh(y, sorted_ap, color=bar_colors,
            height=0.55, zorder=3, edgecolor="white")

    # Mean line
    mean_ap = np.mean(sorted_ap)
    ax.axvline(mean_ap, color=DARK, lw=1.5,
               linestyle="--", zorder=5)
    ax.text(mean_ap + 0.005, len(sorted_cls) - 0.5,
            f"mean\n{mean_ap:.3f}",
            fontsize=8.5, color=DARK, va="top")

    ax.set_yticks(y)
    ax.set_yticklabels(sorted_cls, fontsize=10.5, color=DARK)
    ax.set_xlim(0, 0.60)
    ax.set_xlabel("AP@50  (higher is better)", fontsize=10, color=DARK)

    for i, val in enumerate(sorted_ap):
        ax.text(val + 0.006, i, f"{val:.3f}",
                va="center", fontsize=10,
                fontweight="bold", color=DARK)

    plt.tight_layout()
    save_fig(fig, "08_per_class_ap.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",
                    default="outputs/visualizations/eval_metrics_refined")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving plots to: {args.output_dir}\n")

    plot_detection(args.output_dir)
    plot_calibration_quality(args.output_dir)
    plot_class_accuracy(args.output_dir)
    plot_uncertainty(args.output_dir)
    plot_pdq(args.output_dir)
    plot_lrp(args.output_dir)
    plot_tide(args.output_dir)
    plot_per_class_ap(args.output_dir)

    print(f"\nDone — 8 plots saved.")

if __name__ == "__main__":
    main()