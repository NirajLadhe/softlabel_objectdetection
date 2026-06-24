"""
viz_eval_metrics_hard.py

Plots evaluation metrics for the hard label run.
Available metrics: default (mAP, IoU, ECE) and LRP only.
PDQ, Top-K, JS/KL/Cosine, uncertainty — not available for hard labels.

Usage:
    python viz_eval_metrics_hard.py \
        --output-dir outputs/visualizations/eval_metrics_hard

Outputs (3 PNGs):
    01_detection_ece_hard.png
    02_lrp_hard.png
    03_per_class_ap_hard.png
"""
import argparse, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Colours ────────────────────────────────────────────────────────────────
NAVY   = "#336B87"
TEAL   = "#028090"
SAGE   = "#2E8B57"
ORANGE = "#F59927"
AUTUMN = "#763626"
BEIGE  = "#B5854A"
GREY   = "#95A5A6"
DARK   = "#3E464B"
BG     = "#FFFFFF"
GRID   = "#F0F0F0"

# ── Hard label confirmed values ────────────────────────────────────────────
V = dict(
    map50        = 0.2051,
    map75        = 0.0773,
    map40        = 0.2238,
    avg_iou      = 0.7364,
    ece          = 0.2411,
    precision    = 0.5018,
    recall       = 0.6338,
    f1           = 0.5601,
    molrp        = 0.8940,
    molrp_iou    = 0.5041,
    molrp_fp     = 0.5802,
    molrp_fn     = 0.6756,
)

# ── Averages V8–V13 soft label runs (for reference) ───────────────────────
# Hard labels do not have multi-run averages — soft averages shown for context
SOFT_AVG = dict(
    map50   = 0.165,
    map75   = 0.055,
    avg_iou = 0.728,
    ece     = 0.210,
    molrp   = 0.923,
)

# ── Per-class data (hard label LRP eval) ──────────────────────────────────
LRP_CLASSES = ["Tumor","Fibroblast","Lymphocyte","Apoptotic",
               "Unlabeled","Plasma cell","Macrophage","Vasc. end."]
OLRP     = [0.7941, 0.8385, 0.8005, 0.9282, 0.9447, 0.9220, 0.9538, 0.9698]
OLRP_FP  = [0.2644, 0.3559, 0.2636, 0.8333, 0.8190, 0.7391, 0.5000, 0.8667]
OLRP_FN  = [0.5407, 0.5620, 0.5519, 0.5000, 0.6667, 0.8125, 0.8889, 0.8824]
OLRP_IOU = [0.4778, 0.5420, 0.4835, 0.4975, 0.5836, 0.3630, 0.5385, 0.5472]

CLASSES = ["Tumor","Lymphocyte","Fibroblast","Apoptotic",
           "Unlabeled","Plasma cell","Macrophage","Vasc. end."]
AP50    = [0.4022, 0.3583, 0.3015, 0.0833, 0.0830, 0.0643, 0.0370, 0.0201]


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


def save_fig(fig, name, out_dir):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# 01 — Detection + ECE
# ══════════════════════════════════════════════════════════════════════════
def plot_detection_ece(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=BG)
    fig.suptitle("Hard Label — Detection Quality & Calibration",
                 fontsize=13, color=DARK, fontweight="bold", y=1.01)

    # Left: mAP + IoU
    ax = axes[0]
    base_style(ax)
    ax.set_title("mAP@50 / mAP@75 / Avg IoU",
                 fontsize=11, color=DARK, fontweight="bold", pad=8)

    labels = ["mAP@50", "mAP@75", "Avg IoU"]
    values = [V["map50"], V["map75"], V["avg_iou"]]
    colors = [NAVY, TEAL, SAGE]
    avgs   = [SOFT_AVG["map50"], SOFT_AVG["map75"], SOFT_AVG["avg_iou"]]
    y = np.arange(3)

    ax.barh(y, values, color=colors, height=0.50,
            zorder=3, edgecolor="white")

    for i, avg in enumerate(avgs):
        ax.plot([avg, avg], [i-0.30, i+0.30],
                color=DARK, lw=2.0, zorder=5)

    for i, val in enumerate(values):
        ax.text(val + 0.012, i, f"{val:.3f}",
                va="center", fontsize=10,
                fontweight="bold", color=DARK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, color=DARK)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Value  (higher is better)", fontsize=10, color=DARK)

    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0], color=DARK, lw=2,
                               label="soft avg V8–V13")],
              fontsize=9, loc="lower right",
              framealpha=0.85, edgecolor="#CCCCCC")

    # Right: ECE
    ax = axes[1]
    base_style(ax)
    ax.set_title("Expected Calibration Error — ECE  ↓",
                 fontsize=11, color=DARK, fontweight="bold", pad=8)

    ax.barh([0], [V["ece"]], color=AUTUMN,
            height=0.45, zorder=3, edgecolor="white")
    # Soft average reference
    ax.plot([SOFT_AVG["ece"], SOFT_AVG["ece"]],
            [-0.28, 0.28], color=DARK, lw=2.0, zorder=5)

    ax.set_yticks([0])
    ax.set_yticklabels(["ECE"], fontsize=10.5, color=DARK)
    ax.set_xlim(0, 0.70)
    ax.set_xlabel("Value  (lower is better)", fontsize=10, color=DARK)
    ax.text(V["ece"] + 0.012, 0, f"{V['ece']:.4f}",
            va="center", fontsize=11, fontweight="bold", color=DARK)

    ax.legend(handles=[Line2D([0],[0], color=DARK, lw=2,
                               label="soft avg V8–V13")],
              fontsize=9, loc="lower right",
              framealpha=0.85, edgecolor="#CCCCCC")

    plt.tight_layout()
    save_fig(fig, "01_detection_ece_hard.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 02 — LRP
# ══════════════════════════════════════════════════════════════════════════
def plot_lrp(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=BG)
    fig.suptitle("Hard Label — LRP  (lower is better)",
                 fontsize=13, color=DARK, fontweight="bold", y=1.01)

    # Left: moLRP + sub-metrics
    ax = axes[0]
    base_style(ax)
    ax.set_title("moLRP and Sub-components",
                 fontsize=11, color=DARK, fontweight="bold", pad=8)

    labels = ["moLRP", "moLRP_IoU", "moLRP_FP", "moLRP_FN"]
    values = [V["molrp"], V["molrp_iou"], V["molrp_fp"], V["molrp_fn"]]
    colors = [NAVY, SAGE, ORANGE, AUTUMN]
    y = np.arange(4)

    ax.barh(y, values, color=colors, height=0.50,
            zorder=3, edgecolor="white")

    # Soft average for moLRP
    ax.plot([SOFT_AVG["molrp"], SOFT_AVG["molrp"]],
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
                               label="soft avg V8–V13 (moLRP)")],
              fontsize=9, loc="lower right",
              framealpha=0.85, edgecolor="#CCCCCC")

    # Right: per-class oLRP sorted
    ax = axes[1]
    base_style(ax)
    ax.set_title("Per-class oLRP",
                 fontsize=11, color=DARK, fontweight="bold", pad=8)

    pairs = sorted(zip(OLRP, LRP_CLASSES))
    sorted_olrp, sorted_cls = zip(*pairs)

    n = len(sorted_cls)
    y2 = np.arange(n)
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

    ax.axvline(1.0, color=GREY, lw=1.2,
               linestyle="--", zorder=4)
    ax.text(1.01, n - 0.5, "oLRP=1\n(no detect.)",
            fontsize=7.5, color=GREY, va="top")

    plt.tight_layout()
    save_fig(fig, "02_lrp_hard.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
# 03 — Per-class AP@50
# ══════════════════════════════════════════════════════════════════════════
def plot_per_class_ap(out_dir):
    fig, ax = plt.subplots(figsize=(8, 5.0), facecolor=BG)
    base_style(ax)
    ax.set_title("Hard Label — Per-class AP@50",
                 fontsize=13, color=DARK, fontweight="bold", pad=10)

    pairs = sorted(zip(AP50, CLASSES), reverse=True)
    sorted_ap, sorted_cls = zip(*pairs)

    y = np.arange(len(sorted_cls))
    cmap = plt.cm.Blues
    norm_vals = [(v - min(sorted_ap)) /
                 (max(sorted_ap) - min(sorted_ap) + 1e-8)
                 for v in sorted_ap]
    bar_colors = [cmap(0.35 + 0.55 * nv) for nv in norm_vals]

    ax.barh(y, sorted_ap, color=bar_colors,
            height=0.55, zorder=3, edgecolor="white")

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
    save_fig(fig, "03_per_class_ap_hard.png", out_dir)


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",
                    default="outputs/visualizations/eval_metrics_hard")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nSaving hard label plots to: {args.output_dir}\n")

    plot_detection_ece(args.output_dir)
    plot_lrp(args.output_dir)
    plot_per_class_ap(args.output_dir)

    print(f"\nDone — 3 plots saved.")

if __name__ == "__main__":
    main()