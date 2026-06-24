"""
visualize_metrics.py
Generates four publication-ready charts for the NuCLS soft label detection project.

Usage:
    python visualize_metrics.py --output-dir outputs/visualizations
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── colour palette ────────────────────────────────────────────────────────────
C_NAVY   = "#1A3C5E"
C_TEAL   = "#028090"
C_GREEN  = "#2E8B57"
C_AMBER  = "#E67E22"
C_RED    = "#C0392B"
C_GREY   = "#7F8C8D"
C_LIGHT  = "#ECF0F1"
C_WHITE  = "#FFFFFF"

RUN_COLORS = {
    "Run 2": "#85C1E9",
    "V3":    "#5DADE2",
    "V5":    "#C0392B",
    "V6*":   "#8E44AD",
    "V7":    "#1ABC9C",
    "V8":    C_TEAL,
    "V9":    C_AMBER,
}

# ── known results (true test set 17 images, except V6 which used val set) ─────
RUNS_MAP = {
    "Run 2": 0.059,
    "V3":    0.059,
    "V5":    0.041,
    "V6*":   0.112,   # * val set — not directly comparable
    "V7":    0.055,
    "V8":    0.098,
    "V9":    0.100,
}

PERCLASS_V8 = {
    "tumor":          0.1838,
    "fibroblast":     0.0382,
    "vascular_end":   0.0011,
    "macrophage":     0.3341,
    "lymphocyte":     0.0982,
    "plasma_cell":    0.0242,
    "apoptotic":      0.0000,
    "unlabeled":      0.0125,
}
PERCLASS_V9 = {
    "tumor":          0.1571,
    "fibroblast":     0.0452,
    "vascular_end":   0.0000,
    "macrophage":     0.1733,
    "lymphocyte":     0.1007,
    "plasma_cell":    0.0938,
    "apoptotic":      0.1111,
    "unlabeled":      0.0123,
}

TIDE_DATA = {
    # (missed_gt%, classification%, localization%, other%)
    "V7": (93.1, 25.3, 11.1, 3.5),
    "V8": (70.3, 18.9,  8.4, 2.4),
    "V9": (49.1, 16.2, 16.5, 2.0),
}

CALIBRATION = {
    # (Std-Error Corr, Calib Ratio)
    "Run 2": (0.426, 1.583),
    "V3":    (0.606, 1.280),
    "V7":    (0.664, 1.170),
    "V8":    (0.566, 1.410),
    "V9":    (0.447, 1.402),
}


def fig_map_progression(output_dir):
    """Bar chart: mAP@50 across all runs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_LIGHT)

    labels = list(RUNS_MAP.keys())
    values = list(RUNS_MAP.values())
    colors = [RUN_COLORS[r] for r in labels]

    bars = ax.bar(labels, values, color=colors, width=0.6, zorder=3, edgecolor="white", linewidth=0.8)

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold", color=C_NAVY)

    # Highlight V8 and V9
    bars[5].set_edgecolor(C_NAVY); bars[5].set_linewidth(2.5)
    bars[6].set_edgecolor(C_NAVY); bars[6].set_linewidth(2.5)

    ax.axhline(0.10, color=C_RED, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(6.5, 0.102, "0.10 target", color=C_RED, fontsize=9)

    ax.set_ylabel("mAP@50", fontsize=12, color=C_NAVY)
    ax.set_title("mAP@50 Progression Across Runs\n(V6* evaluated on val set — not directly comparable)",
                 fontsize=13, color=C_NAVY, pad=12)
    ax.set_ylim(0, 0.14)
    ax.tick_params(colors=C_NAVY)
    ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "map_progression.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_perclass_ap(output_dir):
    """Grouped bar chart: per-class AP V8 vs V9."""
    classes = list(PERCLASS_V8.keys())
    v8_vals = [PERCLASS_V8[c] for c in classes]
    v9_vals = [PERCLASS_V9[c] for c in classes]

    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_LIGHT)

    b1 = ax.bar(x - w/2, v8_vals, w, label="V8 (ImageNet backbone)", color=C_TEAL,
                zorder=3, edgecolor="white")
    b2 = ax.bar(x + w/2, v9_vals, w, label="V9 (Lunit backbone + 8px anchor)", color=C_AMBER,
                zorder=3, edgecolor="white")

    # Annotate improvements
    for i, (v8v, v9v) in enumerate(zip(v8_vals, v9_vals)):
        delta = v9v - v8v
        if abs(delta) > 0.01:
            sym = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
            col = C_GREEN if delta > 0 else C_RED
            ax.text(i + w/2, v9v + 0.005, sym, ha="center", va="bottom",
                    fontsize=8, color=col, fontweight="bold")

    # Special annotation for apoptotic
    ax.annotate("First non-zero\napoptotic AP!", xy=(6 + w/2, 0.111),
                xytext=(5.5, 0.18), fontsize=8, color=C_GREEN,
                arrowprops=dict(arrowstyle="->", color=C_GREEN),
                ha="center")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=9)
    ax.set_ylabel("AP@50", fontsize=12, color=C_NAVY)
    ax.set_title("Per-Class AP@50: V8 vs V9", fontsize=13, color=C_NAVY, pad=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 0.42)
    ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=C_NAVY)

    plt.tight_layout()
    path = os.path.join(output_dir, "perclass_ap_v8_v9.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_tide_breakdown(output_dir):
    """Stacked bar: TIDE error composition V7 → V8 → V9."""
    runs   = list(TIDE_DATA.keys())
    missed = [TIDE_DATA[r][0] for r in runs]
    cls    = [TIDE_DATA[r][1] for r in runs]
    loc    = [TIDE_DATA[r][2] for r in runs]
    other  = [TIDE_DATA[r][3] for r in runs]

    x = np.arange(len(runs))
    w = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_LIGHT)

    p1 = ax.bar(x, missed, w, label="Missed GT",       color=C_RED,   zorder=3)
    p2 = ax.bar(x, cls,    w, bottom=missed,            label="Classification", color=C_AMBER, zorder=3)
    p3 = ax.bar(x, loc,    w, bottom=np.array(missed)+np.array(cls),
                label="Localization", color=C_TEAL, zorder=3)
    p4 = ax.bar(x, other,  w, bottom=np.array(missed)+np.array(cls)+np.array(loc),
                label="Other",        color=C_GREY, zorder=3)

    for i, r in enumerate(runs):
        m, c, l, o = TIDE_DATA[r]
        ax.text(i, m/2,           f"{m:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(i, m + c/2,       f"{c:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        ax.text(i, m + c + l/2,   f"{l:.0f}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=12)
    ax.set_ylabel("% of AP Loss", fontsize=12, color=C_NAVY)
    ax.set_title("TIDE Error Composition — What Limits mAP", fontsize=13, color=C_NAVY, pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=C_NAVY)

    plt.tight_layout()
    path = os.path.join(output_dir, "tide_breakdown.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def fig_calibration(output_dir):
    """Scatter: Std-Error Correlation vs Calibration Ratio across runs."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(C_WHITE)
    ax.set_facecolor(C_LIGHT)

    for run, (corr, ratio) in CALIBRATION.items():
        color = RUN_COLORS.get(run, C_GREY)
        ax.scatter(corr, ratio, s=180, color=color, zorder=5, edgecolor=C_NAVY, linewidth=1)
        ax.annotate(run, (corr, ratio), textcoords="offset points",
                    xytext=(8, 4), fontsize=10, color=C_NAVY, fontweight="bold")

    ax.axhline(1.0, color=C_GREEN, linestyle="--", linewidth=1.2, zorder=2)
    ax.text(0.66, 1.01, "Perfect calibration", color=C_GREEN, fontsize=8)

    ax.set_xlabel("Std-Error Correlation ↑ (better)", fontsize=11, color=C_NAVY)
    ax.set_ylabel("Calibration Ratio ↓ (closer to 1.0 = better)", fontsize=11, color=C_NAVY)
    ax.set_title("Bbox Uncertainty Calibration Across Runs", fontsize=13, color=C_NAVY, pad=12)
    ax.tick_params(colors=C_NAVY)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")

    plt.tight_layout()
    path = os.path.join(output_dir, "calibration_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/visualizations")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating visualizations...")
    fig_map_progression(args.output_dir)
    fig_perclass_ap(args.output_dir)
    fig_tide_breakdown(args.output_dir)
    fig_calibration(args.output_dir)
    print(f"\nDone. 4 charts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
