"""
viz_metrics_charts.py
Generates all evaluation metric charts for the NuCLS presentation.
Data is hardcoded from actual evaluation runs V8–V12.

Produces:
  1. map_progression.png       — mAP@50 bar chart V8→V12
  2. perclass_ap.png           — per-class AP grouped bars (best 3 runs)
  3. tide_breakdown.png        — TIDE error stacked bars V9→V12
  4. calibration_metrics.png   — ECE + PDQ + mAP@75 scatter/bars
  5. detection_stats.png       — TP / FP / FN across runs
  6. soft_label_metrics.png    — Top-1, PDQ Label Q, ECE across runs

Usage:
    python viz_metrics_charts.py --output-dir outputs/visualizations/presentation
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Brand palette ──────────────────────────────────────────────────────────────
P   = "#336B87"   # primary navy
S   = "#4F85A0"   # secondary light blue
SUP = "#3E464B"   # support dark
PC  = "#B5854A"   # primary-c beige
SC  = "#F59927"   # secondary-c orange
SPC = "#75644D"   # support-c brown
ACC = "#763626"   # accent autumn red
OK  = "#A9FC6D"   # okay accent green
BG  = "#F7F7F7"   # background

RUN_COLORS = {
    "V8":  S,
    "V9":  PC,
    "V10": P,
    "V11": SC,
    "V12": ACC,
}

# ── Evaluation data (true test set, 17 images) ─────────────────────────────────
DATA = {
    "V8":  {"map50":0.098,"map40":0.116,"map75":0.046,"prec":0.224,"rec":0.593,
            "tp":582,"fp":2014,"fn":399,"iou":0.696,"ece":0.392,
            "pdq":0.097,"labelq":0.442,"spatialq":0.230,
            "top1":64.78,"top3":82.82,"js":0.199,
            "moLRP":0.941,"moLRP_FP":0.736,"moLRP_FN":0.818,
            "tide_tp":380,"tide_cls":202,"tide_loc":61,"tide_dup":9,"tide_bg":35,"tide_miss":399,
            "tumor":0.184,"fibro":0.038,"vasc":0.001,"macro":0.334,
            "lymph":0.098,"plasma":0.024,"apopt":0.000,"unlab":0.013},
    "V9":  {"map50":0.100,"map40":0.123,"map75":0.038,"prec":0.126,"rec":0.632,
            "tp":620,"fp":4307,"fn":361,"iou":0.680,"ece":0.509,
            "pdq":0.059,"labelq":0.484,"spatialq":0.265,
            "top1":66.45,"top3":84.52,"js":0.186,
            "moLRP":0.953,"moLRP_FP":0.829,"moLRP_FN":0.644,
            "tide_tp":414,"tide_cls":206,"tide_loc":1318,"tide_dup":25,"tide_bg":57,"tide_miss":361,
            "tumor":0.157,"fibro":0.045,"vasc":0.000,"macro":0.173,
            "lymph":0.101,"plasma":0.094,"apopt":0.111,"unlab":0.012},
    "V10": {"map50":0.199,"map40":0.208,"map75":0.059,"prec":0.500,"rec":0.511,
            "tp":501,"fp":502,"fn":480,"iou":0.756,"ece":0.064,
            "pdq":0.147,"labelq":0.328,"spatialq":0.370,
            "top1":58.88,"top3":77.25,"js":0.262,
            "moLRP":0.910,"moLRP_FP":0.610,"moLRP_FN":0.718,
            "tide_tp":295,"tide_cls":206,"tide_loc":28,"tide_dup":379,"tide_bg":57,"tide_miss":480,
            "tumor":0.342,"fibro":0.199,"vasc":0.002,"macro":0.266,
            "lymph":0.319,"plasma":0.091,"apopt":0.000,"unlab":0.023},
    "V11": {"map50":0.195,"map40":0.201,"map75":0.038,"prec":0.537,"rec":0.462,
            "tp":453,"fp":391,"fn":528,"iou":0.727,"ece":0.040,
            "pdq":0.140,"labelq":0.345,"spatialq":0.347,
            "top1":65.56,"top3":81.90,"js":0.243,
            "moLRP":0.922,"moLRP_FP":0.622,"moLRP_FN":0.596,
            "tide_tp":297,"tide_cls":156,"tide_loc":23,"tide_dup":276,"tide_bg":43,"tide_miss":528,
            "tumor":0.314,"fibro":0.201,"vasc":0.002,"macro":0.247,
            "lymph":0.263,"plasma":0.036,"apopt":0.067,"unlab":0.026},
    "V12": {"map50":0.190,"map40":0.227,"map75":0.047,"prec":0.361,"rec":0.572,
            "tp":561,"fp":994,"fn":420,"iou":0.741,"ece":0.133,
            "pdq":0.132,"labelq":0.363,"spatialq":0.338,
            "top1":54.90,"top3":74.15,"js":0.236,
            "moLRP":0.917,"moLRP_FP":0.691,"moLRP_FN":0.699,
            "tide_tp":308,"tide_cls":253,"tide_loc":93,"tide_dup":603,"tide_bg":165,"tide_miss":420,
            "tumor":0.402,"fibro":0.215,"vasc":0.006,"macro":0.203,
            "lymph":0.279,"plasma":0.055,"apopt":0.042,"unlab":0.035},
}

RUNS = list(DATA.keys())
CLASSES = ["tumor","fibro","vasc","macro","lymph","plasma","apopt","unlab"]
CLASS_LABELS = ["Tumor","Fibroblast","Vasc. End.","Macrophage",
                "Lymphocyte","Plasma Cell","Apoptotic","Unlabeled"]

# ── Helpers ────────────────────────────────────────────────────────────────────
def style_ax(ax, title="", xlabel="", ylabel="", grid=True):
    ax.set_facecolor("white")
    if title: ax.set_title(title, fontsize=12, color=SUP, pad=8, fontfamily="sans-serif")
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=SUP)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=SUP)
    ax.tick_params(colors=SUP, labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    if grid:
        ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

def save(fig, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved: {path}")

# ── Chart 1: mAP Progression ───────────────────────────────────────────────────
def chart_map_progression(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)
    metrics = [("map50","mAP@50"), ("map40","mAP@40"), ("map75","mAP@75")]

    for ax, (key, label) in zip(axes, metrics):
        vals  = [DATA[r][key] for r in RUNS]
        cols  = [RUN_COLORS[r] for r in RUNS]
        bars  = ax.bar(RUNS, vals, color=cols, width=0.55, zorder=3,
                       edgecolor="white", linewidth=0.8)
        best  = max(range(len(vals)), key=lambda i: vals[i])
        bars[best].set_edgecolor(SUP); bars[best].set_linewidth(2.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.003, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=SUP)
        style_ax(ax, label, ylabel="AP")
        ax.set_ylim(0, max(vals)*1.22)

    plt.tight_layout(pad=2.0)
    save(fig, out)

# ── Chart 2: Per-class AP (V10, V11, V12) ─────────────────────────────────────
def chart_perclass_ap(out):
    highlight_runs = ["V10","V11","V12"]
    x = np.arange(len(CLASSES))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)

    for i, run in enumerate(highlight_runs):
        vals = [DATA[run][c] for c in CLASSES]
        offset = (i - 1) * w
        bars = ax.bar(x + offset, vals, w,
                      label=run, color=RUN_COLORS[run],
                      zorder=3, edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=8.5, rotation=15, ha="right")
    style_ax(ax, "Per-Class AP@50 — V10 / V11 / V12", ylabel="AP@50")
    ax.legend(fontsize=9, framealpha=0.7, edgecolor="#cccccc")
    ax.set_ylim(0, 0.50)

    plt.tight_layout(pad=2.0)
    save(fig, out)

# ── Chart 3: TIDE Error Breakdown ─────────────────────────────────────────────
def chart_tide(out):
    tide_runs = ["V9","V10","V11","V12"]
    keys  = ["tide_tp","tide_cls","tide_loc","tide_dup","tide_bg","tide_miss"]
    labels= ["True Positives","Classification","Localization",
             "Duplicate","Background FP","Missed GT"]
    colors= [OK, SC, S, PC, SPC, ACC]

    totals = {r: sum(DATA[r][k] for k in keys) for r in tide_runs}

    fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
    x = np.arange(len(tide_runs))
    bottoms = np.zeros(len(tide_runs))

    for k, lbl, col in zip(keys, labels, colors):
        vals = np.array([DATA[r][k] / totals[r] * 100 for r in tide_runs])
        bars = ax.bar(x, vals, 0.5, bottom=bottoms, label=lbl, color=col, zorder=3)
        for j, (bar, v) in enumerate(zip(bars, vals)):
            if v > 4:
                ax.text(bar.get_x()+bar.get_width()/2,
                        bottoms[j]+v/2,
                        f"{v:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(tide_runs, fontsize=11)
    style_ax(ax, "TIDE Error Composition", ylabel="% of Total Errors")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.8,
              edgecolor="#cccccc", ncol=2)

    plt.tight_layout(pad=2.0)
    save(fig, out)

# ── Chart 4: Calibration & Probabilistic Metrics ──────────────────────────────
def chart_calibration(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)

    # ECE bar
    ece_vals = [DATA[r]["ece"] for r in RUNS]
    axes[0].bar(RUNS, ece_vals, color=[RUN_COLORS[r] for r in RUNS],
                width=0.55, zorder=3, edgecolor="white")
    axes[0].axhline(0.10, color=ACC, linestyle="--", linewidth=1.2, zorder=2)
    axes[0].text(len(RUNS)-0.5, 0.105, "0.10 target", color=ACC, fontsize=7.5)
    for i, v in enumerate(ece_vals):
        axes[0].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=7.5,
                     fontweight="bold", color=SUP)
    style_ax(axes[0], "ECE (↓ better)", ylabel="ECE")
    axes[0].set_ylim(0, max(ece_vals)*1.25)

    # PDQ bar
    pdq_vals = [DATA[r]["pdq"] for r in RUNS]
    axes[1].bar(RUNS, pdq_vals, color=[RUN_COLORS[r] for r in RUNS],
                width=0.55, zorder=3, edgecolor="white")
    for i, v in enumerate(pdq_vals):
        axes[1].text(i, v+0.003, f"{v:.3f}", ha="center", fontsize=7.5,
                     fontweight="bold", color=SUP)
    style_ax(axes[1], "PDQ Score (↑ better)", ylabel="PDQ")
    axes[1].set_ylim(0, max(pdq_vals)*1.25)

    # PDQ Label Q
    lq_vals = [DATA[r]["labelq"] for r in RUNS]
    axes[2].bar(RUNS, lq_vals, color=[RUN_COLORS[r] for r in RUNS],
                width=0.55, zorder=3, edgecolor="white")
    for i, v in enumerate(lq_vals):
        axes[2].text(i, v+0.005, f"{v:.3f}", ha="center", fontsize=7.5,
                     fontweight="bold", color=SUP)
    style_ax(axes[2], "PDQ Label Quality (↑ better)", ylabel="Label Q")
    axes[2].set_ylim(0, max(lq_vals)*1.25)

    plt.tight_layout(pad=2.0)
    save(fig, out)

# ── Chart 5: Detection Stats TP / FP / FN ─────────────────────────────────────
def chart_detection_stats(out):
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    x = np.arange(len(RUNS))
    w = 0.25

    tp_vals = [DATA[r]["tp"] for r in RUNS]
    fp_vals = [DATA[r]["fp"] for r in RUNS]
    fn_vals = [DATA[r]["fn"] for r in RUNS]

    b1 = ax.bar(x - w, tp_vals, w, label="TP", color=P,   zorder=3, edgecolor="white")
    b2 = ax.bar(x,     fp_vals, w, label="FP", color=ACC, zorder=3, edgecolor="white")
    b3 = ax.bar(x + w, fn_vals, w, label="FN", color=SC,  zorder=3, edgecolor="white")

    ax.axhline(981, color="#aaaaaa", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(len(RUNS)-0.2, 995, "GT=981", color="#888888", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(RUNS, fontsize=11)
    style_ax(ax, "Detection Statistics per Run", ylabel="Count")
    ax.legend(fontsize=9, framealpha=0.8, edgecolor="#cccccc")

    plt.tight_layout(pad=2.0)
    save(fig, out)

# ── Chart 6: Soft Label Metrics ────────────────────────────────────────────────
def chart_soft_metrics(out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), facecolor=BG)

    metrics = [
        ("top1",    "Top-1 Accuracy (%) ↑", True,  "%"),
        ("js",      "JS Divergence ↓",      False, ""),
        ("moLRP_FP","moLRP FP component ↓", False, ""),
    ]

    for ax, (key, label, higher, unit) in zip(axes, metrics):
        vals = [DATA[r][key] for r in RUNS]
        cols = [RUN_COLORS[r] for r in RUNS]
        best = max(range(len(vals)), key=lambda i: vals[i]) if higher else \
               min(range(len(vals)), key=lambda i: vals[i])
        bars = ax.bar(RUNS, vals, color=cols, width=0.55, zorder=3,
                      edgecolor="white", linewidth=0.8)
        bars[best].set_edgecolor(SUP); bars[best].set_linewidth(2.2)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    v + (max(vals)*0.02), f"{v:.3f}{unit}",
                    ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=SUP)
        style_ax(ax, label, ylabel=unit or "Score")
        ax.set_ylim(0, max(vals)*1.25)

    plt.tight_layout(pad=2.0)
    save(fig, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/visualizations/presentation")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    d = args.output_dir

    chart_map_progression   (os.path.join(d, "map_progression.png"))
    chart_perclass_ap       (os.path.join(d, "perclass_ap.png"))
    chart_tide              (os.path.join(d, "tide_breakdown.png"))
    chart_calibration       (os.path.join(d, "calibration_metrics.png"))
    chart_detection_stats   (os.path.join(d, "detection_stats.png"))
    chart_soft_metrics      (os.path.join(d, "soft_label_metrics.png"))

    print(f"\nAll 6 charts saved to: {d}")

if __name__ == "__main__":
    main()
