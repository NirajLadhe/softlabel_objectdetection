"""
viz_cluster.py  (v7)

Produces two images:

1. cluster_disagreement.png  — grid of individual annotator panels (UNCHANGED)
2. cluster_consensus.png     — LEFT: H&E crop with all annotator boxes (labelled
                               by rater_id only) + thick mean box labelled "Mean"
                               RIGHT: pie chart of class_probs distribution

Usage:
    python viz_cluster.py \
        --softlabel-json  outputs/nucls_transform_0.001/1d_gaussian_softlabels/1DGMM_softLabels_iou0_25_with_priors.json \
        --img-dir         /media/klee_ro/public_datasets/nucls/images \
        --num-annotators  10 \
        --crop-padding    80 \
        --output          outputs/visualizations/presentation/cluster.png
"""
import argparse, json, os, math
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# ── Brand palette ──────────────────────────────────────────────────────────────
P  = "#336B87"; S  = "#4F85A0"; PC = "#B5854A"; SC = "#F59927"
ACC= "#763626"; SUP= "#3E464B"; BG = "#F7F7F7"

CLASS_COLOR = {
    "tumor":                "#336B87",
    "fibroblast":           "#2E8B57",
    "vascular_endothelium": "#F59927",
    "macrophage":           "#1ABC9C",
    "lymphocyte":           "#4F85A0",
    "plasma_cell":          "#E67E22",
    "apoptotic_body":       "#763626",
    "unlabeled":            "#95A5A6",
    "mitotic_figure":       "#8E44AD",
    "myoepithelium":        "#D35400",
    "neutrophil":           "#9B59B6",
    "no_finding":           "#cccccc",
}
# Distinct per-rater colours (up to 30 raters)
RATER_COLORS = [
    "#E74C3C","#3498DB","#2ECC71","#F39C12","#9B59B6",
    "#1ABC9C","#E67E22","#34495E","#E91E63","#00BCD4",
    "#8BC34A","#FF5722","#607D8B","#795548","#CDDC39",
    "#FF9800","#03A9F4","#4CAF50","#673AB7","#F44336",
    "#2196F3","#009688","#FFC107","#9C27B0","#FF5252",
    "#69F0AE","#40C4FF","#FFD740","#EA80FC","#FF6D00",
]
def cc(cls): return CLASS_COLOR.get(cls, "#888888")


def load_sl(path):
    with open(path) as f: return json.load(f)


def resolve_img(img_dir, iid):
    for ext in [".png",".jpg",".jpeg",".tif",""]:
        for name in [iid, os.path.splitext(iid)[0]]:
            p = os.path.join(img_dir, name+ext)
            if os.path.exists(p): return p
    return None


def find_cluster(sl, target_n):
    scored = sorted(sl, key=lambda r:(
        abs(r.get("num_annotations_in_cluster",0)-target_n),
        -len(r.get("class_probs",{}))
    ))
    rec = scored[0]
    n   = rec.get("num_annotations_in_cluster", len(rec.get("original_annotations",[])))
    cp  = rec.get("class_probs",{})
    dom = max(cp, key=cp.get) if cp else "—"
    print(f"\nCluster selected:")
    print(f"  image_id   : {rec['image_id']}")
    print(f"  finding_id : {rec.get('finding_id','—')}")
    print(f"  annotators : {n}  (requested {target_n})")
    print(f"  dominant   : {dom}  class_probs: {dict(sorted(cp.items(),key=lambda x:-x[1]))}")
    return rec


def get_geometry(rec):
    iw = rec.get("image_width",  400)
    ih = rec.get("image_height", 400)
    sp = rec.get("spatial",{})
    mn = sp.get("mean",[0,0,0.1,0.1])
    st = sp.get("std", [0.01,0.01,0.01,0.01])
    mx1=mn[0]*iw; my1=mn[1]*ih; mx2=mx1+mn[2]*iw; my2=my1+mn[3]*ih
    sx=st[0]*iw;  sy=st[1]*ih
    return iw, ih, mn, st, mx1, my1, mx2, my2, sx, sy


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Individual annotator panels (unchanged from v6)
# ─────────────────────────────────────────────────────────────────────────────
def fig_disagreement(rec, crop, ox, oy, orig,
                     mx1, my1, mx2, my2, iw, ih, output, pad):
    n      = len(orig)
    total  = n + 1
    ncols  = math.ceil(math.sqrt(total))
    nrows  = math.ceil(total / ncols)

    ch, cw = crop.shape[:2]
    aspect = ch / cw
    panel_w = 3.0
    panel_h = panel_w * aspect

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols*panel_w, nrows*panel_h + 0.6),
                             facecolor=BG)
    axes = np.array(axes).flatten()
    for ax in axes: ax.axis("off"); ax.set_facecolor("#eeeeee")

    # Density heatmap
    hm_full = np.zeros((ih, iw), dtype=np.float32)
    for ann in orig:
        bb = ann["bbox"]
        x1=int(max(0,bb[0]*iw)); y1=int(max(0,bb[1]*ih))
        x2=int(min(iw,(bb[0]+bb[2])*iw)); y2=int(min(ih,(bb[1]+bb[3])*ih))
        if x2>x1 and y2>y1: hm_full[y1:y2,x1:x2] += 1
    if hm_full.max()>0: hm_full /= hm_full.max()
    hm_crop = hm_full[int(oy):int(oy+crop.shape[0]),
                       int(ox):int(ox+crop.shape[1])]
    cmap_heat = LinearSegmentedColormap.from_list(
        "heat", ["#00000000","#F5992799","#76362699"], N=256)

    for i, ann in enumerate(orig):
        ax = axes[i]
        ax.imshow(crop); ax.axis("off")
        bb  = ann["bbox"]
        bx1 = bb[0]*iw-ox; by1 = bb[1]*ih-oy
        bw  = bb[2]*iw;    bh  = bb[3]*ih
        cls = ann.get("class_name","")
        col = cc(cls)
        ax.add_patch(patches.Rectangle((bx1,by1),bw,bh,
            lw=2.2,edgecolor=col,facecolor=col,alpha=0.18,zorder=3))
        ax.add_patch(patches.Rectangle((bx1,by1),bw,bh,
            lw=2.2,edgecolor=col,facecolor="none",alpha=1.0,zorder=4))
        label_y = by1-4 if by1>12 else by1+bh+4
        ax.text(bx1+bw/2, label_y,
                f"{ann['rater_id']}\n{cls}",
                fontsize=7, color="white", fontweight="bold",
                ha="center", va="bottom", zorder=5,
                bbox=dict(facecolor=col,alpha=0.90,
                          pad=2,linewidth=0,boxstyle="round,pad=0.3"))
        weight = ann.get("weight",0)
        ax.set_title(f"w={weight:.4f}", fontsize=7, color=SUP, pad=2)

    # Last panel: density heatmap
    ax_hm = axes[n]
    ax_hm.imshow(crop)
    ax_hm.imshow(hm_crop, cmap=cmap_heat, alpha=0.75,
                 extent=[0,crop.shape[1],crop.shape[0],0], zorder=2)
    ax_hm.add_patch(patches.Rectangle(
        (mx1-ox,my1-oy),mx2-mx1,my2-my1,
        lw=2.5,edgecolor="white",facecolor="none",zorder=3))
    ax_hm.set_title("Box overlap\ndensity + mean",
                    fontsize=8, color=SUP, pad=2, fontweight="bold")
    ax_hm.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap_heat, norm=plt.Normalize(0,1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_hm, fraction=0.04, pad=0.04)
    cb.set_label("Annotator\nagreement", fontsize=6, color=SUP)
    cb.ax.tick_params(labelsize=6)

    fig.suptitle(
        f"Annotator Disagreement — {n} raters, image: {rec['image_id']}",
        fontsize=11, color=SUP, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.4)
    plt.savefig(output, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {output}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: All boxes on image (rater_id labels only) + pie chart
# ─────────────────────────────────────────────────────────────────────────────
def fig_consensus(rec, crop, ox, oy, orig,
                  mx1, my1, mx2, my2, sx, sy,
                  iw, ih, output):

    cp           = rec.get("class_probs", {})
    sorted_probs = sorted(cp.items(), key=lambda x: x[1], reverse=True)
    n_raters     = len(orig)

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1],
                             wspace=0.08, left=0.03, right=0.97,
                             top=0.90, bottom=0.06)
    ax_i  = fig.add_subplot(gs[0])
    ax_pie= fig.add_subplot(gs[1])

    # ── LEFT: H&E + all annotator boxes + mean box ────────────────────
    ax_i.imshow(crop)
    ax_i.set_facecolor("white"); ax_i.axis("off")

    # Assign each unique rater a distinct colour
    rater_ids = [ann["rater_id"] for ann in orig]
    unique_raters = list(dict.fromkeys(rater_ids))   # preserve order
    rater_col = {r: RATER_COLORS[i % len(RATER_COLORS)]
                 for i, r in enumerate(unique_raters)}

    for ann in orig:
        bb  = ann["bbox"]
        bx1 = bb[0]*iw - ox;  by1 = bb[1]*ih - oy
        bw  = bb[2]*iw;        bh  = bb[3]*ih
        rid = ann["rater_id"]
        col = rater_col[rid]

        # Box outline
        ax_i.add_patch(patches.Rectangle(
            (bx1,by1), bw, bh,
            lw=1.5, edgecolor=col, facecolor=col,
            alpha=0.08, zorder=3))
        ax_i.add_patch(patches.Rectangle(
            (bx1,by1), bw, bh,
            lw=1.5, edgecolor=col, facecolor="none",
            alpha=0.90, zorder=4))

        # Rater label — small badge at top-left corner of box
        ax_i.text(bx1+2, by1+2, rid,
                  fontsize=6, color="white", fontweight="bold",
                  va="top", ha="left", zorder=5,
                  bbox=dict(facecolor=col, alpha=0.85,
                            pad=1.2, linewidth=0,
                            boxstyle="round,pad=0.2"))

    # Mean box — thick, dark, prominent
    ax_i.add_patch(patches.Rectangle(
        (mx1-ox, my1-oy), mx2-mx1, my2-my1,
        lw=3.5, edgecolor=SUP, facecolor="none",
        linestyle="-", zorder=6))

    # "Mean" label below the mean box
    ax_i.text((mx1+mx2)/2-ox, my2-oy+5,
              "Mean",
              fontsize=11, color=SUP, fontweight="bold",
              ha="center", va="top", zorder=7,
              bbox=dict(facecolor="white", alpha=0.85,
                        pad=3, linewidth=0))

    ax_i.set_title(
        f"{n_raters} annotators  ·  image: {rec['image_id']}  ·  "
        f"finding_id: {rec.get('finding_id','—')}",
        fontsize=10, color=SUP, pad=7)

    # ── RIGHT: Pie chart of class_probs ──────────────────────────────
    ax_pie.set_facecolor(BG)
    ax_pie.axis("off")

    if sorted_probs:
        labels = [p[0].replace("_","\n") for p in sorted_probs]
        sizes  = [p[1] for p in sorted_probs]
        colors = [cc(p[0]) for p in sorted_probs]

        wedges, texts, autotexts = ax_pie.pie(
            sizes,
            labels=None,
            colors=colors,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 4 else "",
            pctdistance=0.78,
            startangle=90,
            wedgeprops=dict(linewidth=1.2, edgecolor="white"),
        )
        for at in autotexts:
            at.set_fontsize(9)
            at.set_color("white")
            at.set_fontweight("bold")

        # Legend outside pie
        ax_pie.legend(
            wedges, labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.18),
            fontsize=9,
            framealpha=0.0,
            ncol=2,
        )

    ax_pie.set_title("Class Distribution\n(soft label probs)",
                     fontsize=11, color=SUP, pad=8, fontweight="bold")

    plt.savefig(output, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Saved: {output}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softlabel-json", required=True)
    ap.add_argument("--img-dir",        required=True)
    ap.add_argument("--num-annotators", type=int, default=12)
    ap.add_argument("--crop-padding",   type=int, default=80)
    ap.add_argument("--output",
                    default="outputs/visualizations/presentation/cluster.png")
    args = ap.parse_args()

    sl  = load_sl(args.softlabel_json)
    rec = find_cluster(sl, args.num_annotators)

    iw, ih, mn, st, mx1, my1, mx2, my2, sx, sy = get_geometry(rec)
    orig = rec.get("original_annotations", [])

    img_path = resolve_img(args.img_dir, rec["image_id"])
    if img_path is None:
        raise FileNotFoundError(f"Image not found: {rec['image_id']} in {args.img_dir}")
    img  = np.array(Image.open(img_path).convert("RGB"))

    pad  = args.crop_padding
    ox   = max(0, mx1-pad);  oy  = max(0, my1-pad)
    x2c  = min(iw, mx2+pad); y2c = min(ih, my2+pad)
    crop = img[int(oy):int(y2c), int(ox):int(x2c)]

    base = os.path.splitext(args.output)[0]
    out1 = base + "_disagreement.png"
    out2 = base + "_consensus.png"
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",
                exist_ok=True)

    fig_disagreement(rec, crop, ox, oy, orig,
                     mx1, my1, mx2, my2, iw, ih, out1, pad)
    fig_consensus(rec, crop, ox, oy, orig,
                  mx1, my1, mx2, my2, sx, sy,
                  iw, ih, out2)

if __name__ == "__main__":
    main()