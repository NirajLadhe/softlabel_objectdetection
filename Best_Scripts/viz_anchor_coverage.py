"""
viz_anchor_coverage.py  (v2 — DATA DRIVEN)
Computes actual nucleus sizes from the canonical CSV per class,
then overlays the 5 anchor sizes as vertical lines.

This is now backed by real data — no hardcoded size assumptions.

Usage:
    python viz_anchor_coverage.py \
        --canonical-csv  outputs/nucls_transform_0.001/nucls_canonical.csv \
        --output         outputs/visualizations/presentation/anchor_coverage.png

The plot shows a horizontal box+violin per class (actual bounding box sizes
computed as max(width, height) from x_min,y_min,x_max,y_max in the CSV).
"""
import argparse, csv, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

P="#336B87"; S="#4F85A0"; PC="#B5854A"; SC="#F59927"
ACC="#763626"; OK="#A9FC6D"; SUP="#3E464B"; BG="#F7F7F7"

ANCHOR_SIZES=[8,16,32,64,128]
ANCHOR_COLORS=["#763626","#4F85A0","#B5854A","#F59927","#75644D"]

CLASS_ORDER=["tumor","fibroblast","vascular_endothelium","macrophage",
             "lymphocyte","plasma_cell","apoptotic_body","unlabeled",
             "mitotic_figure","myoepithelium","neutrophil"]
CLASS_COLORS={"tumor":P,"fibroblast":"#2E8B57","vascular_endothelium":"#F59927",
              "macrophage":"#1ABC9C","lymphocyte":S,"plasma_cell":"#E67E22",
              "apoptotic_body":ACC,"unlabeled":"#95A5A6",
              "mitotic_figure":"#8E44AD","myoepithelium":"#D35400",
              "neutrophil":"#9B59B6"}

def load_sizes(path):
    sizes=defaultdict(list)
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            w=float(row["x_max"])-float(row["x_min"])
            h=float(row["y_max"])-float(row["y_min"])
            size=max(w,h)  # use max dimension as proxy for "nucleus diameter"
            if size>0:
                sizes[row["class_name"]].append(size)
    return sizes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--canonical-csv",required=True)
    ap.add_argument("--output",default="outputs/visualizations/presentation/anchor_coverage.png")
    ap.add_argument("--max-size",type=float,default=200,
                    help="Clip display range to this max (pixels)")
    args=ap.parse_args()

    sizes=load_sizes(args.canonical_csv)

    # Order by median size ascending
    classes_present=[c for c in CLASS_ORDER if c in sizes]
    classes_present.sort(key=lambda c: np.median(sizes[c]))

    fig,ax=plt.subplots(figsize=(13,6),facecolor=BG)
    ax.set_facecolor("white")

    y_pos=list(range(len(classes_present)))

    for i,cls in enumerate(classes_present):
        data=np.clip(sizes[cls],0,args.max_size)
        col=CLASS_COLORS.get(cls,"#888888")
        q1,med,q3=np.percentile(data,[25,50,75])
        mn,mx=np.percentile(data,[5,95])

        # Whisker
        ax.plot([mn,mx],[i,i],color="#cccccc",lw=1.5,zorder=2)
        # IQR box
        ax.barh(i,q3-q1,left=q1,height=0.45,
                color=col,alpha=0.75,zorder=3,edgecolor="white",linewidth=0.8)
        # Median line
        ax.plot([med,med],[i-0.23,i+0.23],color="white",lw=2.2,zorder=4)
        # Count label
        ax.text(args.max_size*0.98,i,f"n={len(data):,}",
                va="center",ha="right",fontsize=7.5,color="#888888")

    # Anchor lines
    for size,col in zip(ANCHOR_SIZES,ANCHOR_COLORS):
        ax.axvline(size,color=col,lw=2.0,linestyle="--",zorder=5,alpha=0.9)
        ax.text(size,len(classes_present)+0.05,f"{size}px",
                ha="center",va="bottom",fontsize=8.5,color=col,
                fontweight="bold")

    # Apoptotic annotation
    if "apoptotic_body" in classes_present:
        ap_idx=classes_present.index("apoptotic_body")
        ax.annotate("8px anchor added\n(V9 — first detection)",
                    xy=(8,ap_idx),xytext=(35,ap_idx-0.8),
                    fontsize=8,color=ACC,
                    arrowprops=dict(arrowstyle="->",color=ACC,lw=1.5),ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_"," ").title() for c in classes_present],
                       fontsize=9,color=SUP)
    ax.set_xlabel("Nucleus max dimension (pixels) — box shows IQR, whiskers 5-95 pct",
                  fontsize=9,color=SUP)
    ax.set_xlim(0,args.max_size+5)
    ax.set_ylim(-0.7,len(classes_present)+0.5)
    ax.set_title("Nucleus Size Distribution vs Anchor Sizes\n(computed from canonical CSV annotations)",
                 fontsize=12,color=SUP,pad=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd"); ax.spines["bottom"].set_color("#dddddd")
    ax.xaxis.grid(True,color="#eeeeee",lw=0.8,zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=SUP,labelsize=9)

    legend_p=[mpatches.Patch(color=ANCHOR_COLORS[i],label=f"{s}px anchor")
              for i,s in enumerate(ANCHOR_SIZES)]
    ax.legend(handles=legend_p,loc="lower right",fontsize=8,
              framealpha=0.85,edgecolor="#cccccc",title="Anchor sizes",title_fontsize=8)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".",exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.output,dpi=180,bbox_inches="tight",facecolor=BG)
    plt.close()
    print(f"Saved: {args.output}")

    # Print summary stats
    print("\n=== Nucleus size stats (median max-dimension, pixels) ===")
    for cls in classes_present:
        d=np.clip(sizes[cls],0,args.max_size)
        print(f"  {cls:<28} median={np.median(d):>6.1f}  IQR=[{np.percentile(d,25):.1f},{np.percentile(d,75):.1f}]  n={len(d)}")

if __name__=="__main__":
    main()
