"""
viz_training_curves.py  (v2)
Fixes:
  - Removed mAP@50 (not in training logs)
  - Uses metrics actually logged: val_loss, class_loss, reg_loss, avg_iou, top1, ece, uw_map
  - Better error messages when run JSON files are missing
  - Produces two PNGs (2 charts each)

Slide chart 1: Validation Loss (L) + Avg IoU (R)  → convergence quality
Slide chart 2: Classification Loss (L) + Top-1 Accuracy (R)  → learning quality

Place renamed JSON files in one directory:
    training_metrics_phase_V8.json
    training_metrics_phase_V9.json
    training_metrics_phase_V10.json  etc.

Available JSON keys (from your training framework):
    train_loss, val_loss
    val_class_loss  (or: classification_loss, Class)
    val_reg_loss    (or: regression_loss, Reg)
    val_std_loss    (or: std_loss, Std)
    val_avg_iou     (or: avg_iou, Avg IoU, detection_avg_iou)
    val_top1        (or: Top1, top1_accuracy)
    val_ece         (or: ECE, ece)
    val_uw_map      (or: UW-mAP, uw_map)
    lr

Usage:
    python viz_training_curves.py \
        --logs-dir /path/to/renamed/json/dir \
        --runs V8,V9,V10,V11,V12 \
        --output-dir outputs/visualizations/presentation
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

P="#336B87"; S="#4F85A0"; PC="#B5854A"; SC="#F59927"
ACC="#763626"; OK="#A9FC6D"; SUP="#3E464B"; BG="#F7F7F7"
RUN_COLORS={"V8":S,"V9":PC,"V10":P,"V11":SC,"V12":ACC}
DEFAULT_COLORS=[P,SC,ACC,PC,S,"#75644D"]

KEY_ALIASES={
    "val_loss":  ["val_loss","val/loss","total_val_loss"],
    "train_loss":["train_loss","train/loss"],
    "cls_loss":  ["val_class_loss","val_classification_loss","classification_loss",
                  "val/class_loss","Class","class_loss"],
    "reg_loss":  ["val_reg_loss","val_regression_loss","regression_loss",
                  "val/reg_loss","Reg","reg_loss"],
    "avg_iou":   ["val_avg_iou","avg_iou","Avg IoU","detection_avg_iou",
                  "val/avg_iou","BboxCorr","Expected_IoU"],
    "top1":      ["val_top1","Top1","top1_accuracy","val/top1","top1"],
    "ece":       ["val_ece","ECE","ece","val/ece"],
    "uw_map":    ["val_uw_map","UW-mAP","uw_map","val/uw_map","UW_mAP"],
    "lr":        ["lr","learning_rate","LR"],
}

def get(d, key):
    for k in KEY_ALIASES.get(key,[]):
        if k in d: return d[k]
    return None

def smooth(v, w=5):
    if w<=1 or len(v)<w: return np.array(v, dtype=float)
    kernel=np.ones(w)/w
    padded=np.pad(v,(w//2,w//2),mode="edge")
    return np.convolve(padded,kernel,mode="valid")[:len(v)]

def style_ax(ax, title="", xlabel="Epoch", ylabel=""):
    ax.set_facecolor("white")
    if title: ax.set_title(title,fontsize=12,color=SUP,pad=8)
    if xlabel: ax.set_xlabel(xlabel,fontsize=9,color=SUP)
    if ylabel: ax.set_ylabel(ylabel,fontsize=9,color=SUP)
    ax.tick_params(colors=SUP,labelsize=8)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")
    ax.yaxis.grid(True,color="#eeeeee",linewidth=0.8,zorder=0); ax.set_axisbelow(True)
    ax.legend(fontsize=9,framealpha=0.75,edgecolor="#cccccc")

def load_runs(logs_dir, run_names, max_epochs):
    loaded={}
    for name in run_names:
        path=os.path.join(logs_dir,f"training_metrics_phase_{name}.json")
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {name}")
            continue
        with open(path) as f: d=json.load(f)
        # Print available keys for first run
        if not loaded:
            print(f"\nAvailable keys in {name}: {list(d.keys())}\n")
        if max_epochs>0:
            d={k:v[:max_epochs] if isinstance(v,list) else v for k,v in d.items()}
        loaded[name]=d
        print(f"  Loaded {name}: {len(d.get('val_loss',d.get('train_loss',[])))} epochs")
    return loaded

def plot_chart1(loaded, colors, smooth_w, output_dir):
    """Val Loss + Avg IoU — convergence quality"""
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),facecolor=BG)
    for i,(name,d) in enumerate(loaded.items()):
        col=colors.get(name,DEFAULT_COLORS[i%len(DEFAULT_COLORS)])
        vl=get(d,"val_loss")
        if vl:
            ep=range(1,len(vl)+1)
            axes[0].plot(ep,smooth(vl,smooth_w),color=col,lw=2.0,label=name,zorder=3)
        ai=get(d,"avg_iou")
        if ai:
            ep=range(1,len(ai)+1)
            axes[1].plot(ep,smooth(ai,smooth_w),color=col,lw=2.0,label=name,zorder=3)
    style_ax(axes[0],"Validation Loss",ylabel="Loss")
    style_ax(axes[1],"Validation Avg IoU (Matched Predictions)",ylabel="IoU")
    plt.tight_layout(pad=2.0)
    out=os.path.join(output_dir,"training_curve_loss_iou.png")
    plt.savefig(out,dpi=180,bbox_inches="tight",facecolor=BG); plt.close()
    print(f"Saved: {out}")

def plot_chart2(loaded, colors, smooth_w, output_dir):
    """Classification Loss + Top-1 — learning quality"""
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),facecolor=BG)
    for i,(name,d) in enumerate(loaded.items()):
        col=colors.get(name,DEFAULT_COLORS[i%len(DEFAULT_COLORS)])
        cl=get(d,"cls_loss")
        if cl:
            ep=range(1,len(cl)+1)
            axes[0].plot(ep,smooth(cl,smooth_w),color=col,lw=2.0,label=name,zorder=3)
        t1=get(d,"top1")
        if t1:
            ep=range(1,len(t1)+1)
            vals=[v*100 if v<=1.0 else v for v in t1]
            axes[1].plot(ep,smooth(vals,smooth_w),color=col,lw=2.0,label=name,zorder=3)
    style_ax(axes[0],"Classification Loss (Val)",ylabel="Loss")
    style_ax(axes[1],"Top-1 Accuracy on Matched Predictions (Val)",ylabel="%")
    plt.tight_layout(pad=2.0)
    out=os.path.join(output_dir,"training_curve_cls_top1.png")
    plt.savefig(out,dpi=180,bbox_inches="tight",facecolor=BG); plt.close()
    print(f"Saved: {out}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--logs-dir",required=True)
    ap.add_argument("--runs",default="V8,V9,V10,V11,V12")
    ap.add_argument("--output-dir",default="outputs/visualizations/presentation")
    ap.add_argument("--smooth",type=int,default=5)
    ap.add_argument("--max-epochs",type=int,default=0)
    args=ap.parse_args()

    run_names=[r.strip() for r in args.runs.split(",")]
    os.makedirs(args.output_dir,exist_ok=True)
    loaded=load_runs(args.logs_dir,run_names,args.max_epochs)

    if not loaded:
        raise FileNotFoundError(
            f"No JSON files found in {args.logs_dir}. "
            f"Expected: training_metrics_phase_V10.json etc.")

    colors=RUN_COLORS
    plot_chart1(loaded,colors,args.smooth,args.output_dir)
    plot_chart2(loaded,colors,args.smooth,args.output_dir)
    print("Done.")

if __name__=="__main__":
    main()
