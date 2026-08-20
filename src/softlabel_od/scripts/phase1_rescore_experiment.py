"""
Phase 1 — No-retrain inference experiment for the all-824-class model.

Tests two levers against the v3 baseline:
  A. Class-aware rescoring:  score' = fg_confidence * P(predicted_class)
     -> can only change mAP (per-class ranking), not class-agnostic recall.
  B. Relax the post-NMS cap:  detections_per_img 100->300, score_thresh 0.05->0.01
     -> tests whether GT coverage (recall) is throttled by the top-100 cap.

Runs inference up to 2x (default params, relaxed params) and prints a
comparison table. No weights are modified.
"""

import argparse, os, sys, copy
import numpy as np
import torch

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

from softlabel_od.object_detection.utils.config import get_cfg_from_file
from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.core.evaluation.runner import load_model, collect_predictions
from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_detection_summary, compute_map,
)


def rescore_class_aware(predictions):
    """Return a deep-copied prediction list with score = fg_conf * P(pred_class)."""
    out = []
    for p in predictions:
        q = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in p.items()}
        scores = np.asarray(p["scores"], dtype=np.float64)        # fg_confidence
        labels = np.asarray(p["labels"], dtype=np.int64)          # 1-indexed
        cprobs = p.get("class_probs")
        if cprobs is not None and len(scores) > 0:
            cprobs = np.asarray(cprobs, dtype=np.float64)         # [N, num_classes]
            pred_cls_prob = cprobs[np.arange(len(labels)), labels]  # P(pred class)
            q["scores"] = scores * pred_cls_prob
        out.append(q)
    return out


def summarize(tag, predictions, ground_truths):
    ds = compute_detection_summary(predictions, ground_truths)
    mp = compute_map(predictions, ground_truths)
    per = mp.get("per_class_ap", {})
    nonzero = [float(v) for v in per.values() if float(v) > 0]
    return {
        "tag": tag,
        "pred": ds["total_pred_boxes"], "tp": ds["total_matched"],
        "fp": ds["total_false_positives"], "fn": ds["total_missed_gt"],
        "recall": ds["recall"], "precision": ds["precision"], "f1": ds["f1_score"],
        "map40": mp["mAP_40"], "map50": mp["mAP_50"], "map75": mp["mAP_75"],
        "n_ap_pos": len(nonzero), "map50_nz": (np.mean(nonzero) if nonzero else 0.0),
    }


def run_inference(model, val_loader, device):
    preds, gts = collect_predictions(model, val_loader, device=device, is_probabilistic=True)
    return preds, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",      required=True)
    ap.add_argument("--weights",     required=True)
    ap.add_argument("--data-root",   required=True)
    ap.add_argument("--label-file",  required=True)
    ap.add_argument("--num-classes", type=int, default=825)
    ap.add_argument("--backbone",    default="resnet50")
    ap.add_argument("--test-split-ratio", type=float, default=0.20)
    ap.add_argument("--relaxed-dets",   type=int,   default=300)
    ap.add_argument("--relaxed-score",  type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    cfg = get_cfg_from_file(args.config)
    architecture = getattr(cfg.MODEL, "ARCHITECTURE", None) or None

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset_cls = get_dataset_class(cfg.DATA.DATASET)
    val_dataset = dataset_cls(
        root=args.data_root, label_file=args.label_file,
        transforms=Compose([ToTensor()]), train=False,
        test_split_ratio=args.test_split_ratio, label_mode="soft_1d",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=4,
    )
    print(f"Validation images: {len(val_dataset)}")

    # ── model ─────────────────────────────────────────────────────────────────
    model = load_model(
        weights_path=args.weights, num_classes=args.num_classes, model_type="1D",
        backbone=args.backbone, probabilistic=True, device=device,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS * len(cfg.MODEL.ANCHOR_SIZES),
        architecture=architecture,
    )
    base_score = model.roi_heads.score_thresh
    base_dets  = model.roi_heads.detections_per_img
    print(f"Model post-NMS defaults: score_thresh={base_score}  detections_per_img={base_dets}")

    rows = []

    # ── Pass 1: default params ────────────────────────────────────────────────
    print("\n[Pass 1] Inference with DEFAULT post-NMS params...")
    preds, gts = run_inference(model, val_loader, device)
    rows.append(summarize("baseline (fg_conf score)", preds, gts))
    rows.append(summarize("rescored (fg_conf x P_cls)", rescore_class_aware(preds), gts))

    # ── Pass 2: relaxed cap ───────────────────────────────────────────────────
    model.roi_heads.score_thresh        = args.relaxed_score
    model.roi_heads.detections_per_img  = args.relaxed_dets
    print(f"\n[Pass 2] Inference with RELAXED params "
          f"(score_thresh={args.relaxed_score}, detections_per_img={args.relaxed_dets})...")
    preds2, gts2 = run_inference(model, val_loader, device)
    rows.append(summarize(f"relaxed cap (dets={args.relaxed_dets})", preds2, gts2))
    rows.append(summarize("relaxed + rescored", rescore_class_aware(preds2), gts2))

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'Variant':32s} {'Pred':>8} {'TP':>6} {'FP':>7} {'Recall':>8} {'Prec':>7} "
          f"{'mAP@40':>8} {'mAP@50':>8} {'mAP@75':>8} {'AP>0':>6} {'mAP50nz':>8}")
    print("-" * 110)
    for r in rows:
        print(f"{r['tag']:32s} {r['pred']:>8} {r['tp']:>6} {r['fp']:>7} "
              f"{r['recall']:>8.4f} {r['precision']:>7.4f} "
              f"{r['map40']:>8.4f} {r['map50']:>8.4f} {r['map75']:>8.5f} "
              f"{r['n_ap_pos']:>6} {r['map50_nz']:>8.4f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
