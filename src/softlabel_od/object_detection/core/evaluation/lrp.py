"""
lrp.py — Localisation Recall Precision (LRP) Error

Implements LRP Error and Optimal LRP (oLRP) Error for object detection,
as defined in:
"Kemal Oksuz et al. “Localization recall precision (LRP): A new performance metric for object detec- tion”. In: Proceedings of the European conference on computer vision (ECCV). 2018"

"""

import numpy as np

from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_iou_batch,       
    match_predictions_to_gt, 
)

# Adapter: convert match_predictions_to_gt output to - what LRP needs
def _extract_lrp_counts(pred_boxes: np.ndarray,
                         pred_scores: np.ndarray,
                         gt_boxes: np.ndarray,
                         iou_threshold: float):
    """
    Thin wrapper around match_predictions_to_gt that returns the three
    quantities LRP needs instead of the raw match tuples.

    match_predictions_to_gt returns:
        matches          = [(pred_idx, gt_idx, iou_value), ...]
        unmatched_preds  = [pred_idx, ...]   <- FPs
        unmatched_gts    = [gt_idx,  ...]    <- FNs

    This function extracts:
        tp_ious  list[float]  — IoU value of each TP match
        n_fp     int          — number of false positives
        n_fn     int          — number of false negatives
    """
    matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
        pred_boxes, pred_scores, gt_boxes, iou_threshold
    )
    tp_ious = [float(iou) for (_, _, iou) in matches]
    return tp_ious, len(unmatched_preds), len(unmatched_gts)


# LRP First Componenet
def _e_loc(iou_value: float, iou_threshold: float) -> float:
    """
    Normalised localisation error for one TP, in [0, 1].

        E_Loc = (1 - IoU) / (1 - tau)       [ECCV 2018, Eq. 5 First Component]

    A perfect TP (IoU = 1.0) contributes 0.
    A barely-passing TP (IoU = tau) contributes 1.
    """
    return (1.0 - iou_value) / (1.0 - iou_threshold)

# LRP at one confidence threshold
def _lrp_at_threshold(tp_ious: list,
                       n_fp: int,
                       n_fn: int,
                       iou_threshold: float) -> float:
    """
    Compute LRP(s) from the TP/FP/FN summary at one confidence threshold.

        LRP = ( SUM_TP E_Loc(i)  +  N_FP  +  N_FN )
              / ( N_TP + N_FP + N_FN )              [ECCV 2018, Eq. 5]

    Returns NaN when denominator is 0 (nothing to evaluate at this threshold).
    """
    n_tp = len(tp_ious)
    denom = n_tp + n_fp + n_fn
    if denom == 0:
        return float("nan")
    loc_sum = sum(_e_loc(iou, iou_threshold) for iou in tp_ious)
    return (loc_sum + n_fp + n_fn) / denom

# Per-class oLRP sweep (for 101 thresholds by default)
def _compute_class_lrp(cls_preds: list,
                        cls_gts: list,
                        iou_threshold: float,
                        score_thresholds: np.ndarray) -> dict:
    """
    Compute oLRP for a single class by sweeping confidence thresholds.

    Args:
        cls_preds: list of {'image_id', 'box', 'score'} for this class
        cls_gts:   list of {'image_id', 'box'}          for this class
        iou_threshold:    tau  (e.g. 0.5)
        score_thresholds: 1-D array of candidate s values

    Returns dict:
        oLRP, oLRP_IoU, oLRP_FP, oLRP_FN, optimal_threshold, lrp_curve

    Edge cases
    ----------
    Both empty  -> all NaN (nothing to evaluate).
    No GTs      -> oLRP = 1.0 at all thresholds (all predictions are FP).
    No preds    -> oLRP = 1.0 at all thresholds (all GTs are FN).
    """
    n_gt_total = len(cls_gts)
    n_pred_total = len(cls_preds)

    if n_gt_total == 0 and n_pred_total == 0:
        return {
            "oLRP": float("nan"),
            "oLRP_IoU": float("nan"),
            "oLRP_FP": float("nan"),
            "oLRP_FN": float("nan"),
            "optimal_threshold": float("nan"),
            "lrp_curve": [],
        }

    # Build image_id -> list of GT boxes for per-image matching
    gt_by_image: dict = {}
    for g in cls_gts:
        gt_by_image.setdefault(g["image_id"], []).append(g["box"])

    lrp_curve = []
    component_curve = []   # (lrp_iou, lrp_fp, lrp_fn) at each threshold

    for s in score_thresholds:
        # Filter predictions surviving this threshold
        surviving = [p for p in cls_preds if p["score"] >= s]

        # Group surviving predictions by image (same as GT grouping)
        preds_by_image: dict = {}
        for p in surviving:
            preds_by_image.setdefault(p["image_id"], []).append(p)

        # Accumulate TP/FP/FN across all images at this threshold.
        all_tp_ious = []
        total_fp = 0
        total_fn = 0

        # Union of image IDs that have either GTs or surviving predictions
        all_image_ids = set(gt_by_image.keys()) | set(preds_by_image.keys())
        for img_id in all_image_ids:
            img_preds = preds_by_image.get(img_id, [])
            img_gts   = gt_by_image.get(img_id, [])

            if not img_preds and not img_gts:
                continue

            p_boxes  = (np.array([p["box"]   for p in img_preds])
                        if img_preds else np.zeros((0, 4)))
            p_scores = (np.array([p["score"] for p in img_preds])
                        if img_preds else np.zeros(0))
            g_boxes  = (np.array(img_gts)
                        if img_gts else np.zeros((0, 4)))

            # Uses match_predictions_to_gt from metrics.py via adapter
            tp_ious, n_fp, n_fn = _extract_lrp_counts(
                p_boxes, p_scores, g_boxes, iou_threshold
            )
            all_tp_ious.extend(tp_ious) # Gives all TP IOUs individually
            total_fp += n_fp
            total_fn += n_fn

        lrp_val = _lrp_at_threshold(all_tp_ious, total_fp, total_fn,
                                     iou_threshold)
        lrp_curve.append(lrp_val)

        # Store components for retrieval at the optimal point
        n_tp = len(all_tp_ious)

        if n_tp > 0:
            lrp_iou = float(np.mean([_e_loc(v, iou_threshold)
                                      for v in all_tp_ious]))
        else:
            lrp_iou = float("nan")   # IoU component undefined when no TPs

        lrp_fp = total_fp / max(n_tp + total_fp, 1)   # 1 - precision
        lrp_fn = total_fn / max(n_gt_total, 1)         # 1 - recall
        component_curve.append((lrp_iou, lrp_fp, lrp_fn))

    # Find optimal threshold: argmin over non-NaN LRP values
    lrp_array = np.array(lrp_curve, dtype=float)
    if np.all(np.isnan(lrp_array)):
        return {
            "oLRP": float("nan"),
            "oLRP_IoU": float("nan"),
            "oLRP_FP": float("nan"),
            "oLRP_FN": float("nan"),
            "optimal_threshold": float("nan"),
            "lrp_curve": lrp_curve,
        }

    opt_idx = int(np.nanargmin(lrp_array))
    opt_lrp_iou, opt_lrp_fp, opt_lrp_fn = component_curve[opt_idx]

    return {
        "oLRP": float(lrp_array[opt_idx]),
        "oLRP_IoU": opt_lrp_iou,
        "oLRP_FP": opt_lrp_fp,
        "oLRP_FN": opt_lrp_fn,
        "optimal_threshold": float(score_thresholds[opt_idx]),
        "lrp_curve": [float(v) for v in lrp_curve],
    }



# Main entry point: compute oLRP for all classes and aggregate to moLRP
def compute_lrp(predictions: list,
                ground_truths: list,
                iou_threshold: float = 0.5,
                score_thresholds: np.ndarray = None,
                background_index: int = 0) -> dict:
    """
    Compute Optimal LRP Error for all foreground classes in the dataset.

    Args:
        predictions: list of dicts, one per image.
            Required keys:
                'boxes'   np.ndarray [N, 4]  [x1, y1, x2, y2] pixel coords
                'scores'  np.ndarray [N]     confidence scores in [0, 1]
                'labels'  np.ndarray [N]     integer class IDs
        ground_truths: list of dicts, one per image.
            Required keys:
                'boxes'   np.ndarray [M, 4]
                'labels'  np.ndarray [M]
        iou_threshold: float, default 0.5
            Minimum IoU for a detection to count as a TP (tau in the ECCV 2018 paper).
        score_thresholds: 1-D np.ndarray or None.
            Candidate confidence thresholds for the oLRP sweep.
            Default: np.linspace(0.0, 1.0, 101) — step 0.01, matching the
            exhaustive search in the original paper.
        background_index: int or None, default 0
            Class ID to exclude (background / no-finding class).
            Set to None to evaluate all classes including background.

    Returns:
        dict with keys:
            moLRP        float  — mean oLRP across foreground classes
            moLRP_IoU    float  — mean oLRP_IoU
            moLRP_FP     float  — mean oLRP_FP  (mean precision error at s*)
            moLRP_FN     float  — mean oLRP_FN  (mean recall error at s*)
            per_class    dict   — {class_id (int): {oLRP, oLRP_IoU, oLRP_FP,
                                                     oLRP_FN, optimal_threshold,
                                                     lrp_curve}}

    Notes
    -----
    moLRP is computed only over classes that have at least one GT box in the
    dataset (NaN classes excluded).

    lrp_curve in per_class is retained for diagnostic plotting but is typically
    excluded from the saved JSON summary (101 floats per class).
    """
    if score_thresholds is None:
        score_thresholds = np.linspace(0.0, 1.0, 101)
    
    all_classes = set()
    # Discover all foreground classes 
    for gt in ground_truths:
        labels = gt["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        all_classes.update(labels)
    for pred in predictions:
        labels = pred["labels"]
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
        all_classes.update(labels)

    if background_index is not None:
        all_classes.discard(background_index)

    # Collect per-class preds and GTs across the whole dataset
    cls_predictions: dict = {c: [] for c in all_classes}
    cls_gts: dict         = {c: [] for c in all_classes}

    for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes  = np.array(pred["boxes"])
        pred_scores = np.array(pred["scores"])
        pred_labels = np.array(pred["labels"])
        gt_boxes    = np.array(gt["boxes"])
        gt_labels   = np.array(gt["labels"])

        for cls in all_classes:
            pmask = pred_labels == cls
            if pmask.any():
                for box, score in zip(pred_boxes[pmask], pred_scores[pmask]):
                    cls_predictions[cls].append({
                        "image_id": img_id,
                        "box": box,
                        "score": float(score),
                    })

            gmask = gt_labels == cls
            if gmask.any():
                for box in gt_boxes[gmask]:
                    cls_gts[cls].append({
                        "image_id": img_id,
                        "box": box,
                    })

    # Compute oLRP per class
    per_class_results = {}
    for cls in sorted(all_classes):
        per_class_results[cls] = _compute_class_lrp(
            cls_predictions[cls],
            cls_gts[cls],
            iou_threshold,
            score_thresholds,
        )

    # Aggregate to moLRP (only classes with GTs contribute)
    valid_olrp, valid_iou, valid_fp, valid_fn = [], [], [], []
    for cls, r in per_class_results.items():
        if len(cls_gts[cls]) == 0:
            continue
        if np.isnan(r["oLRP"]):
            continue
        valid_olrp.append(r["oLRP"])
        if not np.isnan(r["oLRP_IoU"]):
            valid_iou.append(r["oLRP_IoU"])
        valid_fp.append(r["oLRP_FP"])
        valid_fn.append(r["oLRP_FN"])

    moLRP     = float(np.mean(valid_olrp)) if valid_olrp else float("nan")
    moLRP_IoU = float(np.mean(valid_iou))  if valid_iou  else float("nan")
    moLRP_FP  = float(np.mean(valid_fp))   if valid_fp   else float("nan")
    moLRP_FN  = float(np.mean(valid_fn))   if valid_fn   else float("nan")

    return {
        "moLRP":     moLRP,
        "moLRP_IoU": moLRP_IoU,
        "moLRP_FP":  moLRP_FP,
        "moLRP_FN":  moLRP_FN,
        "per_class": per_class_results,
    }

def _lrp_fmt_float(x) -> str:
    """Format a scalar for console; show 'nan' / 'inf' safely."""
    xf = float(x)
    if np.isnan(xf) or np.isinf(xf):
        return "nan"
    return f"{xf:.4f}"


def _lrp_json_float(x):
    """JSON-safe float: NaN/Inf → None (standard json.dump rejects NaN)."""
    if not isinstance(x, (float, np.floating)):
        x = float(x)
    xf = float(x)
    if np.isnan(xf) or np.isinf(xf):
        return None
    return xf


def _lrp_serialize_per_class_entry(metrics: dict, include_lrp_curve: bool) -> dict:
    out = {}
    for k, v in metrics.items():
        if k == "lrp_curve":
            if not include_lrp_curve:
                continue
            out[k] = [float(t) for t in v] if isinstance(v, list) else v
            continue
        if isinstance(v, (float, np.floating)) or (isinstance(v, (int, np.integer)) and k == "optimal_threshold"):
            out[k] = _lrp_json_float(v)
        else:
            out[k] = v
    return out

def evaluate_lrp(
    predictions,
    ground_truths,
    *,
    class_names=None,
    gmm_type="1D",
    probabilistic=False,
    output_dir=None,
    iou_threshold: float = 0.5,
    background_index=0,
    score_thresholds=None,
    include_lrp_curve: bool = False,
    **kwargs,
):
    """Run LRP / oLRP: print summary (like default metrics / PDQ), return JSON-serializable dict.

    Extra ``kwargs`` (e.g. ``label_file``, ``priors_file``) are ignored so the
    same ``evaluate.py`` call signature works for all metric runners.
    """
    # ---- compute ----
    raw = compute_lrp(
        predictions,
        ground_truths,
        iou_threshold=iou_threshold,
        score_thresholds=score_thresholds,
        background_index=background_index,
    )

    class_names = class_names or {}

    per_class_out = {
        str(cls_id): _lrp_serialize_per_class_entry(dict(metrics), include_lrp_curve)
        for cls_id, metrics in raw["per_class"].items()
    }

    # ---- console (same style as metrics.evaluate_default_metrics / pdq.evaluate_pdq) ----
    print("\n" + "=" * 70)
    print("LOCALISATION RECALL PRECISION (LRP / oLRP)")
    print("=" * 70)
    print(f"  IoU threshold τ:     {iou_threshold:.2f}")
    if background_index is None:
        print("  Background handling: all classes included")
    else:
        print(f"  Background excluded: class index {background_index}")

    print(f"  moLRP:               {_lrp_fmt_float(raw['moLRP'])}")
    print(f"  moLRP_IoU:           {_lrp_fmt_float(raw['moLRP_IoU'])}")
    print(f"  moLRP_FP:            {_lrp_fmt_float(raw['moLRP_FP'])}")
    print(f"  moLRP_FN:            {_lrp_fmt_float(raw['moLRP_FN'])}")

    print(f"\n  Per-class oLRP (classes with ≥1 GT in dataset):")
    for cls_idx in sorted(raw["per_class"].keys(), key=lambda c: int(c)):
        r = raw["per_class"][cls_idx]
        if np.isnan(float(r["oLRP"])):
            continue
        name = class_names.get(int(cls_idx), f"Class {cls_idx}")
        s_star = float(r["optimal_threshold"])
        if np.isnan(s_star):
            s_str = "nan"
        else:
            s_str = f"{s_star:.2f}"
        print(
            f"    {name:25s}: oLRP={_lrp_fmt_float(r['oLRP'])}  "
            f"IoU={_lrp_fmt_float(r['oLRP_IoU'])}  "
            f"FP={_lrp_fmt_float(r['oLRP_FP'])}  "
            f"FN={_lrp_fmt_float(r['oLRP_FN'])}  "
            f"s*={s_str}"
        )

    # ---- return (JSON-safe: no NaN in floats; optional curves stripped by default) ----
    return {
        "moLRP": _lrp_json_float(raw["moLRP"]),
        "moLRP_IoU": _lrp_json_float(raw["moLRP_IoU"]),
        "moLRP_FP": _lrp_json_float(raw["moLRP_FP"]),
        "moLRP_FN": _lrp_json_float(raw["moLRP_FN"]),
        "per_class": per_class_out,
        "iou_threshold": float(iou_threshold),
        "background_index": background_index,
    }