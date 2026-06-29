"""
pdq.py — Probability-based Detection Quality (PDQ)


 (summary for understanding):

Unlike mAP (which uses a hard IoU threshold), PDQ treats every detection as a
probability distribution over space and class labels. It rewards you for being
correctly uncertain and penalizes you for being wrongly confident.

For each matched (GT, Detection) pair, PDQ computes:

    pPDQ(G, D) = sqrt( Q_spatial(G, D)  *  Q_label(G, D) )

Where:
    Q_spatial: How well does the detection's Gaussian heatmap cover the GT box?
               - Foreground loss: log-prob of GT pixels being detected (inside GT box)
               - Background loss: log-prob of non-GT pixels being background (outside)
               Both losses are averaged over GT pixels and exponentiated → [0, 1]

    Q_label:   The raw probability the detector assigned to the correct GT class.
               e.g., if GT = "Cardiomegaly" and P("Cardiomegaly") = 0.7 → Q_L = 0.7

OPTIMAL ASSIGNMENT:
    Instead of greedy matching (used in mAP), PDQ uses the Hungarian algorithm
    (scipy.optimize.linear_sum_assignment) to find the globally optimal one-to-one
    matching between all GTs and all detections.

FINAL SCORE:
    PDQ = sum(pPDQ for all matched TP pairs) / (TP + FP + FN)
    FPs and FNs contribute 0 quality, pulling the score down.

COMPLEXITY NOTE:
    The spatial quality requires generating a per-pixel probability heatmap for
    each detection and computing log-losses over the full image. This is O(H*W*D*G)
    per image. We use numpy vectorization (tensordot) to make this tractable.
    For our pipeline, we use Gaussian bounding boxes (predicted mean + std) which
    are modelled as axis-aligned 2D Gaussians to generate the heatmap.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import erf as scipy_erf
from typing import List, Dict, Any, Optional, Tuple
import os
import json
import glob
from tqdm import tqdm
import warnings

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def _process_image_task(args):
    pred, gt, heatmap_threshold, bbox_std_fallback = args
    per_img_size = gt.get("image_size")
    if per_img_size is None:
        raise ValueError("PDQ requires gt['image_size'] to dynamically generate spatial heatmaps.")
    return _evaluate_image(pred, gt, per_img_size, heatmap_threshold, bbox_std_fallback=bbox_std_fallback)


from softlabel_od.object_detection.core.evaluation.metrics import delta_stds_to_pixel_stds

# Numerical stability constant — prevents log(0) = -inf
_SMALL_VAL = 1e-14

def _median_image_hw(ground_truths: List[Dict[str, Any]]) -> Tuple[float, float]:
    hs, ws = [], []
    for gt in ground_truths:
        sz = gt.get("image_size")
        if sz and len(sz) == 2:
            h, w = float(sz[0]), float(sz[1])
            if h > 0 and w > 0:
                hs.append(h)
                ws.append(w)
    if not hs:
        return 1024.0, 1024.0
    return float(np.median(hs)), float(np.median(ws))


def _resolve_class_priors_json(
    label_file: Optional[str],
    priors_file: Optional[str],
) -> Optional[str]:
    if priors_file:
        path = os.path.abspath(priors_file)
        return path if os.path.isfile(path) else None
    if not label_file:
        return None
    label_dir = os.path.dirname(os.path.abspath(label_file))
    priors_dir = os.path.join(label_dir, "priors")
    if not os.path.isdir(priors_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(priors_dir, "class_priors*.json")))
    return candidates[0] if candidates else None


def _pixel_corner_std_from_prior_block(
    block: Dict[str, Any],
    ref_h: float,
    ref_w: float,
) -> float:
    x_var = float(block["x_var"])
    y_var = float(block["y_var"])
    w_var = float(block["w_var"])
    h_var = float(block["h_var"])
    sigma_cx = np.sqrt(x_var) * ref_w
    sigma_cy = np.sqrt(y_var) * ref_h
    sigma_w = np.sqrt(w_var) * ref_w
    sigma_h = np.sqrt(h_var) * ref_h
    std_x = float(np.sqrt(sigma_cx**2 + 0.25 * sigma_w**2))
    std_y = float(np.sqrt(sigma_cy**2 + 0.25 * sigma_h**2))
    return (std_x + std_y) / 2.0


def _load_bbox_std_fallback_from_class_priors(
    path: str,
    ref_h: float,
    ref_w: float,
) -> Optional[float]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if "global_prior" in data and isinstance(data["global_prior"], dict):
        block = data["global_prior"]
        if all(k in block for k in ("x_var", "y_var", "w_var", "h_var")):
            return _pixel_corner_std_from_prior_block(block, ref_h, ref_w)

    cp = data.get("class_priors")
    if not isinstance(cp, dict):
        return None
    values = []
    for block in cp.values():
        if isinstance(block, dict) and all(
            k in block for k in ("x_var", "y_var", "w_var", "h_var")
        ):
            values.append(_pixel_corner_std_from_prior_block(block, ref_h, ref_w))
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# the main compute function
# ---------------------------------------------------------------------------

def compute_pdq(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    heatmap_threshold: float = 0.01,
    bbox_std_fallback: float = 10.0,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Compute Probability-based Detection Quality (PDQ) for a dataset.

    PDQ is a holistic, threshold-free metric for probabilistic object detectors.
    It jointly evaluates spatial uncertainty (via Gaussian heatmaps) and label
    quality (via the probability assigned to the GT class).

    Args:
        predictions: List of per-image prediction dicts. Each dict must have:
            - 'boxes'      (np.ndarray, shape [N, 4]): Predicted mean boxes [x1, y1, x2, y2]
            - 'scores'     (np.ndarray, shape [N]):    Detection confidence scores
            - 'labels'     (np.ndarray, shape [N]):    Predicted class indices (argmax)
            - 'class_probs'(np.ndarray, shape [N, C]): Full class probability distribution
            - 'bbox_stds'  (np.ndarray, shape [N, 4]): Predicted bbox std deviations
                           in delta-space [std_dx, std_dy, std_dw, std_dh].
                           If absent, a default uncertainty is assumed.

        ground_truths: List of per-image GT dicts. Each dict must have:
            - 'boxes'      (np.ndarray, shape [M, 4]): GT boxes [x1, y1, x2, y2]
            - 'labels'     (np.ndarray, shape [M]):    GT class indices
            - 'image_size' (tuple):                    (height, width) of the image

        heatmap_threshold: Probability values below this are clipped to zero in the
                           heatmap. This mimics the thresholding in the reference
                           implementation for numerical stability.
        bbox_std_fallback: Pixel corner std when bbox_stds are missing (see delta_stds_to_pixel_stds).
        show_progress: tqdm over images when True.

    Returns:
        dict with keys:
            'pdq'              : float [0, 1] — the main PDQ score
            'avg_spatial_quality' : float [0, 1] — mean spatial Q over TPs only
            'avg_label_quality'   : float [0, 1] — mean label Q over TPs only
            'total_tp'         : int
            'total_fp'         : int
            'total_fn'         : int
    """
    assert len(predictions) == len(ground_truths), \
        "Number of prediction and ground-truth images must match."
    tot_overall_quality = 0.0
    tot_spatial_quality = 0.0
    tot_label_quality = 0.0
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    tasks = [(pred, gt, heatmap_threshold, bbox_std_fallback) for pred, gt in zip(predictions, ground_truths)]
    num_workers = min(multiprocessing.cpu_count(), 16) # Use up to 16 cores
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        if show_progress:
            results = list(tqdm(executor.map(_process_image_task, tasks), total=len(tasks), desc="PDQ", unit="img"))
        else:
            results = list(executor.map(_process_image_task, tasks))
    for img_results in results:
        tot_overall_quality += img_results["overall"]
        tot_spatial_quality += img_results["spatial"]
        tot_label_quality   += img_results["label"]
        tot_tp += img_results["TP"]
        tot_fp += img_results["FP"]
        tot_fn += img_results["FN"]
    tot_pairs = tot_tp + tot_fp + tot_fn
    pdq_score = tot_overall_quality / tot_pairs if tot_pairs > 0 else 0.0
    avg_spatial = tot_spatial_quality / tot_tp if tot_tp > 0 else 0.0
    avg_label   = tot_label_quality   / tot_tp if tot_tp > 0 else 0.0
    return {
        "pdq": float(pdq_score),
        "avg_spatial_quality": float(avg_spatial),
        "avg_label_quality": float(avg_label),
        "total_tp": tot_tp,
        "total_fp": tot_fp,
        "total_fn": tot_fn,
    }


# ---------------------------------------------------------------------------
# Per-image evaluation
# ---------------------------------------------------------------------------

def _evaluate_image(
    pred: Dict[str, Any],
    gt: Dict[str, Any],
    img_size: tuple,
    heatmap_threshold: float,
    bbox_std_fallback: float = 10.0,
) -> Dict[str, Any]:
    """
    Run PDQ evaluation for a single image.

    Strategy:
    1. Build a Gaussian heatmap for each detection (spatial probability map).
    2. Compute pairwise (Q_spatial, Q_label) for every possible (GT, Det) pair.
    3. Combine into pPDQ = sqrt(Q_s * Q_l), build a cost matrix (1 - pPDQ).
    4. Run the Hungarian algorithm to find the globally optimal assignment.
    5. Count TPs (quality > 0), FPs, and FNs.
    """
    pred_boxes  = np.array(pred.get("boxes", []))
    pred_stds   = pred.get("bbox_stds", None)
    class_probs = pred.get("class_probs", None)
    pred_scores = np.array(pred.get("scores", []))

    gt_boxes  = np.array(gt.get("boxes", []))
    gt_labels = np.array(gt.get("labels", []))

    n_det = len(pred_boxes)
    n_gt  = len(gt_boxes)

    # Edge case: nothing to compare
    if n_gt == 0 or n_det == 0:
        return {
            "overall": 0.0, "spatial": 0.0, "label": 0.0,
            "TP": 0, "FP": n_det, "FN": n_gt,
        }

    # Convert predicted bbox stds from delta-space to pixel-space
    pixel_stds = delta_stds_to_pixel_stds(pred_boxes, pred_stds, fallback_std=bbox_std_fallback)

    # Build Gaussian heatmaps: shape (H, W, D)
    H, W = img_size
    heatmaps = _build_gaussian_heatmaps(pred_boxes, pixel_stds, H, W, heatmap_threshold)

    # Build GT pixel masks: shape (H, W, G)
    num_fg_pixels = _get_num_fg_pixels(gt_boxes, H, W)
    # gt_fg_masks, gt_bg_masks, num_fg_pixels = _build_gt_masks(gt_boxes, H, W)

    # Vectorised spatial quality: (G, D) matrix
    spatial_qual = _compute_spatial_quality(gt_boxes, heatmaps, num_fg_pixels, H, W)
    # spatial_qual = _compute_spatial_quality(gt_fg_masks, gt_bg_masks, heatmaps, num_fg_pixels)

    # Vectorised label quality: (G, D) matrix
    try:
        label_qual = _compute_label_quality(gt_labels, class_probs, n_det)
    except ValueError as e:
        # class_probs missing → do not corrupt PDQ with a wrong proxy.
        # Count all detections as FP and all GTs as FN, then skip.

        warnings.warn(f"[PDQ] Skipping image: {e}")
        return {
            "overall": 0.0, "spatial": 0.0, "label": 0.0,
            "TP": 0, "FP": n_det, "FN": n_gt,
        }

    # Overall pairwise quality = geometric mean of spatial and label quality
    overall_qual = _geometric_mean(spatial_qual, label_qual)

    # Hungarian optimal assignment (minimise cost = 1 - quality)
    n_pairs = max(n_gt, n_det)
    cost_matrix = np.ones((n_pairs, n_pairs), dtype=np.float32)
    cost_matrix[:n_gt, :n_det] = 1.0 - overall_qual
    row_ids, col_ids = linear_sum_assignment(cost_matrix)

    # Full quality table (padded), needed to read back values
    qual_table = np.zeros((n_pairs, n_pairs), dtype=np.float32)
    qual_table[:n_gt, :n_det] = overall_qual

    spatial_table = np.zeros((n_pairs, n_pairs), dtype=np.float32)
    spatial_table[:n_gt, :n_det] = spatial_qual

    label_table = np.zeros((n_pairs, n_pairs), dtype=np.float32)
    label_table[:n_gt, :n_det] = label_qual

    # Tally TPs, FPs, FNs
    # Rule: a pair is a TP if quality > 0 AND both indices are within the real
    # (non-padded) range.  Padding slots that the Hungarian algo assigns to each
    # other are ignored completely — they are neither FP nor FN.
    tp, fp, fn = 0, 0, 0
    tot_overall = 0.0
    tot_spatial = 0.0
    tot_label   = 0.0

    for r, c in zip(row_ids, col_ids):
        real_gt  = r < n_gt
        real_det = c < n_det
        q = qual_table[r, c]

        if real_gt and real_det:
            if q > 0:
                # TRUE POSITIVE: both exist and have non-zero quality
                tp += 1
                tot_overall += q
                tot_spatial += spatial_table[r, c]
                tot_label   += label_table[r, c]
            else:
                # Both exist but quality is 0 — count as FN+FP (missed match)
                fn += 1
                fp += 1
        elif real_gt and not real_det:
            # GT has no matching detection → False Negative
            fn += 1
        elif real_det and not real_gt:
            # Detection has no matching GT → False Positive
            fp += 1
        # else: padding-to-padding slot — ignore entirely

    return {
        "overall": tot_overall,
        "spatial": tot_spatial,
        "label":   tot_label,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


# ---------------------------------------------------------------------------
# Spatial Quality
# ---------------------------------------------------------------------------

def _build_gt_masks(gt_boxes: np.ndarray, H: int, W: int):
    """
    Build binary foreground and background masks for each GT box.

    Foreground (fg):  pixels INSIDE the GT bounding box.
    Background (bg):  all OTHER pixels in the image.

    Returns:
        fg_masks:      (H, W, G) bool array — True inside GT box
        bg_masks:      (H, W, G) bool array — True outside GT box
        num_fg_pixels: (G, 1) int array — area of each GT box in pixels
    """
    G = len(gt_boxes)
    fg_masks = np.zeros((H, W, G), dtype=bool)
    for g_idx, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Clamp to image bounds — using exclusive end convention [x1, y1, x2, y2]
        # where x2/y2 are the first pixels OUTSIDE the box (standard COCO format).
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        fg_masks[y1:y2, x1:x2, g_idx] = True   # no +1: x2 and y2 are exclusive

    bg_masks = ~fg_masks
    num_fg_pixels = fg_masks.sum(axis=(0, 1)).reshape(G, 1)   # (G, 1)
    # Guard against zero-area boxes
    num_fg_pixels = np.maximum(num_fg_pixels, 1)
    return fg_masks, bg_masks, num_fg_pixels


def _get_num_fg_pixels(gt_boxes: np.ndarray, H: int, W: int) -> np.ndarray:
    """Compute the area (number of foreground pixels) for each GT box."""
    G = len(gt_boxes)
    num_fg_pixels = np.zeros((G, 1), dtype=np.float32)
    for g_idx, box in enumerate(gt_boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        area = max(1, (x2 - x1) * (y2 - y1))
        num_fg_pixels[g_idx, 0] = area
    return num_fg_pixels


def _build_gaussian_heatmaps(
    pred_boxes: np.ndarray,
    pixel_stds: np.ndarray,
    H: int,
    W: int,
    threshold: float,
) -> np.ndarray:
    """
    Build a probability heatmap for each detection.

    Each detection is modelled as two independent axis-aligned Gaussians:
    one for the left/top corner (x1, y1) and one for the right/bottom corner (x2, y2).

    The heatmap value at pixel (r, c) represents the probability that the pixel
    belongs to the detected object, defined as:

        p(pixel) = p(x1 < c) * p(c < x2) * p(y1 < r) * p(r < y2)

    Where each boundary is a Gaussian CDF evaluated at that pixel coordinate.

    This is mathematically equivalent to the approach in the original reference
    implementation (using Gaussian heatmaps for the probability of each pixel
    being inside the detection region).

    Returns:
        heatmaps: (H, W, D) float32 array, values in [0, 1]
    """


    D = len(pred_boxes)
    heatmaps = np.zeros((H, W, D), dtype=np.float32)

    # Pixel coordinate grids
    xs = np.arange(W, dtype=np.float32)   # (W,)
    ys = np.arange(H, dtype=np.float32)   # (H,)

    sqrt2 = np.sqrt(2.0)

    for d_idx, (box, std) in enumerate(zip(pred_boxes, pixel_stds)):
        x1, y1, x2, y2 = box
        std_x1, std_y1, std_x2, std_y2 = std

        # Guard against zero std (perfectly confident → behaves like a hard box)
        std_x1 = max(std_x1, 0.01)
        std_y1 = max(std_y1, 0.01)
        std_x2 = max(std_x2, 0.01)
        std_y2 = max(std_y2, 0.01)

        # P(X > x1): probability that the left edge is to the LEFT of pixel x
        # Using Gaussian CDF: P(X < x) = 0.5 * (1 + erf((x - mu) / (sqrt(2) * sigma)))
        p_right_of_x1 = 0.5 * (1.0 + scipy_erf((xs - x1) / (sqrt2 * std_x1)))  # (W,)
        # P(X < x2): probability that the right edge is to the RIGHT of pixel x
        p_left_of_x2  = 0.5 * (1.0 - scipy_erf((xs - x2) / (sqrt2 * std_x2)))  # (W,)
        # P(Y > y1)
        p_below_y1    = 0.5 * (1.0 + scipy_erf((ys - y1) / (sqrt2 * std_y1)))  # (H,)
        # P(Y < y2)
        p_above_y2    = 0.5 * (1.0 - scipy_erf((ys - y2) / (sqrt2 * std_y2)))  # (H,)

        # Combine: pixel is inside the box with probability = product of all four
        p_x = p_right_of_x1 * p_left_of_x2   # (W,)
        p_y = p_below_y1    * p_above_y2       # (H,)

        heatmap = np.outer(p_y, p_x)           # (H, W) — outer product

        # Threshold: zero out tiny probabilities for efficiency (as in reference code)
        heatmap[heatmap < threshold] = 0.0
        heatmaps[:, :, d_idx] = heatmap.astype(np.float32)

    return heatmaps


def _compute_spatial_quality(
    gt_boxes: np.ndarray,
    heatmaps: np.ndarray,
    num_fg_pixels: np.ndarray,
    H: int,
    W: int,
) -> np.ndarray:
    G = len(gt_boxes)
    D = heatmaps.shape[2]
    
    log_heatmap = np.log(heatmaps + _SMALL_VAL)
    log_bg = np.log(1.0 - heatmaps + _SMALL_VAL) * (heatmaps > 0)
    
    # Total background loss for each detection over the whole image
    total_log_bg = log_bg.sum(axis=(0, 1)) # Shape: (D,)
    
    fg_loss = np.zeros((G, D), dtype=np.float32)
    bg_loss = np.zeros((G, D), dtype=np.float32)
    
    for g in range(G):
        x1, y1, x2, y2 = int(gt_boxes[g, 0]), int(gt_boxes[g, 1]), int(gt_boxes[g, 2]), int(gt_boxes[g, 3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        # Sum over the bounding box region only
        fg_loss[g, :] = log_heatmap[y1:y2, x1:x2, :].sum(axis=(0, 1))
        bg_loss_in_box = log_bg[y1:y2, x1:x2, :].sum(axis=(0, 1))
        
        # Background loss outside the box is total minus inside
        bg_loss[g, :] = total_log_bg - bg_loss_in_box
        
    total_loss = fg_loss + bg_loss
    loss_per_pixel = total_loss / num_fg_pixels
    spatial_qual = np.exp(loss_per_pixel)
    
    spatial_qual[np.isclose(spatial_qual, 0.0)] = 0.0
    spatial_qual[np.isclose(spatial_qual, 1.0)] = 1.0
    
    return spatial_qual.astype(np.float32)


# ---------------------------------------------------------------------------
# Label Quality
# ---------------------------------------------------------------------------

def _compute_label_quality(
    gt_labels: np.ndarray,
    class_probs,
    n_det: int,
) -> np.ndarray:
    """
    Compute label quality Q_L(G, D) for all (GT, Detection) pairs.

    Q_L is simply the probability the detector assigned to the GT class:
        Q_L(G, D) = class_probs_D[ label_G ]

    This directly penalizes overconfident wrong predictions (e.g., P("cat") = 0.99
    when GT is "dog" → Q_L = P("dog") ≈ 0.01).

    Raises:
        ValueError: if class_probs is None or has the wrong number of rows.
                    The soft-label pipeline always provides class_probs, so this
                    should never occur. If it does, skipping the image (rather
                    than using a wrong score-based proxy) is the correct behaviour.

    Args:
        gt_labels:   (G,) int — GT class indices
        class_probs: (D, C) float — full predicted class distributions
        n_det:       int — number of detections D

    Returns:
        label_qual: (G, D) float array, values in [0, 1]
    """
    G = len(gt_labels)

    if class_probs is None or len(class_probs) != n_det:
        raise ValueError(
            f"class_probs missing or has wrong shape (expected {n_det} rows, "
            f"got {len(class_probs) if class_probs is not None else None}). "
            "The soft-label pipeline should always output class_probs. "
            "Skipping this image to avoid corrupting the PDQ score."
        )

    # Full distribution available: look up the GT class column
    class_probs_arr = np.array(class_probs, dtype=np.float32)  # (D, C)
    # Clamp to [0,1] and renormalize for safety
    class_probs_arr = np.clip(class_probs_arr, 0.0, 1.0)
    row_sums = class_probs_arr.sum(axis=1, keepdims=True)
    class_probs_arr = class_probs_arr / np.maximum(row_sums, _SMALL_VAL)

    # label_qual[g, d] = class_probs[d, gt_labels[g]]
    label_qual = class_probs_arr[:, gt_labels].T    # (G, D)
    return label_qual.astype(np.float32)


# ---------------------------------------------------------------------------
# Geometric mean (the pPDQ formula)
# ---------------------------------------------------------------------------

def _geometric_mean(spatial_qual: np.ndarray, label_qual: np.ndarray) -> np.ndarray:
    """
    Compute pPDQ = sqrt( Q_spatial * Q_label ) for all (G, D) pairs.

    The geometric mean ensures BOTH quality components must be high for a good score.
    A perfect spatial fit but zero label probability → pPDQ = 0.
    A perfect label probability but poor spatial coverage → pPDQ ≈ 0.
    """
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.sqrt(spatial_qual * label_qual)
    result = np.nan_to_num(result, nan=0.0)
    return result.astype(np.float32)


def evaluate_pdq(
    predictions,
    ground_truths,
    *,
    class_names=None,
    gmm_type="1D",
    probabilistic=False,
    output_dir=None,
    label_file=None,
    priors_file=None,
    **kwargs,
):
    """Run PDQ evaluation and return JSON-serializable results.

    Raises ValueError upfront if preconditions are not met, rather than
    running the full computation and catching a vague exception at the end.

    The spatial fallback std (used when bbox_stds are missing) is derived
    automatically from the softlabel priors JSON in output_dir, rather than
    using a hardcoded value with no scientific justification.
    """
    ref_h, ref_w = _median_image_hw(ground_truths)
    priors_path = _resolve_class_priors_json(label_file, priors_file)
    fallback_std = None
    if priors_path:
        fallback_std = _load_bbox_std_fallback_from_class_priors(priors_path, ref_h, ref_w)
        if fallback_std is not None:
            print(
                f"  [PDQ] Bbox std fallback from priors: {fallback_std:.2f} px "
                f"({priors_path}; used when bbox_stds missing)"
            )
        else:
            print(f"  [PDQ] Could not parse priors file: {priors_path}")
    if fallback_std is None:
        all_sizes: List[float] = []
        for p in predictions:
            boxes = p.get("boxes", [])
            if len(boxes) == 0:
                continue
            b = np.asarray(boxes, dtype=np.float64)
            widths = b[:, 2] - b[:, 0]
            all_sizes.append(float(np.max(widths)))
        if all_sizes:
            fallback_std = max(float(np.mean(all_sizes) * 0.1), 5.0)
            print(
                f"  [PDQ] No usable class priors; data-derived bbox std fallback: "
                f"{fallback_std:.2f} px"
            )
        else:
            fallback_std = 10.0
            print(
                f"  [PDQ] No priors / boxes; default bbox std fallback: {fallback_std:.2f} px"
            )

    # --- Pre-check 1: PDQ requires a probabilistic model ---
    if gmm_type == "HARD_LABEL" or not probabilistic:
        raise ValueError(
            "PDQ requires probabilistic model outputs (class_probs and bbox_stds). "
            "Run with --gmm-type 1D and a probabilistic model."
        )

    # --- Pre-check 2: Scan predictions for missing required fields ---
    missing = []
    for idx, pred in enumerate(predictions):
        if len(pred.get("boxes", [])) == 0:
            continue  # empty-prediction images are fine — PDQ will count them as FN
        if "class_probs" not in pred:
            missing.append(f"image {idx}: class_probs")
        if "bbox_stds" not in pred:
            missing.append(f"image {idx}: bbox_stds")

    if missing:
        raise ValueError(
            "PDQ missing required prediction fields:\n  "
            + "\n  ".join(missing[:5])
            + (f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )

    # --- All checks passed: run PDQ ---
    print("\n" + "=" * 70)
    print("PDQ — Probabilistic Detection Quality")
    print("=" * 70)
    pdq_results = compute_pdq(
        predictions,
        ground_truths,
        heatmap_threshold=0.01,
        bbox_std_fallback=fallback_std,
        show_progress=True,
    )
    print(f"  PDQ Score:        {pdq_results['pdq']:.4f}  (main metric, higher is better)")
    print(f"  Avg Spatial Q:    {pdq_results['avg_spatial_quality']:.4f}  (localization uncertainty quality)")
    print(f"  Avg Label Q:      {pdq_results['avg_label_quality']:.4f}  (class probability quality)")
    print(f"  TP / FP / FN:     {pdq_results['total_tp']} / {pdq_results['total_fp']} / {pdq_results['total_fn']}")
    return {k: float(v) if isinstance(v, float) else int(v)
            for k, v in pdq_results.items()}

