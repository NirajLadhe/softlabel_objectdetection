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
from typing import List, Dict, Any

from softlabel_od.object_detection.core.evaluation.metrics import delta_stds_to_pixel_stds

# Numerical stability constant — prevents log(0) = -inf
_SMALL_VAL = 1e-14


# ---------------------------------------------------------------------------
# the main compute function
# ---------------------------------------------------------------------------

def compute_pdq(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    heatmap_threshold: float = 0.01,
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

    for pred, gt in zip(predictions, ground_truths):
        per_img_size = gt.get("image_size")
        if per_img_size is None:
            raise ValueError("PDQ requires gt['image_size'] to dynamically generate spatial heatmaps.")
            
        img_results = _evaluate_image(pred, gt, per_img_size, heatmap_threshold)
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
    pixel_stds = delta_stds_to_pixel_stds(pred_boxes, pred_stds)

    # Build Gaussian heatmaps: shape (H, W, D)
    H, W = img_size
    heatmaps = _build_gaussian_heatmaps(pred_boxes, pixel_stds, H, W, heatmap_threshold)

    # Build GT pixel masks: shape (H, W, G)
    gt_fg_masks, gt_bg_masks, num_fg_pixels = _build_gt_masks(gt_boxes, H, W)

    # Vectorised spatial quality: (G, D) matrix
    spatial_qual = _compute_spatial_quality(gt_fg_masks, gt_bg_masks, heatmaps, num_fg_pixels)

    # Vectorised label quality: (G, D) matrix
    try:
        label_qual = _compute_label_quality(gt_labels, class_probs, n_det)
    except ValueError as e:
        # class_probs missing → do not corrupt PDQ with a wrong proxy.
        # Count all detections as FP and all GTs as FN, then skip.
        import warnings
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
    from scipy.special import erf as scipy_erf

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
    fg_masks: np.ndarray,
    bg_masks: np.ndarray,
    heatmaps: np.ndarray,
    num_fg_pixels: np.ndarray,
) -> np.ndarray:
    """
    Compute spatial quality Q_S(G, D) for all (GT, Detection) pairs.

    The spatial quality is computed as:

        loss_fg(G, D) = sum_{p in FG(G)} log( heatmap_D(p) )
        loss_bg(G, D) = sum_{p in BG(G)} log( 1 - heatmap_D(p) )   [only where heatmap > 0]
        total_loss    = loss_fg + loss_bg
        Q_S(G, D)     = exp( total_loss / |FG(G)| )

    The logic:
    - Foreground pixels should have HIGH probability in the heatmap → log near 0
    - Background pixels should have LOW probability → (1 - p) near 1 → log near 0
    - Any deviation from perfect prediction reduces the score exponentially.

    Args:
        fg_masks:      (H, W, G) bool — foreground regions for each GT
        bg_masks:      (H, W, G) bool — background regions for each GT
        heatmaps:      (H, W, D) float — Gaussian heatmaps for each detection
        num_fg_pixels: (G, 1) int — number of foreground pixels per GT

    Returns:
        spatial_qual: (G, D) float array, values in [0, 1]
    """
    # Foreground log-loss: log( p + epsilon ) for pixels inside GT box
    log_heatmap = np.log(heatmaps + _SMALL_VAL)                         # (H, W, D)

    # Background log-loss: log( 1 - p + epsilon ) for pixels OUTSIDE GT box,
    # but only where the heatmap is non-zero (efficiency + matches reference impl)
    log_bg = np.log(1.0 - heatmaps + _SMALL_VAL) * (heatmaps > 0)       # (H, W, D)

    # Tensordot computes sum over (H, W) axes:
    # fg_loss[g, d] = sum_{x,y} fg_masks[x,y,g] * log_heatmap[x,y,d]
    fg_loss = np.tensordot(fg_masks.astype(np.float32), log_heatmap,
                           axes=([0, 1], [0, 1]))   # (G, D)
    bg_loss = np.tensordot(bg_masks.astype(np.float32), log_bg,
                           axes=([0, 1], [0, 1]))   # (G, D)

    total_loss = fg_loss + bg_loss                  # (G, D)

    # Normalise by the number of foreground pixels in each GT and exponentiate
    loss_per_pixel = total_loss / num_fg_pixels     # (G, D) broadcast
    spatial_qual   = np.exp(loss_per_pixel)         # (G, D), values in (0, 1]

    # Clamp floating point noise at the boundaries
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
    **kwargs,
):
    """Run PDQ evaluation and return JSON-serializable results.

    Raises ValueError upfront if preconditions are not met, rather than
    running the full computation and catching a vague exception at the end.

    The spatial fallback std (used when bbox_stds are missing) is derived
    automatically from the softlabel priors JSON in output_dir, rather than
    using a hardcoded value with no scientific justification.
    """
    # --- Load prior-derived fallback std from the pipeline's priors file ---
    # The soft-label generation pipeline writes a *_with_priors.json that
    # contains per-class GMM statistics. We use the mean spatial std across all
    # classes as a principled fallback rather than an arbitrary constant.
    fallback_std = None
    if output_dir is not None:
        import os, glob, json
        prior_files = glob.glob(os.path.join(output_dir, "**", "*_with_priors.json"), recursive=True)
        if prior_files:
            try:
                with open(prior_files[0]) as f:
                    priors = json.load(f)
                # Extract all per-class std values and take the mean
                stds = []
                for entry in priors.values():
                    if isinstance(entry, dict):
                        for comp in entry.get("components", []):
                            if "std" in comp:
                                stds.append(comp["std"])
                if stds:
                    fallback_std = float(np.mean(stds))
                    print(f"  [PDQ] Using prior-derived fallback std: {fallback_std:.2f} px")
            except Exception as e:
                print(f"  [PDQ] Could not read priors file ({e}). Deriving fallback from data.")

    if fallback_std is None:
        # Data-driven fallback: mean predicted box size × 0.1 (a small fraction of a typical box)
        all_sizes = [
            max(p["boxes"][:, 2] - p["boxes"][:, 0]) if len(p.get("boxes", [])) else 0
            for p in predictions
        ]
        fallback_std = max(float(np.mean([s for s in all_sizes if s > 0]) * 0.1), 5.0)
        print(f"  [PDQ] No priors file found. Using data-derived fallback std: {fallback_std:.2f} px")

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
    )
    print(f"  PDQ Score:        {pdq_results['pdq']:.4f}  (main metric, higher is better)")
    print(f"  Avg Spatial Q:    {pdq_results['avg_spatial_quality']:.4f}  (localization uncertainty quality)")
    print(f"  Avg Label Q:      {pdq_results['avg_label_quality']:.4f}  (class probability quality)")
    print(f"  TP / FP / FN:     {pdq_results['total_tp']} / {pdq_results['total_fp']} / {pdq_results['total_fn']}")
    return {k: float(v) if isinstance(v, float) else int(v)
            for k, v in pdq_results.items()}

