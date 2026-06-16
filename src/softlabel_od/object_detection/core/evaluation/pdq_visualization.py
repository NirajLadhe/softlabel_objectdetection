"""
pdq_visualize.py — PDQ Visualization Engine

Generates publication-quality grid visualizations of per-image PDQ results.

Each panel overlays:
  1. Raw image background (percentile-normalised for medical 16-bit images)
  2. Gaussian spatial heatmap (hot colormap) showing detection uncertainty
  3. Ground truth boxes (solid green)
  4. Predicted boxes (dashed red) with per-pair score badges
  5. False positive boxes (dotted amber)
  6. False negative markers (blue cross)
  7. Class probability mini-bar chart per matched detection

Usage:
    from pdq_visualize import collect_pdq_image_results, render_pdq_grid
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless — safe for cluster/server environments
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from softlabel_od.object_detection.core.evaluation.metrics import delta_stds_to_pixel_stds
from softlabel_od.object_detection.core.evaluation.pdq import (
    _build_gaussian_heatmaps,
    _build_gt_masks,
    _compute_spatial_quality,
    _compute_label_quality,
    _geometric_mean,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_GT_COLOR   = "#00e676"   # bright green  — ground truth
_PRED_COLOR = "#ff1744"   # vivid red     — prediction (TP)
_FP_COLOR   = "#ff9100"   # amber         — false positive
_FN_COLOR   = "#40c4ff"   # light blue    — false negative

# Minimum detection confidence to include in the visualisation.
# Lower-confidence predictions are filtered out to reduce visual clutter
# and avoid computing heatmaps for noisy, near-zero detections.
_SCORE_THRESHOLD = 0.25


# ---------------------------------------------------------------------------
# Step 1: collect per-image PDQ statistics
# ---------------------------------------------------------------------------

def collect_pdq_image_results(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    image_paths: Optional[List[str]] = None,
    heatmap_threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    """
    Run per-image PDQ evaluation and return rich result dicts for visualisation.

    Re-implements the Hungarian matching from ``compute_pdq`` but retains
    intermediate per-pair quality scores needed for rendering overlays.

    Args:
        predictions:       List of per-image prediction dicts.
        ground_truths:     List of per-image GT dicts (must contain ``image_size``).
        image_paths:       Optional list of raw image file paths (for background).
        heatmap_threshold: Probability values below this are zeroed in heatmaps.

    Returns:
        List of per-image result dicts with keys:
            ``image_path``, ``image_size``, ``pred_boxes``, ``pixel_stds``,
            ``class_probs``, ``gt_boxes``, ``gt_labels``, ``matches``,
            ``fp_indices``, ``fn_indices``, ``image_pdq``.
    """
    from tqdm import tqdm

    results = []
    n_images = len(predictions)

    for img_idx, (pred, gt) in enumerate(
        tqdm(zip(predictions, ground_truths), total=n_images, desc="Computing PDQ per image")
    ):
        img_path = image_paths[img_idx] if image_paths else None
        img_size = gt.get("image_size")
        if img_size is None:
            warnings.warn(f"[PDQ-VIZ] Image {img_idx}: missing gt['image_size'], skipping.")
            continue

        H, W = img_size
        pred_boxes  = np.array(pred.get("boxes", []))
        pred_stds   = pred.get("bbox_stds", None)
        class_probs = pred.get("class_probs", None)
        gt_boxes    = np.array(gt.get("boxes", []))
        gt_labels   = np.array(gt.get("labels", []))

        n_det = len(pred_boxes)
        n_gt  = len(gt_boxes)

        # Filter low-confidence predictions before expensive heatmap computation
        pred_scores = np.array(pred.get("scores", []))
        if len(pred_scores) == n_det and n_det > 0:
            keep = pred_scores >= _SCORE_THRESHOLD
            pred_boxes = pred_boxes[keep]
            n_det = len(pred_boxes)
            if pred_stds is not None:
                pred_stds = np.array(pred_stds)[keep]
            if class_probs is not None:
                class_probs = np.array(class_probs)[keep]

        # Fast path: skip heatmap computation when there is nothing to match
        if n_gt == 0 or n_det == 0:
            results.append(_make_result(
                img_path, img_size, pred_boxes, pred_stds, class_probs,
                gt_boxes, gt_labels, matches=[], fp=list(range(n_det)),
                fn=list(range(n_gt)), pdq=0.0,
            ))
            continue

        # Convert bbox stds from delta-space to pixel-space
        pixel_stds = delta_stds_to_pixel_stds(pred_boxes, pred_stds)

        # Compute label quality early — if class_probs are missing we
        # cannot proceed with matching, so bail out before the expensive
        # heatmap and spatial quality computation.
        try:
            label_qual = _compute_label_quality(gt_labels, class_probs, n_det)
        except ValueError as exc:
            warnings.warn(f"[PDQ-VIZ] Image {img_idx}: {exc}")
            results.append(_make_result(
                img_path, img_size, pred_boxes, pred_stds, class_probs,
                gt_boxes, gt_labels, matches=[], fp=list(range(n_det)),
                fn=list(range(n_gt)), pdq=0.0, pixel_stds=pixel_stds,
            ))
            continue

        # Build Gaussian heatmaps and compute spatial quality
        heatmaps = _build_gaussian_heatmaps(pred_boxes, pixel_stds, H, W, heatmap_threshold)
        gt_fg_masks, gt_bg_masks, num_fg_pixels = _build_gt_masks(gt_boxes, H, W)
        spatial_qual = _compute_spatial_quality(gt_fg_masks, gt_bg_masks, heatmaps, num_fg_pixels)
        overall_qual = _geometric_mean(spatial_qual, label_qual)

        # Hungarian optimal assignment
        n_pairs = max(n_gt, n_det)
        cost_matrix = np.ones((n_pairs, n_pairs), dtype=np.float32)
        cost_matrix[:n_gt, :n_det] = 1.0 - overall_qual
        row_ids, col_ids = linear_sum_assignment(cost_matrix)

        matches    = []
        fp_indices = set(range(n_det))
        fn_indices = set(range(n_gt))
        tot_pdq    = 0.0

        for r, c in zip(row_ids, col_ids):
            if r < n_gt and c < n_det and overall_qual[r, c] > 0:
                matches.append((
                    int(r), int(c),
                    float(overall_qual[r, c]),
                    float(spatial_qual[r, c]),
                    float(label_qual[r, c]),
                ))
                fp_indices.discard(c)
                fn_indices.discard(r)
                tot_pdq += overall_qual[r, c]

        image_pdq = tot_pdq / max(len(matches) + len(fp_indices) + len(fn_indices), 1)

        results.append(_make_result(
            img_path, img_size, pred_boxes, pred_stds, class_probs,
            gt_boxes, gt_labels, matches=matches, fp=sorted(fp_indices),
            fn=sorted(fn_indices), pdq=float(image_pdq), pixel_stds=pixel_stds,
        ))

    return results


def _make_result(
    img_path, img_size, pred_boxes, pred_stds, class_probs,
    gt_boxes, gt_labels, *, matches, fp, fn, pdq, pixel_stds=None,
) -> Dict[str, Any]:
    """Build a standardised per-image result dict."""
    if pixel_stds is None:
        pixel_stds = (
            delta_stds_to_pixel_stds(pred_boxes, pred_stds)
            if len(pred_boxes) > 0
            else np.zeros((0, 4))
        )
    return {
        "image_path":  img_path,
        "image_size":  img_size,
        "pred_boxes":  pred_boxes,
        "pixel_stds":  pixel_stds,
        "class_probs": class_probs,
        "gt_boxes":    gt_boxes,
        "gt_labels":   gt_labels,
        "matches":     matches,
        "fp_indices":  fp,
        "fn_indices":  fn,
        "image_pdq":   pdq,
    }


# ---------------------------------------------------------------------------
# Step 2: render a single image panel
# ---------------------------------------------------------------------------

def render_single_panel(
    result: Dict[str, Any],
    class_names: Optional[Dict[int, str]] = None,
    ax: Optional[plt.Axes] = None,
    show_heatmap: bool = True,
    heatmap_alpha: float = 0.45,
) -> plt.Axes:
    """
    Draw one full PDQ visualisation panel on the given Axes.

    Layers (bottom → top):
        1. Raw image background (or grey canvas)
        2. Gaussian heatmap overlay (hot colormap, semi-transparent)
        3. GT boxes (solid green) with class label
        4. TP prediction boxes (dashed red) with score badges
        5. FP boxes (dotted amber)
        6. FN markers (blue cross on missed GT)
        7. Per-detection class probability mini-bar
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 8))

    H, W = result["image_size"]

    # --- 1. Background image ---
    _draw_background(ax, result.get("image_path"), H, W)

    # --- 2. Gaussian heatmap overlay ---
    if show_heatmap and len(result["pred_boxes"]) > 0:
        heatmaps = _build_gaussian_heatmaps(
            result["pred_boxes"], result["pixel_stds"], H, W, threshold=0.01,
        )
        if heatmaps.shape[2] > 0:
            composite = heatmaps.max(axis=2)
            if composite.max() > 0:
                norm = mcolors.Normalize(vmin=0, vmax=composite.max())
                rgba = plt.cm.hot(norm(composite))
                rgba[..., 3] = heatmap_alpha * (composite > 0).astype(float)
                ax.imshow(rgba, aspect="auto")

    # --- 3. GT boxes (solid green) ---
    for g_idx, gt_box in enumerate(result["gt_boxes"]):
        x1, y1, x2, y2 = gt_box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=_GT_COLOR, facecolor="none",
            linestyle="-", boxstyle="square,pad=0",
        )
        ax.add_patch(rect)
        label_idx = int(result["gt_labels"][g_idx])
        label_text = class_names.get(label_idx, f"GT:{label_idx}") if class_names else f"GT:{label_idx}"
        ax.text(x1 + 2, y1 - 4, label_text,
                color=_GT_COLOR, fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, ec="none"))

    # --- 4. TP prediction boxes with score badges ---
    for g_idx, d_idx, q_overall, q_spatial, q_label in result["matches"]:
        x1, y1, x2, y2 = result["pred_boxes"][d_idx]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=_PRED_COLOR, facecolor="none",
            linestyle="--", boxstyle="square,pad=0",
        )
        ax.add_patch(rect)
        badge = f"PDQ={q_overall:.2f}  Sp={q_spatial:.2f}  Lb={q_label:.2f}"
        ax.text(x1 + 2, y2 + 10, badge,
                color="white", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.15", fc=_PRED_COLOR, alpha=0.8, ec="none"))
        if result["class_probs"] is not None:
            _draw_prob_bar(ax, result["class_probs"][d_idx], x2, y1, class_names)

    # --- 5. FP boxes (no GT match) ---
    for d_idx in result["fp_indices"]:
        x1, y1, x2, y2 = result["pred_boxes"][d_idx]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor=_FP_COLOR, facecolor="none",
            linestyle=":", boxstyle="square,pad=0",
        )
        ax.add_patch(rect)
        ax.text(x1 + 2, y1 - 4, "FP",
                color=_FP_COLOR, fontsize=6, fontweight="bold")

    # --- 6. FN markers (missed GT) ---
    for g_idx in result["fn_indices"]:
        x1, y1, x2, y2 = result["gt_boxes"][g_idx]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax.plot(cx, cy, "x", color=_FN_COLOR, markersize=10, markeredgewidth=2)
        ax.text(cx + 4, cy, "FN", color=_FN_COLOR, fontsize=6, fontweight="bold")

    # --- 7. Title ---
    ax.set_title(
        f"PDQ={result['image_pdq']:.3f}  "
        f"TP={len(result['matches'])}  "
        f"FP={len(result['fp_indices'])}  "
        f"FN={len(result['fn_indices'])}",
        fontsize=9, color="white", pad=4,
        bbox=dict(fc="black", alpha=0.6, boxstyle="round,pad=0.3", ec="none"),
    )
    ax.axis("off")
    return ax


def _draw_background(ax: plt.Axes, img_path: Optional[str], H: int, W: int):
    """Load and display the raw image, or a grey canvas if unavailable.

    Medical images (e.g. VinDr-CXR 16-bit PNGs) are normalised using
    percentile clipping (1st–99th) to produce visible contrast.
    """
    if img_path and os.path.isfile(img_path):
        try:
            raw_np = np.array(Image.open(img_path)).astype(np.float32)

            # Percentile-based normalisation: clip to 1st–99th percentile
            # to remove scanner-border outliers that compress chest contrast.
            lo, hi = np.percentile(raw_np, (1, 99))
            if hi > lo:
                raw_np = np.clip(raw_np, lo, hi)
                raw_np = (raw_np - lo) / (hi - lo) * 255.0
            else:
                raw_np = np.zeros_like(raw_np)
            raw_u8 = raw_np.astype(np.uint8)

            # Ensure 3-channel RGB
            if raw_u8.ndim == 2:
                raw_u8 = np.stack([raw_u8] * 3, axis=-1)
            elif raw_u8.ndim == 3 and raw_u8.shape[2] == 1:
                raw_u8 = np.concatenate([raw_u8] * 3, axis=2)

            img = Image.fromarray(raw_u8, mode="RGB").resize((W, H), Image.BILINEAR)
            ax.imshow(np.array(img), aspect="auto")
        except Exception as exc:
            warnings.warn(f"[PDQ-VIZ] Could not load image {img_path}: {exc}")
            ax.imshow(np.zeros((H, W, 3), dtype=np.uint8), aspect="auto")
    else:
        ax.imshow(np.zeros((H, W, 3), dtype=np.uint8), aspect="auto")


def _draw_prob_bar(
    ax: plt.Axes,
    probs: np.ndarray,
    anchor_x: float,
    anchor_y: float,
    class_names: Optional[Dict[int, str]],
    width: float = 50,
    height: float = 40,
    top_k: int = 4,
):
    """Draw a tiny horizontal bar chart showing the top-k class probabilities."""
    top_k = min(top_k, len(probs))
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs   = probs[top_indices]

    bar_h = height / top_k
    for rank, (cls_idx, prob) in enumerate(zip(top_indices, top_probs)):
        y_pos = anchor_y + rank * bar_h
        bar_w = prob * width
        ax.barh(y_pos, bar_w, height=bar_h * 0.7, left=anchor_x,
                color="#82b1ff", alpha=0.8)
        label = str(class_names.get(int(cls_idx), cls_idx)) if class_names else str(cls_idx)
        ax.text(anchor_x + 1, y_pos, f"{label[:8]} {prob:.2f}",
                va="center", fontsize=5, color="white")


# ---------------------------------------------------------------------------
# Step 3: render grid panels (best / worst / failures)
# ---------------------------------------------------------------------------

def render_pdq_grid(
    results: List[Dict[str, Any]],
    output_dir: str,
    class_names: Optional[Dict[int, str]] = None,
    n_cols: int = 3,
    top_n: int = 9,
    heatmap_alpha: float = 0.45,
):
    """
    Generate and save three grid panels to *output_dir*:

        - ``best_pdq.png``   — top *n* images by PDQ score
        - ``worst_pdq.png``  — bottom *n* images with at least one TP match
        - ``failures.png``   — images with no TP matches (pure FP / FN)

    Args:
        results:       Output of :func:`collect_pdq_image_results`.
        output_dir:    Directory where PNGs are saved.
        class_names:   Mapping of class index → display name.
        n_cols:        Number of columns in each grid.
        top_n:         Number of images to include per grid.
        heatmap_alpha: Transparency of the Gaussian heatmap overlay.
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_by_pdq = sorted(results, key=lambda r: r["image_pdq"], reverse=True)
    best     = [r for r in sorted_by_pdq if len(r["matches"]) > 0][:top_n]
    worst    = [r for r in reversed(sorted_by_pdq) if len(r["matches"]) > 0][:top_n]
    failures = [r for r in results if len(r["matches"]) == 0
                and (len(r["fp_indices"]) > 0 or len(r["fn_indices"]) > 0)][:top_n]

    _save_grid(best,     os.path.join(output_dir, "best_pdq.png"),
               "Best PDQ Images", class_names, n_cols, heatmap_alpha)
    _save_grid(worst,    os.path.join(output_dir, "worst_pdq.png"),
               "Worst PDQ Images (TP but low score)", class_names, n_cols, heatmap_alpha)
    _save_grid(failures, os.path.join(output_dir, "failures.png"),
               "Complete Failures (no TP matches)", class_names, n_cols, heatmap_alpha)

    print(f"[PDQ-VIZ] Saved visualizations to: {output_dir}")
    print(f"          best_pdq.png    ({len(best)} images)")
    print(f"          worst_pdq.png   ({len(worst)} images)")
    print(f"          failures.png    ({len(failures)} images)")


def _save_grid(
    results: List[Dict[str, Any]],
    save_path: str,
    title: str,
    class_names: Optional[Dict[int, str]],
    n_cols: int,
    heatmap_alpha: float,
):
    """Arrange *N* panels into a grid and save to disk."""
    if not results:
        print(f"[PDQ-VIZ] No images to show for: {title}")
        return

    n_rows = int(np.ceil(len(results) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 5, n_rows * 5),
                             facecolor="#1a1a2e")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=1.01)

    axes_flat = np.array(axes).flatten()

    for ax, result in zip(axes_flat, results):
        render_single_panel(result, class_names=class_names,
                             ax=ax, heatmap_alpha=heatmap_alpha)

    for ax in axes_flat[len(results):]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[PDQ-VIZ] Saved: {save_path}")
