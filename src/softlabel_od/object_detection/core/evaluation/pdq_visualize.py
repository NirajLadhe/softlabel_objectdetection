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
    _get_num_fg_pixels,         # <-- Changed this
    _compute_spatial_quality,
    _compute_label_quality,
    _geometric_mean,
)

# Add these imports at the top
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Add this helper function outside of collect_pdq_image_results
def _process_viz_image_task(args):
    img_idx, pred, gt, img_path, heatmap_threshold = args
    
    img_size = gt.get("image_size")
    if img_size is None:
        return None # Return None instead of warning to avoid cluttering parallel output
        
    H, W = img_size
    pred_boxes  = np.array(pred.get("boxes", []))
    pred_stds   = pred.get("bbox_stds", None)
    class_probs = pred.get("class_probs", None)
    gt_boxes    = np.array(gt.get("boxes", []))
    gt_labels   = np.array(gt.get("labels", []))

    n_det = len(pred_boxes)
    n_gt  = len(gt_boxes)

    pred_scores = np.array(pred.get("scores", []))
    if len(pred_scores) == n_det and n_det > 0:
        keep = pred_scores >= _SCORE_THRESHOLD
        pred_boxes = pred_boxes[keep]
        n_det = len(pred_boxes)
        if pred_stds is not None:
            pred_stds = np.array(pred_stds)[keep]
        if class_probs is not None:
            class_probs = np.array(class_probs)[keep]

    gt_stds = gt.get("gt_stds", None)
    gt_class_probs = gt.get("gt_class_probs", None)

    if n_gt == 0 or n_det == 0:
        return _make_result(
            img_path, img_size, pred_boxes, pred_stds, class_probs,
            gt_boxes, gt_labels, gt_stds, gt_class_probs, matches=[], fp=list(range(n_det)),
            fn=list(range(n_gt)), pdq=0.0,
        )

    pixel_stds = delta_stds_to_pixel_stds(pred_boxes, pred_stds)

    try:
        label_qual = _compute_label_quality(gt_labels, class_probs, n_det)
    except ValueError:
        return _make_result(
            img_path, img_size, pred_boxes, pred_stds, class_probs,
            gt_boxes, gt_labels, gt_stds, gt_class_probs, matches=[], fp=list(range(n_det)),
            fn=list(range(n_gt)), pdq=0.0, pixel_stds=pixel_stds,
        )

    heatmaps = _build_gaussian_heatmaps(pred_boxes, pixel_stds, H, W, heatmap_threshold)
    num_fg_pixels = _get_num_fg_pixels(gt_boxes, H, W)
    spatial_qual = _compute_spatial_quality(gt_boxes, heatmaps, num_fg_pixels, H, W)
    overall_qual = _geometric_mean(spatial_qual, label_qual)

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

    return _make_result(
        img_path, img_size, pred_boxes, pred_stds, class_probs,
        gt_boxes, gt_labels, gt_stds, gt_class_probs, matches=matches, fp=sorted(fp_indices),
        fn=sorted(fn_indices), pdq=float(image_pdq), pixel_stds=pixel_stds,
    )

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_GT_COLOR   = "#A9FC6D"   # OKAY-ACCENT (Bright Green)
_PRED_COLOR = "#336B87"   # PRIMARY (Navy)
_FP_COLOR   = "#F59927"   # SECONDARY-C (Light Orange)
_FN_COLOR   = "#4F85A0"   # SECONDARY (Light Blue)
_TEXT_BG    = "#3E464B"   # SUPPORT (Dark Blue)

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
    from tqdm import tqdm
    n_images = len(predictions)
    tasks = []
    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        img_path = image_paths[img_idx] if image_paths else None
        tasks.append((img_idx, pred, gt, img_path, heatmap_threshold))
    num_workers = min(multiprocessing.cpu_count(), 16)
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(_process_viz_image_task, tasks), total=len(tasks), desc="Computing PDQ per image"):
            if res is not None:
                results.append(res)
    return results


def _make_result(
    img_path, img_size, pred_boxes, pred_stds, class_probs,
    gt_boxes, gt_labels, gt_stds, gt_class_probs, *, matches, fp, fn, pdq, pixel_stds=None,
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
        "gt_stds":     gt_stds,
        "gt_class_probs": gt_class_probs,
        "matches":     matches,
        "fp_indices":  fp,
        "fn_indices":  fn,
        "image_pdq":   pdq,
    }


# ---------------------------------------------------------------------------
# Step 2: render a single image panel
# ---------------------------------------------------------------------------

def render_side_by_side(
    result: Dict[str, Any],
    save_path: str,
    class_names: Optional[Dict[int, str]] = None,
    heatmap_alpha: float = 0.45,
):
    fig, (ax_gt, ax_pred) = plt.subplots(1, 2, figsize=(16, 8), facecolor="#faf8f5")
    
    H, W = result["image_size"]
    
    # --- LEFT: GROUND TRUTH ---
    _draw_background(ax_gt, result.get("image_path"), H, W)
    
    # GT Heatmap
    if result.get("gt_stds") is not None and len(result["gt_boxes"]) > 0:
        gt_stds_px = np.array(result["gt_stds"]) * np.array([W, H, W, H])
        sigma_cx = gt_stds_px[:, 0]
        sigma_cy = gt_stds_px[:, 1]
        sigma_w = gt_stds_px[:, 2]
        sigma_h = gt_stds_px[:, 3]
        std_x = np.sqrt(sigma_cx**2 + 0.25 * sigma_w**2)
        std_y = np.sqrt(sigma_cy**2 + 0.25 * sigma_h**2)
        gt_pixel_stds = np.stack([std_x, std_y, std_x, std_y], axis=1)
        
        gt_heatmaps = _build_gaussian_heatmaps(result["gt_boxes"], gt_pixel_stds, H, W, threshold=0.01)
        if gt_heatmaps.shape[2] > 0:
            composite = gt_heatmaps.max(axis=2)
            if composite.max() > 0:
                norm = mcolors.Normalize(vmin=0, vmax=composite.max())
                rgba = plt.cm.Blues(norm(composite))
                rgba[..., 3] = heatmap_alpha * (composite > 0).astype(float)
                ax_gt.imshow(rgba, aspect="auto")
                
    # GT Boxes
    for g_idx, gt_box in enumerate(result["gt_boxes"]):
        x1, y1, x2, y2 = gt_box
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=_GT_COLOR, facecolor="none",
            linestyle="-", boxstyle="square,pad=0",
        )
        ax_gt.add_patch(rect)
        
        match_id = None
        for m_idx, (mg, md, q_o, q_s, q_l) in enumerate(result["matches"]):
            if mg == g_idx:
                match_id = m_idx + 1
                break
                
        label_idx = int(result["gt_labels"][g_idx])
        label_text = class_names.get(label_idx, f"GT:{label_idx}") if class_names else f"GT:{label_idx}"
        if match_id is not None:
            label_text = f"[M{match_id}] {label_text}"
        elif g_idx in result["fn_indices"]:
            label_text = f"[Missed] {label_text}"
            
        ax_gt.text(x1 + 2, y1 - 4, label_text,
                color=_GT_COLOR, fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc=_TEXT_BG, alpha=0.8, ec="none"))
                
        if result.get("gt_class_probs") is not None:
            _draw_prob_bar(ax_gt, result["gt_class_probs"][g_idx], x2, y1, class_names)
            
    # FN Markers
    for g_idx in result["fn_indices"]:
        x1, y1, x2, y2 = result["gt_boxes"][g_idx]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax_gt.plot(cx, cy, "x", color=_FN_COLOR, markersize=12, markeredgewidth=3)

    ax_gt.set_title("Ground Truth (Annotator Uncertainty)", color="black", fontsize=20)
    ax_gt.axis("off")
    
    # --- RIGHT: PREDICTION ---
    _draw_background(ax_pred, result.get("image_path"), H, W)
    
    if len(result["pred_boxes"]) > 0:
        pred_heatmaps = _build_gaussian_heatmaps(
            result["pred_boxes"], result["pixel_stds"], H, W, threshold=0.01,
        )
        if pred_heatmaps.shape[2] > 0:
            composite = pred_heatmaps.max(axis=2)
            if composite.max() > 0:
                norm = mcolors.Normalize(vmin=0, vmax=composite.max())
                rgba = plt.cm.hot(norm(composite))
                rgba[..., 3] = heatmap_alpha * (composite > 0).astype(float)
                ax_pred.imshow(rgba, aspect="auto")
                
    for m_idx, (g_idx, d_idx, q_overall, q_spatial, q_label) in enumerate(result["matches"]):
        x1, y1, x2, y2 = result["pred_boxes"][d_idx]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=_PRED_COLOR, facecolor="none",
            linestyle="--", boxstyle="square,pad=0",
        )
        ax_pred.add_patch(rect)
        badge = f"[M{m_idx+1}] PDQ={q_overall:.2f} (Sp={q_spatial:.2f}, Lb={q_label:.2f})"
        ax_pred.text(x1 + 2, y1 - 4, badge,
                color="white", fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=_PRED_COLOR, alpha=0.9, ec="none"))
        if result["class_probs"] is not None:
            _draw_prob_bar(ax_pred, result["class_probs"][d_idx], x2, y1, class_names)
            
    for d_idx in result["fp_indices"]:
        x1, y1, x2, y2 = result["pred_boxes"][d_idx]
        rect = mpatches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=_FP_COLOR, facecolor="none",
            linestyle=":", boxstyle="square,pad=0",
        )
        ax_pred.add_patch(rect)
        ax_pred.text(x1 + 2, y1 - 4, "False Positive",
                color=_FP_COLOR, fontsize=12, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc=_TEXT_BG, alpha=0.8, ec="none"))
                
    ax_pred.set_title(f"Prediction (Model Uncertainty) | Image PDQ: {result['image_pdq']:.3f}", color="black", fontsize=20)
    ax_pred.axis("off")
    
    if len(result["matches"]) > 0:
        table_text = "Match Details:\n"
        for m_idx, (g_idx, d_idx, q_overall, q_spatial, q_label) in enumerate(result["matches"]):
            table_text += f"[M{m_idx+1}] PDQ: {q_overall:.3f} = sqrt(Spatial: {q_spatial:.3f} * Label: {q_label:.3f})\n"
        fig.text(0.5, 0.02, table_text, color="white", fontsize=16, ha="center", va="bottom", 
                 bbox=dict(fc=_TEXT_BG, alpha=0.8, boxstyle="round,pad=0.5", ec="none"))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


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
        label = class_names.get(int(cls_idx), str(cls_idx)) if class_names else str(cls_idx)
        ax.text(anchor_x + 1, y_pos, f"{label[:8]} {prob:.2f}",
                va="center", fontsize=5, color="white")


# ---------------------------------------------------------------------------
# Step 3: render grid panels (best / worst / failures)
# ---------------------------------------------------------------------------

def render_pdq_grid(
    results: List[Dict[str, Any]],
    output_dir: str,
    class_names: Optional[Dict[int, str]] = None,
    n_cols: int = 3, # Kept for signature compatibility, but unused
    top_n: int = 9,
    heatmap_alpha: float = 0.45,
):
    """
    Generate and save side-by-side panels to *output_dir*:
        - ``best/``      — top *n* images by PDQ score
        - ``worst/``     — bottom *n* images with at least one TP match
        - ``failures/``  — images with no TP matches (pure FP / FN)
    """
    os.makedirs(output_dir, exist_ok=True)

    sorted_by_pdq = sorted(results, key=lambda r: r["image_pdq"], reverse=True)
    best     = [r for r in sorted_by_pdq if len(r["matches"]) > 0][:top_n]
    worst    = [r for r in reversed(sorted_by_pdq) if len(r["matches"]) > 0][:top_n]
    failures = [r for r in results if len(r["matches"]) == 0
                and (len(r["fp_indices"]) > 0 or len(r["fn_indices"]) > 0)][:top_n]

    _save_category(best,     os.path.join(output_dir, "best"),     "Best PDQ Images", class_names, heatmap_alpha)
    _save_category(worst,    os.path.join(output_dir, "worst"),    "Worst PDQ Images", class_names, heatmap_alpha)
    _save_category(failures, os.path.join(output_dir, "failures"), "Complete Failures", class_names, heatmap_alpha)

    print(f"[PDQ-VIZ] Saved visualizations to: {output_dir}")
    print(f"          best/       ({len(best)} images)")
    print(f"          worst/      ({len(worst)} images)")
    print(f"          failures/   ({len(failures)} images)")


def _save_category(
    results: List[Dict[str, Any]],
    save_dir: str,
    title: str,
    class_names: Optional[Dict[int, str]],
    heatmap_alpha: float,
):
    """Save each result in the category as an individual side-by-side plot."""
    if not results:
        print(f"[PDQ-VIZ] No images to show for: {title}")
        return

    os.makedirs(save_dir, exist_ok=True)
    
    for i, result in enumerate(results):
        # Format: 01_pdq-0.850_imageID.png
        img_id = os.path.basename(result["image_path"]).split(".")[0] if result["image_path"] else f"img_{i}"
        filename = f"{i+1:02d}_pdq-{result['image_pdq']:.3f}_{img_id}.pdf"
        save_path = os.path.join(save_dir, filename)
        
        render_side_by_side(result, save_path, class_names=class_names, heatmap_alpha=heatmap_alpha)
        print(f"[PDQ-VIZ] Saved: {save_path}")
