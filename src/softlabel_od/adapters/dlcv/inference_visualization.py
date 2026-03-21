"""
inference_visualization.py

Unified inference script for:
- Hard Label Model
- 1D GMM Soft Label Model  
- 2D GMM Soft Label Model (future)

Visualizes GT vs Predictions with various uncertainty visualization methods.

Visualization Methods:
- ellipse: Symmetric uncertainty ellipses at box vertices (simple approximation)
- heatmap: Heatmap overlay showing uncertainty spread
- polygon: Asymmetric confidence regions using log-normal statistics (scientifically accurate)
- monte_carlo: Monte Carlo sampled confidence contours (most accurate, slower)

Usage:
    python inference_visualization.py \
        --hard_weights path/to/hard_model.pth \
        --soft_1d_weights path/to/soft_1d_model.pth \
        --viz_method polygon  # or 'ellipse', 'heatmap', 'monte_carlo'
"""

import os
import sys
import json
import random
import argparse
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from typing import Optional, Dict, List, Tuple
from torchvision.transforms import functional as F

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from softlabel_od.adapters.dlcv.model_1d import get_model_1d
from softlabel_od.adapters.dlcv.dataset import Compose, ToTensor

# ============================================================================
# CONSTANTS
# ============================================================================

CLASS_MAPPING = {
    0: 'No Finding', 
    1: 'Aortic enlargement', 
    2: 'Atelectasis',
    3: 'Calcification', 
    4: 'Cardiomegaly', 
    5: 'Consolidation',
    6: 'ILD', 
    7: 'Infiltration', 
    8: 'Lung Opacity', 
    9: 'Nodule/Mass',
    10: 'Other lesion', 
    11: 'Pleural effusion', 
    12: 'Pleural thickening',
    13: 'Pneumothorax', 
    14: 'Pulmonary fibrosis',
}

# Reverse mapping for JSON parsing
CLASS_TO_IDX = {
    'Aortic enlargement': 1,
    'Atelectasis': 2,
    'Calcification': 3,
    'Cardiomegaly': 4,
    'Consolidation': 5,
    'ILD': 6,
    'Infiltration': 7,
    'Lung Opacity': 8,
    'Nodule/Mass': 9,
    'Other lesion': 10,
    'Pleural effusion': 11,
    'Pleural thickening': 12,
    'Pneumothorax': 13,
    'Pulmonary fibrosis': 14,
    'No Finding': 0,
    'NoFindingInCluster': 0,
    'No finding': 0,
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_gt_labels(json_path: str) -> List[Dict]:
    """Load ground truth labels from JSON file."""
    print(f"Loading GT labels from: {json_path}")
    with open(json_path, 'r') as f:
        labels = json.load(f)
    print(f"Loaded {len(labels)} label entries")
    return labels


def get_labels_for_image(labels: List[Dict], image_id: str) -> List[Dict]:
    """Get all labels for a specific image."""
    return [l for l in labels if l['image_id'] == image_id]


def get_random_image_with_annotations(image_dir: str, labels: List[Dict]) -> Tuple[str, str]:
    """Get a random image that has annotations."""
    # Get unique image IDs with annotations
    image_ids_with_annotations = list(set(l['image_id'] for l in labels))
    
    # Filter to only images that exist on disk
    valid_image_ids = []
    for img_id in image_ids_with_annotations:
        img_path = os.path.join(image_dir, f"{img_id}.png")
        if os.path.exists(img_path):
            valid_image_ids.append(img_id)
    
    if not valid_image_ids:
        raise FileNotFoundError(f"No valid images found in {image_dir}")
    
    image_id = random.choice(valid_image_ids)
    image_path = os.path.join(image_dir, f"{image_id}.png")
    return image_id, image_path


def class_probs_to_vector(class_probs: Dict) -> np.ndarray:
    """Convert class_probs dict to 15-element vector."""
    vector = np.zeros(15)
    for class_name, prob in class_probs.items():
        if class_name in CLASS_TO_IDX:
            vector[CLASS_TO_IDX[class_name]] = prob
    # Normalize
    if vector.sum() > 0:
        vector = vector / vector.sum()
    return vector


def format_class_distribution(probs: np.ndarray, top_k: int = 3) -> str:
    """Format class distribution for legend display."""
    # Get top-k classes
    top_indices = np.argsort(probs)[::-1][:top_k]
    parts = []
    for idx in top_indices:
        if probs[idx] > 0.01:  # Only show if > 1%
            parts.append(f"{CLASS_MAPPING[idx]}: {probs[idx]:.2f}")
    return "\n".join(parts)


def load_image(image_path: str) -> Tuple[Image.Image, np.ndarray]:
    """Load image and return both PIL and numpy versions."""
    pil_img = Image.open(image_path).convert("RGB")
    np_img = np.array(pil_img)
    return pil_img, np_img


# ============================================================================
# HARD LABEL PROCESSING
# ============================================================================

def parse_hard_label_gt(labels: List[Dict], img_width: int, img_height: int) -> Dict:
    """
    Parse hard label ground truth from JSON.
    
    Returns:
        Dict with 'boxes' (list of [x1,y1,x2,y2]), 'labels' (list of class indices)
    """
    boxes = []
    class_labels = []
    
    for label in labels:
        # Get bbox in normalized format [x, y, w, h]
        bbox = label['bbox']
        norm_x, norm_y, norm_w, norm_h = bbox
        
        # Skip full-image placeholder boxes
        if norm_w >= 0.99 and norm_h >= 0.99:
            continue
        
        # Convert to pixel coordinates
        x = norm_x * img_width
        y = norm_y * img_height
        w = norm_w * img_width
        h = norm_h * img_height
        
        # Convert to [x1, y1, x2, y2]
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Get class label
        if 'renormalized_class_probs' in label:
            probs = label['renormalized_class_probs']
            best_class = max(probs.items(), key=lambda x: x[1])[0]
            class_idx = CLASS_TO_IDX.get(best_class, 0)
        elif 'class' in label:
            class_idx = CLASS_TO_IDX.get(label['class'], 0)
        else:
            class_idx = 0
        
        if class_idx == 0:  # Skip No Finding
            continue
            
        boxes.append([x1, y1, x2, y2])
        class_labels.append(class_idx)
    
    return {
        'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
        'labels': np.array(class_labels) if class_labels else np.zeros((0,), dtype=int)
    }


# ============================================================================
# 1D SOFT LABEL PROCESSING
# ============================================================================

def parse_1d_soft_label_gt(labels: List[Dict], img_width: int, img_height: int) -> Dict:
    """
    Parse 1D soft label ground truth from JSON.
    
    Returns:
        Dict with 'boxes', 'stds', 'class_probs', 'labels'
    """
    boxes = []
    stds = []
    class_probs_list = []
    class_labels = []
    
    for label in labels:
        # Extract spatial information
        mean_box = label['spatial']['mean']  # [x, y, w, h] normalized
        std_box = label['spatial']['std']    # [σ_x, σ_y, σ_w, σ_h] normalized
        
        # Skip full-image placeholder boxes
        if mean_box[2] >= 0.99 and mean_box[3] >= 0.99:
            continue
        
        # Convert mean to pixel coordinates
        x = mean_box[0] * img_width
        y = mean_box[1] * img_height
        w = mean_box[2] * img_width
        h = mean_box[3] * img_height
        
        # Box in [x1, y1, x2, y2] format
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Convert std to pixel space
        sigma_x = std_box[0] * img_width
        sigma_y = std_box[1] * img_height
        sigma_w = std_box[2] * img_width
        sigma_h = std_box[3] * img_height
        
        # Get class probabilities
        probs = class_probs_to_vector(label['class_probs'])
        
        # Get dominant class (excluding background for label)
        disease_probs = probs.copy()
        disease_probs[0] = 0  # Exclude background
        dominant_class = np.argmax(disease_probs) if disease_probs.max() > 0 else 1
        
        boxes.append([x1, y1, x2, y2])
        stds.append([sigma_x, sigma_y, sigma_w, sigma_h])
        class_probs_list.append(probs)
        class_labels.append(dominant_class)
    
    return {
        'boxes': np.array(boxes) if boxes else np.zeros((0, 4)),
        'stds': np.array(stds) if stds else np.zeros((0, 4)),
        'class_probs': np.array(class_probs_list) if class_probs_list else np.zeros((0, 15)),
        'labels': np.array(class_labels) if class_labels else np.zeros((0,), dtype=int)
    }


def compute_vertex_uncertainties(box: np.ndarray, std: np.ndarray) -> List[Tuple[float, float, float, float]]:
    """
    Compute uncertainty at each vertex of the bounding box.
    
    Args:
        box: [x1, y1, x2, y2] in pixels
        std: [σ_x, σ_y, σ_w, σ_h] in pixels
    
    Returns:
        List of (cx, cy, σ_x, σ_y) for each vertex: [top-left, top-right, bottom-right, bottom-left]
    """
    x1, y1, x2, y2 = box
    sigma_x, sigma_y, sigma_w, sigma_h = std
    
    # Propagate uncertainties to corners
    # Top-left (x1, y1): directly uses σ_x, σ_y
    v1 = (x1, y1, sigma_x, sigma_y)
    
    # Top-right (x2, y1): x2 = x1 + w, so σ_x2 = sqrt(σ_x² + σ_w²)
    v2 = (x2, y1, np.sqrt(sigma_x**2 + sigma_w**2), sigma_y)
    
    # Bottom-right (x2, y2): both corners have propagated uncertainty
    v3 = (x2, y2, np.sqrt(sigma_x**2 + sigma_w**2), np.sqrt(sigma_y**2 + sigma_h**2))
    
    # Bottom-left (x1, y2): y2 = y1 + h
    v4 = (x1, y2, sigma_x, np.sqrt(sigma_y**2 + sigma_h**2))
    
    return [v1, v2, v3, v4]

import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np

def compute_corner_confidence_bounds(box, delta_stds, confidence=0.95):
    """
    Compute asymmetric confidence bounds for each corner.
    
    Args:
        box: [x1, y1, x2, y2] predicted box in pixels
        delta_stds: [σ_dx, σ_dy, σ_dw, σ_dh] in delta space
        confidence: confidence level
    
    Returns:
        Dict with corner bounds: {corner_name: (x, y, x_lower, x_upper, y_lower, y_upper)}
    """
    from scipy import stats
    z = stats.norm.ppf((1 + confidence) / 2)
    
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    sigma_dx, sigma_dy, sigma_dw, sigma_dh = delta_stds
    
    # Clamp delta stds to prevent numerical issues
    sigma_dw = min(sigma_dw, 2.0)  # Prevents exp(3.92) = 50x expansion
    sigma_dh = min(sigma_dh, 2.0)
    sigma_dx = min(sigma_dx, 2.0)
    sigma_dy = min(sigma_dy, 2.0)
    # Center position uncertainty (symmetric, Gaussian)
    sigma_cx_px = sigma_dx * w  # Approximation using pred width
    sigma_cy_px = sigma_dy * h
    
    # Width/height uncertainty (asymmetric, log-normal)
    # CI for width: [w * exp(-z*σ_dw), w * exp(+z*σ_dw)]
    w_lower = w * np.exp(-z * sigma_dw)
    w_upper = w * np.exp(+z * sigma_dw)
    h_lower = h * np.exp(-z * sigma_dh)
    h_upper = h * np.exp(+z * sigma_dh)
    
    # Corner uncertainties
    # Top-Left (x1, y1) = (cx - w/2, cy - h/2)
    # x1_bounds depend on cx uncertainty AND w uncertainty
    corners = {}
    
    # TL corner: x1 = cx - w/2
    corners['TL'] = {
        'x': x1, 'y': y1,
        'x_lower': (cx - z*sigma_cx_px) - w_upper/2,  # cx goes left, w gets bigger
        'x_upper': (cx + z*sigma_cx_px) - w_lower/2,  # cx goes right, w gets smaller
        'y_lower': (cy - z*sigma_cy_px) - h_upper/2,
        'y_upper': (cy + z*sigma_cy_px) - h_lower/2,
    }
    
    # TR corner: x2 = cx + w/2
    corners['TR'] = {
        'x': x2, 'y': y1,
        'x_lower': (cx - z*sigma_cx_px) + w_lower/2,
        'x_upper': (cx + z*sigma_cx_px) + w_upper/2,
        'y_lower': (cy - z*sigma_cy_px) - h_upper/2,
        'y_upper': (cy + z*sigma_cy_px) - h_lower/2,
    }
    
    # BR corner: (cx + w/2, cy + h/2)
    corners['BR'] = {
        'x': x2, 'y': y2,
        'x_lower': (cx - z*sigma_cx_px) + w_lower/2,
        'x_upper': (cx + z*sigma_cx_px) + w_upper/2,
        'y_lower': (cy - z*sigma_cy_px) + h_lower/2,
        'y_upper': (cy + z*sigma_cy_px) + h_upper/2,
    }
    
    # BL corner: (cx - w/2, cy + h/2)
    corners['BL'] = {
        'x': x1, 'y': y2,
        'x_lower': (cx - z*sigma_cx_px) - w_upper/2,
        'x_upper': (cx + z*sigma_cx_px) - w_lower/2,
        'y_lower': (cy - z*sigma_cy_px) + h_lower/2,
        'y_upper': (cy + z*sigma_cy_px) + h_upper/2,
    }
    
    return corners

def draw_asymmetric_confidence_region(ax, box, delta_stds, color='blue', 
                                       alpha=0.2, confidence=0.95):
    """
    Draw the asymmetric confidence region as inner/outer polygons.
    
    This shows the min/max bounding box based on log-normal width/height
    and Gaussian center position.
    """
    corners = compute_corner_confidence_bounds(box, delta_stds, confidence)
    
    # Outer bound (worst case expansion)
    outer_polygon = [
        (corners['TL']['x_lower'], corners['TL']['y_lower']),
        (corners['TR']['x_upper'], corners['TR']['y_lower']),
        (corners['BR']['x_upper'], corners['BR']['y_upper']),
        (corners['BL']['x_lower'], corners['BL']['y_upper']),
    ]
    
    # Inner bound (worst case shrinkage)
    inner_polygon = [
        (corners['TL']['x_upper'], corners['TL']['y_upper']),
        (corners['TR']['x_lower'], corners['TR']['y_upper']),
        (corners['BR']['x_lower'], corners['BR']['y_lower']),
        (corners['BL']['x_upper'], corners['BL']['y_lower']),
    ]
    
    # Draw outer region
    outer_patch = Polygon(outer_polygon, closed=True, 
                          facecolor=color, edgecolor=color,
                          alpha=alpha, linewidth=1, linestyle='--')
    ax.add_patch(outer_patch)
    
    # Draw inner region (optional - shows the "minimum" box)
    inner_patch = Polygon(inner_polygon, closed=True,
                          facecolor='white', edgecolor=color,
                          alpha=0.3, linewidth=1, linestyle=':')
    ax.add_patch(inner_patch)
    
    return outer_polygon, inner_polygon

def sample_boxes_monte_carlo(box, delta_stds, w_prop=None, h_prop=None, 
                              n_samples=1000):
    """
    Sample boxes from the predicted distribution using Monte Carlo.
    
    This properly handles the correlation between corners through
    the shared center and size parameters.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Use predicted dimensions as proxy for proposal (approximation)
    if w_prop is None:
        w_prop = w
    if h_prop is None:
        h_prop = h
    
    sigma_dx, sigma_dy, sigma_dw, sigma_dh = delta_stds
    
    # Sample center offsets (Gaussian)
    # The predicted dx is 0 in the output space (already decoded)
    # So we sample around 0 with the predicted std
    d_cx = np.random.normal(0, sigma_dx * w_prop, n_samples)
    d_cy = np.random.normal(0, sigma_dy * h_prop, n_samples)
    
    # Sample width/height ratios (log-normal)
    # The predicted dw is 0 (we're at the decoded prediction)
    # Sample perturbations around 0 in log-space
    dw_samples = np.random.normal(0, sigma_dw, n_samples)
    dh_samples = np.random.normal(0, sigma_dh, n_samples)
    
    # Convert to actual widths/heights
    w_samples = w * np.exp(dw_samples)
    h_samples = h * np.exp(dh_samples)
    
    # Compute sampled centers
    cx_samples = cx + d_cx
    cy_samples = cy + d_cy
    
    # Compute corner positions
    sampled_boxes = np.stack([
        cx_samples - w_samples/2,  # x1
        cy_samples - h_samples/2,  # y1
        cx_samples + w_samples/2,  # x2
        cy_samples + h_samples/2,  # y2
    ], axis=1)
    
    return sampled_boxes


def draw_confidence_contour(ax, box, delta_stds, color='blue', 
                            n_samples=2000, levels=[0.5, 0.9, 0.95]):
    """
    Draw confidence contours for each corner using Monte Carlo samples.
    """
    from scipy import stats
    
    sampled_boxes = sample_boxes_monte_carlo(box, delta_stds, n_samples=n_samples)
    
    corner_names = ['TL', 'TR', 'BR', 'BL']
    corner_indices = [(0, 1), (2, 1), (2, 3), (0, 3)]  # (x_idx, y_idx) for each corner
    
    for name, (xi, yi) in zip(corner_names, corner_indices):
        x_samples = sampled_boxes[:, xi]
        y_samples = sampled_boxes[:, yi]
        
        # Compute 2D KDE
        try:
            kde = stats.gaussian_kde(np.vstack([x_samples, y_samples]))
            
            # Create grid for contour
            x_min, x_max = x_samples.min() - 10, x_samples.max() + 10
            y_min, y_max = y_samples.min() - 10, y_samples.max() + 10
            xx, yy = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Evaluate KDE
            zz = kde(positions).reshape(xx.shape)
            
            # Draw contours
            ax.contour(xx, yy, zz, levels=3, colors=color, alpha=0.5, linewidths=0.5)
            ax.contourf(xx, yy, zz, levels=3, colors=[color], alpha=0.1)
        except:
            # Fallback if KDE fails
            pass

# ============================================================================
# VISUALIZATION - ELLIPSE METHOD
# ============================================================================

def draw_uncertainty_ellipse(ax, cx: float, cy: float, sigma_x: float, sigma_y: float, 
                              color: str = 'red', alpha: float = 0.3, scale: float = 2.0):
    """
    Draw uncertainty ellipse at a point.
    
    Args:
        ax: matplotlib axis
        cx, cy: center coordinates
        sigma_x, sigma_y: standard deviations
        color: ellipse color
        alpha: transparency
        scale: scale factor (2.0 = 95% confidence interval)
    """
    # Minimum size for visibility
    min_size = 5
    width = max(sigma_x * scale * 2, min_size)
    height = max(sigma_y * scale * 2, min_size)
    
    ellipse = Ellipse(
        (cx, cy), 
        width=width, 
        height=height,
        angle=0,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1
    )
    ax.add_patch(ellipse)


def plot_hard_label_comparison(image: np.ndarray, gt_data: Dict, pred_data: Dict, 
                                save_path: str, image_id: str):
    """
    Plot hard label: GT (red) on left, Predictions (blue) on right.
    Shows vertex coordinates and class names.
    """
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle(f"Hard Label Inference - Image: {image_id}", fontsize=16)
    
    gt_boxes = gt_data['boxes']
    gt_labels = gt_data['labels']
    pred_boxes = pred_data.get('boxes', np.zeros((0, 4)))
    pred_labels = pred_data.get('labels', np.zeros((0,), dtype=int))
    pred_scores = pred_data.get('scores', np.zeros((0,)))
    
    # --- Left: Ground Truth (Red) ---
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth (Red)", fontsize=14)
    axes[0].axis('off')
    
    for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Draw box
        rect = patches.Rectangle((x1, y1), w, h, 
                                   linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Add vertex coordinates
        axes[0].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'ro', markersize=4)
        axes[0].text(x1, y1 - 5, f"({x1:.0f},{y1:.0f})", fontsize=7, color='red', 
                     ha='left', va='bottom')
        axes[0].text(x2, y1 - 5, f"({x2:.0f},{y1:.0f})", fontsize=7, color='red',
                     ha='right', va='bottom')
        axes[0].text(x2, y2 + 5, f"({x2:.0f},{y2:.0f})", fontsize=7, color='red',
                     ha='right', va='top')
        axes[0].text(x1, y2 + 5, f"({x1:.0f},{y2:.0f})", fontsize=7, color='red',
                     ha='left', va='top')
        
        # Class name
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        axes[0].text(x1, y1 - 20, class_name, fontsize=10, color='white', 
                     backgroundcolor='red', fontweight='bold')
    
    # --- Right: Predictions (Blue) ---
    axes[1].imshow(image)
    axes[1].set_title("Predictions (Blue)", fontsize=14)
    axes[1].axis('off')
    
    for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Draw box
        rect = patches.Rectangle((x1, y1), w, h,
                                   linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        axes[1].add_patch(rect)
        
        # Add vertex coordinates
        axes[1].plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'bo', markersize=4)
        axes[1].text(x1, y1 - 5, f"({x1:.0f},{y1:.0f})", fontsize=7, color='blue',
                     ha='left', va='bottom')
        axes[1].text(x2, y1 - 5, f"({x2:.0f},{y1:.0f})", fontsize=7, color='blue',
                     ha='right', va='bottom')
        axes[1].text(x2, y2 + 5, f"({x2:.0f},{y2:.0f})", fontsize=7, color='blue',
                     ha='right', va='top')
        axes[1].text(x1, y2 + 5, f"({x1:.0f},{y2:.0f})", fontsize=7, color='blue',
                     ha='left', va='top')
        
        # Class name with score
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        axes[1].text(x1, y1 - 20, f"{class_name} ({score:.2f})", fontsize=10, 
                     color='white', backgroundcolor='blue', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved hard label visualization to: {save_path}")


def plot_1d_soft_label_ellipse(image: np.ndarray, gt_data: Dict, pred_data: Dict,
                                save_path: str, image_id: str):
    """
    Plot 1D soft label with uncertainty ellipses at vertices.
    
    Left: GT with mean bbox, vertex ellipses, std array and class distribution
    Right: Predictions with same information
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    fig.suptitle(f"1D Soft Label Inference (Ellipse Method) - Image: {image_id}", fontsize=16)
    
    gt_boxes = gt_data['boxes']
    gt_stds = gt_data['stds']
    gt_class_probs = gt_data['class_probs']
    gt_labels = gt_data['labels']
    
    pred_boxes = pred_data.get('boxes', np.zeros((0, 4)))
    pred_stds = pred_data.get('bbox_stds', np.zeros((0, 4)))
    pred_class_probs = pred_data.get('class_probs', np.zeros((0, 15)))
    pred_labels = pred_data.get('labels', np.zeros((0,), dtype=int))
    pred_scores = pred_data.get('scores', np.zeros((0,)))
    
    # Get image dimensions for std conversion
    img_h, img_w = image.shape[:2]
    
    # --- Left: Ground Truth ---
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth", fontsize=14)
    axes[0].axis('off')
    
    legend_entries = []
    
    for i, (box, std, probs, label) in enumerate(zip(gt_boxes, gt_stds, gt_class_probs, gt_labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Draw mean bbox
        rect = patches.Rectangle((x1, y1), w, h, 
                                   linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Draw uncertainty ellipses at vertices
        vertices = compute_vertex_uncertainties(box, std)
        vertex_names = ['TL', 'TR', 'BR', 'BL']
        for (vx, vy, sx, sy), vname in zip(vertices, vertex_names):
            draw_uncertainty_ellipse(axes[0], vx, vy, sx, sy, color='red', alpha=0.25)
            axes[0].plot(vx, vy, 'ro', markersize=3)
        
        # Class name and distribution
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        dist_str = format_class_distribution(probs)
        std_str = f"σ=[{std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}, {std[3]:.1f}]"
        
        axes[0].text(x1, y1 - 25, f"{class_name}\n{std_str}", fontsize=8, 
                     color='white', backgroundcolor='red', fontweight='bold')
        
        # Add to legend
        legend_entries.append(f"Box {i+1}: {class_name}\n{dist_str}")
    
    # Add legend for class distributions
    if legend_entries:
        legend_text = "\n\n".join(legend_entries)
        axes[0].text(0.02, 0.98, f"Class Distributions:\n{legend_text}", 
                     transform=axes[0].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- Right: Predictions ---
    axes[1].imshow(image)
    axes[1].set_title("Predictions", fontsize=14)
    axes[1].axis('off')
    
    legend_entries = []
    
    for i, (box, std, probs, label, score) in enumerate(
        zip(pred_boxes, pred_stds, pred_class_probs, pred_labels, pred_scores)):
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Convert predicted std from delta space to pixel space (approximate)
        # bbox_stds is [σ_dx, σ_dy, σ_dw, σ_dh] in delta space
        # Approximate conversion: multiply by box dimensions
        sigma_cx_px = std[0] * w
        sigma_cy_px = std[1] * h
        sigma_w_px = std[2] * w
        sigma_h_px = std[3] * h
        pixel_std = np.array([sigma_cx_px, sigma_cy_px, sigma_w_px, sigma_h_px])
        
        # Draw mean bbox
        rect = patches.Rectangle((x1, y1), w, h,
                                   linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        axes[1].add_patch(rect)
        
        # Draw uncertainty ellipses at vertices
        # For predictions, we use center-based std, so propagate to corners
        vertices = [
            (x1, y1, sigma_cx_px, sigma_cy_px),  # TL
            (x2, y1, np.sqrt(sigma_cx_px**2 + sigma_w_px**2), sigma_cy_px),  # TR
            (x2, y2, np.sqrt(sigma_cx_px**2 + sigma_w_px**2), np.sqrt(sigma_cy_px**2 + sigma_h_px**2)),  # BR
            (x1, y2, sigma_cx_px, np.sqrt(sigma_cy_px**2 + sigma_h_px**2)),  # BL
        ]
        
        for (vx, vy, sx, sy) in vertices:
            draw_uncertainty_ellipse(axes[1], vx, vy, sx, sy, color='blue', alpha=0.25)
            axes[1].plot(vx, vy, 'bo', markersize=3)
        
        # Class name with score and std
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        std_str = f"σ=[{pixel_std[0]:.1f}, {pixel_std[1]:.1f}, {pixel_std[2]:.1f}, {pixel_std[3]:.1f}]"
        
        axes[1].text(x1, y1 - 25, f"{class_name} ({score:.2f})\n{std_str}", fontsize=8,
                     color='white', backgroundcolor='blue', fontweight='bold')
        
        # Add to legend
        dist_str = format_class_distribution(probs)
        legend_entries.append(f"Box {i+1}: {class_name} ({score:.2f})\n{dist_str}")
    
    # Add legend for class distributions
    if legend_entries:
        legend_text = "\n\n".join(legend_entries)
        axes[1].text(0.02, 0.98, f"Class Distributions:\n{legend_text}",
                     transform=axes[1].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 1D soft label ellipse visualization to: {save_path}")


def plot_1d_soft_label_asymmetric(image: np.ndarray, gt_data: Dict, pred_data: Dict,
                                   save_path: str, image_id: str,
                                   method: str = 'polygon'):
    """
    Plot 1D soft label with proper asymmetric confidence regions.
    
    Uses log-normal statistics for width/height uncertainty, resulting in
    asymmetric confidence bounds that are more scientifically accurate.
    
    Args:
        image: numpy array of the image
        gt_data: dict with 'boxes', 'stds', 'class_probs', 'labels'
        pred_data: dict with 'boxes', 'bbox_stds', 'class_probs', 'labels', 'scores'
        save_path: path to save the visualization
        image_id: image identifier for title
        method: 'polygon' for analytical bounds, 'monte_carlo' for sampled contours
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    fig.suptitle(f"1D Soft Label Inference (Asymmetric {method.title()} Method) - Image: {image_id}", fontsize=16)
    
    # Extract GT data
    gt_boxes = gt_data['boxes']
    gt_stds = gt_data['stds']
    gt_class_probs = gt_data['class_probs']
    gt_labels = gt_data['labels']
    
    # Extract prediction data
    pred_boxes = pred_data.get('boxes', np.zeros((0, 4)))
    pred_stds = pred_data.get('bbox_stds', np.zeros((0, 4)))
    pred_class_probs = pred_data.get('class_probs', np.zeros((0, 15)))
    pred_labels = pred_data.get('labels', np.zeros((0,), dtype=int))
    pred_scores = pred_data.get('scores', np.zeros((0,)))
    
    # Get image dimensions
    img_h, img_w = image.shape[:2]
    
    # =========================================================================
    # LEFT: Ground Truth with Asymmetric Confidence Regions
    # =========================================================================
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth (Asymmetric CI)", fontsize=14)
    axes[0].axis('off')
    
    legend_entries_gt = []
    
    for i, (box, std, probs, label) in enumerate(zip(gt_boxes, gt_stds, gt_class_probs, gt_labels)):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # For GT, stds are already in pixel space
        # Draw asymmetric confidence region
        if method == 'polygon':
            draw_asymmetric_confidence_region(axes[0], box, std, color='red', alpha=0.15)
        elif method == 'monte_carlo':
            draw_confidence_contour(axes[0], box, std, color='red')
        
        # Draw mean bbox on top
        rect = patches.Rectangle((x1, y1), w, h, 
                                   linewidth=2, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Mark corners
        axes[0].plot([x1, x2, x2, x1], [y1, y1, y2, y2], 'ro', markersize=4)
        
        # Class name and asymmetric CI info
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        
        # Compute asymmetric bounds for display (log-normal for w, h)
        # GT std is in pixel space: [σ_x, σ_y, σ_w, σ_h]
        # For GT, we treat these as direct pixel uncertainties
        w_lower = max(1, w - 1.96 * std[2])
        w_upper = w + 1.96 * std[2]
        h_lower = max(1, h - 1.96 * std[3])
        h_upper = h + 1.96 * std[3]
        
        std_str = f"σ=[{std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}, {std[3]:.1f}]"
        axes[0].text(x1, y1 - 25, f"{class_name}\n{std_str}", fontsize=8,
                     color='white', backgroundcolor='red', fontweight='bold')
        
        # Asymmetric CI text below box
        ci_text = f"w: {w:.0f} [{w_lower:.0f}, {w_upper:.0f}]\nh: {h:.0f} [{h_lower:.0f}, {h_upper:.0f}]"
        axes[0].text(x1, y2 + 10, ci_text, fontsize=7, color='darkred')
        
        # Legend entry
        dist_str = format_class_distribution(probs)
        legend_entries_gt.append(f"Box {i+1}: {class_name}\n{dist_str}")
    
    # Add legend for GT
    if legend_entries_gt:
        legend_text = "\n\n".join(legend_entries_gt)
        axes[0].text(0.02, 0.98, f"GT Class Distributions:\n{legend_text}",
                     transform=axes[0].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # =========================================================================
    # RIGHT: Predictions with Asymmetric Confidence Regions
    # =========================================================================
    axes[1].imshow(image)
    axes[1].set_title("Predictions (Asymmetric CI - Log-Normal)", fontsize=14)
    axes[1].axis('off')
    
    legend_entries_pred = []
    
    for i, (box, std, probs, label, score) in enumerate(
        zip(pred_boxes, pred_stds, pred_class_probs, pred_labels, pred_scores)):
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        
        # Convert predicted std from delta space to pixel space
        # bbox_stds is [σ_dx, σ_dy, σ_dw, σ_dh] in delta space
        sigma_cx_px = std[0] * w  # Approximate using pred box width
        sigma_cy_px = std[1] * h
        # For dw, dh we keep in delta space for log-normal computation
        sigma_dw = std[2]
        sigma_dh = std[3]
        
        # Create pixel-space std for visualization functions
        # Note: for width/height, we use the first-order approximation
        sigma_w_px = sigma_dw * w
        sigma_h_px = sigma_dh * h
        pixel_std = np.array([sigma_cx_px, sigma_cy_px, sigma_w_px, sigma_h_px])
        
        # Draw asymmetric confidence region
        if method == 'polygon':
            # Use delta-space stds for proper log-normal handling
            draw_asymmetric_confidence_region(axes[1], box, std, color='blue', alpha=0.15)
        elif method == 'monte_carlo':
            draw_confidence_contour(axes[1], box, std, color='blue')
        
        # Draw mean bbox on top
        rect = patches.Rectangle((x1, y1), w, h,
                                   linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
        axes[1].add_patch(rect)
        
        # Mark corners
        axes[1].plot([x1, x2, x2, x1], [y1, y1, y2, y2], 'bo', markersize=4)
        
        # Class name with score
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        
        # Compute asymmetric bounds using LOG-NORMAL for width/height
        # 95% CI: [w * exp(-1.96*σ_dw), w * exp(+1.96*σ_dw)]
        z = 1.96  # 95% confidence
        w_lower = w * np.exp(-z * sigma_dw)
        w_upper = w * np.exp(+z * sigma_dw)
        h_lower = h * np.exp(-z * sigma_dh)
        h_upper = h * np.exp(+z * sigma_dh)
        
        # For center position (Gaussian): symmetric
        cx_lower = (x1 + x2) / 2 - z * sigma_cx_px
        cx_upper = (x1 + x2) / 2 + z * sigma_cx_px
        
        std_str = f"δ-std=[{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}, {std[3]:.3f}]"
        axes[1].text(x1, y1 - 30, f"{class_name} ({score:.2f})\n{std_str}", fontsize=8,
                     color='white', backgroundcolor='blue', fontweight='bold')
        
        # Asymmetric CI text below box - shows the asymmetry clearly
        ci_text = (f"w: {w:.0f} [{w_lower:.0f}, {w_upper:.0f}] "
                   f"(−{w - w_lower:.0f}/+{w_upper - w:.0f})\n"
                   f"h: {h:.0f} [{h_lower:.0f}, {h_upper:.0f}] "
                   f"(−{h - h_lower:.0f}/+{h_upper - h:.0f})")
        axes[1].text(x1, y2 + 10, ci_text, fontsize=7, color='darkblue')
        
        # Legend entry
        dist_str = format_class_distribution(probs)
        legend_entries_pred.append(f"Box {i+1}: {class_name} ({score:.2f})\n{dist_str}")
    
    # Add legend for predictions
    if legend_entries_pred:
        legend_text = "\n\n".join(legend_entries_pred)
        axes[1].text(0.02, 0.98, f"Pred Class Distributions:\n{legend_text}",
                     transform=axes[1].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Add note about log-normal interpretation
    fig.text(0.5, 0.02, 
             "Note: Width/Height use log-normal CI (asymmetric). "
             "Shaded regions show 95% confidence bounds.",
             ha='center', fontsize=10, style='italic')
    try:
        plt.tight_layout()
    except Exception as e:
        pass
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 1D soft label asymmetric visualization to: {save_path}")

# ============================================================================
# VISUALIZATION - HEATMAP METHOD
# ============================================================================

def create_uncertainty_heatmap(image_shape: Tuple[int, int], boxes: np.ndarray, 
                                stds: np.ndarray) -> np.ndarray:
    """
    Create heatmap showing bbox uncertainty as Gaussian density.
    
    The center line of the bbox is the 'hot' zone, and it gets cooler
    based on the uncertainty (std).
    
    Args:
        image_shape: (H, W) of the image
        boxes: [N, 4] array of boxes in [x1, y1, x2, y2] format
        stds: [N, 4] array of stds in [σ_x, σ_y, σ_w, σ_h] format (in pixels)
    
    Returns:
        heatmap: [H, W] array with values in [0, 1]
    """
    H, W = image_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    if len(boxes) == 0:
        return heatmap
    
    # Create coordinate grids
    y_grid, x_grid = np.ogrid[:H, :W]
    y_grid = y_grid.astype(np.float32)
    x_grid = x_grid.astype(np.float32)
    
    for box, std in zip(boxes, stds):
        x1, y1, x2, y2 = box
        sigma_x, sigma_y, sigma_w, sigma_h = std
        
        # Box center
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Use the average of positional and size uncertainties
        sigma_total_x = np.sqrt(sigma_x**2 + (sigma_w/2)**2)
        sigma_total_y = np.sqrt(sigma_y**2 + (sigma_h/2)**2)
        
        # Minimum sigma for visibility
        sigma_total_x = max(sigma_total_x, 10)
        sigma_total_y = max(sigma_total_y, 10)
        
        # Create 2D Gaussian centered at box center
        # The Gaussian represents uncertainty - wider sigma = more spread = more uncertainty
        gaussian = np.exp(-((x_grid - cx)**2 / (2 * sigma_total_x**2) + 
                           (y_grid - cy)**2 / (2 * sigma_total_y**2)))
        
        # Take maximum with existing heatmap
        heatmap = np.maximum(heatmap, gaussian)
    
    return heatmap


def create_box_edge_heatmap(image_shape: Tuple[int, int], boxes: np.ndarray,
                             stds: np.ndarray) -> np.ndarray:
    """
    Create heatmap where the box edges are 'hot' and uncertainty spreads outward.
    
    Args:
        image_shape: (H, W) of the image
        boxes: [N, 4] array of boxes in [x1, y1, x2, y2] format
        stds: [N, 4] array of stds in [σ_x, σ_y, σ_w, σ_h] format (in pixels)
    
    Returns:
        heatmap: [H, W] array with values in [0, 1]
    """
    H, W = image_shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    
    if len(boxes) == 0:
        return heatmap
    
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    for box, std in zip(boxes, stds):
        x1, y1, x2, y2 = box
        sigma_x, sigma_y, sigma_w, sigma_h = std
        
        # Minimum sigma for stability
        sigma_x = max(sigma_x, 5)
        sigma_y = max(sigma_y, 5)
        sigma_w = max(sigma_w, 5)
        sigma_h = max(sigma_h, 5)
        
        # Distance to each edge
        dist_left = np.abs(x_coords - x1)
        dist_right = np.abs(x_coords - x2)
        dist_top = np.abs(y_coords - y1)
        dist_bottom = np.abs(y_coords - y2)
        
        # Gaussian falloff from edges
        # Left edge uncertainty
        left_heat = np.exp(-(dist_left**2) / (2 * sigma_x**2))
        left_heat *= (y_coords >= y1 - sigma_y) & (y_coords <= y2 + sigma_h)
        
        # Right edge uncertainty  
        right_heat = np.exp(-(dist_right**2) / (2 * np.sqrt(sigma_x**2 + sigma_w**2)**2))
        right_heat *= (y_coords >= y1 - sigma_y) & (y_coords <= y2 + sigma_h)
        
        # Top edge uncertainty
        top_heat = np.exp(-(dist_top**2) / (2 * sigma_y**2))
        top_heat *= (x_coords >= x1 - sigma_x) & (x_coords <= x2 + sigma_w)
        
        # Bottom edge uncertainty
        bottom_heat = np.exp(-(dist_bottom**2) / (2 * np.sqrt(sigma_y**2 + sigma_h**2)**2))
        bottom_heat *= (x_coords >= x1 - sigma_x) & (x_coords <= x2 + sigma_w)
        
        # Combine edges
        box_heat = np.maximum.reduce([left_heat, right_heat, top_heat, bottom_heat])
        heatmap = np.maximum(heatmap, box_heat)
    
    return heatmap


def plot_1d_soft_label_heatmap(image: np.ndarray, gt_data: Dict, pred_data: Dict,
                                save_path: str, image_id: str):
    """
    Plot 1D soft label with heatmap overlay showing uncertainty.
    
    The mean bbox edges are 'hot' and the uncertainty spreads the heat outward.
    """
    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    fig.suptitle(f"1D Soft Label Inference (Heatmap Method) - Image: {image_id}", fontsize=16)
    
    gt_boxes = gt_data['boxes']
    gt_stds = gt_data['stds']
    gt_class_probs = gt_data['class_probs']
    gt_labels = gt_data['labels']
    
    pred_boxes = pred_data.get('boxes', np.zeros((0, 4)))
    pred_stds = pred_data.get('bbox_stds', np.zeros((0, 4)))
    pred_class_probs = pred_data.get('class_probs', np.zeros((0, 15)))
    pred_labels = pred_data.get('labels', np.zeros((0,), dtype=int))
    pred_scores = pred_data.get('scores', np.zeros((0,)))
    
    img_h, img_w = image.shape[:2]
    
    # Create custom colormap (hot = red/yellow, cold = blue/transparent)
    colors = ['#00000000', '#0000FF40', '#00FFFF80', '#FFFF00C0', '#FF0000FF']
    cmap = LinearSegmentedColormap.from_list('uncertainty', colors)
    
    # --- Left: Ground Truth ---
    axes[0].imshow(image)
    axes[0].set_title("Ground Truth - Uncertainty Heatmap", fontsize=14)
    axes[0].axis('off')
    
    if len(gt_boxes) > 0:
        # Create heatmap
        gt_heatmap = create_box_edge_heatmap((img_h, img_w), gt_boxes, gt_stds)
        
        # Overlay heatmap
        axes[0].imshow(gt_heatmap, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        
        # Draw mean bboxes
        for i, (box, label, probs) in enumerate(zip(gt_boxes, gt_labels, gt_class_probs)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Draw mean bbox with solid line
            rect = patches.Rectangle((x1, y1), w, h,
                                       linewidth=3, edgecolor='white', facecolor='none')
            axes[0].add_patch(rect)
            
            # Class name
            class_name = CLASS_MAPPING.get(int(label), 'Unknown')
            axes[0].text(x1, y1 - 10, class_name, fontsize=10,
                         color='black', backgroundcolor='white', fontweight='bold')
    
    # Add legend for GT
    legend_entries = []
    for i, (box, std, probs, label) in enumerate(zip(gt_boxes, gt_stds, gt_class_probs, gt_labels)):
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        std_str = f"σ=[{std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}, {std[3]:.1f}]"
        dist_str = format_class_distribution(probs)
        legend_entries.append(f"Box {i+1}: {class_name}\n{std_str}\n{dist_str}")
    
    if legend_entries:
        legend_text = "\n\n".join(legend_entries)
        axes[0].text(0.02, 0.98, f"GT Details:\n{legend_text}",
                     transform=axes[0].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    # --- Right: Predictions ---
    axes[1].imshow(image)
    axes[1].set_title("Predictions - Uncertainty Heatmap", fontsize=14)
    axes[1].axis('off')
    
    if len(pred_boxes) > 0:
        # Convert predicted stds to pixel space
        pixel_stds = []
        for box, std in zip(pred_boxes, pred_stds):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            pixel_std = np.array([
                std[0] * w,
                std[1] * h, 
                std[2] * w,
                std[3] * h
            ])
            pixel_stds.append(pixel_std)
        pixel_stds = np.array(pixel_stds)
        
        # Create heatmap
        pred_heatmap = create_box_edge_heatmap((img_h, img_w), pred_boxes, pixel_stds)
        
        # Overlay heatmap
        axes[1].imshow(pred_heatmap, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        
        # Draw mean bboxes
        for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            rect = patches.Rectangle((x1, y1), w, h,
                                       linewidth=3, edgecolor='white', facecolor='none', linestyle='--')
            axes[1].add_patch(rect)
            
            class_name = CLASS_MAPPING.get(int(label), 'Unknown')
            axes[1].text(x1, y1 - 10, f"{class_name} ({score:.2f})", fontsize=10,
                         color='black', backgroundcolor='white', fontweight='bold')
    
    # Add legend for predictions
    legend_entries = []
    for i, (box, std, probs, label, score) in enumerate(
        zip(pred_boxes, pred_stds, pred_class_probs, pred_labels, pred_scores)):
        
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        pixel_std = [std[0]*w, std[1]*h, std[2]*w, std[3]*h]
        
        class_name = CLASS_MAPPING.get(int(label), 'Unknown')
        std_str = f"σ=[{pixel_std[0]:.1f}, {pixel_std[1]:.1f}, {pixel_std[2]:.1f}, {pixel_std[3]:.1f}]"
        dist_str = format_class_distribution(probs)
        legend_entries.append(f"Box {i+1}: {class_name} ({score:.2f})\n{std_str}\n{dist_str}")
    
    if legend_entries:
        legend_text = "\n\n".join(legend_entries)
        axes[1].text(0.02, 0.98, f"Pred Details:\n{legend_text}",
                     transform=axes[1].transAxes, fontsize=8, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.02, pad=0.05)
    cbar.set_label('Uncertainty (Low → High)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 1D soft label heatmap visualization to: {save_path}")


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def run_model_inference(model, image: Image.Image, device: str) -> Dict:
    """
    Run inference on a single image.
    
    Args:
        model: trained model
        image: PIL Image
        device: 'cuda' or 'cpu'
    
    Returns:
        Dict with predictions
    """
    model.eval()
    
    # Transform image to tensor
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Move to CPU and convert to numpy
    output = outputs[0]
    result = {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            result[k] = v.cpu().numpy()
        else:
            result[k] = v
    
    return result

def cross_class_nms(pred_data, iou_threshold=0.7):
    """
    Suppress overlapping boxes across classes, keeping the highest-scoring one per region.
    The built-in NMS is per-class, so two overlapping boxes with different class labels
    both survive. This merges them, keeping only the best-scoring detection per spatial region.
    """
    boxes = pred_data.get('boxes', np.zeros((0, 4)))
    scores = pred_data.get('scores', np.zeros((0,)))
    
    if len(boxes) == 0:
        return pred_data
    
    import torch as _torch
    keep = torchvision.ops.nms(
        _torch.tensor(boxes, dtype=_torch.float32),
        _torch.tensor(scores, dtype=_torch.float32),
        iou_threshold
    ).numpy()
    
    return {k: v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v
            for k, v in pred_data.items()}

# ============================================================================
# MAIN INFERENCE FUNCTION
# ============================================================================

def run_inference(
    hard_weights: Optional[str] = None,
    soft_1d_weights: Optional[str] = None,
    soft_2d_weights: Optional[str] = None,
    image_dir: str = "/media/klee_ro/public_datasets/vindr_cxr/train",
    soft_1d_json: str = "/home/haaroon/SoftLabel_Generation/1DGMM/1DGMM_softLabels_clustered_all_with_priors.json",
    hard_json: str = "/home/haaroon/SoftLabel_Generation/HardLabels/1DGMM_HardLabels_Majority.json",
    output_dir: str = "./inference_results",
    device: str = 'cuda',
    score_threshold: float = 0.3,
    viz_method: str = 'polygon', # 'ellipse', 'polygon', 'monte_carlo', 'heatmap'
    num_classes: int = 15,
    backbone: str = 'resnet50',
    seed: Optional[int] = None
):
    """
    Main inference function for all model types.
    
    Args:
        hard_weights: Path to hard label model weights (None to skip)
        soft_1d_weights: Path to 1D soft label model weights (None to skip)
        soft_2d_weights: Path to 2D soft label model weights (None to skip)
        image_dir: Directory containing training images
        soft_1d_json: Path to 1D soft label JSON file
        hard_json: Path to hard label JSON file
        output_dir: Directory to save visualizations
        device: 'cuda' or 'cpu'
        score_threshold: Minimum score for predictions
        viz_method: 'ellipse' or 'heatmap' for soft label visualization
        num_classes: Number of classes (including background)
        backbone: Model backbone name
        seed: Random seed for reproducibility
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if at least one model is provided
    if not any([hard_weights, soft_1d_weights, soft_2d_weights]):
        print("Error: At least one model weights path must be provided!")
        return
    
    # Select device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    print(f"Visualization method: {viz_method}")
    
    # ========================================================================
    # HARD LABEL INFERENCE
    # ========================================================================
    if hard_weights is not None:
        print("\n" + "="*60)
        print("HARD LABEL MODEL INFERENCE")
        print("="*60)
        
        # Load GT labels
        hard_labels = load_gt_labels(hard_json)
        
        # Get random image with actual GT boxes (retry up to 50 times)
        max_retries = 50
        for attempt in range(max_retries):
            image_id, image_path = get_random_image_with_annotations(image_dir, hard_labels)
            
            # Load image
            pil_img, np_img = load_image(image_path)
            img_w, img_h = pil_img.size
            
            # Parse GT
            image_labels = get_labels_for_image(hard_labels, image_id)
            gt_data = parse_hard_label_gt(image_labels, img_w, img_h)
            
            if len(gt_data['boxes']) > 0:
                print(f"Selected image: {image_id} (attempt {attempt + 1})")
                print(f"GT boxes: {len(gt_data['boxes'])}")
                break
            
            if attempt == max_retries - 1:
                print(f"Warning: Could not find image with GT boxes after {max_retries} attempts")
                print(f"Using last selected image: {image_id}")
                print(f"GT boxes: {len(gt_data['boxes'])}")
        
        # Load model
        print(f"Loading hard label model from: {hard_weights}")
        model = get_model_1d(num_classes=num_classes, backbone_name=backbone, 
                            pretrained=False, probabilistic=False)
        
        checkpoint = torch.load(hard_weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
        # Run inference
        pred_data = run_model_inference(model, pil_img, device)
        
        # Filter by score threshold
        if len(pred_data['scores']) > 0:
            keep = pred_data['scores'] > score_threshold
            pred_data = {k: v[keep] if isinstance(v, np.ndarray) else v 
                        for k, v in pred_data.items()}
        print(f"Predictions after filtering: {len(pred_data.get('boxes', []))}")
        
        # Plot
        save_path = os.path.join(output_dir, f"hard_label_{image_id}.png")
        plot_hard_label_comparison(np_img, gt_data, pred_data, save_path, image_id)
    
    # ========================================================================
    # 1D SOFT LABEL INFERENCE
    # ========================================================================
    if soft_1d_weights is not None:
        print("\n" + "="*60)
        print("1D SOFT LABEL MODEL INFERENCE")
        print("="*60)
        
        # Load GT labels
        soft_1d_labels = load_gt_labels(soft_1d_json)
        
        # Get random image with actual GT boxes (retry up to 50 times)
        max_retries = 50
        for attempt in range(max_retries):
            image_id, image_path = get_random_image_with_annotations(image_dir, soft_1d_labels)
            
            # Load image
            pil_img, np_img = load_image(image_path)
            img_w, img_h = pil_img.size
            
            # Parse GT
            image_labels = get_labels_for_image(soft_1d_labels, image_id)
            gt_data = parse_1d_soft_label_gt(image_labels, img_w, img_h)
            
            if len(gt_data['boxes']) > 0:
                print(f"Selected image: {image_id} (attempt {attempt + 1})")
                print(f"GT boxes: {len(gt_data['boxes'])}")
                break
            
            if attempt == max_retries - 1:
                print(f"Warning: Could not find image with GT boxes after {max_retries} attempts")
                print(f"Using last selected image: {image_id} (may have no GT boxes)")
                print(f"GT boxes: {len(gt_data['boxes'])}")
        
        # Load model
        print(f"Loading 1D soft label model from: {soft_1d_weights}")
        model = get_model_1d(num_classes=num_classes, backbone_name=backbone,
                            pretrained=False, probabilistic=True)
        
        checkpoint = torch.load(soft_1d_weights, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        
         # Run inference
        pred_data = run_model_inference(model, pil_img, device)
        
        # Step 1: Cross-class NMS (merge overlapping boxes regardless of class)
        pred_data = cross_class_nms(pred_data, iou_threshold=0.3)
        print(f"Predictions after cross-class NMS: {len(pred_data.get('boxes', []))}")
        
        # Step 2: Score threshold filter
        if len(pred_data.get('scores', [])) > 0:
            keep = pred_data['scores'] > score_threshold
            pred_data = {k: v[keep] if isinstance(v, np.ndarray) and len(v) == len(keep) else v
                        for k, v in pred_data.items()}
        print(f"Predictions after score filter: {len(pred_data.get('boxes', []))}")
        
        # Plot based on visualization method
        save_path = os.path.join(output_dir, f"soft_1d_{viz_method}_{image_id}.png")
        
        if viz_method == 'ellipse':
            plot_1d_soft_label_ellipse(np_img, gt_data, pred_data, save_path, image_id)
        elif viz_method == 'heatmap':
            plot_1d_soft_label_heatmap(np_img, gt_data, pred_data, save_path, image_id)
        elif viz_method == 'polygon':
            plot_1d_soft_label_asymmetric(np_img, gt_data, pred_data, save_path, image_id, method='polygon')
        elif viz_method == 'monte_carlo':
            plot_1d_soft_label_asymmetric(np_img, gt_data, pred_data, save_path, image_id, method='monte_carlo')
        else:
            print(f"Unknown visualization method: {viz_method}")
    
    # ========================================================================
    # 2D SOFT LABEL INFERENCE (placeholder for future)
    # ========================================================================
    if soft_2d_weights is not None:
        print("\n" + "="*60)
        print("2D SOFT LABEL MODEL INFERENCE")
        print("="*60)
        print("2D soft label inference not yet implemented.")
        # TODO: Implement 2D soft label inference
    
    print("\n" + "="*60)
    print("INFERENCE COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("="*60)


# ============================================================================
# ARGUMENT PARSER
# ============================================================================

hard_weights = '/home/haaroon/ObjectDetection_SoftLabel/src/dlcv/saved_models/vindrcxr_hardLabel_resnet50_30Epoch_28012026.pth'
soft_1d_weights = '/home/haaroon/ObjectDetection_SoftLabel/src/dlcv/saved_models/SoftLabel_1D_30Epoch_resnet50_lr0_0005_Mean+Var+Cls.pth'
# soft_2d_weights = '/home/haaroon/ObjectDetection_SoftLabel/src/dlcv/saved_models/SoftLabel_1D_30Epoch_resnet50_lr0_0002_04022026.pth'


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified inference visualization for object detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hard label model only
  python inference_visualization.py --hard_weights path/to/hard.pth
  
  # 1D soft label with ellipse visualization (simple symmetric)
  python inference_visualization.py --soft_1d_weights path/to/soft1d.pth --viz_method ellipse
  
  # 1D soft label with polygon visualization (asymmetric, log-normal - RECOMMENDED)
  python inference_visualization.py --soft_1d_weights path/to/soft1d.pth --viz_method polygon
  
  # 1D soft label with monte carlo visualization (most accurate)
  python inference_visualization.py --soft_1d_weights path/to/soft1d.pth --viz_method monte_carlo
  
  # 1D soft label with heatmap visualization
  python inference_visualization.py --soft_1d_weights path/to/soft1d.pth --viz_method heatmap
  
  # Both models
  python inference_visualization.py --hard_weights path/to/hard.pth --soft_1d_weights path/to/soft1d.pth
        """
    )
    
    # Model weights
    parser.add_argument('--hard_weights', type=str, default=None,
                        help='Path to hard label model weights (skip if None)')
    parser.add_argument('--soft_1d_weights', type=str, default=soft_1d_weights,
                        help='Path to 1D soft label model weights (skip if None)')
    parser.add_argument('--soft_2d_weights', type=str, default=None,
                        help='Path to 2D soft label model weights (skip if None)')
    
    # Data paths
    parser.add_argument('--image_dir', type=str, 
                        default='/media/klee_ro/public_datasets/vindr_cxr/train',
                        help='Directory containing training images')
    parser.add_argument('--soft_1d_json', type=str,
                        default='/home/haaroon/SoftLabel_Generation/1DGMM/1DGMM_softLabels_clustered_all_with_priors.json',
                        help='Path to 1D soft label JSON file')
    parser.add_argument('--hard_json', type=str,
                        default='/home/haaroon/SoftLabel_Generation/HardLabels/1DGMM_HardLabels_Majority.json',
                        help='Path to hard label JSON file')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                        help='Directory to save visualization results')
    
    # Visualization settings
    parser.add_argument('--viz_method', type=str, default='ellipse',
                        choices=['ellipse', 'heatmap', 'polygon', 'monte_carlo'],
                        help='Visualization method for soft labels: ellipse or heatmap or polygon or monte_carlo')
    
    # Model settings
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run inference on')
    parser.add_argument('--score_threshold', type=float, default=0.7,
                        help='Minimum score threshold for predictions (default: 0.7)')
    parser.add_argument('--num_classes', type=int, default=15,
                        help='Number of classes (including background)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'mobilenet'],
                        help='Model backbone architecture')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    args = parse_args()
    
    run_inference(
        hard_weights=args.hard_weights,
        soft_1d_weights=args.soft_1d_weights,
        soft_2d_weights=args.soft_2d_weights,
        image_dir=args.image_dir,
        soft_1d_json=args.soft_1d_json,
        hard_json=args.hard_json,
        output_dir=args.output_dir,
        device=args.device,
        score_threshold=args.score_threshold,
        viz_method=args.viz_method,
        num_classes=args.num_classes,
        backbone=args.backbone,
        seed=args.seed
    )
