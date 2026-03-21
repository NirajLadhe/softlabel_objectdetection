"""
Shared clustering utilities for prior calculation and GMM soft label generation.

This module contains common functions for loading data, clustering annotations,
and organizing clusters by class - used by both 1D and 2D prior calculators.
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from typing import Dict, List, Any, Tuple
from scipy.cluster.hierarchy import linkage, fcluster


# ──────────────────────────────────────────────────────────────────────────────
# Clustering constants
# ──────────────────────────────────────────────────────────────────────────────

# Penalty distance assigned between annotations from the same rater.
# Must be high enough that average linkage can NEVER dilute it below any
# reasonable dendrogram cut threshold. With legitimate distances in [0, 1]
# and a cut threshold at most ~0.95, even in a cluster of 100 members a
# single 100.0 entry keeps the average well above any cut.
SAME_RATER_PENALTY = 100.0

# Small distance bonus applied when two different-rater annotations share
# the same class label. This breaks ties when a rater drew multiple overlapping
# annotations with different class labels (e.g., R9-ILD and R9-PulmonaryFibrosis
# at the same bbox). Without this, the clustering would match R9's annotations
# to other raters' boxes arbitrarily; with this, same-class matches are preferred.
# Must be small enough not to override the spatial metric.
CLASS_MATCH_BONUS = 0.01


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: Bounding boxes in format [x, y, w, h]
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, w1, h1 = box1
    box1_x2 = x1_1 + w1
    box1_y2 = y1_1 + h1
    x2_1, y2_1, w2, h2 = box2
    box2_x2 = x2_1 + w2
    box2_y2 = y2_1 + h2
    
    inter_x1 = max(x1_1, x2_1)
    inter_y1 = max(y1_1, y2_1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0


def calculate_iomin(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Minimum area (IoMin) between two bounding boxes.
    
    IoMin is more robust to size differences than IoU. When a small box is fully
    contained inside a larger box, IoMin = 1.0 regardless of size ratio, whereas
    IoU can be arbitrarily low.
    
    Formula: IoMin(A, B) = Intersection(A, B) / min(Area(A), Area(B))
    
    References:
        - Used in containment-based matching: Kalal et al. (2010) "Forward-Backward
          Error: Automatic Detection of Tracking Failures"
        - Related to "Intersection over Smaller" in Papadopoulos et al. (2017)
          "Training Object Class Detectors with Click Supervision", CVPR
        - Applied in annotation aggregation: Amgad et al. (2022) "NuCLS: A scalable 
          crowdsourcing approach and dataset for nucleus classification and 
          segmentation in breast cancer", GigaScience
    
    Args:
        box1, box2: Bounding boxes in format [x, y, w, h]
    
    Returns:
        IoMin value between 0 and 1
    """
    x1_1, y1_1, w1, h1 = box1
    box1_x2 = x1_1 + w1
    box1_y2 = y1_1 + h1
    x2_1, y2_1, w2, h2 = box2
    box2_x2 = x2_1 + w2
    box2_y2 = y2_1 + h2
    
    inter_x1 = max(x1_1, x2_1)
    inter_y1 = max(y1_1, y2_1)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)
    
    intersection_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    min_area = min(box1_area, box2_area)
    
    return intersection_area / min_area if min_area > 0 else 0.0


def calculate_normalized_center_distance(box1: List[float], box2: List[float]) -> float:
    """
    Calculate center distance between two bboxes, normalized by the smaller bbox's diagonal.
    
    This provides a scale-invariant measure of how far apart two bbox centers are
    relative to their size. A value of 0 means identical centers; a value > 1 means
    the centers are farther apart than the smaller box's diagonal.
    
    References:
        - Distance-IoU (DIoU) loss: Zheng et al. (2020) "Distance-IoU Loss: Faster 
          and Better Learning for Bounding Box Regression", AAAI 2020
        - Center distance is a key component in CIoU: Zheng et al. (2021) "Enhancing
          Geometric Factors in Model Learning and Inference for Object Detection and 
          Instance Segmentation", IEEE TGRS
    
    Args:
        box1, box2: Bounding boxes in format [x, y, w, h]
    
    Returns:
        Normalized center distance (0 = identical centers, >1 = far apart relative to size)
    """
    cx1, cy1 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    cx2, cy2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    
    center_dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    # Normalize by the smaller bbox diagonal for scale invariance
    diag1 = np.sqrt(box1[2]**2 + box1[3]**2)
    diag2 = np.sqrt(box2[2]**2 + box2[3]**2)
    min_diag = min(diag1, diag2)
    
    if min_diag <= 0:
        return float('inf')
    
    return center_dist / min_diag


def cluster_annotations_by_iou(annotations: List[Dict], iou_threshold: float = 0.5) -> List[List[Dict]]:
    """
    Cluster annotations using NuCLS methodology (hierarchical clustering with participant constraint).
    
    Args:
        annotations: List of annotation dictionaries with 'bbox' and 'rater_id' keys
        iou_threshold: Minimum IoU for annotations to be considered part of same cluster
    
    Returns:
        List of clusters, where each cluster is a list of annotations
    """
    if not annotations:
        return []

    num_annotations = len(annotations)
    if num_annotations <= 1:
        return [annotations]

    # Create condensed distance matrix from IoU
    distances = []
    for i in range(num_annotations):
        for j in range(i + 1, num_annotations):
            ann_i = annotations[i]
            ann_j = annotations[j]

            # Apply participant constraint: same rater annotations cannot cluster
            if ann_i.get('rater_id') == ann_j.get('rater_id'):
                distance = SAME_RATER_PENALTY
            else:
                iou = calculate_iou(ann_i['bbox'], ann_j['bbox'])
                distance = 1 - iou
                # Class-aware tie-breaking: prefer same-class matches
                if ann_i.get('class_name') == ann_j.get('class_name'):
                    distance = max(0, distance - CLASS_MATCH_BONUS)
            
            distances.append(distance)

    if not distances:
        return [[ann] for ann in annotations]

    # Perform agglomerative hierarchical clustering
    Z = linkage(distances, method='average')

    # Form flat clusters by cutting dendrogram
    distance_threshold = 1 - iou_threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Format output
    num_clusters = cluster_labels.max()
    clusters = [[] for _ in range(num_clusters)]
    for i, ann in enumerate(annotations):
        cluster_idx = cluster_labels[i] - 1  # fcluster labels are 1-based
        clusters[cluster_idx].append(ann)

    return clusters


def cluster_annotations_by_iomin(annotations: List[Dict], iomin_threshold: float = 0.5) -> List[List[Dict]]:
    """
    Cluster annotations using IoMin (Intersection over Minimum) similarity metric.
    
    IoMin is more robust than IoU when annotators agree on the same finding but
    draw bounding boxes of different sizes. The smaller box being substantially
    contained within the larger box yields high IoMin regardless of size difference.
    
    Uses the same hierarchical clustering framework (NuCLS methodology) as the IoU
    variant, with the same participant constraint.
    
    Args:
        annotations: List of annotation dictionaries with 'bbox' and 'rater_id' keys
        iomin_threshold: Minimum IoMin for annotations to be considered part of same cluster
    
    Returns:
        List of clusters, where each cluster is a list of annotations
    """
    if not annotations:
        return []

    num_annotations = len(annotations)
    if num_annotations <= 1:
        return [annotations]

    # Create condensed distance matrix from IoMin
    distances = []
    for i in range(num_annotations):
        for j in range(i + 1, num_annotations):
            ann_i = annotations[i]
            ann_j = annotations[j]

            # Apply participant constraint: same rater annotations cannot cluster
            if ann_i.get('rater_id') == ann_j.get('rater_id'):
                distance = SAME_RATER_PENALTY
            else:
                iomin = calculate_iomin(ann_i['bbox'], ann_j['bbox'])
                distance = 1 - iomin
                # Class-aware tie-breaking: prefer same-class matches
                if ann_i.get('class_name') == ann_j.get('class_name'):
                    distance = max(0, distance - CLASS_MATCH_BONUS)
            
            distances.append(distance)

    if not distances:
        return [[ann] for ann in annotations]

    # Perform agglomerative hierarchical clustering
    Z = linkage(distances, method='average')

    # Form flat clusters by cutting dendrogram
    distance_threshold = 1 - iomin_threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Format output
    num_clusters = cluster_labels.max()
    clusters = [[] for _ in range(num_clusters)]
    for i, ann in enumerate(annotations):
        cluster_idx = cluster_labels[i] - 1
        clusters[cluster_idx].append(ann)

    return clusters


def cluster_annotations_by_iomin_center_gate(
    annotations: List[Dict], 
    iomin_threshold: float = 0.5, 
    center_ratio_threshold: float = 0.5
) -> List[List[Dict]]:
    """
    Cluster annotations using IoMin with center-distance gating.
    
    Combines IoMin (robust to size differences) with a center-distance gate
    that prevents merging genuinely distinct nearby findings. Two annotations
    can only cluster together if BOTH conditions are met:
      1. Their normalized center distance is within center_ratio_threshold
      2. Their IoMin exceeds iomin_threshold (enforced via dendrogram cut)
    
    The center gate addresses the key weakness of pure IoMin: a small finding
    bbox that happens to fall within a larger finding's bbox would have high IoMin
    but displaced centers — the gate correctly separates them.
    
    References:
        - Center-distance gating concept from DIoU: Zheng et al. (2020) 
          "Distance-IoU Loss: Faster and Better Learning for Bounding Box 
          Regression", AAAI 2020
        - Multi-criteria clustering for annotations: Amgad et al. (2022) "NuCLS: 
          A scalable crowdsourcing approach and dataset for nucleus classification 
          and segmentation in breast cancer", GigaScience
        - Hierarchical clustering with constraints: Davidson & Ravi (2005) 
          "Agglomerative hierarchical clustering with constraints: Theoretical and 
          empirical results", PKDD 2005
    
    Args:
        annotations: List of annotation dictionaries with 'bbox' and 'rater_id' keys
        iomin_threshold: Minimum IoMin for clustering (applied via dendrogram cut)
        center_ratio_threshold: Maximum normalized center distance for clustering.
            Normalized by the smaller bbox's diagonal. 0.5 means centers must be
            within half a diagonal of each other.
    
    Returns:
        List of clusters, where each cluster is a list of annotations
    """
    if not annotations:
        return []

    num_annotations = len(annotations)
    if num_annotations <= 1:
        return [annotations]

    # Create condensed distance matrix with center-distance gating
    distances = []
    for i in range(num_annotations):
        for j in range(i + 1, num_annotations):
            ann_i = annotations[i]
            ann_j = annotations[j]

            # Apply participant constraint: same rater annotations cannot cluster
            if ann_i.get('rater_id') == ann_j.get('rater_id'):
                distance = SAME_RATER_PENALTY
            else:
                # Center-distance gate: check proximity first
                center_ratio = calculate_normalized_center_distance(
                    ann_i['bbox'], ann_j['bbox']
                )
                
                if center_ratio > center_ratio_threshold:
                    # Gate fails: centers too far apart, prevent clustering
                    # regardless of how much the boxes overlap
                    distance = SAME_RATER_PENALTY
                else:
                    # Gate passes: use IoMin as similarity metric
                    iomin = calculate_iomin(ann_i['bbox'], ann_j['bbox'])
                    distance = 1 - iomin
                    # Class-aware tie-breaking: prefer same-class matches
                    if ann_i.get('class_name') == ann_j.get('class_name'):
                        distance = max(0, distance - CLASS_MATCH_BONUS)
            
            distances.append(distance)

    if not distances:
        return [[ann] for ann in annotations]

    # Perform agglomerative hierarchical clustering
    Z = linkage(distances, method='average')

    # Form flat clusters by cutting dendrogram
    distance_threshold = 1 - iomin_threshold
    cluster_labels = fcluster(Z, t=distance_threshold, criterion='distance')

    # Format output
    num_clusters = cluster_labels.max()
    clusters = [[] for _ in range(num_clusters)]
    for i, ann in enumerate(annotations):
        cluster_idx = cluster_labels[i] - 1
        clusters[cluster_idx].append(ann)

    return clusters


def enforce_participant_constraint(clusters: List[List[Dict]]) -> List[List[Dict]]:
    """
    Post-clustering safety net: ensure each cluster has at most one annotation per rater.
    
    If a rater contributed multiple annotations to the same cluster (which can happen
    when they drew overlapping bboxes with different class labels), this function:
    1. Determines the cluster's majority class from non-duplicate rater contributions
    2. Keeps the annotation from the duplicate rater that matches the majority class
    3. Ejects non-matching annotations to their own singleton clusters
    
    This preserves all annotations in the output (nothing is discarded) while ensuring
    each cluster respects the one-annotation-per-rater invariant.
    
    Args:
        clusters: List of clusters from hierarchical clustering
    
    Returns:
        List of clusters with the participant constraint enforced
    """
    enforced_clusters = []
    ejected = []
    
    for cluster in clusters:
        if len(cluster) <= 1:
            enforced_clusters.append(cluster)
            continue
        
        # Group annotations by rater
        rater_annotations = defaultdict(list)
        for ann in cluster:
            rater_annotations[ann.get('rater_id')].append(ann)
        
        # If no rater has duplicates, keep cluster as-is
        if all(len(anns) <= 1 for anns in rater_annotations.values()):
            enforced_clusters.append(cluster)
            continue
        
        # Determine majority class from unique-rater contributions first,
        # since those are unambiguous signals of what this cluster represents
        unique_rater_votes = []
        for rater_id, anns in rater_annotations.items():
            if len(anns) == 1:
                unique_rater_votes.append(anns[0].get('class_name'))
        
        if unique_rater_votes:
            majority_class = max(set(unique_rater_votes), key=unique_rater_votes.count)
        else:
            # All raters have duplicates — use overall majority
            all_votes = [ann.get('class_name') for ann in cluster]
            majority_class = max(set(all_votes), key=all_votes.count)
        
        # Resolve duplicates: keep the annotation matching majority class
        kept = []
        for rater_id, anns in rater_annotations.items():
            if len(anns) == 1:
                kept.append(anns[0])
            else:
                # Prefer annotation whose class matches the cluster majority
                matching = [a for a in anns if a.get('class_name') == majority_class]
                non_matching = [a for a in anns if a.get('class_name') != majority_class]
                
                if matching:
                    kept.append(matching[0])
                    # Eject extra same-class duplicates (true duplicates / corrections)
                    ejected.extend(matching[1:])
                    # Eject different-class annotations
                    ejected.extend(non_matching)
                else:
                    # No annotation matches majority — keep first, eject rest
                    kept.append(anns[0])
                    ejected.extend(anns[1:])
        
        enforced_clusters.append(kept)
    
    # Ejected annotations become singleton clusters (not discarded)
    for ann in ejected:
        enforced_clusters.append([ann])
    
    return enforced_clusters


# Valid clustering method identifiers
VALID_CLUSTERING_METHODS = ['iou', 'iomin', 'iomin_center_gate']


def cluster_annotations(
    annotations: List[Dict],
    method: str = 'iou',
    iou_threshold: float = 0.5,
    iomin_threshold: float = 0.5,
    center_ratio_threshold: float = 0.5
) -> List[List[Dict]]:
    """
    Generic annotation clustering dispatcher.
    
    Routes to the appropriate clustering function based on the specified method.
    All methods use the same NuCLS-style hierarchical clustering framework with
    participant constraints, but differ in their similarity metric.
    
    Methods:
        'iou': Standard IoU-based clustering (original method)
        'iomin': IoMin-based clustering (robust to bbox size differences)
        'iomin_center_gate': IoMin with center-distance gating (robust to both
            size differences and nearby distinct findings)
    
    Args:
        annotations: List of annotation dictionaries with 'bbox' and 'rater_id' keys
        method: Clustering method ('iou', 'iomin', or 'iomin_center_gate')
        iou_threshold: IoU threshold (used when method='iou')
        iomin_threshold: IoMin threshold (used when method='iomin' or 'iomin_center_gate')
        center_ratio_threshold: Center-distance gate threshold 
            (used when method='iomin_center_gate')
    
    Returns:
        List of clusters, where each cluster is a list of annotations.
        Post-processed by enforce_participant_constraint() to guarantee
        at most one annotation per rater per cluster.
    
    Raises:
        ValueError: If an unknown clustering method is specified
    """
    if method == 'iou':
        clusters = cluster_annotations_by_iou(annotations, iou_threshold)
    elif method == 'iomin':
        clusters = cluster_annotations_by_iomin(annotations, iomin_threshold)
    elif method == 'iomin_center_gate':
        clusters = cluster_annotations_by_iomin_center_gate(
            annotations, iomin_threshold, center_ratio_threshold
        )
    else:
        raise ValueError(
            f"Unknown clustering method: '{method}'. "
            f"Valid methods: {VALID_CLUSTERING_METHODS}"
        )
    
    # Safety net: enforce one-annotation-per-rater invariant
    return enforce_participant_constraint(clusters)


def normalize_annotator_weights(csv_path: str) -> Dict[str, float]:
    """
    Load and normalize annotator reliability weights from CSV.
    
    Args:
        csv_path: Path to CSV file with annotator reliability scores
    
    Returns:
        Dictionary mapping annotator IDs to normalized weights
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return {}
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        return {}

    # Verify expected columns exist
    expected_columns = ['Annotator', 'Avg K-alpha']
    if not all(col in df.columns for col in expected_columns):
        print(f"Error: CSV file must contain columns: {expected_columns}")
        print(f"Found columns: {df.columns.tolist()}")
        return {}
        
    annotator_ids = df['Annotator'].astype(str)
    weights = df['Avg K-alpha'].astype(float).fillna(0)

    # Normalize weights (sum to 1)
    if weights.sum() == 0:
        print("Warning: Sum of weights is zero, cannot normalize.")
        return {}
    else:
        normalized_weights = weights / weights.sum()
    
    return dict(zip(annotator_ids, normalized_weights))


def load_and_cluster_data(
    train_csv_path: str, 
    raters_json_path: str, 
    weights_csv_path: str,
    iou_threshold: float = 0.5,
    max_images: int = None,
    clustering_method: str = 'iou',
    iomin_threshold: float = 0.5,
    center_ratio_threshold: float = 0.5
) -> Dict[str, List[List[Dict]]]:
    """
    Load VinDr-CXR data, cluster annotations, and organize by class.
    
    Args:
        train_csv_path: Path to train_modified.csv
        raters_json_path: Path to train_vindrcxr_all.json
        weights_csv_path: Path to annotator reliability scores CSV
        iou_threshold: IoU threshold for clustering (used when clustering_method='iou')
        max_images: Maximum number of images to process (None = all)
        clustering_method: Clustering method ('iou', 'iomin', or 'iomin_center_gate')
        iomin_threshold: IoMin threshold (used when clustering_method='iomin' or 'iomin_center_gate')
        center_ratio_threshold: Center-distance gate threshold 
            (used when clustering_method='iomin_center_gate')
    
    Returns:
        Dictionary mapping class names to lists of clusters
    """
    print(f"Loading data from {train_csv_path}...")
    
    # Load annotator weights
    normalized_weights = normalize_annotator_weights(weights_csv_path)
    if not normalized_weights:
        print("Warning: Could not load annotator weights, using equal weights")
    
    # Load CSV data
    try:
        df_train = pd.read_csv(train_csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return {}

    # Determine class column name
    class_name_col = None
    if 'class_name' in df_train.columns:
        class_name_col = 'class_name'
    elif 'label' in df_train.columns:
        class_name_col = 'label'
    else:
        print("Error: No class label column found")
        return {}

    # Get unique image IDs
    all_image_ids = df_train['image_id'].unique()
    if max_images:
        all_image_ids = all_image_ids[:max_images]
    
    print(f"Processing {len(all_image_ids)} images...")
    
    # Load annotations per image
    clusters_by_class = defaultdict(list)
    
    for img_id in all_image_ids:
        img_annotations_df = df_train[df_train['image_id'] == img_id]
        if img_annotations_df.empty:
            continue

        current_image_annotations = []
        for _, row in img_annotations_df.iterrows():
            class_name = row[class_name_col]
            
            # Skip 'No Finding' annotations for prior calculation
            if class_name.lower() in ['no finding', 'no findings']:
                continue
            
            # Handle regular findings
            x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            if x_max <= x_min or y_max <= y_min:
                continue
            
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            rater_weight = normalized_weights.get(str(row['rad_id']), 1.0/len(normalized_weights) if normalized_weights else 1.0)
            
            current_image_annotations.append({
                'rater_id': str(row['rad_id']),
                'bbox': bbox,
                'class_name': class_name,
                'weight': rater_weight
            })
        
        if not current_image_annotations:
            continue
        
        # Cluster annotations for this image
        annotation_clusters = cluster_annotations(
            current_image_annotations,
            method=clustering_method,
            iou_threshold=iou_threshold,
            iomin_threshold=iomin_threshold,
            center_ratio_threshold=center_ratio_threshold
        )
        
        # Organize clusters by class (majority vote)
        for cluster in annotation_clusters:
            if not cluster:
                continue
            
            # Determine cluster class by majority vote
            class_votes = [ann['class_name'] for ann in cluster]
            cluster_class = max(set(class_votes), key=class_votes.count)
            
            # Add cluster to the appropriate class
            clusters_by_class[cluster_class].append(cluster)
    
    print(f"Loaded {sum(len(clusters) for clusters in clusters_by_class.values())} total clusters")
    print(f"Classes found: {list(clusters_by_class.keys())}")
    
    return dict(clusters_by_class)


def print_clustering_statistics(clusters_by_class: Dict[str, List[List[Dict]]]):
    """
    Print detailed statistics about the clustering results.
    
    Args:
        clusters_by_class: Dictionary mapping class names to lists of clusters
    """
    print("\n" + "="*80)
    print("Clustering Statistics")
    print("="*80)
    
    total_clusters = sum(len(clusters) for clusters in clusters_by_class.values())
    total_annotations = sum(
        sum(len(cluster) for cluster in clusters) 
        for clusters in clusters_by_class.values()
    )
    
    print(f"Total classes: {len(clusters_by_class)}")
    print(f"Total clusters: {total_clusters}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per cluster: {total_annotations/total_clusters:.2f}")
    
    print("\nPer-class breakdown:")
    print(f"{'Class':<25} {'Clusters':<10} {'Multi-ann':<10} {'Single-ann':<10} {'Avg Size':<10}")
    print("-" * 80)
    
    for class_name, clusters in sorted(clusters_by_class.items(), key=lambda x: len(x[1]), reverse=True):
        multi_ann = len([c for c in clusters if len(c) >= 2])
        single_ann = len([c for c in clusters if len(c) == 1])
        avg_size = np.mean([len(c) for c in clusters])
        
        print(f"{class_name:<25} {len(clusters):<10} {multi_ann:<10} {single_ann:<10} {avg_size:<10.2f}")
    
    print("="*80 + "\n")
