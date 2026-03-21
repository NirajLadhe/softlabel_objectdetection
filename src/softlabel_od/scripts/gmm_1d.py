"""
1D GMM Soft Label Generation with Class-Specific Priors.

This script generates 1D GMM soft labels using class-specific variance priors to provide
meaningful uncertainty estimates for single-annotation instances. It extends the baseline
1DGMM_all.py by integrating Bayesian prior adjustment.

Prerequisites:
1. Run calculate_priors_1d.py to pre-compute class-specific priors
2. Ensure annotator reliability scores are available

Usage:
    python 1DGMM_all_with_priors.py
    python 1DGMM_all_with_priors.py --clustering-method iomin_center_gate --iomin-threshold 0.5 --center-ratio-threshold 0.5
"""

import os
import argparse
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from softlabel_od.softlabel_generation.core.label_assembly.priors import Prior1DCalculator
from softlabel_od.softlabel_generation.core.preprocessing.data_loader import load_all_data_for_gmm
from softlabel_od.softlabel_generation.utils.clustering import (
    normalize_annotator_weights, cluster_annotations, VALID_CLUSTERING_METHODS
)
from softlabel_od.shared.utils.serialization import NpEncoder

# Constants
NO_FINDING_IN_CLUSTER_VOTE = "NoFindingInCluster"


def compute_class_probs(class_votes, vote_weights):
    """Compute normalized probability distribution for classes based on weighted votes."""
    if not class_votes or not vote_weights or len(class_votes) != len(vote_weights):
        return {}
    
    valid_votes = []
    valid_weights = []
    for vote, weight in zip(class_votes, vote_weights):
        if weight > 1e-6:
            valid_votes.append(vote)
            valid_weights.append(weight)
            
    if not valid_votes:
        return {}

    class_p = {}
    total_weight = sum(valid_weights)

    if total_weight == 0:
        if valid_votes:
            num_unique_classes = len(set(valid_votes))
            if num_unique_classes > 0:
                return {cls: 1.0/num_unique_classes for cls in set(valid_votes)}
        return {}

    for class_name, weight in zip(valid_votes, valid_weights):
        class_p[class_name] = class_p.get(class_name, 0) + weight
    
    for class_name in class_p:
        class_p[class_name] /= total_weight
        
    return class_p


def main():
    parser = argparse.ArgumentParser(
        description="Generate 1D GMM soft labels with class-specific priors."
    )
    parser.add_argument('--train-csv', type=str, default='vindr_cxr/train_enriched_normalized.csv')
    parser.add_argument('--raters-json', type=str, default='vindr_cxr/train_vindrcxr_all.json')
    parser.add_argument('--weights-csv', type=str, default=None)
    parser.add_argument('--priors', type=str, default='priors/class_priors_1d.json')
    parser.add_argument('--output', type=str, default='outputs/1DGMM_softLabels_with_priors.json')
    
    # Prior hyperparameters
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Prior strength / pseudo-observation count (default: 1.0)')
    
    # Clustering parameters
    parser.add_argument('--clustering-method', type=str, default='iou',
                        choices=VALID_CLUSTERING_METHODS,
                        help=f'Clustering method (default: iou). Options: {VALID_CLUSTERING_METHODS}')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                        help='IoU threshold for clustering (used when method=iou, default: 0.5)')
    parser.add_argument('--iomin-threshold', type=float, default=0.5,
                        help='IoMin threshold for clustering (used when method=iomin or iomin_center_gate, default: 0.5)')
    parser.add_argument('--center-ratio-threshold', type=float, default=0.5,
                        help='Center-distance gate threshold (used when method=iomin_center_gate, default: 0.5)')
    
    # Reliability IoU (for path resolution)
    parser.add_argument('--iou-reliability', type=float, default=0.5,
                        help='IoU threshold that was used for reliability calculation (for resolving weights CSV path)')
    
    args = parser.parse_args()
    
    if args.weights_csv is None:
        safe_iou_str = str(args.iou_reliability).replace('.', '_')
        args.weights_csv = f'annotator_reliability_vindrcxr/annotator_reliability_scores_vindrcxr_iou{safe_iou_str}.csv'
    
    ANNOTATOR_WEIGHTS_CSV = args.weights_csv
    TRAIN_CSV_PATH = args.train_csv
    RATERS_JSON_PATH = args.raters_json
    PRIORS_PATH = args.priors
    OUTPUT_JSON = args.output
    CLUSTERING_METHOD = args.clustering_method
    IOU_THRESHOLD = args.iou_threshold
    IOMIN_THRESHOLD = args.iomin_threshold
    CENTER_RATIO_THRESHOLD = args.center_ratio_threshold
    
    # Load pre-computed priors
    print("="*80)
    print("1D GMM Soft Label Generation with Class-Specific Priors")
    print("="*80)
    print(f"Configuration:")
    print(f"  Clustering method: {CLUSTERING_METHOD}")
    if CLUSTERING_METHOD == 'iou':
        print(f"  IoU threshold: {IOU_THRESHOLD}")
    elif CLUSTERING_METHOD == 'iomin':
        print(f"  IoMin threshold: {IOMIN_THRESHOLD}")
    elif CLUSTERING_METHOD == 'iomin_center_gate':
        print(f"  IoMin threshold: {IOMIN_THRESHOLD}")
        print(f"  Center-ratio threshold: {CENTER_RATIO_THRESHOLD}")
    print(f"  Prior alpha: {args.alpha}")
    print(f"  Priors path: {PRIORS_PATH}")
    print(f"  Output: {OUTPUT_JSON}")
    print("="*80)
    
    print("Loading pre-computed class priors...")
    
    if not os.path.exists(PRIORS_PATH):
        print(f"Error: Priors file not found: {PRIORS_PATH}")
        print("Please run calculate_priors_1d.py first to generate priors.")
        return
    
    prior_calculator = Prior1DCalculator(alpha=args.alpha)
    prior_calculator.load_priors(PRIORS_PATH)
    
    # Load annotator weights
    print("Loading annotator reliability weights...")
    normalized_rater_weights = normalize_annotator_weights(ANNOTATOR_WEIGHTS_CSV)
    
    if not normalized_rater_weights:
        print("Error: Could not load annotator weights!")
        return
    
    print(f"Loaded weights for {len(normalized_rater_weights)} annotators")
    
    # Load image metadata from JSON
    print("Loading image metadata...")
    with open(RATERS_JSON_PATH, 'r') as f:
        raters_json = json.load(f)
    
    # Create a mapping: image_id -> {width, height}
    image_metadata = {}
    for img_entry in raters_json.get('images', []):
        file_name = img_entry.get('file_name', '')
        image_id = file_name.replace('.dicom', '')
        image_metadata[image_id] = {
            'width': img_entry.get('width'),
            'height': img_entry.get('height'),
            'rater_list': img_entry.get('rater_list', [])
        }
    
    print(f"Loaded metadata for {len(image_metadata)} images")
    
    # Load data
    print("\nLoading dataset...")
    subset_image_ids, subset_image_data = load_all_data_for_gmm(
        TRAIN_CSV_PATH, RATERS_JSON_PATH, normalized_rater_weights
    )
    
    if not subset_image_ids:
        print("Error: No data loaded!")
        return
    
    # Generate soft labels with priors
    print(f"\nGenerating 1D GMM soft labels with priors for {len(subset_image_ids)} images...")
    soft_labels = []
    
    for image_id in tqdm(subset_image_ids, desc="Processing Images"):
        all_annotations_for_image = subset_image_data[image_id]
        all_rater_ids_on_image = set(
            ann['rater_id'] for ann in all_annotations_for_image if 'rater_id' in ann
        )
    
        # Cluster annotations for this image using selected method
        annotation_clusters = cluster_annotations(
            all_annotations_for_image,
            method=CLUSTERING_METHOD,
            iou_threshold=IOU_THRESHOLD,
            iomin_threshold=IOMIN_THRESHOLD,
            center_ratio_threshold=CENTER_RATIO_THRESHOLD
        )
        
        for cluster_idx, cluster_annotations_list in enumerate(annotation_clusters):
            valid_annotations = [
                ann for ann in cluster_annotations_list if ann['weight'] > 1e-6
            ]
            
            if not valid_annotations:
                soft_label = {
                    'image_id': image_id,
                    'image_width': image_metadata[image_id]['width'],
                    'image_height': image_metadata[image_id]['height'],
                    'finding_id': cluster_idx,
                    'original_annotations': cluster_annotations_list,
                    'spatial': None,
                    'class_probs': {},
                    'num_annotations_in_cluster': 0
                }
                soft_labels.append(soft_label)
                continue
    
            # Check if this is a 'No Finding' cluster
            is_no_finding_cluster = all(
                ann['class_name'].lower() in ['no finding', 'no findings'] 
                for ann in valid_annotations
            )
            
            # Get cluster class (majority vote)
            class_votes = [ann['class_name'] for ann in valid_annotations]
            cluster_class = max(set(class_votes), key=class_votes.count)
            
            # Calculate class probabilities (with NoFindingInCluster logic)
            raters_in_cluster = set(ann['rater_id'] for ann in valid_annotations)
            raters_absent_from_cluster = all_rater_ids_on_image - raters_in_cluster
            
            final_class_votes = list(class_votes)
            final_vote_weights = [ann['weight'] for ann in valid_annotations]
            
            for rater_id in raters_absent_from_cluster:
                if rater_id in normalized_rater_weights:
                    final_class_votes.append(NO_FINDING_IN_CLUSTER_VOTE)
                    final_vote_weights.append(normalized_rater_weights[rater_id])
            
            class_probs = compute_class_probs(final_class_votes, final_vote_weights)
            
            # Handle 'No Finding' clusters - no spatial parameters
            if is_no_finding_cluster:
                soft_label = {
                    'image_id': image_id,
                    'image_width': image_metadata[image_id]['width'],
                    'image_height': image_metadata[image_id]['height'],
                    'finding_id': cluster_idx,
                    'original_annotations': cluster_annotations_list,
                    'spatial': {
                        'mean': [0.0, 0.0, 1.0, 1.0],
                        'std': [0.0, 0.0, 0.0, 0.0],
                    },
                    'prior_used': 'no_finding',
                    'class_probs': class_probs,
                    'num_annotations_in_cluster': len(valid_annotations)
                }
                soft_labels.append(soft_label)
                continue
            
            # Calculate spatial parameters with PRIOR ADJUSTMENT
            bboxes = [ann['bbox'] for ann in valid_annotations]
            weights = [ann['weight'] for ann in valid_annotations]
            
            # Calculate weighted means
            x_coords = [bbox[0] for bbox in bboxes]
            y_coords = [bbox[1] for bbox in bboxes]
            w_coords = [bbox[2] for bbox in bboxes]
            h_coords = [bbox[3] for bbox in bboxes]
            
            x_mean = np.average(x_coords, weights=weights)
            y_mean = np.average(y_coords, weights=weights)
            w_mean = np.average(w_coords, weights=weights)
            h_mean = np.average(h_coords, weights=weights)
            
            # Calculate PRIOR-ADJUSTED variances
            x_var = prior_calculator.adjust_variance(x_coords, cluster_class, 'x', weights)
            y_var = prior_calculator.adjust_variance(y_coords, cluster_class, 'y', weights)
            w_var = prior_calculator.adjust_variance(w_coords, cluster_class, 'w', weights)
            h_var = prior_calculator.adjust_variance(h_coords, cluster_class, 'h', weights)
            
            # Convert to standard deviations
            x_std = np.sqrt(x_var)
            y_std = np.sqrt(y_var)
            w_std = np.sqrt(w_var)
            h_std = np.sqrt(h_var)
            
            soft_label = {
                'image_id': image_id,
                'image_width': image_metadata[image_id]['width'],
                'image_height': image_metadata[image_id]['height'],
                'finding_id': cluster_idx,
                'original_annotations': cluster_annotations_list,
                'spatial': {
                    'mean': [x_mean, y_mean, w_mean, h_mean],
                    'std': [x_std, y_std, w_std, h_std],
                },
                'prior_used': cluster_class,
                'class_probs': class_probs,
                'num_annotations_in_cluster': len(valid_annotations)
            }
            soft_labels.append(soft_label)
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_JSON) if os.path.dirname(OUTPUT_JSON) else '.', exist_ok=True)
    print(f"\nSaving results to {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(soft_labels, f, indent=2, cls=NpEncoder)
    
    print(f"Successfully saved {len(soft_labels)} soft labels (1D GMM with priors)")
    
    # Print statistics
    single_ann_count = len([sl for sl in soft_labels if sl.get('num_annotations_in_cluster', 0) == 1])
    multi_ann_count = len([sl for sl in soft_labels if sl.get('num_annotations_in_cluster', 0) > 1])
    no_finding_count = len([sl for sl in soft_labels if sl.get('spatial') is None])
    
    print(f"\nStatistics:")
    print(f"  Total soft labels: {len(soft_labels)}")
    print(f"  Single-annotation clusters: {single_ann_count} (placeholder variance: class specific priors)")
    print(f"  Multi-annotation clusters: {multi_ann_count} (variance prior-adjusted)")
    print(f"  'No Finding' clusters: {no_finding_count}")
    
    if soft_labels:
        print(f"\nSample soft label (with prior adjustment):")
        sample = next((sl for sl in soft_labels if sl.get('spatial') is not None), soft_labels[0])
        print(json.dumps(sample, indent=2, cls=NpEncoder)[:500] + "...")
    
    print("\n" + "="*80)
    print("1D GMM with Priors - Generation Complete!")
    print(f"Output: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
