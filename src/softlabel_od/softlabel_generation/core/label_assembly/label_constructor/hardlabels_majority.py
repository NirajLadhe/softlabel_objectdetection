"""
Extract hard labels from 1D Gaussian Uncertainty soft labels - MAJORITY VOTE approach.

This script respects majority vote decisions.
IMPORTANT: True "No Finding" images (where all annotators agreed) are INCLUDED.

Usage: python -m softlabel_od.softlabel_generation.core.label_assembly.label_constructor.hardlabels_majority
"""

import json
import os
import sys
from collections import Counter
import argparse

def is_placeholder_bbox(bbox):
    """Check if bbox is the [0,0,1,1] placeholder for 'No Finding'."""
    return bbox == [0.0, 0.0, 1.0, 1.0]

def is_true_no_finding_cluster(soft_label):
    """Check if this is a true 'No Finding' cluster (all annotators agreed on no pathology)."""
    original_annotations = soft_label['original_annotations']
    
    # Check if all original annotations are 'No Finding' with placeholder bboxes
    all_no_finding = all(
        ann['class_name'].lower() in ['no finding', 'no findings'] and 
        is_placeholder_bbox(ann['bbox'])
        for ann in original_annotations
    )
    
    return all_no_finding

def extract_majority_hard_labels(soft_labels, min_confidence=0.3):
    """
    Extract hard labels respecting majority vote decisions.
    INCLUDES true No Finding images as negative examples.
    
    Logic:
    1. TRUE NO FINDING: All annotators agreed on no pathology → Include as "No Finding"
    2. MAJORITY NO FINDING: NoFindingInCluster wins vote → Include as "No Finding" 
    3. MAJORITY PATHOLOGY: Pathology class wins vote → Include with spatial bbox
    4. LOW CONFIDENCE: Exclude uncertain cases
    """
    hard_labels = []
    
    stats = {
        'total_soft_labels': len(soft_labels),
        'true_no_finding': 0,
        'majority_pathology': 0,
        'majority_no_finding_disputed': 0,
        'low_confidence_excluded': 0
    }
    
    for soft_label in soft_labels:
        image_id = soft_label['image_id']
        image_width = soft_label['image_width']
        image_height = soft_label['image_height']
        finding_id = soft_label['finding_id']
        original_annotations = soft_label['original_annotations']
        spatial = soft_label.get('spatial')
        class_probs = soft_label['class_probs']
        num_annotations = soft_label['num_annotations_in_cluster']
        
        # Case 1: TRUE NO FINDING - All annotators agreed on no pathology
        if is_true_no_finding_cluster(soft_label):
            hard_label = {
                'image_id': image_id,
                'image_width': image_width,
                'image_height': image_height,
                'finding_id': finding_id,
                'original_annotations': original_annotations,
                'bbox': [0.0, 0.0, 1.0, 1.0],  # Placeholder bbox for No Finding
                'class': 'No Finding',
                'confidence': 1.0,  # Perfect agreement
                'extraction_method': 'true_no_finding',
                'original_class_probs': class_probs,
                'renormalized_class_probs': {'No finding': 1.0},
                'num_annotations': num_annotations
            }
            
            hard_labels.append(hard_label)
            stats['true_no_finding'] += 1
            continue
        
        # Case 2: Use majority vote from class probabilities
        # Guard: skip clusters with empty class_probs (spatial=None records)
        if not class_probs:
            stats['low_confidence_excluded'] += 1
            continue
        majority_class = max(class_probs.keys(), key=class_probs.get)
        majority_confidence = class_probs[majority_class]
        
        # Only proceed if confidence meets threshold
        if majority_confidence < min_confidence:
            stats['low_confidence_excluded'] += 1
            continue
        
        # Case 3: Majority voted for NoFindingInCluster (disputed case)
        if majority_class == 'NoFindingInCluster':
            # hard_label = {
            #     'image_id': image_id,
            #     'image_width': image_width,
            #     'image_height': image_height,
            #     'finding_id': finding_id,
            #     'original_annotations': original_annotations,
            #     'bbox': [0.0, 0.0, 1.0, 1.0],  # Placeholder bbox for No Finding
            #     'class': 'No Finding',
            #     'confidence': majority_confidence,
            #     'extraction_method': 'majority_no_finding_disputed',
            #     'original_class_probs': class_probs,
            #     'renormalized_class_probs': {'No finding': class_probs['NoFindingInCluster']},
            #     'num_annotations': num_annotations
            # }
            
            # hard_labels.append(hard_label)
            stats['majority_no_finding_disputed'] += 1
            continue
        
        # Case 4: Majority voted for a pathology class
        else:
            # Use spatial mean if available, otherwise skip
            if spatial and spatial.get('mean'):
                hard_label = {
                    'image_id': image_id,
                    'image_width': image_width,
                    'image_height': image_height,
                    'finding_id': finding_id,
                    'original_annotations': original_annotations,
                    'bbox': spatial['mean'],  # Use spatial mean as hard bbox
                    'class': majority_class,
                    'confidence': majority_confidence,
                    'extraction_method': 'majority_pathology',
                    'original_class_probs': class_probs,
                    # renormalized class probs should be the argmax of the class probs
                    'renormalized_class_probs': {majority_class: class_probs[majority_class]},
                    'num_annotations': num_annotations
                }
                
                hard_labels.append(hard_label)
                stats['majority_pathology'] += 1
            else:
                stats['low_confidence_excluded'] += 1
    
    return hard_labels, stats

def main():
    parser = argparse.ArgumentParser(description="Extract hard labels from 1D Gaussian Uncertainty soft labels - MAJORITY VOTE approach.")
    parser.add_argument("--input-file", type=str, required=True, help="Input file path for 1D Gaussian soft labels.")
    parser.add_argument("--min-confidence", type=float, default=0.3, help="Minimum confidence threshold for hard labels.")
    parser.add_argument("--clustering-method", type=str, required=True, choices=['iou', 'iomin', 'iomin_center_gate'], help="Clustering method used to generate soft labels.")
    args = parser.parse_args()
    # File paths
    project_root_ext = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '..'))   
    input_file = args.input_file
    output_dir = os.path.join(project_root_ext, 'outputs/nucls_transform_0.001/hard_labels')
    output_file = os.path.join(output_dir, f'HardLabels_MajorityVote_{args.clustering_method}.json')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("1D GMM Hard Labels Extraction - MAJORITY VOTE Approach")
    print("="*80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print("✅ TRUE NO FINDING images will be INCLUDED")
    print()
    
    # Load soft labels
    print("Loading soft labels...")
    try:
        with open(input_file, 'r') as f:
            soft_labels = json.load(f)
        print(f"Loaded {len(soft_labels)} soft labels")
    except FileNotFoundError:
        print(f"ERROR: Input file not found: {input_file}")
        return
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in input file: {e}")
        return
    
    # Extract hard labels
    print("\nExtracting hard labels (majority vote approach)...")
    hard_labels, stats = extract_majority_hard_labels(soft_labels)
    
    # Save hard labels
    print(f"\nSaving {len(hard_labels)} hard labels...")
    with open(output_file, 'w') as f:
        json.dump(hard_labels, f, indent=2)
    
    # Print statistics
    print("\n" + "="*80)
    print("EXTRACTION STATISTICS")
    print("="*80)
    print(f"Total soft labels processed:        {stats['total_soft_labels']:,}")
    print(f"✅ True No Finding (included):      {stats['true_no_finding']:,}")
    print(f"Majority pathology:                 {stats['majority_pathology']:,}")
    print(f"Majority No Finding (disputed):     {stats['majority_no_finding_disputed']:,}")
    print(f"Low confidence excluded:            {stats['low_confidence_excluded']:,}")
    print(f"TOTAL HARD LABELS:                  {len(hard_labels):,}")
    print(f"Conversion rate:                    {len(hard_labels)/stats['total_soft_labels']*100:.1f}%")
    print("="*80)
    print(f"✅ Hard labels saved to: {output_file}")

if __name__ == "__main__":
    main()