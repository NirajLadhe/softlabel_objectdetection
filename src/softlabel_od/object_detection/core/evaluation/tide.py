"""
evaluation_metrics.py - Comprehensive probabilistic evaluation metrics
This file contains the functions to compute the evaluation metrics for the probabilistic model.

"""

import numpy as np
from copy import deepcopy
from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_map,
    match_predictions_to_gt,
    compute_iou_batch
)

def classify_errors_detailed(predictions, ground_truths, iou_threshold=0.5, bg_threshold=0.1):
    """
    Classify detection results into error types.
    Returns detailed error information for oracle operations.

    Error types (from TIDE paper):
        - 'tp': True positives (correct detections)
        - 'classification': IoU >= iou_threshold but wrong class
        - 'localization': 0.1 <= IoU < iou_threshold with correct class
        - 'both': 0.1 <= IoU < iou_threshold with wrong class
        - 'duplicate': IoU >= iou_threshold but GT already matched
        - 'background': IoU < bg_threshold
        - 'missed_gt': Ground truths with no matching detection
    """
    error_details = {
        'tp': [],           # (pred_idx, img_idx, score, pred_box, gt_box, gt_label, iou)
        'classification': [],  # (pred_idx, img_idx, score, pred_box, gt_box, gt_label)
        'localization': [],    # (pred_idx, img_idx, score, pred_box, gt_box, gt_label)
        'both': [],            # (pred_idx, img_idx, score, pred_box)
        'duplicate': [],       # (pred_idx, img_idx, score, pred_box)
        'background': [],      # (pred_idx, img_idx, score, pred_box)
        'missed_gt': [],       # (gt_idx, img_idx, gt_box, gt_label)
        'counts': {}
    }
    
    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        if 'gt_class_probs' in gt and len(gt['gt_class_probs']) > 0:
            gt_labels = np.argmax(gt['gt_class_probs'], axis=1)
        elif 'labels' in gt and len(gt['labels']) > 0:
            gt_labels = gt['labels']
        else:
            gt_labels = np.array([])

        # Handle empty cases
        if len(pred_boxes) == 0:
            for gt_idx in range(len(gt_boxes)):
                error_details['missed_gt'].append((gt_idx, img_idx, gt_boxes[gt_idx], gt_labels[gt_idx]))
            continue
        
        if len(gt_boxes) == 0:
            for pred_idx in range(len(pred_boxes)):
                error_details['background'].append((pred_idx, img_idx, pred_scores[pred_idx], pred_boxes[pred_idx]))
            continue
        
        # Standard matching
        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            pred_boxes, pred_scores, gt_boxes, iou_threshold
        )
        
        # Track matched GTs to identify duplicates
        matched_gt_to_pred = {}
        for pred_idx, gt_idx, iou in matches:
            matched_gt_to_pred[gt_idx] = (pred_idx, iou)
        
        # Process matches (potential TPs, classification, or localization errors)
        for pred_idx, gt_idx, iou in matches:
            pred_label = pred_labels[pred_idx]
            gt_label = gt_labels[gt_idx]
            
            if pred_label == gt_label and iou >= iou_threshold:
                # True positive
                error_details['tp'].append((
                    pred_idx, img_idx, pred_scores[pred_idx], 
                    pred_boxes[pred_idx], gt_boxes[gt_idx], gt_label, iou
                ))
            elif pred_label != gt_label and iou >= iou_threshold:
                # Classification error
                error_details['classification'].append((
                    pred_idx, img_idx, pred_scores[pred_idx],
                    pred_boxes[pred_idx], gt_boxes[gt_idx], gt_label
                ))
            elif pred_label == gt_label and iou < iou_threshold:
                # Localization error
                error_details['localization'].append((
                    pred_idx, img_idx, pred_scores[pred_idx],
                    pred_boxes[pred_idx], gt_boxes[gt_idx], gt_label
                ))
            else:  # pred_label != gt_label and iou < iou_threshold
                # Both classification and localization error
                error_details['both'].append((
                    pred_idx, img_idx, pred_scores[pred_idx], pred_boxes[pred_idx]
                ))
        
        # Process unmatched predictions (False Positives)
        for pred_idx in unmatched_preds:
            pred_box = pred_boxes[pred_idx]
            pred_label = pred_labels[pred_idx]
            pred_score = pred_scores[pred_idx]
            
            # Find best IoU with any GT (ignoring label) to classify the FP
            best_iou = 0
            best_gt_idx = -1
            best_gt_label = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou_batch(pred_box[None], gt_box[None])[0, 0]
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_gt_label = gt_labels[gt_idx]
            
            if best_iou >= iou_threshold:
                # High IoU but unmatched → duplicate or both
                if best_gt_idx in matched_gt_to_pred:
                    # GT already matched → duplicate detection
                    error_details['duplicate'].append((
                        pred_idx, img_idx, pred_score, pred_box
                    ))
                else:
                    # Should not happen if matching is correct
                    error_details['both'].append((
                        pred_idx, img_idx, pred_score, pred_box
                    ))
            elif best_iou >= bg_threshold:
                # Medium IoU but unmatched
                if pred_label == best_gt_label:
                    error_details['localization'].append((
                        pred_idx, img_idx, pred_score, pred_box,
                        gt_boxes[best_gt_idx], best_gt_label
                    ))
                else:
                    error_details['both'].append((
                        pred_idx, img_idx, pred_score, pred_box
                    ))
            else:
                # Low IoU → background
                error_details['background'].append((
                    pred_idx, img_idx, pred_score, pred_box
                ))
        
        # Process unmatched GTs (False Negatives)
        for gt_idx in unmatched_gts:
            error_details['missed_gt'].append((
                gt_idx, img_idx, gt_boxes[gt_idx], gt_labels[gt_idx]
            ))
    
    # Store counts
    for key in error_details:
        if key != 'counts':
            error_details['counts'][key] = len(error_details[key])
    
    return error_details


def apply_classification_oracle(predictions, ground_truths, error_details, iou_threshold=0.5):
    """
    Oracle: Fix classification errors by correcting the class label.
    """
    fixed_predictions = deepcopy(predictions)
    
    for pred_idx, img_idx, score, pred_box, gt_box, correct_label in error_details['classification']:
        fixed_predictions[img_idx]['labels'][pred_idx] = correct_label
    
    map_results = compute_map(fixed_predictions, ground_truths, iou_thresholds=[iou_threshold])
    return map_results[f'mAP_{int(iou_threshold*100)}']


def apply_localization_oracle(predictions, ground_truths, error_details, iou_threshold=0.5):
    """
    Oracle: Fix localization errors by moving box to GT location.
    Suppress duplicates if they would be created.
    """
    fixed_predictions = deepcopy(predictions)
    
    # Track which GTs we've already fixed to avoid duplicates
    used_gts = set()
    to_suppress = set()
    
    for pred_idx, img_idx, score, pred_box, gt_box, gt_label in error_details['localization']:
        gt_key = (img_idx, tuple(gt_box))
        
        if gt_key not in used_gts:
            # Fix localization
            fixed_predictions[img_idx]['boxes'][pred_idx] = gt_box
            used_gts.add(gt_key)
        else:
            # Would create duplicate, suppress this detection
            to_suppress.add((img_idx, pred_idx))
    
    # Suppress duplicates
    for img_idx, pred_idx in to_suppress:
        fixed_predictions[img_idx]['scores'][pred_idx] = -1
    
    # Filter out suppressed detections
    for img_idx in range(len(fixed_predictions)):
        if 'scores' in fixed_predictions[img_idx]:
            keep_mask = fixed_predictions[img_idx]['scores'] >= 0
            for key in ['boxes', 'labels', 'scores']:
                if key in fixed_predictions[img_idx]:
                    fixed_predictions[img_idx][key] = fixed_predictions[img_idx][key][keep_mask]
    
    map_results = compute_map(fixed_predictions, ground_truths, iou_thresholds=[iou_threshold])
    return map_results[f'mAP_{int(iou_threshold*100)}']


def apply_both_oracle(predictions, ground_truths, error_details, iou_threshold=0.5):
    """
    Oracle: Suppress detections that have both classification AND localization errors.
    """
    fixed_predictions = deepcopy(predictions)
    
    for pred_idx, img_idx, score, pred_box in error_details['both']:
        fixed_predictions[img_idx]['scores'][pred_idx] = -1
    
    # Filter out suppressed detections
    for img_idx in range(len(fixed_predictions)):
        if 'scores' in fixed_predictions[img_idx]:
            keep_mask = fixed_predictions[img_idx]['scores'] >= 0
            for key in ['boxes', 'labels', 'scores']:
                if key in fixed_predictions[img_idx]:
                    fixed_predictions[img_idx][key] = fixed_predictions[img_idx][key][keep_mask]
    
    map_results = compute_map(fixed_predictions, ground_truths, iou_thresholds=[iou_threshold])
    return map_results[f'mAP_{int(iou_threshold*100)}']


def apply_duplicate_oracle(predictions, ground_truths, error_details, iou_threshold=0.5):
    """
    Oracle: Suppress duplicate detections.
    """
    fixed_predictions = deepcopy(predictions)
    
    for pred_idx, img_idx, score, pred_box in error_details['duplicate']:
        fixed_predictions[img_idx]['scores'][pred_idx] = -1
    
    # Filter out suppressed detections
    for img_idx in range(len(fixed_predictions)):
        if 'scores' in fixed_predictions[img_idx]:
            keep_mask = fixed_predictions[img_idx]['scores'] >= 0
            for key in ['boxes', 'labels', 'scores']:
                if key in fixed_predictions[img_idx]:
                    fixed_predictions[img_idx][key] = fixed_predictions[img_idx][key][keep_mask]
    
    map_results = compute_map(fixed_predictions, ground_truths, iou_thresholds=[iou_threshold])
    return map_results[f'mAP_{int(iou_threshold*100)}']


def apply_background_oracle(predictions, ground_truths, error_details, iou_threshold=0.5):
    """
    Oracle: Suppress background false positives.
    """
    fixed_predictions = deepcopy(predictions)
    
    for pred_idx, img_idx, score, pred_box in error_details['background']:
        fixed_predictions[img_idx]['scores'][pred_idx] = -1
    
    # Filter out suppressed detections
    for img_idx in range(len(fixed_predictions)):
        if 'scores' in fixed_predictions[img_idx]:
            keep_mask = fixed_predictions[img_idx]['scores'] >= 0
            for key in ['boxes', 'labels', 'scores']:
                if key in fixed_predictions[img_idx]:
                    fixed_predictions[img_idx][key] = fixed_predictions[img_idx][key][keep_mask]
    
    map_results = compute_map(fixed_predictions, ground_truths, iou_thresholds=[iou_threshold])
    return map_results[f'mAP_{int(iou_threshold*100)}']


def apply_missed_gt_oracle(predictions, ground_truths, error_details, iou_threshold=0.5):
    """
    Oracle: Add perfect detections for missed ground truths.
    """
    fixed_predictions = deepcopy(predictions)
    
    for gt_idx, img_idx, gt_box, gt_label in error_details['missed_gt']:
        # Add a perfect detection for this missed GT
        if 'boxes' not in fixed_predictions[img_idx]:
            fixed_predictions[img_idx]['boxes'] = []
            fixed_predictions[img_idx]['labels'] = []
            fixed_predictions[img_idx]['scores'] = []
        
        # Append perfect detection (score=1.0)
        if isinstance(fixed_predictions[img_idx]['boxes'], list):
            fixed_predictions[img_idx]['boxes'].append(gt_box)
            fixed_predictions[img_idx]['labels'].append(gt_label)
            fixed_predictions[img_idx]['scores'].append(1.0)
        else:
            # Convert to list, add, then convert back
            fixed_predictions[img_idx]['boxes'] = fixed_predictions[img_idx]['boxes'].tolist()
            fixed_predictions[img_idx]['labels'] = fixed_predictions[img_idx]['labels'].tolist()
            fixed_predictions[img_idx]['scores'] = fixed_predictions[img_idx]['scores'].tolist()
            fixed_predictions[img_idx]['boxes'].append(gt_box)
            fixed_predictions[img_idx]['labels'].append(gt_label)
            fixed_predictions[img_idx]['scores'].append(1.0)
    
    # Convert lists back to numpy arrays
    for img_idx in range(len(fixed_predictions)):
        for key in ['boxes', 'labels', 'scores']:
            if key in fixed_predictions[img_idx] and isinstance(fixed_predictions[img_idx][key], list):
                fixed_predictions[img_idx][key] = np.array(fixed_predictions[img_idx][key])
    
    map_results = compute_map(fixed_predictions, ground_truths, iou_thresholds=[iou_threshold])
    return map_results[f'mAP_{int(iou_threshold*100)}']


def compute_tide_error_impacts(predictions, ground_truths, iou_threshold=0.5, bg_threshold=0.1):
    """
    Compute TIDE-style error impacts using ΔAP for each error type.
    
    This is the main function that quantifies how much each error type
    is hurting the model's performance.
    
    Args:
        predictions: list of dicts with 'boxes', 'scores', 'labels'
        ground_truths: list of dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for positive matches (default: 0.5)
        bg_threshold: IoU threshold for background vs. foreground (default: 0.1)
    
    Returns:
        dict containing:
            - error_impacts: ΔAP for each error type
            - error_counts: Raw counts for reference
            - original_ap: Original model AP
            - max_possible_ap: AP after fixing all errors (should be 100)
            - error_percentages: Relative contribution of each error
    """
    print("\n" + "="*60)

    # Step 1: Classify all errors
    print("\n📋 Classifying errors...")
    error_details = classify_errors_detailed(predictions, ground_truths, iou_threshold, bg_threshold)
    
    print(f"   True Positives: {error_details['counts']['tp']}")
    print(f"   Classification errors: {error_details['counts']['classification']}")
    print(f"   Localization errors: {error_details['counts']['localization']}")
    print(f"   Both errors: {error_details['counts']['both']}")
    print(f"   Duplicate errors: {error_details['counts']['duplicate']}")
    print(f"   Background errors: {error_details['counts']['background']}")
    print(f"   Missed GT: {error_details['counts']['missed_gt']}")
    
    # Step 2: Compute original AP
    print("\n📊 Computing original AP...")
    map_results = compute_map(predictions, ground_truths, iou_thresholds=[iou_threshold])
    original_ap = map_results[f'mAP_{int(iou_threshold*100)}']
    print(f"   Original AP@{int(iou_threshold*100)}: {original_ap:.3f}")
    
    # Step 3: Apply each oracle independently and compute ΔAP
    print("\n🔮 Applying oracles and computing ΔAP...")
    
    error_impacts = {}
    
    # Classification oracle
    ap_fixed = apply_classification_oracle(predictions, ground_truths, error_details, iou_threshold)
    error_impacts['classification'] = ap_fixed - original_ap
    print(f"   Classification oracle: AP = {ap_fixed:.3f} (Δ = +{error_impacts['classification']:.3f})")
    
    # Localization oracle
    ap_fixed = apply_localization_oracle(predictions, ground_truths, error_details, iou_threshold)
    error_impacts['localization'] = ap_fixed - original_ap
    print(f"   Localization oracle:   AP = {ap_fixed:.3f} (Δ = +{error_impacts['localization']:.3f})")
    
    # Both oracle
    ap_fixed = apply_both_oracle(predictions, ground_truths, error_details, iou_threshold)
    error_impacts['both'] = ap_fixed - original_ap
    print(f"   Both oracle:           AP = {ap_fixed:.3f} (Δ = +{error_impacts['both']:.3f})")
    
    # Duplicate oracle
    ap_fixed = apply_duplicate_oracle(predictions, ground_truths, error_details, iou_threshold)
    error_impacts['duplicate'] = ap_fixed - original_ap
    print(f"   Duplicate oracle:      AP = {ap_fixed:.3f} (Δ = +{error_impacts['duplicate']:.3f})")
    
    # Background oracle
    ap_fixed = apply_background_oracle(predictions, ground_truths, error_details, iou_threshold)
    error_impacts['background'] = ap_fixed - original_ap
    print(f"   Background oracle:     AP = {ap_fixed:.3f} (Δ = +{error_impacts['background']:.3f})")
    
    # Missed GT oracle
    ap_fixed = apply_missed_gt_oracle(predictions, ground_truths, error_details, iou_threshold)
    error_impacts['missed_gt'] = ap_fixed - original_ap
    print(f"   Missed GT oracle:      AP = {ap_fixed:.3f} (Δ = +{error_impacts['missed_gt']:.3f})")
    
    # Step 4: Calculate error percentages (relative contribution)
    total_impact = sum(error_impacts.values())
    error_percentages = {
        k: (v / total_impact * 100) if total_impact > 0 else 0 
        for k, v in error_impacts.items()
    }
    
    # Step 5: Verify all oracles together achieve 100 AP (optional - computationally expensive)
    # Uncomment if you want to verify
    # print("\n✅ Verifying all oracles together...")
    # all_fixed = apply_all_oracles_together(predictions, ground_truths, error_details, iou_threshold)
    # print(f"   All oracles applied: AP = {all_fixed:.3f} (expected 100)")
    
    return {
        'error_impacts': error_impacts,
        'error_counts': error_details['counts'],
        'error_percentages': error_percentages,
        'original_ap': original_ap,
        'total_impact': total_impact,
        'max_possible_ap': original_ap + total_impact,
    }

def evaluate_tide(
    predictions,
    ground_truths,
    *,
    class_names=None,
    gmm_type="1D",
    probabilistic=False,
    output_dir=None,
    iou_threshold=0.5,
    bg_threshold=0.1,
    **kwargs,
):
    """Run TIDE-style evaluation and return JSON-serializable results."""
    use_soft_labels = gmm_type != "HARD_LABEL"
    return compute_tide_error_impacts(
        predictions,
        ground_truths,
        iou_threshold=iou_threshold,
        bg_threshold=bg_threshold,
        # use_soft_labels=use_soft_labels,
    )

def print_tide_summary(tide_results):
    """
    Pretty print TIDE analysis results.
    """
    print("\n" + "="*60)
    print("TIDE ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n📊 Baseline Performance:")
    print(f"   Original AP@{int(tide_results.get('iou_threshold', 50))}: {tide_results['original_ap']:.3f}")
    print(f"   Maximum achievable AP: {tide_results['max_possible_ap']:.3f}")
    print(f"   Total improvement potential: +{tide_results['total_impact']:.3f}")
    
    print(f"\n🔍 Error Impacts (ΔAP):")
    sorted_errors = sorted(tide_results['error_impacts'].items(), key=lambda x: x[1], reverse=True)
    for error_type, impact in sorted_errors:
        percentage = tide_results['error_percentages'][error_type]
        bar = "█" * int(percentage / 2)
        print(f"   {error_type:15s}: +{impact:.3f} AP  ({percentage:5.1f}%) {bar}")
    
    print(f"\n📈 Raw Error Counts:")
    for error_type, count in tide_results['error_counts'].items():
        if error_type != 'tp':
            print(f"   {error_type:15s}: {count}")
        else:
            print(f"   {error_type:15s}: {count} (correct detections)")