"""
evaluation_metrics.py - Comprehensive probabilistic evaluation metrics
This file contains the functions to compute the evaluation metrics for the probabilistic model.

"""

import numpy as np
import torch
from scipy import stats
from scipy.special import erf
from sklearn.metrics import average_precision_score
import warnings


def compute_iou_batch(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: [N, 4] array of boxes [x1, y1, x2, y2]
        boxes2: [M, 4] array of boxes [x1, y1, x2, y2]
    
    Returns:
        ious: [N, M] array of IoU values
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1[:, None, :]  # [N, 1, 4]
    boxes2 = boxes2[None, :, :]  # [1, M, 4]
    
    # Intersection coordinates
    inter_xmin = np.maximum(boxes1[..., 0], boxes2[..., 0])
    inter_ymin = np.maximum(boxes1[..., 1], boxes2[..., 1])
    inter_xmax = np.minimum(boxes1[..., 2], boxes2[..., 2])
    inter_ymax = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    # Intersection area
    inter_width = np.maximum(0, inter_xmax - inter_xmin)
    inter_height = np.maximum(0, inter_ymax - inter_ymin)
    inter_area = inter_width * inter_height
    
    # Box areas
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Union area
    union_area = boxes1_area + boxes2_area - inter_area
    
    # IoU
    iou = inter_area / np.maximum(union_area, 1e-10)
    
    # Handle the case where inputs might be 1D
    if iou.ndim == 2:
        return iou
    else:
        return iou.reshape(len(boxes1), len(boxes2))


def match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Match predictions to ground truth boxes using Hungarian matching.
    
    Returns:
        matches: list of (pred_idx, gt_idx, iou) tuples
        unmatched_preds: list of unmatched prediction indices
        unmatched_gts: list of unmatched GT indices
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
    
    # Compute IoU matrix
    iou_matrix = compute_iou_batch(pred_boxes, gt_boxes)
    
    # Greedy matching (sorted by confidence)
    matches = []
    matched_gt = set()
    
    # Sort predictions by score
    sorted_indices = np.argsort(-pred_scores)
    
    for pred_idx in sorted_indices:
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx in matched_gt:
                continue
            
            iou = iou_matrix[pred_idx, gt_idx]
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
    
    # Find unmatched
    matched_preds = {m[0] for m in matches}
    matched_gts_set = {m[1] for m in matches}
    
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gts_set]
    
    return matches, unmatched_preds, unmatched_gts


def compute_map(predictions, ground_truths, iou_thresholds=[0.4, 0.5, 0.75]):
    """
    Compute mean Average Precision at specified IoU thresholds.
    
    Args:
        predictions: list of dicts with 'boxes', 'scores', 'labels'
        ground_truths: list of dicts with 'boxes', 'labels'
        iou_thresholds: list of IoU thresholds to evaluate
    
    Returns:
        dict with mAP metrics
    """
    all_classes = set()
    for gt in ground_truths:
        all_classes.update(gt['labels'].tolist())
    
    ap_per_class = {cls: [] for cls in all_classes}
    
    for iou_thresh in iou_thresholds:
        for cls in all_classes:
            cls_preds = []
            cls_gts = []
            
            for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                pred_mask = pred['labels'] == cls
                if pred_mask.sum() > 0:
                    for box, score in zip(pred['boxes'][pred_mask], pred['scores'][pred_mask]):
                        cls_preds.append({
                            'image_id': img_idx,
                            'box': box,
                            'score': score
                        })
                
                gt_mask = gt['labels'] == cls
                for box in gt['boxes'][gt_mask]:
                    cls_gts.append({
                        'image_id': img_idx,
                        'box': box,
                        'matched': False
                    })
            
            if len(cls_preds) == 0 or len(cls_gts) == 0:
                ap_per_class[cls].append(0.0)
                continue
            
            cls_preds = sorted(cls_preds, key=lambda x: x['score'], reverse=True)
            
            tp = []
            fp = []
            
            for pred in cls_preds:
                img_gts = [g for g in cls_gts if g['image_id'] == pred['image_id'] and not g['matched']]
                
                if len(img_gts) == 0:
                    fp.append(1)
                    tp.append(0)
                    continue
                
                ious = [compute_iou_batch(pred['box'][None], g['box'][None])[0] 
                       for g in img_gts]
                best_iou_idx = np.argmax(ious)
                best_iou = ious[best_iou_idx]
                
                if best_iou >= iou_thresh:
                    img_gts[best_iou_idx]['matched'] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            
            # === CHANGED: interpolated PR curve, anchored at (0,1) ===
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)

            recalls    = np.concatenate([[0.0], tp_cumsum / len(cls_gts)])
            precisions = np.concatenate([[1.0], tp_cumsum / (tp_cumsum + fp_cumsum)])

            # Precision envelope: replace each point with max precision to its right
            for i in range(len(precisions) - 2, -1, -1):
                precisions[i] = max(precisions[i], precisions[i + 1])

            # Area under the enveloped curve
            ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
            # === END CHANGE ===

            ap_per_class[cls].append(ap)
    
    results = {}
    for i, thresh in enumerate(iou_thresholds):
        aps = [ap_per_class[cls][i] for cls in all_classes]
        results[f'mAP_{int(thresh*100)}'] = np.mean(aps)
    
    results['mAP'] = np.mean([results[f'mAP_{int(t*100)}'] for t in iou_thresholds])
    results['per_class_ap'] = {cls: np.mean(aps) for cls, aps in ap_per_class.items()}
    
    return results


def compute_expected_iou(predictions, ground_truths, num_samples=100, iou_threshold=0.5):
    """
    Compute Expected IoU by sampling from predicted uncertainty distributions.
    
    Only considers matched predictions (true positives) so the metric
    measures variance calibration quality, not detector precision.
    
    The predicted stds are in box-coder delta space [std_dx, std_dy, std_dw, std_dh]
    but the predicted means (boxes) are in pixel space [x1, y1, x2, y2].
    We convert stds to pixel space before sampling.
    """
    expected_ious = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        if 'bbox_stds' not in pred:
            continue
        
        pred_stds_delta = pred['bbox_stds']  # delta space: [std_dx, std_dy, std_dw, std_dh]
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Only evaluate on matched (true positive) predictions
        matches, _, _ = match_predictions_to_gt(
            pred_boxes, pred_scores, gt_boxes, iou_threshold
        )
        
        if len(matches) == 0:
            continue
        
        for pred_idx, gt_idx, det_iou in matches:
            pred_mean = pred_boxes[pred_idx]      # pixel space [x1, y1, x2, y2]
            std_delta = pred_stds_delta[pred_idx]  # delta space [std_dx, std_dy, std_dw, std_dh]
            gt_box = gt_boxes[gt_idx]
            
            # --- Convert delta-space stds to pixel-space stds ---
            # Approximate using the predicted box dimensions
            pred_w = max(pred_mean[2] - pred_mean[0], 1.0)
            pred_h = max(pred_mean[3] - pred_mean[1], 1.0)
            
            # Delta-to-pixel conversion:
            #   dx = (cx - prop_cx) / prop_w  →  sigma_cx ≈ std_dx * pred_w
            #   dy = (cy - prop_cy) / prop_h  →  sigma_cy ≈ std_dy * pred_h
            #   dw = log(w / prop_w)           →  sigma_w  ≈ std_dw * pred_w
            #   dh = log(h / prop_h)           →  sigma_h  ≈ std_dh * pred_h
            sigma_cx = std_delta[0] * pred_w
            sigma_cy = std_delta[1] * pred_h
            sigma_w  = std_delta[2] * pred_w
            sigma_h  = std_delta[3] * pred_h
            
            # Convert center-format stds to corner-format [x1, y1, x2, y2]
            # x1 = cx - w/2  →  std(x1) = sqrt(std(cx)^2 + (std(w)/2)^2)
            # y1 = cy - h/2  →  std(y1) = sqrt(std(cy)^2 + (std(h)/2)^2)
            # x2 = cx + w/2  →  std(x2) = same as std(x1) by symmetry
            # y2 = cy + h/2  →  std(y2) = same as std(y1)
            pixel_std = np.array([
                np.sqrt(sigma_cx**2 + 0.25 * sigma_w**2),  # std_x1
                np.sqrt(sigma_cy**2 + 0.25 * sigma_h**2),  # std_y1
                np.sqrt(sigma_cx**2 + 0.25 * sigma_w**2),  # std_x2
                np.sqrt(sigma_cy**2 + 0.25 * sigma_h**2),  # std_y2
            ])
            
            # Sample boxes from the Gaussian and compute IoU with matched GT
            sampled_boxes = np.random.normal(
                pred_mean, pixel_std, size=(num_samples, 4)
            )
            
            # Ensure valid boxes (x2 > x1, y2 > y1)
            sampled_boxes[:, 2] = np.maximum(sampled_boxes[:, 2], sampled_boxes[:, 0] + 1)
            sampled_boxes[:, 3] = np.maximum(sampled_boxes[:, 3], sampled_boxes[:, 1] + 1)
            
            # Compute IoU of each sample with the matched GT box
            ious = compute_iou_batch(sampled_boxes, gt_box[None])  # [num_samples, 1]
            expected_ious.append(np.mean(ious))
    
    return np.mean(expected_ious) if len(expected_ious) > 0 else 0.0


def compute_calibration_error(predictions, ground_truths, num_bins=10):
    """
    Compute Expected Calibration Error (ECE).
    
    Measures: Do 70%-confident predictions succeed 70% of the time?
    """
    confidences = []
    correctness = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_scores = pred['scores']
        pred_labels = pred['labels']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']
        
        if len(pred_boxes) == 0:
            continue
        
        # Match predictions to GTs
        matches, _, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes)
        
        # For each prediction
        class_scores = pred.get('class_scores', pred_scores)
        for i, (box, score, label) in enumerate(zip(pred_boxes, class_scores, pred_labels)):
            confidences.append(score)
            
            # Check if this prediction was correct
            matched = any(m[0] == i for m in matches)
            if matched:
                match = [m for m in matches if m[0] == i][0]
                gt_idx = match[1]
                correct = (label == gt_labels[gt_idx])
            else:
                correct = False
            
            correctness.append(float(correct))
    
    if len(confidences) == 0:
        return 0.0
    
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    # Bin by confidence
    bins = np.linspace(0, 1, num_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    
    for i in range(num_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if mask.sum() > 0:
            bin_acc = correctness[mask].mean()
            bin_conf = confidences[mask].mean()
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            bin_counts.append(mask.sum())
    
    # Weighted ECE
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    bin_counts = np.array(bin_counts)
    
    ece = np.sum(bin_counts * np.abs(bin_accs - bin_confs)) / np.sum(bin_counts)
    
    return ece


def compute_soft_class_metrics(predictions, ground_truths):
    """
    Compute metrics for soft classification labels.
    
    Requires 'class_probs' in predictions and 'gt_class_probs' in ground_truths.
    """
    kl_divs = []
    js_divs = []
    l1_dists = []
    
    for pred, gt in zip(predictions, ground_truths):
        if 'class_probs' not in pred or 'gt_class_probs' not in gt:
            continue
        
        pred_probs = pred['class_probs']
        gt_probs = gt['gt_class_probs']
        
        # Match by class assignment
        for p_prob, g_prob in zip(pred_probs, gt_probs):
            # KL divergence: KL(GT || Pred)
            p_prob = np.clip(p_prob, 1e-10, 1.0)
            g_prob = np.clip(g_prob, 1e-10, 1.0)
            
            kl = np.sum(g_prob * np.log(g_prob / p_prob))
            kl_divs.append(kl)
            
            # JS divergence (symmetric)
            m = 0.5 * (p_prob + g_prob)
            js = 0.5 * (np.sum(g_prob * np.log(g_prob / m)) + 
                       np.sum(p_prob * np.log(p_prob / m)))
            js_divs.append(js)
            
            # L1 distance
            l1 = np.sum(np.abs(p_prob - g_prob))
            l1_dists.append(l1)
    
    return {
        'kl_div': np.mean(kl_divs) if len(kl_divs) > 0 else 0.0,
        'js_div': np.mean(js_divs) if len(js_divs) > 0 else 0.0,
        'l1_dist': np.mean(l1_dists) if len(l1_dists) > 0 else 0.0,
        'soft_acc': 1.0 - (np.mean(l1_dists) / 2.0) if len(l1_dists) > 0 else 0.0
    }


def compute_uncertainty_weighted_map(predictions, ground_truths, iou_threshold=0.5, alpha=1.0):
    """
    Compute mAP weighted by uncertainty calibration.
    
    Rewards predictions where GT falls within predicted uncertainty.
    """
    weighted_precisions = []
    
    for pred, gt in zip(predictions, ground_truths):
        if 'bbox_means' not in pred or 'bbox_stds' not in pred:
            pred_means = pred['boxes']
            pred_stds = np.ones_like(pred_means) * 5  # Default uncertainty
        else:
            pred_means = pred['bbox_means']
            pred_stds = pred['bbox_stds']
        
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        if len(pred_means) == 0 or len(gt_boxes) == 0:
            continue
        
        # Match predictions
        matches, _, _ = match_predictions_to_gt(pred_means, pred_scores, gt_boxes, iou_threshold)
        
        for match in matches:
            pred_idx, gt_idx, iou = match
            
            # Compute uncertainty weight
            pred_mean = pred_means[pred_idx]
            pred_std = pred_stds[pred_idx]
            gt_box = gt_boxes[gt_idx]
            
            # Z-score: how many std away is GT?
            z_scores = np.abs(gt_box - pred_mean) / (pred_std + 1e-6)
            avg_z = z_scores.mean()
            
            # Confidence: probability GT falls within distribution
            confidence = np.exp(-0.5 * avg_z ** 2)
            
            # Weighted precision
            weighted_precisions.append(confidence ** alpha)
    
    return np.mean(weighted_precisions) if len(weighted_precisions) > 0 else 0.0

def compute_soft_detection_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute soft label metrics for object detection.
    Matches predictions to GT boxes, then compares class distributions.
    
    Args:
        predictions: list of dicts with 'boxes', 'labels', 'scores'
        ground_truths: list of dicts with 'boxes', 'labels', 'gt_class_probs'
    
    Returns:
        dict with soft_accuracy, top1_accuracy, top3_accuracy, weighted_accuracy
    """
    top1_correct = 0
    top3_correct = 0
    weighted_acc_sum = 0.0
    total_matched = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        
        gt_boxes = gt['boxes']
        gt_labels = gt['labels']  # Hard labels (dominant class)
        gt_class_probs = gt.get('gt_class_probs', None)
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Match predictions to GT
        matches, _, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        
        for match in matches:
            pred_idx, gt_idx, iou = match
            pred_label = pred_labels[pred_idx]
            gt_label = gt_labels[gt_idx]
            
            # Top-1 accuracy (standard)
            if pred_label == gt_label:
                top1_correct += 1
            
            # Weighted/Soft accuracy (if GT soft labels available)
            if gt_class_probs is not None and len(gt_class_probs) > gt_idx:
                gt_probs = gt_class_probs[gt_idx]
                
                # How much probability mass does GT assign to predicted class?
                if pred_label < len(gt_probs):
                    weighted_acc_sum += gt_probs[pred_label]
                
                # Top-3 accuracy: is GT dominant class in top-3 predicted classes?
                # (requires full pred distribution - using pred_label as top-1 for now)
                top3_probs = np.argsort(gt_probs)[-3:][::-1]  # Top 3 GT classes
                if pred_label in top3_probs:
                    top3_correct += 1
            else:
                # Fallback to hard label
                top3_correct += (pred_label == gt_label)
            
            total_matched += 1
    
    return {
        'top1_accuracy': top1_correct / total_matched if total_matched > 0 else 0.0,
        'top3_accuracy': top3_correct / total_matched if total_matched > 0 else 0.0,
        'weighted_accuracy': weighted_acc_sum / total_matched if total_matched > 0 else 0.0,
        'total_matched': total_matched,
    }


def compute_class_distribution_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Compare predicted class scores with GT soft label distributions.
    Requires model to output 'class_probs' (full distribution per detection).
    
    If model only outputs 'scores' (single value), this creates a one-hot proxy.
    """
    kl_divs = []
    js_divs = []
    cosine_sims = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        pred_class_probs = pred.get('class_probs', None)  # Full distribution if available
        
        gt_boxes = gt['boxes']
        gt_class_probs = gt.get('gt_class_probs', None)
        
        if len(pred_boxes) == 0 or len(gt_boxes) == 0 or gt_class_probs is None:
            continue
        
        # Match predictions to GT
        matches, _, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        
        for match in matches:
            pred_idx, gt_idx, iou = match
            
            gt_probs = np.array(gt_class_probs[gt_idx])
            gt_probs = np.clip(gt_probs, 1e-10, 1.0)
            gt_probs = gt_probs / gt_probs.sum()  # Ensure normalized
            
            # Get predicted distribution
            if pred_class_probs is not None and len(pred_class_probs) > pred_idx:
                p_probs = np.array(pred_class_probs[pred_idx])
            else:
                # Create proxy distribution from label and score
                num_classes = len(gt_probs)
                p_probs = np.ones(num_classes) * (1 - pred_scores[pred_idx]) / (num_classes - 1)
                pred_label = pred_labels[pred_idx]
                if pred_label < num_classes:
                    p_probs[pred_label] = pred_scores[pred_idx]
            
            p_probs = np.clip(p_probs, 1e-10, 1.0)
            p_probs = p_probs / p_probs.sum()  # Ensure normalized
            
            # KL divergence: KL(GT || Pred)
            kl = np.sum(gt_probs * np.log(gt_probs / p_probs))
            kl_divs.append(kl)
            
            # JS divergence (symmetric)
            m = 0.5 * (p_probs + gt_probs)
            js = 0.5 * (np.sum(gt_probs * np.log(gt_probs / m)) + 
                       np.sum(p_probs * np.log(p_probs / m)))
            js_divs.append(js)
            
            # Cosine similarity
            cos_sim = np.dot(p_probs, gt_probs) / (np.linalg.norm(p_probs) * np.linalg.norm(gt_probs) + 1e-10)
            cosine_sims.append(cos_sim)
    
    return {
        'kl_divergence': np.mean(kl_divs) if kl_divs else 0.0,
        'js_divergence': np.mean(js_divs) if js_divs else 0.0,
        'cosine_similarity': np.mean(cosine_sims) if cosine_sims else 0.0,
        'num_comparisons': len(kl_divs),
    }

def compute_bbox_uncertainty_calibration(predictions, ground_truths, iou_threshold=0.5):
    """
    Measure if predicted bbox uncertainty correlates with actual localization error.
    
    Good calibration: High predicted std → High actual error
    
    Returns:
        dict with correlation metrics and calibration error
    """
    pred_stds_all = []
    actual_errors_all = []
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred['boxes']
        pred_stds = pred.get('bbox_stds', None)
        pred_scores = pred['scores']
        gt_boxes = gt['boxes']
        
        if pred_stds is None or len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue
        
        # Match predictions to GT
        matches, _, _ = match_predictions_to_gt(pred_boxes, pred_scores, gt_boxes, iou_threshold)
        
        for match in matches:
            pred_idx, gt_idx, iou = match

            pred_box = pred_boxes[pred_idx]       # pixels [x1,y1,x2,y2]
            gt_box   = gt_boxes[gt_idx]           # pixels [x1,y1,x2,y2]
            std_delta = pred_stds[pred_idx]       # delta space [std_dx, std_dy, std_dw, std_dh]

            # Convert delta-space stds to pixel-space (same logic as compute_expected_iou)
            pred_w = max(pred_box[2] - pred_box[0], 1.0)
            pred_h = max(pred_box[3] - pred_box[1], 1.0)

            sigma_cx = std_delta[0] * pred_w
            sigma_cy = std_delta[1] * pred_h
            sigma_w  = std_delta[2] * pred_w
            sigma_h  = std_delta[3] * pred_h

            pixel_std = np.array([
                np.sqrt(sigma_cx**2 + 0.25 * sigma_w**2),  # std_x1
                np.sqrt(sigma_cy**2 + 0.25 * sigma_h**2),  # std_y1
                np.sqrt(sigma_cx**2 + 0.25 * sigma_w**2),  # std_x2
                np.sqrt(sigma_cy**2 + 0.25 * sigma_h**2),  # std_y2
            ])

            actual_error = np.abs(pred_box - gt_box)        # pixels, per coordinate

            pred_stds_all.append(np.mean(pixel_std))
            actual_errors_all.append(np.mean(actual_error))
    
    if len(pred_stds_all) < 2:
        return {
            'std_error_correlation': 0.0,
            'avg_pred_std': 0.0,
            'avg_actual_error': 0.0,
            'calibration_ratio': 0.0,
        }
    
    pred_stds_all = np.array(pred_stds_all)
    actual_errors_all = np.array(actual_errors_all)
    
    # Correlation: should be positive (high std → high error)
    correlation = np.corrcoef(pred_stds_all, actual_errors_all)[0, 1]
    
    # Calibration ratio: pred_std / actual_error (should be ~1.0)
    calibration_ratio = np.mean(pred_stds_all) / (np.mean(actual_errors_all) + 1e-6)
    
    return {
        'std_error_correlation': correlation if not np.isnan(correlation) else 0.0,
        'avg_pred_std': np.mean(pred_stds_all),
        'avg_actual_error': np.mean(actual_errors_all),
        'calibration_ratio': calibration_ratio,
    }


def compute_detection_summary(predictions, ground_truths):
    """Compute per-image detection counts: matches, missed GT, false positives."""
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_missed = 0
    total_fp = 0

    for pred, gt in zip(predictions, ground_truths):
        n_gt = len(gt['boxes'])
        n_pred = len(pred['boxes'])
        total_gt += n_gt
        total_pred += n_pred

        if n_pred == 0 or n_gt == 0:
            total_missed += n_gt
            total_fp += n_pred
            continue

        matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
            pred['boxes'], pred['scores'], gt['boxes'], iou_threshold=0.5
        )
        total_matched += len(matches)
        total_missed += len(unmatched_gts)
        total_fp += len(unmatched_preds)

    recall = total_matched / max(total_gt, 1)
    precision = total_matched / max(total_pred, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        'total_gt_boxes': total_gt,
        'total_pred_boxes': total_pred,
        'total_matched': total_matched,
        'total_missed_gt': total_missed,
        'total_false_positives': total_fp,
        'recall': recall,
        'precision': precision,
        'f1_score': f1,
    }


def compute_iou_stats(predictions, ground_truths):
    """Compute IoU statistics over matched predictions."""
    all_ious = []
    for pred, gt in zip(predictions, ground_truths):
        if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
            continue
        matches, _, _ = match_predictions_to_gt(
            pred['boxes'], pred['scores'], gt['boxes'], iou_threshold=0.5
        )
        for _, _, iou in matches:
            all_ious.append(iou)

    if not all_ious:
        return {'avg_iou': 0.0, 'median_iou': 0.0, 'std_iou': 0.0, 'num_matches': 0}

    ious = np.array(all_ious)
    return {
        'avg_iou': float(np.mean(ious)),
        'median_iou': float(np.median(ious)),
        'std_iou': float(np.std(ious)),
        'num_matches': len(ious),
    }


def evaluate_default_metrics(
    predictions,
    ground_truths,
    *,
    class_names=None,
    gmm_type="1D",
    probabilistic=False,
    **kwargs,
):
    """Run the repository's standard evaluation metrics."""
    class_names = class_names or {}
    results = {}
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    det_summary = compute_detection_summary(predictions, ground_truths)
    results["detection_summary"] = det_summary
    print(f"  GT boxes:         {det_summary['total_gt_boxes']}")
    print(f"  Predicted boxes:  {det_summary['total_pred_boxes']}")
    print(f"  Matched (TP):     {det_summary['total_matched']}")
    print(f"  Missed GT (FN):   {det_summary['total_missed_gt']}")
    print(f"  False Pos (FP):   {det_summary['total_false_positives']}")
    print(f"  Precision:        {det_summary['precision']:.4f}")
    print(f"  Recall:           {det_summary['recall']:.4f}")
    print(f"  F1 Score:         {det_summary['f1_score']:.4f}")
    print("\n" + "=" * 70)
    print("IoU STATISTICS (matched predictions)")
    print("=" * 70)
    iou_stats = compute_iou_stats(predictions, ground_truths)
    results["iou_stats"] = iou_stats
    print(f"  Avg IoU:          {iou_stats['avg_iou']:.4f}")
    print(f"  Median IoU:       {iou_stats['median_iou']:.4f}")
    print(f"  Std IoU:          {iou_stats['std_iou']:.4f}")
    print(f"  Num matches:      {iou_stats['num_matches']}")
    print("\n" + "=" * 70)
    print("MEAN AVERAGE PRECISION")
    print("=" * 70)
    map_results = compute_map(predictions, ground_truths, iou_thresholds=[0.4, 0.5, 0.75])
    results["mAP"] = {k: float(v) if not isinstance(v, dict) else v for k, v in map_results.items()}
    print(f"  mAP:              {map_results['mAP']:.4f}")
    print(f"  mAP@40:           {map_results['mAP_40']:.4f}")
    print(f"  mAP@50:           {map_results['mAP_50']:.4f}")
    print(f"  mAP@75:           {map_results['mAP_75']:.4f}")
    print(f"\n  Per-class AP@50:")
    for cls_idx, ap in sorted(map_results.get("per_class_ap", {}).items()):
        cls_name = class_names.get(int(cls_idx), f"Class {cls_idx}")
        print(f"    {cls_name:25s}: {float(ap):.4f}")
    print("\n" + "=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    ece = compute_calibration_error(predictions, ground_truths)
    results["ece"] = float(ece)
    print(f"  ECE:              {ece:.4f}")
    if probabilistic:
        print("\n" + "=" * 70)
        print("SOFT LABEL METRICS")
        print("=" * 70)
        soft_metrics = compute_soft_detection_metrics(predictions, ground_truths)
        results["soft_detection"] = {k: float(v) for k, v in soft_metrics.items()}
        print(f"  Top-1 Accuracy:   {soft_metrics['top1_accuracy']:.4f}")
        print(f"  Top-3 Accuracy:   {soft_metrics['top3_accuracy']:.4f}")
        print(f"  Weighted Acc:     {soft_metrics['weighted_accuracy']:.4f}")
        print(f"  Total matched:    {soft_metrics['total_matched']}")
        dist_metrics = compute_class_distribution_metrics(predictions, ground_truths)
        results["distribution"] = {
            k: float(v) if not isinstance(v, (list, dict)) else v
            for k, v in dist_metrics.items()
        }
        print(f"  JS Divergence:    {dist_metrics['js_divergence']:.4f}")
        print(f"  KL Divergence:    {dist_metrics['kl_divergence']:.4f}")
        print(f"  Cosine Sim:       {dist_metrics['cosine_similarity']:.4f}")
        print("\n" + "=" * 70)
        print("BBOX UNCERTAINTY METRICS")
        print("=" * 70)
        expected_iou = compute_expected_iou(predictions, ground_truths, num_samples=100)
        results["expected_iou"] = float(expected_iou)
        print(f"  Expected IoU:     {expected_iou:.4f}")
        bbox_cal = compute_bbox_uncertainty_calibration(predictions, ground_truths)
        results["bbox_calibration"] = {k: float(v) for k, v in bbox_cal.items()}
        print(f"  Std-Error Corr:   {bbox_cal['std_error_correlation']:.4f}")
        print(f"  Avg Pred Std:     {bbox_cal['avg_pred_std']:.4f}")
        print(f"  Avg Actual Err:   {bbox_cal['avg_actual_error']:.4f}")
        print(f"  Calibration Rat:  {bbox_cal['calibration_ratio']:.4f}")
    return results