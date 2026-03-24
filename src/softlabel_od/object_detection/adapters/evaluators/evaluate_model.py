"""
evaluate_model.py

Standalone evaluation script for Hard Label and 1D Soft Label models.
Produces comprehensive metrics with a fixed, deterministic validation set.

Usage:
    # Evaluate soft label (1D) model
    python evaluate_model.py \
        --weights path/to/model.pth \
        --gmm_type 1D \
        --label_file path/to/soft_labels.json

    # Evaluate hard label model
    python evaluate_model.py \
        --weights path/to/model.pth \
        --gmm_type HARD_LABEL \
        --label_file path/to/hard_labels.json
"""

import argparse
import os
import json
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm

from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_map,
    compute_expected_iou,
    compute_calibration_error,
    compute_bbox_uncertainty_calibration,
    compute_soft_detection_metrics,
    compute_class_distribution_metrics,
    match_predictions_to_gt,
)
from softlabel_od.object_detection.utils.config import get_cfg_defaults
cfg = get_cfg_defaults()

CLASS_MAPPING = {
    0: 'No Finding', 1: 'Aortic enlargement', 2: 'Atelectasis',
    3: 'Calcification', 4: 'Cardiomegaly', 5: 'Consolidation',
    6: 'ILD', 7: 'Infiltration', 8: 'Lung Opacity', 9: 'Nodule/Mass',
    10: 'Other lesion', 11: 'Pleural effusion', 12: 'Pleural thickening',
    13: 'Pneumothorax', 14: 'Pulmonary fibrosis',
}


def load_model(weights_path, num_classes, backbone, probabilistic, device):
    """Load model from checkpoint."""
    model = get_model_1d(
        num_classes=num_classes,
        backbone_name=backbone,
        pretrained=False,
        probabilistic=probabilistic,
    )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter out unexpected keys (e.g. class_weights buffer from fine-tune runs)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"Note: {len(missing)} keys not in checkpoint (expected for buffers added post-training)")

    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, data_loader, device, is_probabilistic):
    """Run inference on the full validation set and collect predictions + GT."""
    all_predictions = []
    all_ground_truths = []

    pbar = tqdm(data_loader, desc="Running inference")
    for images, targets in pbar:
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for pred, target, img in zip(outputs, targets, images):
            img_h, img_w = img.shape[1], img.shape[2]

            pred_entry = {
                'boxes': pred['boxes'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy(),
                'scores': pred['scores'].cpu().numpy(),
            }
            if 'bbox_stds' in pred:
                pred_entry['bbox_stds'] = pred['bbox_stds'].cpu().numpy()
                pred_entry['bbox_means'] = pred['boxes'].cpu().numpy()
            if 'class_probs' in pred:
                probs_np = pred['class_probs'].cpu().numpy()        # [N, 15]
                pred_entry['class_probs'] = probs_np
                # Class confidence for ECE: P(predicted class) from the full distribution
                pred_labels_np = pred['labels'].cpu().numpy()       # [N]
                pred_entry['class_scores'] = probs_np[np.arange(len(pred_labels_np)), pred_labels_np]         # [N]

            gt_entry = {
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy(),
                'image_size': (img_h, img_w),
            }
            if 'gt_class_probs' in target:
                gt_entry['gt_class_probs'] = target['gt_class_probs'].cpu().numpy()
            if 'gt_stds' in target:
                gt_entry['gt_stds'] = target['gt_stds'].cpu().numpy()

            all_predictions.append(pred_entry)
            all_ground_truths.append(gt_entry)

    return all_predictions, all_ground_truths


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


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Device: {device}")

    # ── Determine dataset / model type ──
    is_hard = args.gmm_type == 'HARD_LABEL'
    probabilistic = not is_hard
    label_mode = "hard" if is_hard else f"soft_{args.gmm_type.lower()}"

    # ── Build validation dataset (deterministic split via random_state=42) ──
    print(f"\nCreating {label_mode} validation dataset...")
    dataset_cls = get_dataset_class(args.dataset)
    val_dataset = dataset_cls(
        root=args.data_root,
        label_file=args.label_file,
        transforms=Compose([ToTensor()]),
        train=False,
        test_split_ratio=args.test_split_ratio,
        label_mode=label_mode,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    print(f"Validation set: {len(val_dataset)} images")

    # ── Load model ──
    print(f"\nLoading model from: {args.weights}")
    model = load_model(args.weights, args.num_classes, args.backbone, probabilistic, device)

    # ── Collect predictions ──
    predictions, ground_truths = collect_predictions(model, val_loader, device, probabilistic)
    print(f"Collected predictions for {len(predictions)} images")

    # ══════════════════════════════════════════════════════════════════════
    # METRICS
    # ══════════════════════════════════════════════════════════════════════
    results = {}

    # ── 1. Detection summary (precision / recall / F1) ──
    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    det_summary = compute_detection_summary(predictions, ground_truths)
    results['detection_summary'] = det_summary
    print(f"  GT boxes:         {det_summary['total_gt_boxes']}")
    print(f"  Predicted boxes:  {det_summary['total_pred_boxes']}")
    print(f"  Matched (TP):     {det_summary['total_matched']}")
    print(f"  Missed GT (FN):   {det_summary['total_missed_gt']}")
    print(f"  False Pos (FP):   {det_summary['total_false_positives']}")
    print(f"  Precision:        {det_summary['precision']:.4f}")
    print(f"  Recall:           {det_summary['recall']:.4f}")
    print(f"  F1 Score:         {det_summary['f1_score']:.4f}")

    # ── 2. IoU statistics ──
    print("\n" + "=" * 70)
    print("IoU STATISTICS (matched predictions)")
    print("=" * 70)
    iou_stats = compute_iou_stats(predictions, ground_truths)
    results['iou_stats'] = iou_stats
    print(f"  Avg IoU:          {iou_stats['avg_iou']:.4f}")
    print(f"  Median IoU:       {iou_stats['median_iou']:.4f}")
    print(f"  Std IoU:          {iou_stats['std_iou']:.4f}")
    print(f"  Num matches:      {iou_stats['num_matches']}")

    # ── 3. mAP ──
    print("\n" + "=" * 70)
    print("MEAN AVERAGE PRECISION")
    print("=" * 70)
    map_results = compute_map(predictions, ground_truths, iou_thresholds=[0.5, 0.75])
    results['mAP'] = {k: float(v) if not isinstance(v, dict) else v
                      for k, v in map_results.items()}
    print(f"  mAP:              {map_results['mAP']:.4f}")
    print(f"  mAP@50:           {map_results['mAP_50']:.4f}")
    print(f"  mAP@75:           {map_results['mAP_75']:.4f}")
    print(f"\n  Per-class AP@50:")
    for cls_idx, ap in sorted(map_results.get('per_class_ap', {}).items()):
        cls_name = CLASS_MAPPING.get(int(cls_idx), f'Class {cls_idx}')
        print(f"    {cls_name:25s}: {float(ap):.4f}")

    # ── 4. Calibration (ECE) ──
    print("\n" + "=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    ece = compute_calibration_error(predictions, ground_truths)
    results['ece'] = float(ece)
    print(f"  ECE:              {ece:.4f}")

    # ── 5. Soft-label-specific metrics ──
    if probabilistic:
        print("\n" + "=" * 70)
        print("SOFT LABEL METRICS")
        print("=" * 70)

        soft_metrics = compute_soft_detection_metrics(predictions, ground_truths)
        results['soft_detection'] = {k: float(v) for k, v in soft_metrics.items()}
        print(f"  Top-1 Accuracy:   {soft_metrics['top1_accuracy']:.4f}")
        print(f"  Top-3 Accuracy:   {soft_metrics['top3_accuracy']:.4f}")
        print(f"  Weighted Acc:     {soft_metrics['weighted_accuracy']:.4f}")
        print(f"  Total matched:    {soft_metrics['total_matched']}")

        dist_metrics = compute_class_distribution_metrics(predictions, ground_truths)
        results['distribution'] = {k: float(v) if not isinstance(v, (list, dict)) else v
                                   for k, v in dist_metrics.items()}
        print(f"  JS Divergence:    {dist_metrics['js_divergence']:.4f}")
        print(f"  KL Divergence:    {dist_metrics['kl_divergence']:.4f}")
        print(f"  Cosine Sim:       {dist_metrics['cosine_similarity']:.4f}")

        # ── 6. Bbox uncertainty metrics ──
        print("\n" + "=" * 70)
        print("BBOX UNCERTAINTY METRICS")
        print("=" * 70)

        expected_iou = compute_expected_iou(predictions, ground_truths, num_samples=100)
        results['expected_iou'] = float(expected_iou)
        print(f"  Expected IoU:     {expected_iou:.4f}")

        bbox_cal = compute_bbox_uncertainty_calibration(predictions, ground_truths)
        results['bbox_calibration'] = {k: float(v) for k, v in bbox_cal.items()}
        print(f"  Std-Error Corr:   {bbox_cal['std_error_correlation']:.4f}")
        print(f"  Avg Pred Std:     {bbox_cal['avg_pred_std']:.4f}")
        print(f"  Avg Actual Err:   {bbox_cal['avg_actual_error']:.4f}")
        print(f"  Calibration Rat:  {bbox_cal['calibration_ratio']:.4f}")

    # ── Save results ──
    print("\n" + "=" * 70)
    if args.output:
        output_path = args.output
    else:
        model_name = os.path.splitext(os.path.basename(args.weights))[0]
        output_path = f"eval_results_{model_name}.json"

    # Convert numpy types for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {str(k): to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"Results saved to: {output_path}")
    print("=" * 70)

data_root = cfg.DATA.ROOT
evaluation_results_dir = cfg.MISC.EVALUATION_RESULTS_DIR
1d_gaussian_softlabels_dir = cfg.DATA.SOFT_LABEL_FILE_1D
2d_gaussian_softlabels_dir = cfg.DATA.SOFT_LABEL_FILE_2D
hard_labels_dir = cfg.DATA.HARD_LABEL_FILE

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate object detection model")

    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model .pth file')
    parser.add_argument('--gmm_type', type=str, required=True,
                        choices=['1D', 'HARD_LABEL'],
                        help='Model type: 1D (soft label) or HARD_LABEL')
    parser.add_argument('--label_file', type=str, default=None,
                        help='Path to label JSON (auto-selected if not provided)')
    parser.add_argument('--data_root', type=str,
                        default=data_root,
                        help='Dataset root directory')
    parser.add_argument('--dataset', type=str, default=cfg.DATA.DATASET,
                        help='Registered dataset adapter name')
    parser.add_argument('--output', type=str, default=evaluation_results_dir,
                        help='Output JSON path (auto-named if not provided)')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--test_split_ratio', type=float, default=0.20)
    parser.add_argument('--cpu', action='store_true', help='Force CPU')

    args = parser.parse_args()

    # Auto-select label file based on gmm_type
    if args.label_file is None:
        if args.gmm_type == '1D':
            args.label_file = 1d_gaussian_softlabels_dir
        else:
            args.label_file = hard_labels_dir
        print(f"Auto-selected label file: {args.label_file}")

    evaluate(args)