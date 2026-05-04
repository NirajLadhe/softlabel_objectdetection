"""
CLI entry point: softlabel-evaluate
"""

import argparse
import json
import os

import numpy as np
import torch

from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_map,
    compute_expected_iou,
    compute_calibration_error,
    compute_bbox_uncertainty_calibration,
    compute_soft_detection_metrics,
    compute_class_distribution_metrics,
    compute_detection_summary,
    compute_iou_stats,
)
from softlabel_od.object_detection.core.evaluation.runner import load_model, collect_predictions
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.utils.config import get_cfg_from_file

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))



def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    cfg = get_cfg_from_file(args.config)
    is_hard = args.gmm_type == "HARD_LABEL"
    probabilistic = not is_hard
    label_mode = "hard" if is_hard else f"soft_{args.gmm_type.lower()}"

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

    # class_names comes from the dataset adapter — no hard-coded mapping needed
    class_names = val_dataset.class_names

    print(f"\nLoading model from: {args.weights}")
    model = load_model(weights_path=args.weights, 
    num_classes=args.num_classes, 
    model_type=args.gmm_type,
    backbone=args.backbone, 
    probabilistic=probabilistic, 
    device=device,
    anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
    anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS*len(cfg.MODEL.ANCHOR_SIZES))

    predictions, ground_truths = collect_predictions(model, val_loader, device=device, is_probabilistic=probabilistic)
    print(f"Collected predictions for {len(predictions)} images")

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
    map_results = compute_map(predictions, ground_truths, iou_thresholds=[0.4,0.5, 0.75])
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

    print("\n" + "=" * 70)
    
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

    output_dir= args.output
    os.makedirs(output_dir, exist_ok=True)
    output_json = os.path.join(output_dir, f"eval_results_{args.eval_tag}.json")


    with open(output_json, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    print(f"Results saved to: {output_json}")
    print("=" * 70)


label_file = os.path.join(PROJECT_ROOT, "outputs/vindrcxr/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained detection model")
    parser.add_argument("--config", type=str, default=os.path.join(PROJECT_ROOT, "configs/1d_softlabel_config_albers.yaml"), required=True)
    parser.add_argument("--eval-tag", type=str, default="1D_Gaussian_softlabels_iou0_5", required=True)
    parser.add_argument("--weights", type=str, default=weights, required=True)
    parser.add_argument("--gmm-type", type=str, default="1D", required=True, choices=["1D", "HARD_LABEL"])
    parser.add_argument("--dataset", type=str, default="vindrcxr", help="Registered dataset adapter name")
    parser.add_argument("--label-file", type=str, default=label_file)
    parser.add_argument("--data-root", type=str, default="/media/klee_ro/public_datasets/vindr_cxr")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "outputs/evaluation_results"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--test-split-ratio", type=float, default=0.20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.weights is None:
        args.weights = weights
    if args.label_file is None:
        args.label_file = label_file
    
    if args.label_file is None:
        if args.gmm_type == "1D":
            args.label_file = label_file
        else:
            args.label_file = label_file
        print(f"Auto-selected label file: {args.label_file}")

    evaluate(args)


if __name__ == "__main__":
    main()