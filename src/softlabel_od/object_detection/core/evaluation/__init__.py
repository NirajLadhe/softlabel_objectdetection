"""
Evaluation metrics — single authoritative module for all detection metrics.

Probabilistic metrics (require model to output 'bbox_stds' / 'class_probs'):
    compute_expected_iou
    compute_bbox_uncertainty_calibration
    compute_soft_detection_metrics
    compute_class_distribution_metrics

Standard detection metrics:
    compute_map
    compute_calibration_error
    match_predictions_to_gt
    compute_iou_batch
"""

from softlabel_od.object_detection.core.evaluation.metrics import (
    compute_iou_batch,
    match_predictions_to_gt,
    compute_map,
    compute_calibration_error,
    compute_expected_iou,
    compute_soft_detection_metrics,
    compute_class_distribution_metrics,
    compute_bbox_uncertainty_calibration,
    compute_uncertainty_weighted_map,
    compute_soft_class_metrics,
    evaluate_default_metrics,
)
