"""
CLI entry point: softlabel-evaluate
"""

import argparse
import json
import os

import numpy as np
import torch

from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.core.evaluation import evaluate_default_metrics, evaluate_tide, evaluate_pdq, evaluate_lrp
from softlabel_od.object_detection.core.evaluation.runner import load_model, collect_predictions
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.utils.config import get_cfg_from_file

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

# Evalaution metrics Registry
METRIC_RUNNERS = {
    "default": evaluate_default_metrics,
    "tide": evaluate_tide,
    "pdq": evaluate_pdq,
    "lrp": evaluate_lrp,
}

NO_FINDING_CLASS_ID = 1

PRED_ALIGNED_FIELDS = (
    "boxes",
    "labels",
    "scores",
    "bbox_stds",
    "bbox_means",
    "class_probs",
    "class_scores",
)

GT_ALIGNED_FIELDS = (
    "boxes",
    "labels",
    "gt_class_probs",
    "gt_stds",
)


def _filter_entry_rows(entry, keep_mask, aligned_fields):
    """Filter row-aligned arrays in a prediction/GT dict while preserving metadata."""
    filtered = dict(entry)
    keep_mask = np.asarray(keep_mask, dtype=bool)

    for key in aligned_fields:
        if key not in filtered:
            continue

        value = filtered[key]
        if value is None:
            continue

        arr = np.asarray(value)
        if arr.ndim == 0:
            continue

        if arr.shape[0] == keep_mask.shape[0]:
            filtered[key] = arr[keep_mask]

    return filtered


def apply_no_finding_eval_policy(
    predictions,
    ground_truths,
    *,
    policy: str,
    no_finding_class_id: int = NO_FINDING_CLASS_ID,
):
    """
    Eval-only No-Finding handling.

    policy="keep": leave predictions/GT unchanged.
    policy="drop": remove class-1 No-Finding rows from both predictions and GT.
    """
    if policy == "keep":
        return predictions, ground_truths

    if policy != "drop":
        raise ValueError(f"Unsupported no-finding eval policy: {policy}")

    filtered_predictions = []
    filtered_ground_truths = []

    for pred, gt in zip(predictions, ground_truths):
        pred_labels = np.asarray(pred.get("labels", []))
        gt_labels = np.asarray(gt.get("labels", []))

        pred_keep = pred_labels != no_finding_class_id
        gt_keep = gt_labels != no_finding_class_id

        filtered_predictions.append(
            _filter_entry_rows(pred, pred_keep, PRED_ALIGNED_FIELDS)
        )
        filtered_ground_truths.append(
            _filter_entry_rows(gt, gt_keep, GT_ALIGNED_FIELDS)
        )

    return filtered_predictions, filtered_ground_truths

def resolve_metric_types(metric_types: list[str]) -> list[str]:
    """Expand 'all', validate names, and deduplicate while preserving order."""
    if not metric_types:
        return ["default"]

    if "all" in metric_types:
        if len(metric_types) > 1:
            raise ValueError(
                "'all' cannot be combined with other metric types. "
                f"Got: {metric_types}"
            )
        return list(METRIC_RUNNERS.keys())

    resolved = []
    for name in metric_types:
        if name not in METRIC_RUNNERS:
            raise ValueError(
                f"Metric '{name}' is not implemented. "
                f"Available metrics: {sorted(METRIC_RUNNERS)}"
            )
        if name not in resolved:
            resolved.append(name)
    return resolved

def evaluate(args):
    cfg = get_cfg_from_file(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    is_hard = args.gmm_type == "HARD_LABEL"
    probabilistic = not is_hard
    label_mode = "hard" if is_hard else f"soft_{args.gmm_type.lower()}"

    print(f"\nCreating {label_mode} test dataset...")
    dataset_cls = get_dataset_class(args.dataset)
    test_dataset = dataset_cls(
        root=args.data_root,
        label_file=args.label_file,
        transforms=Compose([ToTensor()]),
        split="test",
        test_split_ratio=args.test_split_ratio,
        val_split_ratio=args.val_split_ratio,
        label_mode=label_mode,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
<<<<<<< HEAD
    print(f"Validation set: {len(val_dataset)} images")

    # Handle class_names for both original dataset and Subset wrapper
    if hasattr(val_dataset, 'class_names'):
        class_names = val_dataset.class_names
    else:
        # Subset wrapper - access underlying dataset
        class_names = val_dataset.dataset.class_names

    architecture = getattr(cfg.MODEL, 'ARCHITECTURE', None) or None
    print(f"\nLoading model from: {args.weights}")
    print(f"Architecture: {architecture or args.gmm_type}")
    model = load_model(weights_path=args.weights,
    num_classes=args.num_classes,
    model_type=args.gmm_type,
    backbone=args.backbone,
    probabilistic=probabilistic,
    device=device,
    anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
    anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS*len(cfg.MODEL.ANCHOR_SIZES),
    architecture=architecture)

    predictions, ground_truths = collect_predictions(model, val_loader, device=device, is_probabilistic=probabilistic)
    print(f"Collected predictions for {len(predictions)} images")

    selected_metrics = (
        list(METRIC_RUNNERS.keys())
        if args.metric_type == "all"
        else [args.metric_type]
=======
    print(f"Test set: {len(test_dataset)} images")
    class_names = test_dataset.class_names
    model = load_model(
        weights_path=args.weights,
        num_classes=args.num_classes,
        model_type=args.gmm_type,
        backbone=args.backbone,
        probabilistic=probabilistic,
        device=device,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS,
>>>>>>> origin/main
    )

    predictions, ground_truths = collect_predictions(model, test_loader, device=device, is_probabilistic=probabilistic)
    print(f"Collected predictions for {len(predictions)} images")
    predictions, ground_truths = apply_no_finding_eval_policy(
        predictions,
        ground_truths,
        policy=args.no_finding_eval_policy,
    )
    if args.no_finding_eval_policy == "drop":
        print("No-Finding eval policy: dropped class_id=1 from predictions and ground truth")
    else:
        print("No-Finding eval policy: kept class_id=1 in predictions and ground truth")

    selected_metrics = resolve_metric_types(args.metric_type)

    results = {}

    for metric_name in selected_metrics:
        if metric_name not in METRIC_RUNNERS:
            raise ValueError(
                f"Metric '{metric_name}' is not implemented yet. "
                f"Available metrics: {sorted(METRIC_RUNNERS)}"
            )

        metric_runner = METRIC_RUNNERS[metric_name]
        results[metric_name] = metric_runner(
            predictions,
            ground_truths,
            class_names=class_names,
            gmm_type=args.gmm_type,
            probabilistic=probabilistic,
            output_dir=args.output,
            label_file=args.label_file,
            priors_file=args.priors_file,
        )

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained detection model")
    parser.add_argument("--config", type=str, default=None, help="Model initialization config file (.yaml)")
    parser.add_argument("--eval-tag", type=str, default=None, help="Evaluation tag for the results")
    parser.add_argument("--metric-type", type=str, nargs="+", default=["default"], choices=["default", "tide", "pdq", "lrp", "all"], help="Which evaluation metric group to run.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--gmm-type", type=str, default="1D", choices=["1D", "HARD_LABEL"])
    parser.add_argument("--dataset", type=str, default="vindrcxr", help="Registered dataset adapter name")
    parser.add_argument("--label-file", type=str, default=None, help="Path to label file (.json)")
    parser.add_argument("--priors-file", type=str, default=None, help="Path to priors file (.json)")
    parser.add_argument("--data-root", type=str, default="/media/klee_ro/public_datasets/vindr_cxr")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "outputs/vindrcxr/evaluation_results"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=16)
    parser.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet101", "mobilenet", "swin_v2_b", "swin_v2_t"])
    parser.add_argument("--test-split-ratio", type=float, default=0.20)
    parser.add_argument("--val-split-ratio", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers for dataloader (0 = single-process)")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--no-finding-eval-policy",
        type=str,
        default="keep",
        choices=["keep", "drop"],
        help=(
            "Eval-only handling for class_id=1 No Finding. "
            "'keep' leaves predictions/GT unchanged; "
            "'drop' removes No-Finding rows from both predictions and GT."
        ),
    )
    args = parser.parse_args()

    cfg = get_cfg_from_file(args.config)
    
    # Handle default values for num_workers
    if args.num_workers is None:
        if args.metric_type in ["pdq"]:
            args.num_workers = 1
        else:
            args.num_workers = 4

    # Handle default values for label file
    if args.label_file is None:
        if args.gmm_type == "1D":
            args.label_file = os.path.join(PROJECT_ROOT, "outputs/vindrcxr/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json")
        elif args.gmm_type == "HARD_LABEL":
            args.label_file = os.path.join(PROJECT_ROOT, "outputs/vindrcxr/hard_labels/HardLabels_MajorityVote_iou0_5.json")
        else:
            raise ValueError(f"No label file provided for gmm type: {args.gmm_type}")
        print(f"Auto-selected label file: {args.label_file}")
    
    # Handle default values for config file
    if args.config is None:
        if args.gmm_type == "1D":
            args.config = os.path.join(PROJECT_ROOT, "configs/vindrcxr_1d_softlabel_config_albers.yaml")
        elif args.gmm_type == "HARD_LABEL":
            args.config = os.path.join(PROJECT_ROOT, "configs/vindrcxr_hardlabel_config_albers.yaml")
        else:
            raise ValueError(f"No config file provided for gmm type: {args.gmm_type}")

    # Handle default values for eval tag
    if args.eval_tag is None:
        args.eval_tag = "1D_Gaussian_softlabels_iou0_5"
    

    # Handle default values for priors file
    if args.priors_file is None:
        if args.gmm_type == "1D":
            args.priors_file = os.path.join(PROJECT_ROOT, "outputs/vindrcxr/1d_gaussian_softlabels/priors/class_priors_1d_iou0_5.json")
        elif args.gmm_type == "HARD_LABEL":
            pass
    
    # Handle default values for test and val split ratio
    if args.test_split_ratio is None:
        args.test_split_ratio = cfg.DATA.TEST_SPLIT_RATIO
    if args.val_split_ratio is None:
        args.val_split_ratio = cfg.DATA.VAL_SPLIT_RATIO

    evaluate(args)



if __name__ == "__main__":
    main()