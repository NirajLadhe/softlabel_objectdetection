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

    # Apply no-finding filtering/sampling for evaluation
    all_indices = list(range(len(val_dataset)))
    valid_indices = all_indices
    
    if cfg.DATA.IGNORE_NOFINDING_IMAGES:
        print("Filtering pure No-Finding images (healthy CXR) from validation set...")
        valid_indices = [
            idx for idx in all_indices
            if val_dataset.has_pathology_clusters(val_dataset.image_ids[idx])
        ]
    elif cfg.DATA.NOFINDING_INCLUDE_FRACTION < 1.0:
        fraction = cfg.DATA.NOFINDING_INCLUDE_FRACTION
        print(f"Sampling {fraction*100:.1f}% of pure No-Finding images for validation...")
        
        pathology_indices = []
        nofinding_indices = []
        for idx in all_indices:
            if val_dataset.has_pathology_clusters(val_dataset.image_ids[idx]):
                pathology_indices.append(idx)
            else:
                nofinding_indices.append(idx)
        
        if nofinding_indices:
            np.random.seed(42)
            n_nofinding_to_keep = int(len(nofinding_indices) * fraction)
            sampled_nofinding = np.random.choice(
                nofinding_indices, 
                size=n_nofinding_to_keep, 
                replace=False
            ).tolist()
            valid_indices = pathology_indices + sampled_nofinding
            valid_indices.sort()
        else:
            valid_indices = pathology_indices
        
        print(f"  Validation: {len(pathology_indices)} pathology + {len(sampled_nofinding if nofinding_indices else [])} no-finding = {len(valid_indices)} total")
    
    if len(valid_indices) < len(all_indices):
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, valid_indices)

    # Filter out pure no-finding images if configured
    if cfg.DATA.IGNORE_NOFINDING_IMAGES:
        print("Filtering pure No-Finding images (healthy CXR) from validation set...")
        valid_indices = [
            idx for idx in range(len(val_dataset))
            if val_dataset.has_pathology_clusters(val_dataset.image_ids[idx])
        ]
        kept = len(valid_indices)
        total = len(val_dataset)
        print(f"  Kept {kept}/{total} images ({kept/total*100:.1f}% have pathology).")
        from torch.utils.data import Subset
        val_dataset = Subset(val_dataset, valid_indices)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    print(f"Validation set: {len(val_dataset)} images")

    # Handle class_names for both original dataset and Subset wrapper
    if hasattr(val_dataset, 'class_names'):
        class_names = val_dataset.class_names
    else:
        # Subset wrapper - access underlying dataset
        class_names = val_dataset.dataset.class_names

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

    selected_metrics = (
        list(METRIC_RUNNERS.keys())
        if args.metric_type == "all"
        else [args.metric_type]
    )

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
    parser.add_argument("--metric-type", type=str, default="default", choices=["default", "tide", "pdq", "lrp", "all"], help="Which evaluation metric group to run.")
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights (.pth)")
    parser.add_argument("--gmm-type", type=str, default="1D", choices=["1D", "HARD_LABEL"])
    parser.add_argument("--dataset", type=str, default="vindrcxr", help="Registered dataset adapter name")
    parser.add_argument("--label-file", type=str, default=None, help="Path to label file (.json)")
    parser.add_argument("--priors-file", type=str, default=None, help="Path to priors file (.json)")
    parser.add_argument("--data-root", type=str, default="/media/klee_ro/public_datasets/vindr_cxr")
    parser.add_argument("--output", type=str, default=os.path.join(PROJECT_ROOT, "outputs/vindrcxr/evaluation_results"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=15)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--test-split-ratio", type=float, default=0.20)
    parser.add_argument("--num-workers", type=int, default=None, help="Number of workers for dataloader (0 = single-process)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.num_workers is None:
        if args.metric_type in ["pdq"]:
            args.num_workers = 1
        else:
            args.num_workers = 4
    if args.label_file is None:
        if args.gmm_type == "1D":
            args.label_file = os.path.join(PROJECT_ROOT, "outputs/vindrcxr/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json")
        else:
            raise ValueError(f"No label file provided for gmm type: {args.gmm_type}")
        print(f"Auto-selected label file: {args.label_file}")
    if args.config is None:
        args.config = os.path.join(PROJECT_ROOT, "configs/vindrcxr_1d_softlabel_config_albers.yaml")
    if args.eval_tag is None:
        args.eval_tag = "1D_Gaussian_softlabels_iou0_5"
    
    if args.priors_file is None:
        if args.gmm_type == "1D":
            args.priors_file = os.path.join(PROJECT_ROOT, "outputs/vindrcxr/1d_gaussian_softlabels/priors/class_priors_1d_iou0_5.json")
        else:
            raise ValueError(f"No priors file provided for gmm type: {args.gmm_type}")
        print(f"Auto-selected priors file: {args.priors_file}")
    

    evaluate(args)



if __name__ == "__main__":
    main()