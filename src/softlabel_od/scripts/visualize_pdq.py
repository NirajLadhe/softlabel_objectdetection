"""
visualize_pdq.py — CLI entry point for PDQ visualisation

Generates grid panels showing:
  - Best PDQ predictions (high spatial + label quality)
  - Worst PDQ predictions (low quality TP matches)
  - Complete failures (pure FP / FN images)

Example usage:
    python visualize_pdq.py \\
        --weights    outputs/models/best_model.pth \\
        --config     configs/vindrcxr_1d_softlabel_config.yaml \\
        --label-file outputs/vindrcxr/.../softLabels.json \\
        --data-root  /data/vindr_cxr \\
        --image-dir  /data/vindr_cxr \\
        --output     outputs/pdq_visualizations \\
        --top-n 9 --n-cols 3
"""

import argparse
import os

import torch
import pickle

from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.core.evaluation.pdq_visualize import (
    collect_pdq_image_results,
    render_pdq_grid,
)
from softlabel_od.object_detection.core.evaluation.runner import load_model, collect_predictions
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.utils.config import get_cfg_from_file
from softlabel_od.scripts.evaluate import apply_no_finding_eval_policy



def main():
    parser = argparse.ArgumentParser(
        description="Visualize PDQ evaluation results with Gaussian heatmaps"
    )
    parser.add_argument("--weights",    type=str, required=True,
                        help="Path to model weights (.pth)")
    parser.add_argument("--config",     type=str, required=True,
                        help="Model initialisation config file (.yaml)")
    parser.add_argument("--label-file", type=str, required=True,
                        help="Soft-label JSON file with class_probs and bbox_stds")
    parser.add_argument("--output",     type=str, required=True,
                        help="Directory to save visualisation PNGs")
    parser.add_argument("--image-dir",  type=str, default='/media/klee_ro/public_datasets/vindr_cxr/train',
                        help="Root directory of raw images; subdirs are searched automatically")
    parser.add_argument("--dataset",    type=str, default="vindrcxr",
                        help="Registered dataset adapter name")
    parser.add_argument("--data-root",  type=str, default='/media/klee_ro/public_datasets/vindr_cxr',
                        help="Dataset root directory")
    parser.add_argument("--gmm-type",   type=str, default="1D",
                        choices=["1D"],
                        help="GMM type (only 1D supported — PDQ requires probabilistic outputs)")
    parser.add_argument("--num-classes",     type=int, default=16)
    parser.add_argument("--backbone",        type=str, default="resnet50")
    parser.add_argument("--batch-size",      type=int, default=4)
    parser.add_argument("--num-workers",     type=int, default=4)
    parser.add_argument("--test-split-ratio",type=float, default=0.20)
    parser.add_argument("--top-n",           type=int, default=9,
                        help="Number of best/worst images per grid")
    parser.add_argument("--n-cols",          type=int, default=3,
                        help="Number of columns in the output grid")
    parser.add_argument("--heatmap-alpha",   type=float, default=0.45,
                        help="Transparency of the Gaussian heatmap overlay [0,1]")
    parser.add_argument("--cache-file", type=str, default='/home/haaroon/softlabel_object_detection/outputs/vindrcxr/pdq_viz/s-r50-iou/pdq.pkl',
                        help="Path to a .pkl file to save/load cached PDQ results")
    parser.add_argument("--cpu",             action="store_true")
    parser.add_argument(
        "--no-finding-eval-policy",
        type=str,
        default="drop",
        choices=["keep", "drop"],
        help=(
            "Eval-only handling for class_id=1 No Finding. "
            "'keep' leaves predictions/GT unchanged; "
            "'drop' removes No-Finding rows from both predictions and GT."
        ),
    )
    args = parser.parse_args()

    if args.cache_file and os.path.exists(args.cache_file):
        print(f"\nLoading cached PDQ results from: {args.cache_file}")
        with open(args.cache_file, "rb") as f:
            cache_data = pickle.load(f)
            results = cache_data["results"]
            class_names = cache_data["class_names"]
        print(f"Loaded {len(results)} cached images.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        print(f"Device: {device}")
        # --- Load config and dataset ---
        cfg = get_cfg_from_file(args.config)
        label_mode = f"soft_{args.gmm_type.lower()}"
        print(f"\nCreating {label_mode} validation dataset...")
        dataset_cls = get_dataset_class(args.dataset)
        
        val_dataset = dataset_cls(
            root=args.data_root,
            label_file=args.label_file,
            transforms=Compose([ToTensor()]),
            split="test",
            test_split_ratio=args.test_split_ratio,
            val_split_ratio=cfg.DATA.VAL_SPLIT_RATIO,
            label_mode=label_mode,
        )
        print(f"Test set: {len(val_dataset)} images")
        class_names = val_dataset.class_names
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
        )
        # --- Load model and collect predictions ---
        print(f"\nLoading model from: {args.weights}")
        model = load_model(
            weights_path=args.weights,
            num_classes=args.num_classes,
            model_type=args.gmm_type,
            backbone=args.backbone,
            probabilistic=True,
            device=device,
            anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
            anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS * len(cfg.MODEL.ANCHOR_SIZES),
        )
        print("\nCollecting predictions...")
        predictions, ground_truths = collect_predictions(
            model, val_loader, device=device, is_probabilistic=True
        )
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
        # --- Resolve image paths (optional) ---
        image_paths = _build_image_index(args.image_dir, ground_truths) if args.image_dir else None
        # --- Collect per-image PDQ results ---
        print("\nComputing per-image PDQ statistics...")
        results = collect_pdq_image_results(
            predictions, ground_truths,
            image_paths=image_paths,
        )
        print(f"Processed {len(results)} images")
        # --- Save to cache if requested ---
        if args.cache_file:
            print(f"\nSaving PDQ results to cache: {args.cache_file}")
            os.makedirs(os.path.dirname(os.path.abspath(args.cache_file)), exist_ok=True)
            with open(args.cache_file, "wb") as f:
                pickle.dump({
                    "results": results,
                    "class_names": class_names
                }, f)
    # --- Render and save grids ---
    print(f"\nRendering visualisation grids → {args.output}")
    render_pdq_grid(
        results,
        output_dir=args.output,
        class_names=class_names,
        n_cols=args.n_cols,
        top_n=args.top_n,
        heatmap_alpha=args.heatmap_alpha,
    )
    print("\nDone.")


def _build_image_index(
    image_dir: str, ground_truths: list,
) -> list:
    """Build a filename index and resolve image paths for each ground truth.

    Searches the root directory and all immediate subdirectories (e.g.
    ``train/``, ``test/``) for image files.
    """
    print(f"\nBuilding image index from: {image_dir}")
    image_index: dict = {}
    ext_set = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

    search_dirs = [image_dir] + [
        os.path.join(image_dir, d)
        for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d))
    ]
    for search_dir in search_dirs:
        try:
            for fname in os.listdir(search_dir):
                stem, ext = os.path.splitext(fname)
                if ext in ext_set:
                    image_index[stem] = os.path.join(search_dir, fname)
        except PermissionError:
            pass
    print(f"  Indexed {len(image_index)} images across {len(search_dirs)} directories")

    image_paths = [
        image_index.get(str(gt.get("image_id", "")), None)
        for gt in ground_truths
    ]
    found_count = sum(1 for p in image_paths if p is not None)
    print(f"  Resolved {found_count}/{len(image_paths)} image paths")
    return image_paths


if __name__ == "__main__":
    main()
