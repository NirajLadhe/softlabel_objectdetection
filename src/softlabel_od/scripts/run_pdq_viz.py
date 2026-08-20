"""
CLI driver for PDQ visualization.

Loads the model, runs inference on the validation set, then writes three
grid images to --output using pdq_visualization.py:

    best_pdq.png    -- top-N images by PDQ score
    worst_pdq.png   -- images with TPs but lowest PDQ score
    failures.png    -- images with zero TPs (pure FP / FN)

Usage:
    python run_pdq_viz.py \
        --config   configs/config_lvis.yaml \
        --weights  outputs/saved_models/.../best_model.pth \
        --data-root /media/klee_ro/public_datasets/coco_lvis/val2017 \
        --label-file outputs/lvis/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json \
        --num-classes 825 \
        --output outputs/lvis/pdq_viz \
        --top-n 9
"""

import argparse
import os
import sys

import torch

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))

from softlabel_od.object_detection.utils.config import get_cfg_from_file
from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.core.evaluation.runner import load_model, collect_predictions
from softlabel_od.object_detection.core.evaluation.pdq_visualization import (
    collect_pdq_image_results,
    render_pdq_grid,
)


def build_image_paths(image_ids, data_root):
    """Map val image IDs to absolute .jpg paths."""
    paths = []
    for iid in image_ids:
        num = str(iid).replace("lvis_", "")
        try:
            fname = f"{int(num):012d}.jpg"
        except ValueError:
            fname = f"{num}.jpg"
        full = os.path.join(data_root, fname)
        paths.append(full if os.path.exists(full) else None)
    return paths


def main():
    parser = argparse.ArgumentParser(description="PDQ Visualization")
    parser.add_argument("--config",            required=True,  help="Path to YAML config")
    parser.add_argument("--weights",           required=True,  help="Path to _best_model.pth")
    parser.add_argument("--data-root",         required=True,  help="Val image dir (ends in /val2017)")
    parser.add_argument("--label-file",        required=True,  help="All-classes soft-label JSON")
    parser.add_argument("--num-classes",       type=int, default=825)
    parser.add_argument("--output",            default=os.path.join(PROJECT_ROOT,
                                               "outputs/lvis/1d_gaussian_softlabels/pdq_viz"))
    parser.add_argument("--top-n",             type=int, default=9,
                                               help="Images per grid panel")
    parser.add_argument("--backbone",          default="resnet50")
    parser.add_argument("--test-split-ratio",  type=float, default=0.20)
    parser.add_argument("--heatmap-threshold", type=float, default=0.01,
                                               help="Minimum heatmap prob to render")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Config ────────────────────────────────────────────────────────────────
    cfg = get_cfg_from_file(args.config)
    architecture = getattr(cfg.MODEL, "ARCHITECTURE", None) or None

    # ── Validation dataset ────────────────────────────────────────────────────
    print("Building validation dataset...")
    dataset_cls = get_dataset_class(cfg.DATA.DATASET)
    val_dataset = dataset_cls(
        root=args.data_root,
        label_file=args.label_file,
        transforms=Compose([ToTensor()]),
        train=False,
        test_split_ratio=args.test_split_ratio,
        label_mode="soft_1d",
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=1,
    )
    print(f"Validation images: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model from: {args.weights}")
    model = load_model(
        weights_path=args.weights,
        num_classes=args.num_classes,
        model_type="1D",
        backbone=args.backbone,
        probabilistic=True,
        device=device,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS * len(cfg.MODEL.ANCHOR_SIZES),
        architecture=architecture,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    print("Running inference on validation set...")
    predictions, ground_truths = collect_predictions(
        model, val_loader, device=device, is_probabilistic=True
    )
    print(f"Collected predictions for {len(predictions)} images")

    # ── Image paths (parallel to predictions) ─────────────────────────────────
    image_ids = (val_dataset.image_ids if hasattr(val_dataset, "image_ids")
                 else val_dataset.dataset.image_ids)
    image_paths = build_image_paths(image_ids, args.data_root)
    missing = sum(1 for p in image_paths if p is None)
    if missing:
        print(f"Warning: {missing} image paths not found on disk")

    # ── Class name map: index -> display name ─────────────────────────────────
    raw_names = (val_dataset.class_names if hasattr(val_dataset, "class_names")
                 else val_dataset.dataset.class_names)
    class_names = {i + 1: name for i, name in enumerate(raw_names)}

    # ── PDQ per-image evaluation ───────────────────────────────────────────────
    print("Computing per-image PDQ results...")
    results = collect_pdq_image_results(
        predictions,
        ground_truths,
        image_paths=image_paths,
        heatmap_threshold=args.heatmap_threshold,
    )

    # ── Render grids ──────────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    print(f"Rendering PDQ grids (top_n={args.top_n}) → {args.output}")
    render_pdq_grid(
        results,
        output_dir=args.output,
        class_names=class_names,
        top_n=args.top_n,
    )

    print("\nDone. Files written:")
    for fname in ("best_pdq.png", "worst_pdq.png", "failures.png"):
        path = os.path.join(args.output, fname)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1e6
            print(f"  {path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
