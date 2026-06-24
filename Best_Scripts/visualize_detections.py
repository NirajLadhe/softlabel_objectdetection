"""
visualize_detections.py
Draws ground-truth and model predictions on NuCLS test images.

Usage:
    python visualize_detections.py \
        --config configs/nucls_1d_softlabel_config.yaml \
        --weights outputs/saved_models/.../best_model.pth \
        --data-root /media/klee_ro/public_datasets/nucls \
        --label-file outputs/nucls_transform_0.001/1d_gaussian_softlabels/1DGMM_softLabels_iou0_25_with_priors.json \
        --output-dir outputs/visualizations/detections \
        --num-images 6 \
        --score-thresh 0.15
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from PIL import Image

# Add project src to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "src"))

from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.core.evaluation.runner import load_model, collect_predictions
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor
from softlabel_od.object_detection.utils.config import get_cfg_from_file

# ── class metadata ─────────────────────────────────────────────────────────────
CLASS_NAMES = {
    2:  "tumor",      3:  "mitotic",    4:  "fibroblast",
    5:  "vasc_end",   6:  "macrophage", 7:  "lymphocyte",
    8:  "plasma",     9:  "neutrophil", 10: "eosinophil",
    11: "myoepi",     12: "apoptotic",  13: "unlabeled",
}

# Distinct colours per class (hex without #)
CLASS_COLORS = {
    2:  "#E74C3C",   # tumor — red
    3:  "#8E44AD",   # mitotic — purple
    4:  "#2ECC71",   # fibroblast — green
    5:  "#F39C12",   # vascular — amber
    6:  "#1ABC9C",   # macrophage — teal
    7:  "#3498DB",   # lymphocyte — blue
    8:  "#E67E22",   # plasma — orange
    9:  "#9B59B6",   # neutrophil — violet
    10: "#16A085",   # eosinophil — dark teal
    11: "#D35400",   # myoepi — dark orange
    12: "#F1C40F",   # apoptotic — yellow
    13: "#95A5A6",   # unlabeled — grey
}


def draw_boxes(ax, boxes, labels, scores=None, class_names=CLASS_NAMES,
               class_colors=CLASS_COLORS, linewidth=1.5, alpha=0.85,
               show_scores=True, score_thresh=0.0):
    """Draw bounding boxes on a matplotlib axis."""
    for i, (box, label) in enumerate(zip(boxes, labels)):
        score = scores[i] if scores is not None else None
        if score is not None and score < score_thresh:
            continue
        x1, y1, x2, y2 = box
        color = class_colors.get(label, "#FFFFFF")
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color,
            facecolor="none", alpha=alpha
        )
        ax.add_patch(rect)
        name = class_names.get(label, str(label))
        label_str = f"{name}" if score is None else f"{name} {score:.2f}"
        ax.text(x1, y1 - 2, label_str,
                color="white", fontsize=5.5,
                bbox=dict(facecolor=color, alpha=0.75, pad=1, linewidth=0),
                va="bottom")


def visualize_image(image_path, gt_boxes, gt_labels, pred_boxes, pred_labels,
                    pred_scores, output_path, score_thresh=0.15):
    """Side-by-side: original | GT boxes | predicted boxes."""
    img = np.array(Image.open(image_path).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#1A1A2E")

    titles = ["Original", "Ground Truth", f"Predictions (thresh={score_thresh})"]
    for ax, title in zip(axes, titles):
        ax.imshow(img)
        ax.set_title(title, color="white", fontsize=11, pad=6)
        ax.axis("off")
        ax.set_facecolor("#1A1A2E")

    # GT boxes (green outlines)
    draw_boxes(axes[1], gt_boxes, gt_labels, scores=None,
               linewidth=1.8, alpha=0.9, show_scores=False)

    # Predicted boxes (colour per class + score)
    draw_boxes(axes[2], pred_boxes, pred_labels, scores=pred_scores,
               linewidth=1.5, alpha=0.85, score_thresh=score_thresh)

    # Stats annotation
    n_gt = len(gt_boxes)
    n_pred = sum(1 for s in pred_scores if s >= score_thresh)
    axes[1].set_title(f"Ground Truth ({n_gt} boxes)", color="white", fontsize=11, pad=6)
    axes[2].set_title(f"Predictions ({n_pred} boxes, thresh={score_thresh})",
                      color="white", fontsize=11, pad=6)

    # Legend
    legend_patches = [
        patches.Patch(color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        for c in sorted(CLASS_NAMES.keys())
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=6,
               fontsize=7.5, framealpha=0.3,
               labelcolor="white", facecolor="#1A1A2E",
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()


def make_gallery(image_paths, output_path, title="Detection Gallery"):
    """Combine individual visualizations into a single gallery PNG."""
    from PIL import Image as PILImage
    images = [PILImage.open(p) for p in image_paths if os.path.exists(p)]
    if not images:
        return
    w, h = images[0].size
    cols = 2
    rows = (len(images) + cols - 1) // cols
    gallery = PILImage.new("RGB", (cols * w, rows * h), (26, 26, 46))
    for i, img in enumerate(images):
        row, col = divmod(i, cols)
        gallery.paste(img, (col * w, row * h))
    gallery.save(output_path)
    print(f"Gallery saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--weights",     required=True)
    parser.add_argument("--data-root",   default="/media/klee_ro/public_datasets/nucls")
    parser.add_argument("--label-file",  required=True)
    parser.add_argument("--output-dir",  default="outputs/visualizations/detections")
    parser.add_argument("--num-images",  type=int, default=6)
    parser.add_argument("--score-thresh",type=float, default=0.12)
    parser.add_argument("--test-split-ratio", type=float, default=0.10)
    parser.add_argument("--val-split-ratio",  type=float, default=0.10)
    parser.add_argument("--backbone",    default="resnet50")
    parser.add_argument("--num-classes", type=int, default=14)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = get_cfg_from_file(args.config)

    # Dataset
    dataset_cls = get_dataset_class("nucls")
    test_dataset = dataset_cls(
        root=args.data_root,
        label_file=args.label_file,
        transforms=Compose([ToTensor()]),
        split="test",
        test_split_ratio=args.test_split_ratio,
        val_split_ratio=args.val_split_ratio,
        label_mode="soft_1d",
    )
    print(f"Test set: {len(test_dataset)} images")

    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0
    )

    # Model
    model = load_model(
        weights_path=args.weights,
        num_classes=args.num_classes,
        model_type="1D",
        backbone=args.backbone,
        probabilistic=True,
        device=device,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS * len(cfg.MODEL.ANCHOR_SIZES),
    )

    # Inference
    predictions, ground_truths = collect_predictions(
        model, loader, device=device, is_probabilistic=True
    )

    # Resolve image paths
    image_ids = test_dataset.image_ids[:args.num_images]
    individual_paths = []

    for i, image_id in enumerate(image_ids):
        image_path = test_dataset.resolve_image_path(image_id)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        pred = predictions[i]
        gt   = ground_truths[i]

        out_path = os.path.join(args.output_dir, f"detection_{i:02d}_{image_id[:20]}.png")
        visualize_image(
            image_path=image_path,
            gt_boxes=gt["boxes"],
            gt_labels=gt["labels"],
            pred_boxes=pred["boxes"],
            pred_labels=pred["labels"],
            pred_scores=pred["scores"],
            output_path=out_path,
            score_thresh=args.score_thresh,
        )
        individual_paths.append(out_path)
        print(f"  [{i+1}/{args.num_images}] {image_id[:30]} → {out_path}")

    # Gallery
    gallery_path = os.path.join(args.output_dir, "detection_gallery.png")
    make_gallery(individual_paths, gallery_path)
    print(f"\nDone. {len(individual_paths)} images visualized.")


if __name__ == "__main__":
    main()
