#!/usr/bin/env python3
"""4-panel clustering comparison: raw boxes vs IoU / IoMin / IoMin-CG mean boxes."""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

SOFT_LABEL_FILES = {
    "IoU": "1DGMM_softLabels_iou0_5_with_priors.json",
    "IoMin": "1DGMM_softLabels_iomin0_5_with_priors.json",
    "IoMin-CG": "1DGMM_softLabels_iomin_cg0_5_0_5_with_priors.json",
}

# Project color theme
THEME = {
    "PRIMARY": "#336B87",       # Navy
    "SECONDARY": "#4F85A0",     # Light Blue
    "SUPPORT": "#3E464B",       # Dark Blue
    "PRIMARY_C": "#B5854A",     # Beige
    "SECONDARY_C": "#F59927",   # Light Orange
    "SUPPORT_C": "#75644D",     # Brown
    "ACCENT": "#763626",        # Autumn
    "OKAY_ACCENT": "#A9FC6D",   # Bright Green
}

RATER_COLORS = [
    THEME["PRIMARY"],
    THEME["SECONDARY_C"],
    THEME["PRIMARY_C"],
    THEME["ACCENT"],
    THEME["SECONDARY"],
    THEME["SUPPORT_C"],
]
MEAN_COLORS = {
    "IoU": THEME["PRIMARY"],
    "IoMin": THEME["ACCENT"],
    "IoMin-CG": THEME["SECONDARY_C"],
}


def load_image_rgb(path: str) -> np.ndarray:
    """Percentile-normalise 16-bit CXR for display."""
    raw = np.array(Image.open(path)).astype(np.float32)
    lo, hi = np.percentile(raw, (1, 99))
    if hi > lo:
        raw = np.clip(raw, lo, hi)
        raw = (raw - lo) / (hi - lo) * 255.0
    else:
        raw = np.zeros_like(raw)
    u8 = raw.astype(np.uint8)
    if u8.ndim == 2:
        u8 = np.stack([u8] * 3, axis=-1)
    elif u8.ndim == 3 and u8.shape[2] == 1:
        u8 = np.concatenate([u8] * 3, axis=2)
    return u8


def norm_xywh_to_pixel(bbox, width, height):
    x, y, w, h = bbox
    return x * width, y * height, w * width, h * height


def is_no_finding_placeholder(entry):
    mean = entry.get("spatial", {}).get("mean")
    prior = entry.get("prior_used")
    return prior == "no_finding" or mean == [0.0, 0.0, 1.0, 1.0]


def load_entries(label_dir: Path, filename: str, image_id: str):
    with open(label_dir / filename) as f:
        data = json.load(f)
    return [e for e in data if e["image_id"] == image_id and not is_no_finding_placeholder(e)]


def collect_raw_annotations(entries):
    """Flatten original_annotations across all clusters (dedupe by rater+bbox)."""
    seen = set()
    raw = []
    for entry in entries:
        for ann in entry["original_annotations"]:
            key = (ann["rater_id"], tuple(round(v, 6) for v in ann["bbox"]), ann["class_name"])
            if key not in seen:
                seen.add(key)
                raw.append(ann)
    return raw


def draw_raw_boxes(ax, raw_anns, width, height):
    rater_colors = {}
    color_i = 0
    for ann in raw_anns:
        rater = ann["rater_id"]
        if rater not in rater_colors:
            rater_colors[rater] = RATER_COLORS[color_i % len(RATER_COLORS)]
            color_i += 1
        color = rater_colors[rater]
        x, y, w, h = norm_xywh_to_pixel(ann["bbox"], width, height)
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.85,
        ))
    return rater_colors


def draw_mean_boxes(ax, entries, width, height, color, linewidth=2.0, label_multi=False):
    for entry in entries:
        mean = entry["spatial"]["mean"]
        x, y, w, h = norm_xywh_to_pixel(mean, width, height)
        n = entry.get("num_annotations_in_cluster", 1)
        lw = linewidth + (1.0 if n > 1 else 0.0)
        ax.add_patch(patches.Rectangle(
            (x, y), w, h, linewidth=lw, edgecolor=color, facecolor="none",
            linestyle="--" if n > 1 else "-", alpha=0.95,
        ))
        if label_multi and n > 1:
            ax.text(x, y - 4, str(n), color=color, fontsize=12, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1))

def bbox_crop_limits(raw_anns, entries_list, width, height, pad=80):
    """Return (x0, x1, y0, y1) pixel limits covering all raw + mean boxes."""
    boxes = []
    for ann in raw_anns:
        boxes.append(ann["bbox"])
    for entries in entries_list:
        for e in entries:
            boxes.append(e["spatial"]["mean"])
    xs, ys = [], []
    for x, y, w, h in boxes:
        px, py, pw, ph = norm_xywh_to_pixel([x, y, w, h], width, height)
        xs.extend([px, px + pw])
        ys.extend([py, py + ph])
    return (
        max(0, min(xs) - pad),
        min(width, max(xs) + pad),
        max(0, min(ys) - pad),
        min(height, max(ys) + pad),
    )


def resolve_crop(crop_mode, raw_anns, entries_list, width, height, pad):
    """Build (x0, x1, y0, y1) crop tuple for the requested mode."""
    if crop_mode is None:
        return None
    x0, x1, y0, y1 = bbox_crop_limits(raw_anns, entries_list, width, height, pad=pad)
    if crop_mode == "full":
        return x0, x1, y0, y1
    if crop_mode == "y":
        return 0, width, y0, y1
    if crop_mode == "bottom":
        return 0, width, 0, y1
    raise ValueError(f"Unknown crop mode: {crop_mode}")


def setup_axis(ax, img_rgb, title, crop=None):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=16, fontweight="bold", color=THEME["SUPPORT"])
    if crop:
        x0, x1, y0, y1 = crop
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)   # inverted y for image coords
    else:
        ax.set_xlim(0, img_rgb.shape[1])
        ax.set_ylim(img_rgb.shape[0], 0)
    ax.axis("off")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", default="03e6ecfa6f6fb33dfeac6ca4f9b459c9")
    parser.add_argument("--image-dir", default="/media/klee_ro/public_datasets/vindr_cxr/train")
    parser.add_argument(
        "--label-dir",
        default="outputs/vindrcxr/1d_gaussian_softlabels",
    )
    crop_group = parser.add_mutually_exclusive_group()
    crop_group.add_argument(
        "--crop-to-boxes", action="store_true",
        help="Crop x and y to the annotation region",
    )
    crop_group.add_argument(
        "--crop-y-to-boxes", action="store_true",
        help="Crop y only (top and bottom); keep full image width",
    )
    crop_group.add_argument(
        "--crop-bottom-to-boxes", action="store_true",
        help="Crop y only below annotations; keep full width and top of image",
    )
    parser.add_argument("--crop-pad", type=int, default=80,
                        help="Padding around annotation region when cropping")
    parser.add_argument(
        "--output",
        default="outputs/vindrcxr/visualizations/fig_clustering_comparison.pdf",
    )
    parser.add_argument("--label-multi", action="store_true",
                        help="Annotate multi-rater clusters with annotation count")
    args = parser.parse_args()

    image_path = os.path.join(args.image_dir, f"{args.image_id}.png")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    label_dir = Path(args.label_dir)
    iou_entries = load_entries(label_dir, SOFT_LABEL_FILES["IoU"], args.image_id)
    if not iou_entries:
        raise ValueError(f"No pathology clusters found for image_id={args.image_id}")

    width = iou_entries[0]["image_width"]
    height = iou_entries[0]["image_height"]
    raw_anns = collect_raw_annotations(iou_entries)

    method_entries = {
        "IoU": iou_entries,
        "IoMin": load_entries(label_dir, SOFT_LABEL_FILES["IoMin"], args.image_id),
        "IoMin-CG": load_entries(label_dir, SOFT_LABEL_FILES["IoMin-CG"], args.image_id),
    }

    img_rgb = load_image_rgb(image_path)

    crop_mode = None
    if args.crop_to_boxes:
        crop_mode = "full"
    elif args.crop_y_to_boxes:
        crop_mode = "y"
    elif args.crop_bottom_to_boxes:
        crop_mode = "bottom"

    crop = resolve_crop(
        crop_mode,
        raw_anns,
        list(method_entries.values()),
        width,
        height,
        args.crop_pad,
    )
    if crop:
        x0, x1, y0, y1 = crop
        print(f"Crop ({crop_mode}): x=[{x0:.0f}, {x1:.0f}], y=[{y0:.0f}, {y1:.0f}]")

    fig_h = 10 if crop_mode in ("y", "bottom") else 14
    fig, axes = plt.subplots(2, 2, figsize=(14, fig_h))
    panels = [
        (axes[0, 0], "raw", f"(a) Raw annotations (n={len(raw_anns)})"),
        (axes[0, 1], "IoU", f"(b) IoU clustering (n={len(method_entries['IoU'])} clusters)"),
        (axes[1, 0], "IoMin", f"(c) IoMin clustering (n={len(method_entries['IoMin'])} clusters)"),
        (axes[1, 1], "IoMin-CG", f"(d) IoMin + center gate (n={len(method_entries['IoMin-CG'])} clusters)"),
    ]

    for ax, key, title in panels:
        setup_axis(ax, img_rgb, title, crop=crop)
        if key == "raw":
            rater_colors = draw_raw_boxes(ax, raw_anns, width, height)
            handles = [
                patches.Patch(edgecolor=c, facecolor="none", linewidth=2, label=r)
                for r, c in sorted(rater_colors.items())
            ]
            legend = ax.legend(
                handles=handles, loc="lower right", fontsize=12,
                framealpha=0.9, ncol=2, edgecolor=THEME["SUPPORT"],
            )
            for text in legend.get_texts():
                text.set_color(THEME["SUPPORT"])
        else:
            draw_mean_boxes(
                ax, method_entries[key], width, height,
                color=MEAN_COLORS[key], label_multi=args.label_multi,
            )

    # fig.suptitle(
    #     f"Clustering comparison — VinDr-CXR ({args.image_id[:8]}…)",
    #     fontsize=18, fontweight="bold", y=0.98,
    # )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved {out} and {out.with_suffix('.png')}")
    print(f"Raw boxes: {len(raw_anns)}")
    for k, v in method_entries.items():
        multi = sum(1 for e in v if e["num_annotations_in_cluster"] > 1)
        print(f"  {k}: {len(v)} clusters ({multi} multi-rater)")


if __name__ == "__main__":
    main()