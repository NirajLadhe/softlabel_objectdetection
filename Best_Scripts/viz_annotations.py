"""
viz_annotations.py
Visualizes raw multi-rater annotations on a NuCLS H&E image.

Usage:
    python viz_annotations.py \
        --json     /media/klee_ro/public_datasets/nucls/nucls_standardized_all.json \
        --img-dir  /media/klee_ro/public_datasets/nucls/images \
        --image-id 1 \
        --max-raters 8 \
        --output   outputs/visualizations/presentation/annotation_viz.png

Parameters you can tune:
    --image-id    : integer image_id from the COCO JSON (try 1, 5, 12, ...)
    --max-raters  : cap how many distinct raters to show (for clarity)
    --min-annots  : skip images with fewer than this many annotations
    --output      : output PNG path
"""

import argparse
import json
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# ── Brand palette ─────────────────────────────────────────────────────────────
BRAND = {
    "primary":   "#336B87",
    "secondary": "#4F85A0",
    "support":   "#3E464B",
    "primary_c": "#B5854A",
    "secondary_c":"#F59927",
    "support_c": "#75644D",
    "accent":    "#763626",
    "okay":      "#A9FC6D",
    "bg1":       "#F7F7F7",
}

# Enough distinct colors for up to 12 raters (cycles through brand + extras)
RATER_COLORS = [
    "#336B87","#F59927","#763626","#B5854A","#4F85A0",
    "#75644D","#A9FC6D","#3E464B","#E07B39","#5B8FA8",
    "#D4A853","#8B3A3A",
]

def load_json(path):
    with open(path) as f:
        return json.load(f)

def resolve_image_path(img_dir, file_name):
    for ext in ["", ".png", ".jpg", ".jpeg", ".tif"]:
        p = os.path.join(img_dir, file_name + ext)
        if os.path.exists(p):
            return p
        p = os.path.join(img_dir, file_name)
        if os.path.exists(p):
            return p
    return None

def pick_image(data, min_annots=10):
    """Auto-select an image that has a good annotation density."""
    counts = {}
    for ann in data["annotations"]:
        counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    candidates = [iid for iid, c in counts.items() if c >= min_annots]
    return random.choice(candidates) if candidates else data["images"][0]["id"]

def draw(data, img_dir, image_id, max_raters, output):
    # ── Find image record ──────────────────────────────────────────────────────
    img_record = next((i for i in data["images"] if i["id"] == image_id), None)
    if img_record is None:
        raise ValueError(f"image_id={image_id} not found in JSON")

    img_path = resolve_image_path(img_dir, img_record["file_name"])
    if img_path is None:
        # try stripping extension from file_name
        base = os.path.splitext(img_record["file_name"])[0]
        img_path = resolve_image_path(img_dir, base)
    if img_path is None:
        raise FileNotFoundError(f"Image not found: {img_record['file_name']} in {img_dir}")

    img = np.array(Image.open(img_path).convert("RGB"))

    # ── Gather annotations for this image ─────────────────────────────────────
    annots = [a for a in data["annotations"] if a["image_id"] == image_id]
    raters = sorted(set(a["rater_id"] for a in annots))
    if max_raters and len(raters) > max_raters:
        raters = raters[:max_raters]
        annots = [a for a in annots if a["rater_id"] in raters]

    rater_color = {r: RATER_COLORS[i % len(RATER_COLORS)] for i, r in enumerate(raters)}

    # ── Figure ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=BRAND["bg1"])
    ax.imshow(img)
    ax.set_facecolor(BRAND["bg1"])
    ax.axis("off")

    for ann in annots:
        x, y, w, h = ann["bbox"]  # COCO: [x, y, width, height]
        color = rater_color[ann["rater_id"]]
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.6, edgecolor=color, facecolor="none", alpha=0.85
        )
        ax.add_patch(rect)
        # Class label above box
        ax.text(
            x, y - 3, ann["category_name"],
            fontsize=6, color=color, fontweight="bold",
            fontfamily="sans-serif",
            bbox=dict(facecolor="white", alpha=0.55, pad=1, linewidth=0),
            va="bottom",
        )

    # ── Legend: raters ─────────────────────────────────────────────────────────
    legend_patches = [
        patches.Patch(color=rater_color[r], label=f"Rater {r}")
        for r in raters
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right", fontsize=7,
        framealpha=0.85, edgecolor="#cccccc",
        ncol=2 if len(raters) > 6 else 1,
    )

    # ── Stats annotation ───────────────────────────────────────────────────────
    ax.set_title(
        f"Image: {img_record['file_name'].split('.')[0]}\n"
        f"{len(annots)} annotations · {len(raters)} raters",
        fontsize=11, color=BRAND["support"], pad=10,
        fontfamily="sans-serif",
    )

    os.makedirs(os.path.dirname(output) if os.path.dirname(output) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180, bbox_inches="tight", facecolor=BRAND["bg1"])
    plt.close()
    print(f"Saved: {output}")
    print(f"  Image: {img_record['file_name']}")
    print(f"  Annotations: {len(annots)}  |  Raters: {len(raters)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json",       required=True,
                        help="Path to nucls_standardized_all.json")
    parser.add_argument("--img-dir",    required=True,
                        help="Directory containing NuCLS images")
    parser.add_argument("--image-id",   type=int, default=None,
                        help="COCO image_id to visualize (omit for auto-select)")
    parser.add_argument("--max-raters", type=int, default=8,
                        help="Maximum raters to display (default: 8)")
    parser.add_argument("--min-annots", type=int, default=10,
                        help="Minimum annotations for auto-selection")
    parser.add_argument("--output",     default="outputs/visualizations/presentation/annotation_viz.png")
    args = parser.parse_args()

    data = load_json(args.json)
    image_id = args.image_id if args.image_id is not None else pick_image(data, args.min_annots)
    print(f"Using image_id: {image_id}")
    draw(data, args.img_dir, image_id, args.max_raters, args.output)


if __name__ == "__main__":
    main()
