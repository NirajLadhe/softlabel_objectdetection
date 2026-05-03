import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


SOFT_PATH = Path("/home/niraj/softlabel_objectdetection/outputs/lvis/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json")
LOOKUP_PATH = Path("/home/niraj/softlabel_objectdetection/outputs/lvis/lvis_image_lookup.json")
IMAGE_ROOT = Path("/media/klee_ro/public_datasets/coco_lvis/val2017")
OUT_DIR = Path("outputs/lvis/visualizations")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def entropy(class_probs):
    if not class_probs:
        return 0.0
    vals = [v for v in class_probs.values() if v > 0]
    return -sum(v * math.log(v) for v in vals)


def box_norm_to_abs(box, width, height):
    x, y, w, h = box
    return [x * width, y * height, w * width, h * height]


soft_labels = json.loads(SOFT_PATH.read_text(encoding="utf-8"))
lookup = json.loads(LOOKUP_PATH.read_text(encoding="utf-8"))

candidates = []

for sl in soft_labels:
    spatial = sl.get("spatial")
    anns = sl.get("original_annotations", [])
    if not spatial or len(anns) < 2:
        continue

    std_score = sum(spatial.get("std", [0, 0, 0, 0]))
    class_score = entropy(sl.get("class_probs", {}))
    disagreement_score = std_score + class_score

    candidates.append((disagreement_score, sl))

candidates = sorted(candidates, key=lambda x: x[0], reverse=True)[:5]

for idx, (score, sl) in enumerate(candidates, start=1):
    image_id = sl["image_id"]
    meta = lookup[image_id]

    img_path = IMAGE_ROOT / meta["file_name"]
    if not img_path.exists():
        print(f"Missing image: {img_path}")
        continue

    img = Image.open(img_path).convert("RGB")
    width, height = img.size

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(f"{image_id} | disagreement score={score:.4f}")
    ax.axis("off")

    rater_to_color = {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ann in sl.get("original_annotations", []):
        rater = ann.get("rater_id", "unknown")
        if rater not in rater_to_color:
            rater_to_color[rater] = color_cycle[len(rater_to_color) % len(color_cycle)]

        color = rater_to_color[rater]
        x, y, w, h = box_norm_to_abs(ann["bbox"], width, height)
        label = f"{ann.get('class_name', 'class')} | {rater}"

        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=1.5,
            edgecolor=color,
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x, max(0, y - 4),
            label,
            fontsize=7,
            color=color,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

    mean_box = sl["spatial"]["mean"]
    mx, my, mw, mh = box_norm_to_abs(mean_box, width, height)

    mean_rect = patches.Rectangle(
        (mx, my), mw, mh,
        linewidth=3,
        edgecolor="black",
        facecolor="none",
        linestyle="--"
    )
    ax.add_patch(mean_rect)

    out_path = OUT_DIR / f"lvis_disagreement_{idx}_{image_id}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {out_path}")