import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_lvis(input_json, output_dir, max_images=None):
    input_json = Path(input_json)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(input_json)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    if max_images is not None:
        keep_ids = {img["id"] for img in images[:max_images]}
        images = [img for img in images if img["id"] in keep_ids]
        annotations = [
            ann for ann in annotations
            if ann.get("image_id") in keep_ids
        ]

    image_by_id = {img["id"]: img for img in images}

    cat_by_id = {
        cat["id"]: cat.get("name", str(cat["id"]))
        for cat in categories
    }

    rows = []

    skipped = {
        "missing_image": 0,
        "missing_rater": 0,
        "missing_category": 0,
        "invalid_bbox": 0,
    }

    for ann in annotations:
        raw_img_id = ann.get("image_id")

        if raw_img_id not in image_by_id:
            skipped["missing_image"] += 1
            continue

        img = image_by_id[raw_img_id]
        width = float(img["width"])
        height = float(img["height"])
        image_key = f"lvis_{raw_img_id}"

        rater_id = ann.get("rater_id", ann.get("rater"))
        if rater_id is None:
            skipped["missing_rater"] += 1
            continue

        rater_id = str(rater_id)

        cat_id = ann.get("category_id")
        if cat_id not in cat_by_id:
            skipped["missing_category"] += 1
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4:
            skipped["invalid_bbox"] += 1
            continue

        x, y, w, h = map(float, bbox)

        x1 = max(0.0, x)
        y1 = max(0.0, y)
        x2 = min(width, x + w)
        y2 = min(height, y + h)

        if x2 <= x1 or y2 <= y1:
            skipped["invalid_bbox"] += 1
            continue

        x_min = x1 / width
        y_min = y1 / height
        x_max = x2 / width
        y_max = y2 / height

        class_name = cat_by_id[cat_id]

        rows.append({
            "image_id": image_key,
            "rad_id": rater_id,
            "class_id": int(cat_id),
            "class_name": class_name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        })

    csv_columns = [
        "image_id",
        "rad_id",
        "class_id",
        "class_name",
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ]

    df = pd.DataFrame(rows, columns=csv_columns)

    csv_path = output_dir / "lvis_train_enriched_normalized.csv"
    df.to_csv(csv_path, index=False)

    print(f"Saved CSV: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-json",
        default="/media/klee_ro/public_datasets/coco_lvis/lvis_standardized_lvis_val.json",
    )

    parser.add_argument(
        "--output-dir",
        default="/home/niraj/softlabel_objectdetection/outputs/lvis", # /home/niraj/softlabel_objectdetection/outputs/lvis
    )

    parser.add_argument("--max-images", type=int, default=None)

    args = parser.parse_args()

    preprocess_lvis(
        input_json=args.input_json,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()