#!/usr/bin/env python3
"""
NuCLS → Canonical CSV Preprocessing.

Converts nucls_standardized_all.json to the canonical CSV format
required as input to run_pipeline.py step 1:

    image_id, rad_id, class_name, class_id,
    x_min, y_min, x_max, y_max, width, height

Coordinates are in PIXEL space (not normalized).
Step 1 of run_pipeline.py (manipulate_csv.py) handles normalization.

Usage:
    python preprocess_nucls_canonical.py \
        --input-json /path/to/nucls_standardized_all.json \
        --output-csv /path/to/outputs/nucls/nucls_canonical.csv
"""

import argparse
import csv
import json
import os


FIELDNAMES = [
    "image_id",
    "rad_id",
    "class_name",
    "class_id",
    "x_min",
    "y_min",
    "x_max",
    "y_max",
    "width",
    "height",
]


def convert(input_json: str, output_csv: str) -> None:
    print(f"Loading: {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    images      = {img["id"]: img for img in data["images"]}
    categories  = {c["id"]: c for c in data["categories"]}
    annotations = data["annotations"]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    written = 0
    skipped = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for ann in annotations:
            img = images.get(ann["image_id"])
            if img is None:
                skipped += 1
                continue

            cat = categories.get(ann["category_id"], {})

            # bbox is COCO [x, y, w, h] in pixel space
            bx, by, bw, bh = ann["bbox"]
            x_min = bx
            y_min = by
            x_max = bx + bw
            y_max = by + bh

            if x_max <= x_min or y_max <= y_min:
                skipped += 1
                continue

            image_id = os.path.splitext(img["file_name"])[0]

            writer.writerow({
                "image_id":  image_id,
                "rad_id":    ann.get("rater_id", ""),
                "class_name": ann.get("category_name", cat.get("name", "")),
                "class_id":  ann["category_id"],
                "x_min":     x_min,
                "y_min":     y_min,
                "x_max":     x_max,
                "y_max":     y_max,
                "width":     img["width"],
                "height":    img["height"],
            })
            written += 1

    print(f" Canonical CSV saved: {output_csv}")
    print(f"  Written: {written}  |  Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NuCLS JSON to canonical CSV for run_pipeline.py"
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default="/media/klee_ro/public_datasets/nucls/nucls_standardized_all.json",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/nucls/nucls_canonical.csv",
    )
    args = parser.parse_args()
    convert(args.input_json, args.output_csv)


if __name__ == "__main__":
    main()
