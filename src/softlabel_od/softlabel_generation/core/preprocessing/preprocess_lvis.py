import argparse
import json
import os
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

    image_by_id = {img["id"]: img for img in images}
    cat_by_id = {
        cat["id"]: cat.get("name", str(cat["id"]))
        for cat in categories
    }

    if max_images is not None:
        keep_ids = set(img["id"] for img in images[:max_images])
        images = [img for img in images if img["id"] in keep_ids]
        annotations = [ann for ann in annotations if ann.get("image_id") in keep_ids]
        image_by_id = {img["id"]: img for img in images}

    rows = []
    coco_images = []
    coco_annotations = []
    image_lookup = {}

    ann_out_id = 1
    skipped = {
        "missing_image": 0,
        "missing_rater": 0,
        "missing_category": 0,
        "invalid_bbox": 0,
    }

    for img in images:
        raw_img_id = img["id"]
        image_key = f"lvis_{raw_img_id}"

        width = int(img["width"])
        height = int(img["height"])
        original_file_name = img.get("file_name", f"{raw_img_id:012d}.jpg")

        image_lookup[image_key] = {
            "source_image_id": raw_img_id,
            "file_name": original_file_name,
            "width": width,
            "height": height,
        }

        coco_images.append({
            "id": image_key,
            "width": width,
            "height": height,

            # This fake .dicom name is intentional:
            # the existing GMM code strips ".dicom" to recover image_id.
            "file_name": f"{image_key}.dicom",

            # Extra fields for your visualization/debugging
            "original_file_name": original_file_name,
            "source_image_id": raw_img_id,
            "license": img.get("license", 0),
            "coco_url": img.get("coco_url", ""),
            "flickr_url": img.get("flickr_url", ""),
            "date_captured": img.get("date_captured", ""),
            "rater_list": sorted([str(r) for r in img.get("rater_list", [])]),
        })

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

        # Normalize to [0, 1], because the current VinDr pipeline expects normalized boxes.
        x_min = x1 / width
        y_min = y1 / height
        x_max = x2 / width
        y_max = y2 / height

        class_name = cat_by_id[cat_id]

        rows.append({
            "image_id": image_key,
            "source_image_id": raw_img_id,
            "file_name": img.get("file_name", ""),
            "rad_id": rater_id,
            "class_id": int(cat_id),
            "class_name": class_name,
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
            "width": int(width),
            "height": int(height),
        })

        coco_annotations.append({
            "id": ann_out_id,
            "image_id": image_key,
            "category_id": int(cat_id),
            "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
            "area": (x_max - x_min) * (y_max - y_min),
            "segmentation": ann.get("segmentation", []),
            "iscrowd": ann.get("iscrowd", 0),
            "rater": rater_id,
            "rater_id": rater_id,
        })
        ann_out_id += 1

    df = pd.DataFrame(rows)

    csv_path = output_dir / "lvis_train_enriched_normalized.csv"
    coco_path = output_dir / "lvis_train_all.json"
    lookup_path = output_dir / "lvis_image_lookup.json"
    summary_path = output_dir / "lvis_preprocess_summary.json"

    df.to_csv(csv_path, index=False)

    coco_out = {
        "info": {"description": "COCO-LVIS converted for soft-label generation"},
        "licenses": data.get("licenses", []),
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    coco_path.write_text(json.dumps(coco_out, indent=2), encoding="utf-8")
    lookup_path.write_text(json.dumps(image_lookup, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "input_json": str(input_json),
        "num_images": len(coco_images),
        "num_annotations": len(coco_annotations),
        "num_rows_csv": len(df),
        "skipped": skipped,
        "csv_path": str(csv_path),
        "coco_path": str(coco_path),
    }, indent=2), encoding="utf-8")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved COCO JSON: {coco_path}")
    print(f"Saved image lookup: {lookup_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Skipped: {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-dir", default="outputs/lvis")
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    preprocess_lvis(
        input_json=args.input_json,
        output_dir=args.output_dir,
        max_images=args.max_images,
    )


if __name__ == "__main__":
    main()