"""
Convert a normalized multi-annotator CSV to COCO JSON format.

The CSV must already be in the canonical column format produced by Step 1
(manipulate_csv.py). This step is dataset-agnostic; dataset-specific details
(e.g. file extension, no-finding class id) are passed as arguments.
"""

import csv
import json
import os
import argparse


def convert_csv_to_coco_json(
    csv_file_path: str,
    output_json_path: str,
    no_finding_class_id: int = None,
    image_file_extension: str = "",
) -> None:
    """Convert a canonical multi-annotator CSV to COCO JSON format."""
    print(f"Converting {csv_file_path} → {output_json_path}")
    if no_finding_class_id is not None:
        print(f"  No-finding class id: {no_finding_class_id}")

    images_data = {}
    annotations_data = []
    categories_data = {}
    annotation_id_counter = 1

    with open(csv_file_path, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                image_id_csv = row["image_id"]
                class_id = int(row["class_id"])
                class_name = row["class_name"]
                rad_id = row["rad_id"]
                img_width = int(float(row["width"]))
                img_height = int(float(row["height"]))

                if image_id_csv not in images_data:
                    images_data[image_id_csv] = {
                        "width": img_width,
                        "height": img_height,
                        "rater_list": set(),
                        "file_name": f"{image_id_csv}{image_file_extension}",
                    }
                images_data[image_id_csv]["rater_list"].add(rad_id)

                if class_id not in categories_data:
                    categories_data[class_id] = class_name

                annotation_entry = {
                    "id": annotation_id_counter,
                    "image_id": image_id_csv,
                    "category_id": class_id,
                    "segmentation": [],
                    "iscrowd": 0,
                    "rater": rad_id,
                    "rater_id": rad_id,
                }

                is_no_finding = (
                    no_finding_class_id is not None and class_id == no_finding_class_id
                )
                if is_no_finding:
                    annotation_entry["bbox"] = [0, 0, 1, 1]
                    annotation_entry["area"] = 1
                else:
                    x_min = float(row["x_min"])
                    y_min = float(row["y_min"])
                    x_max = float(row["x_max"])
                    y_max = float(row["y_max"])
                    if x_max < x_min:
                        x_min, x_max = x_max, x_min
                    if y_max < y_min:
                        y_min, y_max = y_max, y_min
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min
                    if bbox_w <= 0 or bbox_h <= 0:
                        continue
                    annotation_entry["bbox"] = [x_min, y_min, bbox_w, bbox_h]
                    annotation_entry["area"] = bbox_w * bbox_h

                annotations_data.append(annotation_entry)
                annotation_id_counter += 1

            except (ValueError, KeyError):
                continue

    # Assign sequential integer image ids
    coco_images = []
    str_to_int_id: dict = {}
    for int_id, (img_id_csv, data) in enumerate(images_data.items(), start=1):
        str_to_int_id[img_id_csv] = int_id
        coco_images.append({
            "id": int_id,
            "width": data["width"],
            "height": data["height"],
            "file_name": data["file_name"],
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": "2023-01-01",
            "rater_list": sorted(list(data["rater_list"])),
        })

    final_annotations = []
    for ann in annotations_data:
        if ann["image_id"] in str_to_int_id:
            ann["image_id"] = str_to_int_id[ann["image_id"]]
            final_annotations.append(ann)

    coco_categories = [
        {"id": cid, "name": name, "supercategory": "finding"}
        for cid, name in categories_data.items()
    ]

    coco_output = {
        "info": {"description": "Multi-annotator dataset in COCO format"},
        "licenses": [{"url": "", "id": 1, "name": "Placeholder License"}],
        "images": coco_images,
        "annotations": final_annotations,
        "categories": sorted(coco_categories, key=lambda x: x["id"]),
    }

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Conversion complete. Saved to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert canonical multi-annotator CSV to COCO JSON."
    )
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument(
        "--no-finding-id", type=int, default=None,
        help="class_id treated as No Finding. Omit if dataset has no No Finding class.",
    )
    parser.add_argument(
        "--image-file-extension", type=str, default="",
        help="Extension appended to image_id to form file_name (e.g. '.dicom', '.jpg').",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found: {args.input_csv}")
        return

    convert_csv_to_coco_json(
        args.input_csv,
        args.output_json,
        no_finding_class_id=args.no_finding_id,
        image_file_extension=args.image_file_extension,
    )


if __name__ == "__main__":
    main()