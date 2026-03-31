"""
Convert VinDr-CXR CSV to COCO JSON format with 'No Finding' support.
"""

import csv
import json
import os
import argparse


def convert_vindrcxr_to_coco_json(csv_file_path, output_json_path, no_finding_class_id=14):
    """Convert VinDr-CXR CSV to COCO Option A JSON format."""
    print(f"Starting conversion of {csv_file_path} to {output_json_path} (No Finding ID: {no_finding_class_id})...")
    images_data = {}
    annotations_data = []
    categories_data = {}
    annotation_id_counter = 1

    with open(csv_file_path, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_num, row in enumerate(reader):
            try:
                image_id_csv = row['image_id']
                class_id = int(row['class_id'])
                class_name = row['class_name']
                rad_id = row['rad_id']
                img_width = int(float(row['width']))
                img_height = int(float(row['height']))

                if image_id_csv not in images_data:
                    images_data[image_id_csv] = {
                        "width": img_width, "height": img_height,
                        "rater_list": set(), "file_name": f"{image_id_csv}.dicom"
                    }
                images_data[image_id_csv]['rater_list'].add(rad_id)
                if class_id not in categories_data:
                    categories_data[class_id] = class_name

                annotation_entry = {
                    "id": annotation_id_counter, "image_id": image_id_csv,
                    "category_id": class_id, "segmentation": [], "iscrowd": 0, "rater": rad_id, "rater_id": rad_id,
                }
                if class_id == no_finding_class_id:
                    annotation_entry["bbox"] = [0, 0, 1, 1]
                    annotation_entry["area"] = 1
                else:
                    x_min, y_min = float(row['x_min']), float(row['y_min'])
                    x_max, y_max = float(row['x_max']), float(row['y_max'])
                    if x_max < x_min: x_min, x_max = x_max, x_min
                    if y_max < y_min: y_min, y_max = y_max, y_min
                    bbox_width, bbox_height = x_max - x_min, y_max - y_min
                    if bbox_width <= 0 or bbox_height <= 0:
                        continue
                    annotation_entry["bbox"] = [x_min, y_min, bbox_width, bbox_height]
                    annotation_entry["area"] = bbox_width * bbox_height

                annotations_data.append(annotation_entry)
                annotation_id_counter += 1
            except (ValueError, KeyError) as e:
                continue

    coco_images = []
    image_id_csv_to_int_map = {}
    current_int_image_id = 1
    for img_id_csv, data in images_data.items():
        image_id_csv_to_int_map[img_id_csv] = current_int_image_id
        coco_images.append({
            "id": current_int_image_id, "width": data['width'], "height": data['height'],
            "file_name": data['file_name'], "license": 1, "flickr_url": "", "coco_url": "",
            "date_captured": "2023-01-01", "rater_list": sorted(list(data['rater_list']))
        })
        current_int_image_id += 1

    final_annotations = []
    for ann in annotations_data:
        if ann['image_id'] in image_id_csv_to_int_map:
            ann['image_id'] = image_id_csv_to_int_map[ann['image_id']]
            final_annotations.append(ann)

    coco_categories = [{"id": cid, "name": name, "supercategory": "finding"}
                      for cid, name in categories_data.items()]

    coco_output = {
        "info": {"description": "VinDr-CXR Dataset in COCO Option A format"},
        "licenses": [{"url": "", "id": 1, "name": "Placeholder License"}],
        "images": coco_images, "annotations": final_annotations,
        "categories": sorted(coco_categories, key=lambda x: x['id'])
    }

    os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=2)
    print(f"Conversion complete. Output saved to {output_json_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert VinDr-CXR CSV to COCO JSON format.")
    parser.add_argument('--input-csv', type=str, default=None)
    parser.add_argument('--output-json', type=str, default=None)
    parser.add_argument('--no-finding-id', type=int, default=14)
    args = parser.parse_args()

    csv_file = args.input_csv or 'vindr_cxr/train_enriched_normalized.csv'
    json_output_file = args.output_json or 'vindr_cxr/train_vindrcxr_all.json'

    if not os.path.exists(csv_file):
        print(f"Error: Input CSV file not found at {csv_file}")
    else:
        convert_vindrcxr_to_coco_json(csv_file, json_output_file, no_finding_class_id=args.no_finding_id)


if __name__ == "__main__":
    main()
