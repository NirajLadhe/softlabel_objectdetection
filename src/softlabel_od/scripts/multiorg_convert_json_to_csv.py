# generate_csv_from_json.py
"""
From COCO JSON get train_enriched_normalized.csv
"""

import json
import pandas as pd
import os
import argparse
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', '..'))


def coco_to_vindr_csv(coco_json_path, output_csv_path, category_mapping=None):
    """
    convert COCO JSON to VinDr-CXR format CSV
    
    Args:
        coco_json_path: COCO JSON file path
        output_csv_path: output CSV file path
        category_mapping: category name mapping {category_id: class_name}
    """
    
    print(f"Loading COCO JSON from: {coco_json_path}")
    
    with open(coco_json_path, 'r') as f:
        data = json.load(f)
    
    # category mapping
    if category_mapping is None:
        category_mapping = {}
        for cat in data.get('categories', []):
            cat_id = cat['id']
            cat_name = cat.get('name', f'class_{cat_id}')
            category_mapping[cat_id] = cat_name
    
    # image size mapping
    image_sizes = {}
    for img in data.get('images', []):
        img_id = img['id']
        image_sizes[img_id] = {
            'width': img.get('width', 0),
            'height': img.get('height', 0),
            'file_name': img.get('file_name', f'image_{img_id}.jpg'),
            'session': img.get('session_id', None)
        }
    
    records = []
    
    for ann in data.get('annotations', []):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        bbox = ann.get('bbox', [0, 0, 0, 0])
        
        img_info = image_sizes.get(img_id, {})
        img_width = img_info.get('width', 0)
        img_height = img_info.get('height', 0)
        
        # convert bbox format: [x, y, width, height] -> [x_min, y_min, x_max, y_max]
        if len(bbox) == 4:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
        else:
            x_min, y_min, x_max, y_max = 0, 0, 0, 0

        # get annotator ID
        rad_id = ann.get('annotator_id', ann.get('rater_id', 0))
        
        record = {
            'image_id': img_id,
            'class_name': category_mapping.get(cat_id, f'class_{cat_id}'),
            'class_id': cat_id,
            'rad_id': rad_id,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': img_width,
            'height': img_height,
        }
        records.append(record)
    
    # make DataFrame
    df = pd.DataFrame(records)
    
    # make sure columns are in desired order
    column_order = ['image_id', 'class_name', 'class_id', 'rad_id', 
                    'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height']
    df = df[column_order]
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    # save CSV
    df.to_csv(output_csv_path, index=False)
    
    print(f"\n=== CSV Generation Summary ===")
    print(f"Output file: {output_csv_path}")
    print(f"Total annotations: {len(df)}")
    print(f"Unique images: {df['image_id'].nunique()}")
    print(f"Unique radiologists (annotators): {df['rad_id'].nunique()}")
    print(f"Unique classes: {df['class_id'].nunique()}")
    print(f"\nClasses:")
    for class_id, group in df.groupby('class_id'):
        print(f"  {class_id}: {group['class_name'].iloc[0]} ({len(group)} annotations)")
    print(f"===============================\n")
    
    return df

def create_category_mapping_from_list(class_list):
    """
    create category mapping from list or dict
    
    Args:
        class_list: [(class_id, class_name), ...] 或 {"class_id": "class_name"}
    """
    if isinstance(class_list, dict):
        return class_list
    elif isinstance(class_list, list):
        return {cid: name for cid, name in class_list}
    else:
        return {}

def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to VinDr-CXR format CSV")
    parser.add_argument('--coco-json', type=str, default=os.path.join(PROJECT_ROOT, "outputs/multiorg/multiorg_all.json"),
                       help='Path to COCO JSON file')
    parser.add_argument('--output-csv', type=str, default=os.path.join(PROJECT_ROOT, "outputs/multiorg/train_enriched_normalized.csv"),
                       help='Path to output CSV file')
    parser.add_argument('--category-mapping', type=str, default=None,
                       help='JSON file with category mapping {category_id: "class_name"}')
    parser.add_argument('--class-list', type=str, default=None,
                       help='Comma-separated list of class names in order (class_id starts from 1)')
    
    args = parser.parse_args()
    
    # load category mapping
    category_mapping = None
    
    if args.category_mapping and os.path.exists(args.category_mapping):
        import json
        with open(args.category_mapping, 'r') as f:
            category_mapping = json.load(f)
        print(f"Loaded category mapping from {args.category_mapping}")
    
    elif args.class_list:
        class_names = args.class_list.split(',')
        category_mapping = {i+1: name.strip() for i, name in enumerate(class_names)}
        print(f"Created category mapping from list: {category_mapping}")
    
    # convert
    coco_to_vindr_csv(args.coco_json, args.output_csv, category_mapping)

if __name__ == "__main__":
    main()