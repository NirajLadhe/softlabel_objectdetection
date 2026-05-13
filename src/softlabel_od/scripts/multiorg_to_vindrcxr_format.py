import json
import os
from collections import defaultdict
import argparse

def convert_multiorg_to_vindr(
    multiorg_json_path,
    output_path,
    normalize_bbox=True,
    keep_segmentation=False
):
    """
    Args:
        multiorg_json_path: MultiOrg JSON file path
        output_path: output JSON filepath
        normalize_bbox: whether to normalize bbox coordinates to [0,1] range
        keep_segmentation: whether to keep the original segmentation data
    """
    
    with open(multiorg_json_path, 'r') as f:
        data = json.load(f)
    
    # get image dimensions
    img_dimensions = {}
    for img in data.get('images', []):
        img_dimensions[img['id']] = (img['width'], img['height'])

    img_rater_list = {}
    for img in data.get('images', []):
        img_rater_list[img['id']] = img['rater_list'] if 'rater_list' in img else set()

    for img in data.get('images', []):
        if 'date_captured' not in img:
            img['date_captured'] = "1970-01-01"
        if 'flickr_url' not in img:
            img['flickr_url'] = ""
        if 'coco_url' not in img:
            img['coco_url'] = ""
        if 'license' not in img:
            img['license'] = 0
    
    # "annotations" conversion: add rater/rater_id, normalize bbox, handle segmentation
    for ann in data.get('annotations', []):

        ann['rater'] = ann['rater_id']
        
        # normalize bbox 
        if normalize_bbox and 'bbox' in ann:
            img_id = ann['image_id']
            if img_id in img_dimensions:
                img_w, img_h = img_dimensions[img_id]
                x, y, w, h = ann['bbox']
                
                ann['bbox'] = [
                    x / img_w,      
                    y / img_h,      
                    w / img_w,      
                    h / img_h       
                ]

                ann['area'] = (w / img_w) * (h / img_h)
        
        
        if 'segmentation' not in ann:
            ann['segmentation'] = []
        
        if not keep_segmentation:
            ann['segmentation'] = []
        
        if 'iscrowd' not in ann:
            ann['iscrowd'] = 0
    
    
    # save converted data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Conversion completed!")
    print(f"  - Image count: {len(data.get('images', []))}")
    print(f"  - Annotation count: {len(data.get('annotations', []))}")
    print(f"  - Category count: {len(data.get('categories', []))}")
    
    return data


if __name__ == "__main__":

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', '..'))

    parser = argparse.ArgumentParser(description="Convert MultiOrg JSON to VinDr-CXR JSON format.")
    parser.add_argument('--input-json', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/multiorg/multiorg_all_before_conversion.json'))
    parser.add_argument('--output-json', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/multiorg/multiorg_all.json'))
    parser.add_argument('--no-finding-id', type=int, default=14)
    args = parser.parse_args()
    convert_multiorg_to_vindr(
        multiorg_json_path=args.input_json,
        output_path=args.output_json,
        normalize_bbox=True,                   
        keep_segmentation=False                 
    )