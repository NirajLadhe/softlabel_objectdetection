import json
import copy
import argparse
from collections import defaultdict

def simplify_rater_list(data):
    """
    Convert rater_list format from dict to list, and add annotator_id to each annotation
    From: {"Annotator_A": [1], "Annotator_B": [2, 1]}
    To: ["Annotator_A", "Annotator_B"]
    
    Add annotator_id field for each annotation
    """
    
    print("\n" + "="*50)
    print("Simplifying rater_list structure")
    print("="*50)
    
    for img in data.get('images', []):
        rater_list = img.get('rater_list', {})
        
        if isinstance(rater_list, dict):
            # 提取所有 rater 名称（去重）
            raters = list(rater_list.keys())
            # 转换为简单列表
            img['rater_list'] = raters
            print(f"  Image {img['id']}: {rater_list} -> {raters}")
        elif isinstance(rater_list, list):
            print(f"  Image {img['id']}: already list format - {rater_list}")
    
    # 为每个 annotation 添加 annotator_id
    # 由于每个图像只有一个标注者（从你的数据看只有 Annotator_A）
    # 直接使用该图像的 rater_list 中的第一个标注者
    for img in data.get('images', []):
        img_id = img['id']
        raters = img.get('rater_list', [])
        
        if raters:
            # 使用第一个标注者（你的数据中只有 Annotator_A）
            default_rater = raters[0]
            
            # 找到该图像的所有 annotations
            for ann in data.get('annotations', []):
                if ann['image_id'] == img_id:
                    ann['annotator_id'] = default_rater
                    # 可选：也添加 rater_id 字段
                    ann['rater_id'] = default_rater
    
    print(f"\n  Added annotator_id to all annotations")
    return data

def merge_vindr_json(train_path, test_path, output_path, offset_images=None, offset_anns=None, simplify=True):
    """
    Merge training and test JSON files into a single JSON file.
    """
    
    print(f"\n{'='*60}")
    print(f"Merging JSON files")
    print(f"{'='*60}")
    
    # Load the two files
    print(f"\nLoading training data: {train_path}")
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    print(f"Loading test data: {test_path}")
    with open(test_path, 'r') as f:
        test_data = json.load(f)
    
    print(f"\nTraining set: {len(train_data.get('images', []))} images, {len(train_data.get('annotations', []))} annotations")
    print(f"Test set: {len(test_data.get('images', []))} images, {len(test_data.get('annotations', []))} annotations")
    
    merged = copy.deepcopy(train_data)
    
    max_image_id = max([img['id'] for img in train_data.get('images', [])]) if train_data.get('images') else 0
    max_ann_id = max([ann['id'] for ann in train_data.get('annotations', [])]) if train_data.get('annotations') else 0
    
    if offset_images is None:
        offset_images = max_image_id + 1
    if offset_anns is None:
        offset_anns = max_ann_id + 1
    
    print(f"\nID offsets:")
    print(f"  - Image ID offset: {offset_images} (train max: {max_image_id})")
    print(f"  - Annotation ID offset: {offset_anns} (train max: {max_ann_id})")
    
    # 1. merge images
    old_to_new_image_id = {}
    for img in test_data.get('images', []):
        old_id = img['id']
        new_id = old_id + offset_images
        old_to_new_image_id[old_id] = new_id
        
        img_copy = copy.deepcopy(img)
        img_copy['id'] = new_id
        merged['images'].append(img_copy)
    
    print(f"\nMerged images: {len(merged['images'])} total")
    
    # 2. merge annotations
    for ann in test_data.get('annotations', []):
        ann_copy = copy.deepcopy(ann)
        
        ann_copy['id'] = ann_copy['id'] + offset_anns
        
        old_img_id = ann_copy['image_id']
        if old_img_id in old_to_new_image_id:
            ann_copy['image_id'] = old_to_new_image_id[old_img_id]
        else:
            ann_copy['image_id'] = old_img_id + offset_images
        
        merged['annotations'].append(ann_copy)
    
    print(f"Merged annotations: {len(merged['annotations'])} total")
    
    # 3. merge categories
    categories_dict = {}
    for cat in merged.get('categories', []):
        categories_dict[cat['id']] = cat
    
    for cat in test_data.get('categories', []):
        if cat['id'] not in categories_dict:
            categories_dict[cat['id']] = cat
        else:
            if categories_dict[cat['id']]['name'] != cat['name']:
                print(f"Warning: category_id {cat['id']} name inconsistency")
    
    merged['categories'] = list(categories_dict.values())
    print(f"Merged categories: {len(merged['categories'])}")
    
    # 4. update info
    merged['info'] = {
        "description": f"Merge training and test sets | Training: {len(train_data.get('images', []))} images, "
                       f"Test: {len(test_data.get('images', []))} images",
        "version": "1.0",
        "date_created": "2023-01-01"
    }
    
    if test_data.get('licenses'):
        license_ids = {lic['id'] for lic in merged.get('licenses', [])}
        for lic in test_data['licenses']:
            if lic['id'] not in license_ids:
                merged['licenses'].append(lic)
    
    # 5. Simplify rater_list structure (remove session info)
    if simplify:
        print(f"\n{'='*60}")
        print(f"Simplifying data structure")
        print(f"{'='*60}")
        merged = simplify_rater_list(merged)
    
    # save merged data
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Merge completed successfully!")
    print(f"{'='*60}")
    print(f"  - Total images: {len(merged['images'])}")
    print(f"  - Total annotations: {len(merged.get('annotations', []))}")
    print(f"  - Total categories: {len(merged.get('categories', []))}")
    print(f"  - Output file: {output_path}")
    print(f"{'='*60}\n")
    
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON files and simplify structure")
    parser.add_argument('--train-json', type=str, 
                       default="/home/yumeng/yumeng/softlabel_objectdetection/outputs/multiorg/multiorg_train.json", 
                       help="Training set JSON file path")
    parser.add_argument('--test-json', type=str, 
                       default="/home/yumeng/yumeng/softlabel_objectdetection/outputs/multiorg/multiorg_test.json", 
                       help="Test set JSON file path")
    parser.add_argument('--output-json', type=str, 
                       default="/home/yumeng/yumeng/softlabel_objectdetection/outputs/multiorg/multiorg_all_before_conversion.json", 
                       help="Merged output JSON file path")
    parser.add_argument('--no-simplify', action='store_true',
                       help="Skip simplifying rater_list (keep original structure)")

    args = parser.parse_args()

    merge_vindr_json(
        train_path=args.train_json,
        test_path=args.test_json,
        output_path=args.output_json,
        simplify=not args.no_simplify
    )