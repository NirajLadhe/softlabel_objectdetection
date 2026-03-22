import json
import os

def extract_annotators(json_path):
    """Extract unique annotators from the given JSON file."""
    print(f"Opening file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract unique annotators from rater_list in images
    image_annotators = set()
    for img in data.get('images', []):
        if 'rater_list' in img:
            image_annotators.update(img['rater_list'])
    
    # Extract unique annotators from rater in annotations
    annotation_annotators = set()
    for ann in data.get('annotations', []):
        if 'rater' in ann:
            annotation_annotators.add(ann['rater'])
    
    # Get a sample image to check format
    sample_image = data.get('images', [])[0] if data.get('images', []) else None
    
    return {
        'image_annotators': sorted(list(image_annotators)),
        'annotation_annotators': sorted(list(annotation_annotators)),
        'sample_image': sample_image
    }

if __name__ == "__main__":
    json_path = os.path.join("texBig_2023", "train_optionA.json")
    result = extract_annotators(json_path)
    
    print("\nImage annotators (from rater_list):")
    print(result['image_annotators'])
    
    print("\nAnnotation annotators (from rater field):")
    print(result['annotation_annotators'])
    
    print("\nSample image format:")
    print(json.dumps(result['sample_image'], indent=2)) 