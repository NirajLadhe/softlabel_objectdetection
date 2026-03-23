""""
Standalone script to enrich the raw Vindr-CXR CSV file with the image dimensions.
** Run Once Only **
"""
import csv
import json
from PIL import Image
import os

# Configuration
JSON_FILE = '/media/klee_ro/public_datasets/vindr_cxr/train.json'
INPUT_CSV = '/media/klee_ro/public_datasets/vindr_cxr/train.csv'
OUTPUT_CSV = '/home/haaroon/softlabel_object_detection/outputs/vindrcxr/train_enriched.csv'
IMAGE_DIR = '/media/klee_ro/public_datasets/vindr_cxr/train/'

# Load JSON data
with open(JSON_FILE, 'r') as f:
    data = json.load(f)

# Create a lookup dictionary: (file_name_without_ext, rad_id) -> (width, height)
image_lookup = {}
for img in data['images']:
    file_name_key = img['file_name'].replace('.png', '')
    rad_id = img['rad_id']
    key = (file_name_key, rad_id)
    image_lookup[key] = (img['width'], img['height'])

def get_image_dimensions(image_id, image_dir):
    """
    Get image dimensions by reading the PNG file directly.
    
    Args:
        image_id: The image ID (filename without extension)
        image_dir: Directory containing the images
    
    Returns:
        tuple: (width, height) or (None, None) if file not found
    """
    image_path = os.path.join(image_dir, f"{image_id}.png")
    try:
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None, None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None, None

# Process CSV
with open(INPUT_CSV, 'r') as infile, open(OUTPUT_CSV, 'w', newline='') as outfile:
    reader = csv.DictReader(infile)
    
    fieldnames = reader.fieldnames + ['width', 'height']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()
    
    matches_from_json = 0
    matches_from_image = 0
    no_matches = 0
    
    for row in reader:
        key = (row['image_id'], row['rad_id'])
        
        # Try JSON lookup first
        if key in image_lookup:
            row['width'], row['height'] = image_lookup[key]
            matches_from_json += 1
        else:
            # Fall back to reading from image file
            print(f"No JSON match for {key}, reading from image file...")
            width, height = get_image_dimensions(row['image_id'], IMAGE_DIR)
            
            if width is not None and height is not None:
                row['width'], row['height'] = width, height
                matches_from_image += 1
            else:
                row['width'], row['height'] = '', ''
                no_matches += 1
        
        writer.writerow(row)
    
    # Print summary
    print(f"\nSummary:")
    print(f"  Matched from JSON: {matches_from_json}")
    print(f"  Matched from image files: {matches_from_image}")
    print(f"  No matches found: {no_matches}")