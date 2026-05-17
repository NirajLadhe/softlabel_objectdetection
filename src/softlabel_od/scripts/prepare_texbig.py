import json
import csv
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert TexBig JSON to VinDr-CXR style CSV")
    parser.add_argument('--input-json', type=str, required=True, help="Path to texbig_standardized_train_val.json")
    parser.add_argument('--output-csv', type=str, default='outputs/texbig/train_enriched.csv', help="Path to save the output CSV")
    args = parser.parse_args()

    # Create the folder for the output file if it doesn't exist
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    print(f"Loading JSON from {args.input_json}...")
    with open(args.input_json, 'r') as f:
        data = json.load(f)
        
    # 1. Create a dictionary to easily find image width and height
    images_dict = {}
    for img in data['images']:
        images_dict[img['id']] = img
        
    # 2. Create a dictionary to easily translate category_id (e.g., 1) to a name (e.g., "paragraph")
    categories_dict = {}
    for cat in data['categories']:
        categories_dict[cat['id']] = cat['name']
        
    print("Translating JSON boxes to CSV format...")
    # 3. Open our new CSV file to start writing
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the header row (the titles of the columns)
        writer.writerow(['image_id', 'class_name', 'class_id', 'rad_id', 'x_min', 'y_min', 'x_max', 'y_max', 'width', 'height'])
        
        # 4. Loop through every single box drawn by the annotators
        for ann in data['annotations']:
            image_id = ann['image_id']
            img = images_dict[image_id]
            
            cat_id = ann['category_id']
            class_name = categories_dict[cat_id]
            
            # Get the coder name (e.g. "coder_a"). If missing, just put "unknown"
            rad_id = ann.get('rater_id', 'unknown') 
            
            # The JSON gives us [x_start, y_start, width, height]
            x, y, w, h = ann['bbox']
            
            # We need to calculate the actual corners for the CSV format
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            
            width = img['width']
            height = img['height']
            
            # Save this box as a single row in the CSV
            writer.writerow([
                image_id, class_name, cat_id, rad_id, 
                x_min, y_min, x_max, y_max, 
                width, height
            ])
            
    print(f"Successfully created {args.output_csv} with {len(data['annotations'])} boxes translated!")

if __name__ == '__main__':
    main()
