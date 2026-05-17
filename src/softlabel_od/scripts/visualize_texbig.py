import os
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def find_image_file(base_dir, file_name):
    """Recursively search for the image file."""
    for root, dirs, files in os.walk(base_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

def calculate_disagreement_score(cluster):
    """
    Calculate a disagreement score for a cluster.
    High score = high spatial variance (coders disagreeing heavily on borders).
    """
    if cluster.get('num_annotations_in_cluster', 0) < 2:
        return 0.0 # No disagreement if 1 or 0 annotations
    
    spatial = cluster.get('spatial')
    if not spatial or 'std' not in spatial:
        return 0.0
        
    # spatial['std'] contains [x_std, y_std, w_std, h_std] in normalized [0, 1] scale
    # We sum the standard deviations to get a sense of total spatial disagreement
    return sum(spatial['std'])

def plot_cluster(image_path, cluster, output_path):
    """Plot the raw boxes and soft label box for a specific cluster."""
    img = Image.open(image_path).convert("RGB")
    width, height = img.size
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(img)
    
    # Track min/max to crop the image around the cluster
    min_x, min_y = width, height
    max_x, max_y = 0, 0
    
    # Beautiful distinct colors for different coders
    colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
    rater_color_map = {}
    color_idx = 0
    
    # 1. Plot raw annotator boxes
    for ann in cluster['original_annotations']:
        rater = ann.get('rater_id', 'unknown')
        if rater not in rater_color_map:
            rater_color_map[rater] = colors[color_idx % len(colors)]
            color_idx += 1
            
        color = rater_color_map[rater]
        
        # The coordinates in the JSON are normalized [0.0 to 1.0], so we multiply by image size
        n_x, n_y, n_w, n_h = ann['bbox']
        x = n_x * width
        y = n_y * height
        w = n_w * width
        h = n_h * height
        
        # Draw the solid colored box for the human coder
        rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none', alpha=0.8)
        ax.add_patch(rect)
        
        # Add the text label
        class_name = ann.get('class_name', 'unknown')
        ax.text(x, y - 10, f"{class_name} ({rater})", color='black', fontsize=11, fontweight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2))
                
        # Update crop bounds
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # 2. Plot the Soft Label (Mean Box) as a BOLD DASHED line
    spatial = cluster.get('spatial', {})
    if spatial and 'mean' in spatial:
        n_x, n_y, n_w, n_h = spatial['mean']
        x = n_x * width
        y = n_y * height
        w = n_w * width
        h = n_h * height
        
        # Draw a double line (white and black dashed) so it stands out on any background
        rect_bg = patches.Rectangle((x, y), w, h, linewidth=5, edgecolor='white', linestyle='--', facecolor='none')
        rect_fg = patches.Rectangle((x, y), w, h, linewidth=5, edgecolor='black', linestyle='--', facecolor='none')
        ax.add_patch(rect_bg)
        ax.add_patch(rect_fg)
        
        ax.text(x, y + h + 30, "SOFT LABEL (AI COMPROMISE)", color='white', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.9, edgecolor='white', pad=3))

    # Crop the image to just the cluster area, with a 200 pixel padding
    pad = 200
    crop_min_x = max(0, min_x - pad)
    crop_min_y = max(0, min_y - pad)
    crop_max_x = min(width, max_x + pad)
    crop_max_y = min(height, max_y + pad)
    
    ax.set_xlim(crop_min_x, crop_max_x)
    ax.set_ylim(crop_max_y, crop_min_y) # Invert y axis for image plotting
    
    plt.axis('off')
    plt.title(f"Disagreement Visualization\nImage ID: {cluster['image_id']} | Cluster ID: {cluster['finding_id']}", pad=20, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Visualize TexBig soft label disagreements.")
    parser.add_argument('--soft-labels', type=str, required=True, help="Path to 1DGMM_softLabels json")
    parser.add_argument('--original-json', type=str, required=True, help="Path to texbig_standardized_train_val.json (to get real filenames)")
    parser.add_argument('--images-dir', type=str, required=True, help="Base directory containing the actual TexBig PNG images")
    parser.add_argument('--output-dir', type=str, default='outputs/texbig/visualizations', help="Directory to save PNGs")
    parser.add_argument('--top-k', type=int, default=5, help="Number of images to visualize")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Loading Original JSON to map Image IDs to Filenames...")
    with open(args.original_json, 'r') as f:
        orig_data = json.load(f)
        
    # Create cheat sheet mapping image_id to its actual file_name (e.g. '1' -> 'kunstgewerbe_094.png')
    id_to_filename = {str(img['id']): img['file_name'] for img in orig_data['images']}

    print(f"Loading Soft Labels from {args.soft_labels}...")
    with open(args.soft_labels, 'r') as f:
        soft_labels = json.load(f)
        
    print("Calculating disagreement scores...")
    for cluster in soft_labels:
        cluster['disagreement_score'] = calculate_disagreement_score(cluster)
        
    # Filter out boring clusters
    disputed_clusters = [c for c in soft_labels if c['disagreement_score'] > 0]
    
    # Sort so the MESSIEST clusters are at the top
    disputed_clusters.sort(key=lambda x: x['disagreement_score'], reverse=True)
    
    top_clusters = disputed_clusters[:args.top_k]
    print(f"\nFound {len(top_clusters)} clusters with extremely high disagreement.")
    
    for i, cluster in enumerate(top_clusters):
        image_id_str = str(cluster['image_id'])
        file_name = id_to_filename.get(image_id_str)
        
        if not file_name:
            print(f"Could not find filename for image ID {image_id_str}. Skipping.")
            continue
            
        print(f"[{i+1}/{args.top_k}] Searching for actual image file: {file_name}...")
        image_path = find_image_file(args.images_dir, file_name)
        
        if not image_path:
            print(f"  ❌ Could not find {file_name} inside {args.images_dir}. Skipping.")
            continue
            
        output_png = os.path.join(args.output_dir, f"disagreement_rank_{i+1}_img{image_id_str}_cluster{cluster['finding_id']}.png")
        print(f"  ✅ Found image! Plotting and saving to {output_png}...")
        plot_cluster(image_path, cluster, output_png)

    print("\nAll done! Check the visualizations folder.")

if __name__ == "__main__":
    main()
