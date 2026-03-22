"""
Anchor Analysis Script for VinDr-CXR Dataset

Extracts bounding box statistics per class to help configure 
Faster R-CNN anchor sizes and aspect ratios.

Output includes:
- Per-class bbox statistics (size, ratio, area)
- Global statistics across all classes
- Suggested anchor configurations
- Distribution analysis for clustering
"""

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from pathlib import Path

# Configuration
CSV_PATH = '/home/haaroon/SoftLabel_Generation/vindr_cxr/train_enriched.csv'
OUTPUT_JSON = '/home/haaroon/SoftLabel_Generation/vindr_cxr/anchor_analysis.json'

print("="*80)
print("ANCHOR ANALYSIS FOR VinDr-CXR DATASET")
print("="*80)

# Load data
print(f"\nLoading data from {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)
print(f"Total annotations: {len(df)}")

# Filter out 'No Finding' annotations (they have NaN or placeholder bboxes)
df_findings = df[df['class_id'] != 14].copy()  # class_id 14 = No Finding
df_findings = df_findings.dropna(subset=['x_min', 'y_min', 'x_max', 'y_max'])

# Filter out invalid bboxes
df_findings = df_findings[
    (df_findings['x_max'] > df_findings['x_min']) & 
    (df_findings['y_max'] > df_findings['y_min'])
]

print(f"Valid pathology annotations: {len(df_findings)}")
print(f"Unique classes: {df_findings['class_name'].nunique()}")

# Calculate bbox properties
print("\nCalculating bbox properties...")

# Absolute pixel dimensions
df_findings['bbox_width'] = df_findings['x_max'] - df_findings['x_min']
df_findings['bbox_height'] = df_findings['y_max'] - df_findings['y_min']
df_findings['bbox_area'] = df_findings['bbox_width'] * df_findings['bbox_height']

# Normalized dimensions (relative to image size)
df_findings['bbox_width_norm'] = df_findings['bbox_width'] / df_findings['width']
df_findings['bbox_height_norm'] = df_findings['bbox_height'] / df_findings['height']
df_findings['bbox_area_norm'] = df_findings['bbox_width_norm'] * df_findings['bbox_height_norm']

# Aspect ratio (width / height)
df_findings['aspect_ratio'] = df_findings['bbox_width'] / df_findings['bbox_height']

# Scale relative to image (sqrt of area ratio)
df_findings['scale'] = np.sqrt(df_findings['bbox_area'] / (df_findings['width'] * df_findings['height']))

# Center position (normalized)
df_findings['center_x_norm'] = ((df_findings['x_min'] + df_findings['x_max']) / 2) / df_findings['width']
df_findings['center_y_norm'] = ((df_findings['y_min'] + df_findings['y_max']) / 2) / df_findings['height']

# ============================================================================
# PER-CLASS STATISTICS
# ============================================================================
print("\nCalculating per-class statistics...")

class_stats = {}

for class_name in df_findings['class_name'].unique():
    class_df = df_findings[df_findings['class_name'] == class_name]
    
    stats = {
        'class_name': class_name,
        'class_id': int(class_df['class_id'].iloc[0]),
        'count': len(class_df),
        
        # Absolute pixel dimensions
        'bbox_width_px': {
            'mean': float(class_df['bbox_width'].mean()),
            'std': float(class_df['bbox_width'].std()),
            'min': float(class_df['bbox_width'].min()),
            'max': float(class_df['bbox_width'].max()),
            'median': float(class_df['bbox_width'].median()),
            'q25': float(class_df['bbox_width'].quantile(0.25)),
            'q75': float(class_df['bbox_width'].quantile(0.75)),
        },
        'bbox_height_px': {
            'mean': float(class_df['bbox_height'].mean()),
            'std': float(class_df['bbox_height'].std()),
            'min': float(class_df['bbox_height'].min()),
            'max': float(class_df['bbox_height'].max()),
            'median': float(class_df['bbox_height'].median()),
            'q25': float(class_df['bbox_height'].quantile(0.25)),
            'q75': float(class_df['bbox_height'].quantile(0.75)),
        },
        
        # # Normalized dimensions (0-1 range)
        # 'bbox_width_norm': {
        #     'mean': float(class_df['bbox_width_norm'].mean()),
        #     'std': float(class_df['bbox_width_norm'].std()),
        #     'min': float(class_df['bbox_width_norm'].min()),
        #     'max': float(class_df['bbox_width_norm'].max()),
        #     'median': float(class_df['bbox_width_norm'].median()),
        # },
        # 'bbox_height_norm': {
        #     'mean': float(class_df['bbox_height_norm'].mean()),
        #     'std': float(class_df['bbox_height_norm'].std()),
        #     'min': float(class_df['bbox_height_norm'].min()),
        #     'max': float(class_df['bbox_height_norm'].max()),
        #     'median': float(class_df['bbox_height_norm'].median()),
        # },
        
        # Aspect ratios
        'aspect_ratio': {
            'mean': float(class_df['aspect_ratio'].mean()),
            'std': float(class_df['aspect_ratio'].std()),
            'min': float(class_df['aspect_ratio'].min()),
            'max': float(class_df['aspect_ratio'].max()),
            'median': float(class_df['aspect_ratio'].median()),
            'q25': float(class_df['aspect_ratio'].quantile(0.25)),
            'q75': float(class_df['aspect_ratio'].quantile(0.75)),
        },
        
        # # Scale (relative to image)
        # 'scale': {
        #     'mean': float(class_df['scale'].mean()),
        #     'std': float(class_df['scale'].std()),
        #     'min': float(class_df['scale'].min()),
        #     'max': float(class_df['scale'].max()),
        #     'median': float(class_df['scale'].median()),
        # },
        
        # # Area
        # 'area_px': {
        #     'mean': float(class_df['bbox_area'].mean()),
        #     'std': float(class_df['bbox_area'].std()),
        #     'median': float(class_df['bbox_area'].median()),
        # },
        # 'area_norm': {
        #     'mean': float(class_df['bbox_area_norm'].mean()),
        #     'std': float(class_df['bbox_area_norm'].std()),
        #     'median': float(class_df['bbox_area_norm'].median()),
        # },
        
        # Typical center position (useful for understanding where findings appear)
        # 'center_position_norm': {
        #     'x_mean': float(class_df['center_x_norm'].mean()),
        #     'y_mean': float(class_df['center_y_norm'].mean()),
        #     'x_std': float(class_df['center_x_norm'].std()),
        #     'y_std': float(class_df['center_y_norm'].std()),
        # },
    }
    
    class_stats[class_name] = stats

# ============================================================================
# GLOBAL STATISTICS
# ============================================================================
print("Calculating global statistics...")

global_stats = {
    'total_annotations': len(df_findings),
    'num_classes': df_findings['class_name'].nunique(),
    
    # Overall bbox dimensions
    'bbox_width_px': {
        'mean': float(df_findings['bbox_width'].mean()),
        'std': float(df_findings['bbox_width'].std()),
        'min': float(df_findings['bbox_width'].min()),
        'max': float(df_findings['bbox_width'].max()),
        'median': float(df_findings['bbox_width'].median()),
        'q10': float(df_findings['bbox_width'].quantile(0.10)),
        'q25': float(df_findings['bbox_width'].quantile(0.25)),
        'q75': float(df_findings['bbox_width'].quantile(0.75)),
        'q90': float(df_findings['bbox_width'].quantile(0.90)),
    },
    'bbox_height_px': {
        'mean': float(df_findings['bbox_height'].mean()),
        'std': float(df_findings['bbox_height'].std()),
        'min': float(df_findings['bbox_height'].min()),
        'max': float(df_findings['bbox_height'].max()),
        'median': float(df_findings['bbox_height'].median()),
        'q10': float(df_findings['bbox_height'].quantile(0.10)),
        'q25': float(df_findings['bbox_height'].quantile(0.25)),
        'q75': float(df_findings['bbox_height'].quantile(0.75)),
        'q90': float(df_findings['bbox_height'].quantile(0.90)),
    },
    
    # # Normalized dimensions
    # 'bbox_width_norm': {
    #     'mean': float(df_findings['bbox_width_norm'].mean()),
    #     'std': float(df_findings['bbox_width_norm'].std()),
    #     'median': float(df_findings['bbox_width_norm'].median()),
    #     'q10': float(df_findings['bbox_width_norm'].quantile(0.10)),
    #     'q90': float(df_findings['bbox_width_norm'].quantile(0.90)),
    # },
    # 'bbox_height_norm': {
    #     'mean': float(df_findings['bbox_height_norm'].mean()),
    #     'std': float(df_findings['bbox_height_norm'].std()),
    #     'median': float(df_findings['bbox_height_norm'].median()),
    #     'q10': float(df_findings['bbox_height_norm'].quantile(0.10)),
    #     'q90': float(df_findings['bbox_height_norm'].quantile(0.90)),
    # },
    
    # Aspect ratios
    'aspect_ratio': {
        'mean': float(df_findings['aspect_ratio'].mean()),
        'std': float(df_findings['aspect_ratio'].std()),
        'min': float(df_findings['aspect_ratio'].min()),
        'max': float(df_findings['aspect_ratio'].max()),
        'median': float(df_findings['aspect_ratio'].median()),
        'q10': float(df_findings['aspect_ratio'].quantile(0.10)),
        'q25': float(df_findings['aspect_ratio'].quantile(0.25)),
        'q75': float(df_findings['aspect_ratio'].quantile(0.75)),
        'q90': float(df_findings['aspect_ratio'].quantile(0.90)),
    },
    
    # Scales
    # 'scale': {
    #     'mean': float(df_findings['scale'].mean()),
    #     'std': float(df_findings['scale'].std()),
    #     'min': float(df_findings['scale'].min()),
    #     'max': float(df_findings['scale'].max()),
    #     'median': float(df_findings['scale'].median()),
    #     'q10': float(df_findings['scale'].quantile(0.10)),
    #     'q25': float(df_findings['scale'].quantile(0.25)),
    #     'q75': float(df_findings['scale'].quantile(0.75)),
    #     'q90': float(df_findings['scale'].quantile(0.90)),
    # },
    
    # Image size statistics
    # 'image_width': {
    #     'mean': float(df_findings['width'].mean()),
    #     'std': float(df_findings['width'].std()),
    #     'min': float(df_findings['width'].min()),
    #     'max': float(df_findings['width'].max()),
    # },
    # 'image_height': {
    #     'mean': float(df_findings['height'].mean()),
    #     'std': float(df_findings['height'].std()),
    #     'min': float(df_findings['height'].min()),
    #     'max': float(df_findings['height'].max()),
    # },
}

# ============================================================================
# ANCHOR SUGGESTIONS
# ============================================================================
print("Generating anchor suggestions...")

# Common aspect ratios in the dataset
aspect_ratios = df_findings['aspect_ratio'].values
aspect_ratio_percentiles = [10, 25, 50, 75, 90]
suggested_ratios = [float(np.percentile(aspect_ratios, p)) for p in aspect_ratio_percentiles]

# Round to common values for practical use
def round_to_common_ratio(r):
    common = [0.25, 0.33, 0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0, 3.0, 4.0]
    return min(common, key=lambda x: abs(x - r))

suggested_ratios_rounded = list(set([round_to_common_ratio(r) for r in suggested_ratios]))
suggested_ratios_rounded.sort()

# Suggested scales based on bbox sizes
# For input image size of 800-1024px (common for Faster R-CNN)
reference_size = 800  # Assume images will be resized to ~800px
scales = df_findings['scale'].values

# Convert scales to absolute anchor sizes at reference resolution
anchor_sizes_at_ref = scales * reference_size
suggested_sizes = [
    int(np.percentile(anchor_sizes_at_ref, 10)),
    int(np.percentile(anchor_sizes_at_ref, 25)),
    int(np.percentile(anchor_sizes_at_ref, 50)),
    int(np.percentile(anchor_sizes_at_ref, 75)),
    int(np.percentile(anchor_sizes_at_ref, 90)),
]

# Round to powers of 2 or common values
def round_to_anchor_size(s):
    common = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512]
    return min(common, key=lambda x: abs(x - s))

suggested_sizes_rounded = list(set([round_to_anchor_size(s) for s in suggested_sizes]))
suggested_sizes_rounded.sort()

# K-means clustering for optimal anchors (if scikit-learn available)
try:
    from sklearn.cluster import KMeans
    
    # Cluster aspect ratios
    ar_data = aspect_ratios.reshape(-1, 1)
    kmeans_ar = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_ar.fit(ar_data)
    clustered_ratios = sorted([float(c[0]) for c in kmeans_ar.cluster_centers_])
    
    # Cluster scales
    scale_data = scales.reshape(-1, 1)
    kmeans_scale = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans_scale.fit(scale_data)
    clustered_scales = sorted([float(c[0]) for c in kmeans_scale.cluster_centers_])
    
    # Convert clustered scales to anchor sizes at reference resolution
    clustered_anchor_sizes = [int(s * reference_size) for s in clustered_scales]
    
    kmeans_suggestions = {
        'aspect_ratios': clustered_ratios,
        'scales': clustered_scales,
        'anchor_sizes_at_800px': clustered_anchor_sizes,
    }
except ImportError:
    kmeans_suggestions = None
    print("  (sklearn not available, skipping K-means clustering)")

anchor_suggestions = {
    'reference_image_size': reference_size,
    'percentile_based': {
        'aspect_ratios': suggested_ratios,
        'aspect_ratios_rounded': suggested_ratios_rounded,
        'anchor_sizes_at_ref': suggested_sizes,
        'anchor_sizes_rounded': suggested_sizes_rounded,
    },
    'kmeans_based': kmeans_suggestions,
    'recommended_config': {
        'anchor_sizes': suggested_sizes_rounded if len(suggested_sizes_rounded) >= 3 else [32, 64, 128, 256],
        'aspect_ratios': suggested_ratios_rounded if len(suggested_ratios_rounded) >= 3 else [0.5, 1.0, 2.0],
        'note': 'These are data-driven suggestions. Adjust based on your specific model architecture and input resolution.'
    }
}

# ============================================================================
# CLASS DISTRIBUTION (for understanding imbalance)
# ============================================================================
class_distribution = df_findings['class_name'].value_counts().to_dict()
class_distribution = {k: int(v) for k, v in class_distribution.items()}

# ============================================================================
# COMPILE OUTPUT
# ============================================================================
output = {
    'dataset_info': {
        'source': CSV_PATH,
        'total_pathology_annotations': len(df_findings),
        'unique_images': df_findings['image_id'].nunique(),
        'unique_classes': df_findings['class_name'].nunique(),
    },
    'class_distribution': class_distribution,
    'global_statistics': global_stats,
    'per_class_statistics': class_stats,
    'anchor_suggestions': anchor_suggestions,
}

# Save to JSON
print(f"\nSaving results to {OUTPUT_JSON}...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

# ============================================================================
# PRINT SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"\n📊 Dataset Overview:")
print(f"   Total pathology annotations: {len(df_findings)}")
print(f"   Unique images: {df_findings['image_id'].nunique()}")
print(f"   Number of classes: {df_findings['class_name'].nunique()}")

print(f"\n📐 Global Bbox Statistics:")
print(f"   Width (px):  mean={global_stats['bbox_width_px']['mean']:.1f}, median={global_stats['bbox_width_px']['median']:.1f}")
print(f"   Height (px): mean={global_stats['bbox_height_px']['mean']:.1f}, median={global_stats['bbox_height_px']['median']:.1f}")
# print(f"   Width (norm):  mean={global_stats['bbox_width_norm']['mean']:.3f}")
# print(f"   Height (norm): mean={global_stats['bbox_height_norm']['mean']:.3f}")

print(f"\n📏 Aspect Ratio Distribution:")
print(f"   Mean: {global_stats['aspect_ratio']['mean']:.2f}")
print(f"   Median: {global_stats['aspect_ratio']['median']:.2f}")
print(f"   Range: [{global_stats['aspect_ratio']['min']:.2f}, {global_stats['aspect_ratio']['max']:.2f}]")
print(f"   Q10-Q90: [{global_stats['aspect_ratio']['q10']:.2f}, {global_stats['aspect_ratio']['q90']:.2f}]")

# print(f"\n📐 Scale Distribution (relative to image):")
# print(f"   Mean: {global_stats['scale']['mean']:.3f}")
# print(f"   Median: {global_stats['scale']['median']:.3f}")
# print(f"   Q10-Q90: [{global_stats['scale']['q10']:.3f}, {global_stats['scale']['q90']:.3f}]")

print(f"\n🎯 RECOMMENDED ANCHOR CONFIGURATION (for {reference_size}px input):")
print(f"   Anchor sizes: {anchor_suggestions['recommended_config']['anchor_sizes']}")
print(f"   Aspect ratios: {anchor_suggestions['recommended_config']['aspect_ratios']}")

if kmeans_suggestions:
    print(f"\n🔬 K-Means Optimized Anchors:")
    print(f"   Aspect ratios: {[f'{r:.2f}' for r in kmeans_suggestions['aspect_ratios']]}")
    print(f"   Anchor sizes: {kmeans_suggestions['anchor_sizes_at_800px']}")

print(f"\n📁 Class Distribution (top 10):")
sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1], reverse=True)[:10]
for class_name, count in sorted_classes:
    pct = count / len(df_findings) * 100
    print(f"   {class_name}: {count} ({pct:.1f}%)")

print(f"\n✅ Full analysis saved to: {OUTPUT_JSON}")
print("="*80)