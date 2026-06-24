"""
softlabel_stats.py
Computes key statistics from the soft label JSON file.

Reports:
  1. #Clusters          — total clusters across all images
  2. Mean Entropy       — mean class distribution entropy per cluster
  3. Mean Spatial Std   — mean [std_x, std_y, std_w, std_h] in pixels and normalised
  4. Multi-annotator %  — % of clusters with more than 1 annotator bbox

Usage:
    python softlabel_stats.py \
        --softlabel-json outputs/nucls_transform_0.001/1d_gaussian_softlabels/1DGMM_softLabels_iou0_25_with_priors.json
"""

import argparse
import json
import math
import numpy as np
from collections import Counter


def entropy(class_probs_dict):
    """Shannon entropy (bits) of a probability dict. Ignores zero-prob entries."""
    h = 0.0
    for p in class_probs_dict.values():
        if p > 1e-10:
            h -= p * math.log2(p)
    return h


def max_entropy(n_classes):
    """Entropy of a uniform distribution over n_classes."""
    return math.log2(n_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--softlabel-json", required=True,
                        help="Path to 1DGMM_softLabels_*.json")
    args = parser.parse_args()

    with open(args.softlabel_json) as f:
        data = json.load(f)

    # ── 1. Total clusters ─────────────────────────────────────────────
    n_clusters = len(data)

    # ── 2. Class entropy ──────────────────────────────────────────────
    entropies = []
    for rec in data:
        cp = rec.get("class_probs", {})
        if cp:
            entropies.append(entropy(cp))

    mean_entropy    = np.mean(entropies)
    median_entropy  = np.median(entropies)
    std_entropy     = np.std(entropies)

    # Max possible entropy: log2(number of unique classes in dataset)
    all_classes = set()
    for rec in data:
        all_classes.update(rec.get("class_probs", {}).keys())
    n_classes   = len(all_classes)
    max_ent     = max_entropy(n_classes)
    mean_ent_norm = mean_entropy / max_ent  # normalised [0,1]

    # Distribution of entropy
    zero_entropy  = sum(1 for e in entropies if e < 0.01)   # single-class clusters
    high_entropy  = sum(1 for e in entropies if e >= 1.0)    # genuinely ambiguous

    # ── 3. Spatial std ────────────────────────────────────────────────
    stds_norm = []   # [std_x, std_y, std_w, std_h] normalised
    stds_px   = []   # in pixels

    for rec in data:
        sp  = rec.get("spatial", {})
        std = sp.get("std")
        if std and len(std) == 4:
            stds_norm.append(std)
            iw  = rec.get("image_width",  400)
            ih  = rec.get("image_height", 400)
            stds_px.append([
                std[0] * iw,
                std[1] * ih,
                std[2] * iw,
                std[3] * ih,
            ])

    stds_norm_arr = np.array(stds_norm)  # shape [N, 4]
    stds_px_arr   = np.array(stds_px)

    mean_std_norm = stds_norm_arr.mean(axis=0).tolist()
    mean_std_px   = stds_px_arr.mean(axis=0).tolist()
    std_std_px    = stds_px_arr.std(axis=0).tolist()

    # ── 4. Multi-annotator % ──────────────────────────────────────────
    multi_ann_counts = []
    for rec in data:
        n = rec.get("num_annotations_in_cluster",
                    len(rec.get("original_annotations", [])))
        multi_ann_counts.append(n)

    multi_ann_counts = np.array(multi_ann_counts)
    single_ann   = int((multi_ann_counts == 1).sum())
    multi_ann    = int((multi_ann_counts  > 1).sum())
    multi_pct    = 100.0 * multi_ann / n_clusters

    # Distribution breakdown
    bins = [1, 2, 3, 5, 10, 15, 20, 50]
    dist = Counter()
    for n in multi_ann_counts:
        for b in reversed(bins):
            if n >= b:
                dist[f">={b}"] = dist.get(f">={b}", 0) + 1
                break

    # Annotator count stats
    mean_ann  = multi_ann_counts.mean()
    max_ann   = multi_ann_counts.max()
    med_ann   = np.median(multi_ann_counts)

    # ── 5. Per-image stats ────────────────────────────────────────────
    image_ids     = list(set(r.get("image_id","") for r in data))
    n_images      = len(image_ids)
    clusters_per_image = n_clusters / n_images

    # ── Print ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SOFT LABEL DATASET STATISTICS")
    print("="*60)

    print(f"\n{'1. CLUSTER COUNTS':}")
    print(f"   Total clusters         : {n_clusters:,}")
    print(f"   Total images           : {n_images:,}")
    print(f"   Clusters per image     : {clusters_per_image:.1f}")

    print(f"\n2. CLASS ENTROPY (per cluster)")
    print(f"   Unique classes         : {n_classes}")
    print(f"   Max possible entropy   : {max_ent:.3f} bits (uniform over {n_classes} classes)")
    print(f"   Mean entropy           : {mean_entropy:.4f} bits")
    print(f"   Median entropy         : {median_entropy:.4f} bits")
    print(f"   Std entropy            : {std_entropy:.4f} bits")
    print(f"   Mean entropy (normalised): {mean_ent_norm:.4f}  ({mean_ent_norm*100:.1f}% of max)")
    print(f"   Zero-entropy clusters  : {zero_entropy:,} ({100*zero_entropy/n_clusters:.1f}%)  [single dominant class]")
    print(f"   High-entropy clusters  : {high_entropy:,} ({100*high_entropy/n_clusters:.1f}%)  [entropy ≥ 1.0 bit]")

    print(f"\n3. SPATIAL STD (mean per coordinate)")
    labels = ["std_x", "std_y", "std_w", "std_h"]
    print(f"   {'Coord':<8} {'Normalised':>14} {'Pixels (mean)':>15} {'Pixels (std)':>14}")
    print("   " + "─"*52)
    for i, lbl in enumerate(labels):
        print(f"   {lbl:<8} {mean_std_norm[i]:>14.5f} {mean_std_px[i]:>15.3f}px {std_std_px[i]:>13.3f}px")
    print(f"\n   Mean std vector (norm) : [{', '.join(f'{v:.5f}' for v in mean_std_norm)}]")
    print(f"   Mean std vector (px)   : [{', '.join(f'{v:.2f}' for v in mean_std_px)}]")

    print(f"\n4. MULTI-ANNOTATOR CLUSTERS")
    print(f"   Single-annotator       : {single_ann:,} ({100*single_ann/n_clusters:.1f}%)")
    print(f"   Multi-annotator        : {multi_ann:,} ({multi_pct:.1f}%)")
    print(f"   Mean annotators/cluster: {mean_ann:.2f}")
    print(f"   Median annotators      : {med_ann:.1f}")
    print(f"   Max annotators         : {int(max_ann)}")
    print(f"\n   Annotator count distribution:")
    for b in bins:
        key = f">={b}"
        c = dist.get(key, 0)
        print(f"     ≥{b:<4} annotators : {c:,} ({100*c/n_clusters:.1f}%)")

    print("\n" + "="*60)
    print("SUMMARY (for presentation)")
    print("="*60)
    print(f"  #Clusters                = {n_clusters:,}")
    print(f"  Mean Entropy             = {mean_entropy:.4f} bits  "
          f"({mean_ent_norm*100:.1f}% of max entropy)")
    print(f"  Mean Spatial Std (px)    = [{mean_std_px[0]:.2f}, {mean_std_px[1]:.2f}, "
          f"{mean_std_px[2]:.2f}, {mean_std_px[3]:.2f}]")
    print(f"                             [σx, σy, σw, σh]")
    print(f"  Multi-annotator clusters = {multi_pct:.1f}%  ({multi_ann:,} of {n_clusters:,})")
    print("="*60)


if __name__ == "__main__":
    main()
