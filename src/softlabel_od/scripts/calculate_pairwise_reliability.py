"""
Calculate pairwise annotator reliability (K-alpha).
Uses unified core; supports VinDr-CXR and TexBig via --annotation-path.
"""

import json
import os
import csv
import numpy as np
import argparse

from softlabel_od.softlabel_generation.core.reliability_pairwise import (
    calculate_pairwise_kalpha,
    compute_reliability_scores,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def save_pairwise_matrix(matrix, annotators, output_path):
    if not annotators:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([''] + annotators)
        for i, name in enumerate(annotators):
            row = [name] + [f"{matrix[i, j]:.4f}" if not np.isnan(matrix[i, j]) else "N/A"
                   for j in range(len(annotators))]
            writer.writerow(row)


def save_reliability_scores(scores, output_path):
    if not scores:
        return
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fieldnames = list(scores[0].keys())
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scores:
            row = {k: f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else v for k, v in s.items()}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Calculate pairwise annotator reliability (K-alpha) for VinDr-CXR.")
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--no_finding_id', type=int, default=14)
    parser.add_argument('--annotation-path', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/raters_reliability/annotator_reliability_vindrcxr'))
    args = parser.parse_args()

    annotation_path = args.annotation_path or 'vindr_cxr/train_vindrcxr_all.json'
    results_dir = args.output_dir or 'annotator_reliability_vindrcxr'
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(annotation_path):
        print(f"Error: Annotation file not found at {annotation_path}")
        return

    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    pairwise_kalpha, pair_total_assigned_images, annotators = calculate_pairwise_kalpha(
        data, mode="bbox", iou_threshold=args.iou
    )
    if not annotators:
        print("No annotators extracted.")
        return

    reliability_scores = compute_reliability_scores(
        pairwise_kalpha, pair_total_assigned_images, annotators
    )
    safe_iou_str = str(args.iou).replace('.', '_')
    pairwise_output = os.path.join(results_dir, f"pairwise_kalpha_matrix_vindrcxr_iou{safe_iou_str}.csv")
    reliability_output = os.path.join(results_dir, f"annotator_reliability_scores_vindrcxr_iou{safe_iou_str}.csv")
    save_pairwise_matrix(pairwise_kalpha, annotators, pairwise_output)
    save_reliability_scores(reliability_scores, reliability_output)
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
