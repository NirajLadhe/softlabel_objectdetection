"""
Pre-compute class-specific variance priors for 1D GMM soft labels.
"""

import os
import argparse
import numpy as np

from softlabel_od.softlabel_generation.core.label_assembly.priors import Prior1DCalculator
from softlabel_od.softlabel_generation.utils.clustering import (
    load_and_cluster_data,
    print_clustering_statistics,
    VALID_CLUSTERING_METHODS,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute class-specific variance priors for 1D GMM soft labels."
    )
    parser.add_argument('--train-csv', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/vindrcxr/train_enriched_normalized.csv'))
    parser.add_argument('--raters-json', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/vindrcxr/train_vindrcxr_all.json'))
    parser.add_argument('--weights-csv', type=str, default=None)
    parser.add_argument('--output', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/vindrcxr/1d_gaussian_softlabels/priors/class_priors_1d.json'))
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--clustering-method', type=str, default='iou', choices=VALID_CLUSTERING_METHODS)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--iomin-threshold', type=float, default=0.5)
    parser.add_argument('--center-ratio-threshold', type=float, default=0.5)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--iou-reliability', type=float, default=0.5)
    args = parser.parse_args()

    if args.weights_csv is None:
        safe_iou_str = str(args.iou_reliability).replace('.', '_')
        args.weights_csv = os.path.join(PROJECT_ROOT, 'outputs/raters_reliability/annotator_reliability_scores_vindrcxr_iou{safe_iou_str}.csv')

    if not os.path.exists(args.train_csv) or not os.path.exists(args.weights_csv):
        print(f"Error: Required file not found.")
        return

    clustered_data = load_and_cluster_data(
        args.train_csv, args.raters_json, args.weights_csv,
        iou_threshold=args.iou_threshold, max_images=args.max_images,
        clustering_method=args.clustering_method,
        iomin_threshold=args.iomin_threshold,
        center_ratio_threshold=args.center_ratio_threshold,
    )
    if not clustered_data:
        print("Error: No clustered data loaded!")
        return

    print_clustering_statistics(clustered_data)
    calculator = Prior1DCalculator(alpha=args.alpha, min_samples_per_class=args.min_samples)
    priors = calculator.calculate_class_priors(clustered_data)
    if not priors:
        print("Error: No priors could be calculated!")
        return

    calculator.print_statistics()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    calculator.save_priors(args.output)
    print(f"Priors saved to {args.output}")


if __name__ == "__main__":
    main()
