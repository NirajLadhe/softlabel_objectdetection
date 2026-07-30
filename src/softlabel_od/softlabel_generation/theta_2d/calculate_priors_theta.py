"""
calculate_priors_theta.py

Pre-compute class-specific Sigma_theta (4x4, over x,y,w,h) covariance priors.
Parallel script to calculate_priors_2d.py -- does NOT modify or replace it.
Uses PriorThetaCalculator instead of Prior2DCalculator; everything else
(data loading, clustering) reuses the same, already-working infrastructure.

CLI contract matches exactly what run_pipeline.py's build_step4_command()
sends -- including --dataset-name, which the original 1D script accepts
but the original 2D script does not (that mismatch was diagnosed and
fixed once already; this script is written to avoid repeating it).
"""

import os
import argparse

from .prior_theta_calculator import PriorThetaCalculator
from softlabel_od.softlabel_generation.utils.clustering import (
    load_and_cluster_data,
    print_clustering_statistics,
    VALID_CLUSTERING_METHODS,
)
from softlabel_od.softlabel_generation.utils.semantic_mapping import get_semantic_map


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute class-specific Sigma_theta (4x4) covariance priors."
    )
    parser.add_argument('--train-csv', type=str, required=True)
    parser.add_argument('--raters-json', type=str, required=True)
    parser.add_argument('--weights-csv', type=str, default=None)
    parser.add_argument('--output', type=str, default='priors/class_priors_theta.json')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--clustering-method', type=str, default='iou',
                        choices=VALID_CLUSTERING_METHODS)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--semantic-map', type=str, default=None)
    parser.add_argument('--iomin-threshold', type=float, default=0.5)
    parser.add_argument('--center-ratio-threshold', type=float, default=0.5)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--iou-reliability', type=float, default=0.5)
    parser.add_argument('--no-finding-labels', type=str, nargs='*',
                        default=['No finding', 'No findings'])
    # Present on the 1D script, required by run_pipeline.py's build_step4_command --
    # accepted here (even if unused downstream) to avoid the earlier argparse mismatch.
    parser.add_argument('--dataset-name', type=str, default=None)
    parser.add_argument('--reg-eps', type=float, default=1e-4,
                        help="Diagonal regularisation floor added to Sigma_theta priors")
    args = parser.parse_args()

    if args.weights_csv is None:
        raise ValueError(
            "--weights-csv is required. Pass it explicitly or run via run_pipeline.py."
        )

    if not os.path.exists(args.train_csv) or not os.path.exists(args.weights_csv):
        print("Error: Required file not found.")
        return

    clustered_data = load_and_cluster_data(
        args.train_csv, args.raters_json, args.weights_csv,
        iou_threshold=args.iou_threshold, max_images=args.max_images,
        clustering_method=args.clustering_method,
        iomin_threshold=args.iomin_threshold,
        center_ratio_threshold=args.center_ratio_threshold,
        semantic_map=get_semantic_map(args.semantic_map) if args.semantic_map else None,
    )
    if not clustered_data:
        print("Error: No clustered data loaded!")
        return

    print_clustering_statistics(clustered_data)

    calculator = PriorThetaCalculator(
        alpha=args.alpha,
        min_samples_per_class=args.min_samples,
        reg_eps=args.reg_eps,
    )
    priors = calculator.calculate_class_priors(clustered_data)
    if not priors:
        print("Error: No priors could be calculated!")
        return

    calculator.print_statistics()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    calculator.save_priors(args.output)
    print(f"Theta-space priors saved to {args.output}")


if __name__ == "__main__":
    main()
