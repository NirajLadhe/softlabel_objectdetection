"""
Unified annotator reliability CLI.

Replaces calculate_pairwise_reliability.py as the step-3 entry point.
Supports multiple backends via --backend flag.

Usage:
    python -m softlabel_od.softlabel_generation.core.reliability.calculate_reliability \
        --annotation-path outputs/vindrcxr/train_vindrcxr_all.json \
        --iou 0.5 \
        --output-dir outputs/raters_reliability/annotator_reliability_vindrcxr \
        --backend kalos \
        --method shm
"""

import argparse
import csv
import os
import numpy as np
import json
from softlabel_od.softlabel_generation.core.reliability.kalos_reliability import KalosReliabilityCalculator

def save_reliability_scores(scores: list[dict], output_path: str) -> None:
    if not scores:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(scores[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in scores:
            row = {
                k: f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else v
                for k, v in s.items()
            }
            writer.writerow(row)


def _get_calculator(backend: str, method: str):
    """Factory: return a ReliabilityCalculator for the requested backend."""
    if backend == "kalos":
        
        return KalosReliabilityCalculator(method=method)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choices: 'kalos'."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Calculate annotator reliability scores."
    )
    parser.add_argument("--annotation-path", type=str, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--no_finding_id", type=int, default=14)   # kept for compat
    parser.add_argument(
        "--output-dir", type=str,
        default="outputs/raters_reliability/annotator_reliability_vindrcxr"
    )
    parser.add_argument("--backend", type=str, default="kalos", choices=["kalos"], help="Reliability calculation backend.")
    parser.add_argument("--method", type=str, default="shm", choices=["greedy", "shm", "ahc"], help="Matching algorithm (kalos backend only).")
    args = parser.parse_args()

    if not os.path.exists(args.annotation_path):
        print(f"Error: Annotation file not found at {args.annotation_path}")
        return

    calculator = _get_calculator(args.backend, args.method)
    scores = calculator.calculate(annotation_path=args.annotation_path, iou_threshold=args.iou)

    if not scores:
        print("No annotators found or no scores computed.")
        return

    safe_iou_str = str(args.iou).replace(".", "_")
    output_path = os.path.join(args.output_dir, f"annotator_reliability_scores_vindrcxr_iou{safe_iou_str}.csv")
    save_reliability_scores(scores, output_path)
    print(f"Reliability scores saved to {output_path}")


if __name__ == "__main__":
    main()