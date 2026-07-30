"""
gmm_theta.py

Generate soft labels with the 4x4 Sigma_theta covariance (over x,y,w,h) as
the canonical spatial uncertainty representation. Parallel script to
gmm_2d.py -- does NOT modify or replace it.

Output JSON structure per cluster:
    {
        "image_id": ...,
        "finding_id": ...,
        "original_annotations": [...],          # unchanged, for traceability
        "theta": {                               # <-- NEW canonical field
            "mean": [x, y, w, h],
            "covariance": [[...4x4...]]
        },
        "vertices": {                            # kept for backward compatibility --
            "top_left": {"mean": [...], "covariance": [...]},   # DERIVED from theta
            "top_right": {...},                                  # via the proved,
            "bottom_right": {...},                                # verified transform
            "bottom_left": {...}                                  # Sigma_v = A.Sigma_theta.A^T
        },
        "class_probs": {...},
        "num_annotations_in_cluster": ...,
        "prior_used": ...
    }

CLI contract matches exactly what run_pipeline.py's build_step5_command()
sends, including --image-id-key (present on the 1D script; the original 2D
script omits it, which was the second half of the earlier argparse mismatch
-- this script avoids repeating it).
"""

import os
import argparse
import json
from tqdm import tqdm

import numpy as np

from .prior_theta_calculator import PriorThetaCalculator
from .vertex_transform import all_vertex_covariances, all_vertex_means, VERTEX_NAMES
from softlabel_od.softlabel_generation.core.preprocessing.data_loader import load_all_data_for_gmm
from softlabel_od.softlabel_generation.utils.clustering import (
    normalize_annotator_weights,
    cluster_annotations,
    VALID_CLUSTERING_METHODS,
)
from softlabel_od.softlabel_generation.utils.semantic_mapping import get_semantic_map
from softlabel_od.shared.utils.serialization import NpEncoder

ABSENT_RATER_SENTINEL = "NoFindingInCluster"


def compute_class_probs(class_votes, vote_weights):
    """Unchanged from gmm_2d.py -- same weighted class-vote normalisation."""
    if not class_votes or not vote_weights or len(class_votes) != len(vote_weights):
        return {}
    valid_votes, valid_weights = [], []
    for vote, weight in zip(class_votes, vote_weights):
        if weight > 1e-6:
            valid_votes.append(vote)
            valid_weights.append(weight)
    if not valid_votes:
        return {}
    total_weight = sum(valid_weights)
    if total_weight == 0:
        unique = set(valid_votes)
        return {cls: 1.0 / len(unique) for cls in unique} if unique else {}
    class_p = {}
    for class_name, weight in zip(valid_votes, valid_weights):
        class_p[class_name] = class_p.get(class_name, 0) + weight
    for class_name in class_p:
        class_p[class_name] /= total_weight
    return class_p


def main():
    parser = argparse.ArgumentParser(
        description="Generate soft labels with theta-space (4x4) covariance."
    )
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--raters-json", type=str, required=True)
    parser.add_argument("--weights-csv", type=str, required=True)
    parser.add_argument("--priors", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--clustering-method", type=str, default="iou",
                        choices=VALID_CLUSTERING_METHODS)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--iomin-threshold", type=float, default=0.5)
    parser.add_argument("--center-ratio-threshold", type=float, default=0.5)
    parser.add_argument("--semantic-map", type=str, default=None)
    parser.add_argument("--iou-reliability", type=float, default=0.5)
    parser.add_argument("--no-finding-labels", type=str, nargs="*",
                        default=["No finding", "No findings"])
    parser.add_argument("--absent-rater-vote", dest="absent_rater_vote", action="store_true")
    parser.add_argument("--no-absent-rater-vote", dest="absent_rater_vote", action="store_false")
    parser.set_defaults(absent_rater_vote=False)
    # Present on the 1D script, required by run_pipeline.py's build_step5_command:
    parser.add_argument("--image-id-key", type=str, default="file_name")
    parser.add_argument("--reg-eps", type=float, default=1e-4)

    args = parser.parse_args()
    no_finding_labels_lower = {lbl.lower() for lbl in (args.no_finding_labels or [])}

    print("=" * 80)
    print("Theta-space (4x4) Soft Label Generation with Class-Specific Priors")
    print("=" * 80)
    print(f"  Clustering method: {args.clustering_method}")
    print(f"  Prior alpha:       {args.alpha}")
    print(f"  No-finding labels: {args.no_finding_labels}")
    print(f"  Absent rater vote: {args.absent_rater_vote}")
    print(f"  Output:            {args.output}")
    print("=" * 80)

    if not os.path.exists(args.priors):
        print(f"Error: Priors file not found: {args.priors}")
        print("Run calculate_priors_theta.py first.")
        return

    prior_calculator = PriorThetaCalculator(alpha=args.alpha, reg_eps=args.reg_eps)
    prior_calculator.load_priors(args.priors)

    print("Loading annotator reliability weights...")
    normalized_rater_weights = normalize_annotator_weights(args.weights_csv)
    if not normalized_rater_weights:
        print("Error: Could not load annotator weights.")
        return
    print(f"  Loaded weights for {len(normalized_rater_weights)} annotators.")

    print("\nLoading dataset annotations...")
    subset_image_ids, subset_image_data = load_all_data_for_gmm(
        args.train_csv, args.raters_json, normalized_rater_weights
    )
    if not subset_image_ids:
        print("Error: No data loaded.")
        return

    semantic_map = get_semantic_map(args.semantic_map) if args.semantic_map else None

    print(f"\nGenerating theta-space soft labels for {len(subset_image_ids)} images...")
    soft_labels = []

    for image_id in tqdm(subset_image_ids, desc="Processing Images"):
        all_annotations_for_image = subset_image_data[image_id]
        all_rater_ids_on_image = {
            ann["rater_id"] for ann in all_annotations_for_image if "rater_id" in ann
        }

        annotation_clusters = cluster_annotations(
            all_annotations_for_image,
            method=args.clustering_method,
            iou_threshold=args.iou_threshold,
            iomin_threshold=args.iomin_threshold,
            center_ratio_threshold=args.center_ratio_threshold,
            semantic_map=semantic_map,
        )

        for cluster_idx, cluster_anns in enumerate(annotation_clusters):
            valid_anns = [a for a in cluster_anns if a["weight"] > 1e-6]

            if not valid_anns:
                soft_labels.append({
                    "image_id": image_id,
                    "finding_id": cluster_idx,
                    "original_annotations": cluster_anns,
                    "theta": None,
                    "vertices": None,
                    "class_probs": {},
                    "num_annotations_in_cluster": 0,
                })
                continue

            is_no_finding_cluster = all(
                a["class_name"].lower() in no_finding_labels_lower for a in valid_anns
            )

            class_votes = [a["class_name"] for a in valid_anns]
            cluster_class = max(set(class_votes), key=class_votes.count)

            final_class_votes = list(class_votes)
            final_vote_weights = [a["weight"] for a in valid_anns]

            if args.absent_rater_vote:
                raters_in_cluster = {a["rater_id"] for a in valid_anns}
                for rater_id in (all_rater_ids_on_image - raters_in_cluster):
                    if rater_id in normalized_rater_weights:
                        final_class_votes.append(ABSENT_RATER_SENTINEL)
                        final_vote_weights.append(normalized_rater_weights[rater_id])

            class_probs = compute_class_probs(final_class_votes, final_vote_weights)

            if is_no_finding_cluster:
                zero4 = np.zeros((4, 4)).tolist()
                soft_labels.append({
                    "image_id": image_id,
                    "finding_id": cluster_idx,
                    "original_annotations": cluster_anns,
                    "theta": {"mean": [0.0, 0.0, 1.0, 1.0], "covariance": zero4},
                    "vertices": {
                        "top_left":     {"mean": [0.0, 0.0], "covariance": [[0.0, 0.0], [0.0, 0.0]]},
                        "top_right":    {"mean": [1.0, 0.0], "covariance": [[0.0, 0.0], [0.0, 0.0]]},
                        "bottom_left":  {"mean": [0.0, 1.0], "covariance": [[0.0, 0.0], [0.0, 0.0]]},
                        "bottom_right": {"mean": [1.0, 1.0], "covariance": [[0.0, 0.0], [0.0, 0.0]]},
                    },
                    "class_probs": class_probs,
                    "num_annotations_in_cluster": len(valid_anns),
                    "prior_used": "no_finding",
                })
                continue

            boxes = [a["bbox"] for a in valid_anns]   # each [x, y, w, h]
            weights = [a["weight"] for a in valid_anns]

            # --- Canonical theta mean ---
            boxes_arr = np.array(boxes, dtype=float)
            weights_arr = np.array(weights, dtype=float)
            w_norm = weights_arr / weights_arr.sum()
            mean_theta = np.average(boxes_arr, axis=0, weights=w_norm)

            # --- Canonical theta covariance: ONE blend, not four ---
            sigma_theta = prior_calculator.adjust_covariance(
                boxes, cluster_class, weights
            )

            # --- Derive vertex covariances for backward compatibility ---
            # (exact projection, verified: Sigma_v = A_v . Sigma_theta . A_v^T)
            vertex_covs = all_vertex_covariances(sigma_theta)
            vertex_means = all_vertex_means(mean_theta)
            vertex_results = {
                v: {"mean": vertex_means[v].tolist(), "covariance": vertex_covs[v].tolist()}
                for v in VERTEX_NAMES
            }

            soft_labels.append({
                "image_id": image_id,
                "finding_id": cluster_idx,
                "original_annotations": cluster_anns,
                "theta": {
                    "mean": mean_theta.tolist(),
                    "covariance": sigma_theta.tolist(),
                },
                "vertices": vertex_results,
                "class_probs": class_probs,
                "num_annotations_in_cluster": len(valid_anns),
                "prior_used": cluster_class,
            })

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    print(f"\nSaving {len(soft_labels)} soft labels to {args.output}")
    with open(args.output, "w") as f:
        json.dump(soft_labels, f, indent=2, cls=NpEncoder)

    single = sum(1 for sl in soft_labels if sl.get("num_annotations_in_cluster", 0) == 1)
    multi = sum(1 for sl in soft_labels if sl.get("num_annotations_in_cluster", 0) > 1)
    no_find = sum(1 for sl in soft_labels if sl.get("theta") is None)
    print(f"\nStatistics:")
    print(f"  Total:              {len(soft_labels)}")
    print(f"  Single-annotation:  {single}")
    print(f"  Multi-annotation:   {multi}")
    print(f"  No-finding/empty:   {no_find}")
    print("\n" + "=" * 80)
    print("Theta-space GMM with Priors — complete.")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
