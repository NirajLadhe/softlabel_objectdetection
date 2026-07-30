"""
prior_theta_calculator.py

Replaces the vertex-independent prior computation in Prior2DCalculator with
a single, canonical 4x4 covariance prior over theta = (x, y, w, h) per class.

WHY THIS EXISTS (see project specification document for full derivation):
    The original Prior2DCalculator computes and blends a covariance prior
    SEPARATELY for each of the four box corners. Nothing in that computation
    guarantees the four resulting vertex priors correspond to any single,
    geometrically valid box's uncertainty -- they are four independently-fit
    2x2 approximations, not four views of one true 4x4 object.

    This class fixes that by fitting exactly one Sigma_theta prior per class,
    directly from raw (x, y, w, h) annotator data pooled across every
    multi-annotated cluster of that class. Vertex-level covariances -- still
    useful for visualisation, Soft-IoU heatmaps, PDQ spatial quality, etc. --
    are obtained afterwards via the proved, verified transform
    Sigma_v = A_v . Sigma_theta . A_v^T (see vertex_transform.py), which
    guarantees the four vertex outputs are always mutually consistent by
    construction.

Public interface deliberately mirrors Prior2DCalculator's:
    calculate_class_priors(clusters_by_class) -> Dict[class_name, prior_info]
    get_prior_for_class(class_name, fallback_to_global=True) -> Sigma_theta prior
    adjust_covariance(instance_boxes, class_name, weights) -> blended Sigma_theta
    save_priors(path) / load_priors(path)
so it can be swapped in with minimal changes to calling code.
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Any, Optional

import numpy as np

from .vertex_transform import (
    weighted_mean_cov_theta,
    sigma_theta_to_named_dict,
    named_dict_to_sigma_theta,
)


class PriorThetaCalculator:
    """
    Calculate a single 4x4 covariance prior over (x, y, w, h) per class.

    For each class, we:
        1. Collect all multi-annotated clusters (n >= 2 annotations).
        2. Convert each cluster's raw annotator boxes -- already stored as
           [x, y, w, h] -- into a weighted 4x4 instance covariance.
        3. Average these instance covariances (element-wise, over clusters)
           to get one class-specific prior Sigma_theta.

    Then for each new instance:
        - If n == 1: use the class's Sigma_theta prior directly (matches
          the original code's n==1 branch, which returns the prior with no
          blending or regularisation).
        - If n >= 2: Bayesian-blend the instance Sigma_theta with the class
          prior, then add a small diagonal floor for numerical stability --
          exactly the same two-step logic as the original per-vertex code,
          just applied once, to the true 4x4 object, instead of four times.
    """

    def __init__(self, alpha: float = 1.0, min_samples_per_class: int = 5,
                reg_eps: float = 1e-4):
        self.alpha = alpha
        self.min_samples_per_class = min_samples_per_class
        self.reg_eps = reg_eps
        self.class_priors: Dict[str, Dict[str, Any]] = {}
        self.global_prior: Optional[Dict[str, Any]] = None
        self.has_global_fallback = False

    # ------------------------------------------------------------------
    # Prior computation
    # ------------------------------------------------------------------

    def calculate_class_priors(
        self, clusters_by_class: Dict[str, List[List[Dict]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Args:
            clusters_by_class: {class_name: [cluster, ...]}, each cluster a
                list of annotation dicts with 'bbox': [x, y, w, h] and
                optionally 'weight' (defaults to 1.0 if absent).

        Returns:
            {class_name: {"sigma_theta": 4x4 list, "n_samples": int}}
        """
        print("\n[Theta Prior Calculation] Processing clusters by class...")

        multi_ann = {
            cls: [c for c in clusters if len(c) >= 2]
            for cls, clusters in clusters_by_class.items()
        }
        multi_ann = {cls: clusters for cls, clusters in multi_ann.items() if clusters}

        if not multi_ann:
            print("  Warning: No multi-annotated clusters found!")
            return {}

        print(f"   Found {len(multi_ann)} classes with multi-annotated data")

        for class_name, clusters in multi_ann.items():
            if len(clusters) < self.min_samples_per_class:
                print(f"   Skipping {class_name}: only {len(clusters)} clusters "
                      f"(min required: {self.min_samples_per_class})")
                continue

            instance_sigmas = []
            for cluster in clusters:
                boxes = np.array([ann["bbox"] for ann in cluster], dtype=float)
                # NOTE: matches the ORIGINAL Prior2DCalculator exactly -- prior
                # computation uses UNWEIGHTED covariance (np.cov equivalent).
                # Only instance-level adjust_covariance() below is weighted,
                # exactly as in the original code. This isolates the
                # vertex-vs-theta representation change from any change in
                # weighting behaviour.
                uniform_weights = np.ones(len(boxes))
                _, sigma_theta = weighted_mean_cov_theta(boxes, uniform_weights)
                instance_sigmas.append(sigma_theta)

            if not instance_sigmas:
                continue

            avg_sigma = np.mean(instance_sigmas, axis=0)
            # Light regularisation for numerical stability, mirroring the
            # original code's "+= 1e-4 * eye" treatment on stored priors.
            avg_sigma = avg_sigma + 1e-4 * np.eye(4)

            self.class_priors[class_name] = {
                "sigma_theta": avg_sigma.tolist(),
                "n_samples": len(instance_sigmas),
            }

            named = sigma_theta_to_named_dict(avg_sigma)
            print(f"   {class_name}: {len(instance_sigmas)} clusters, "
                  f"Var(x)={named['Var(x)']:.3e}  Var(y)={named['Var(y)']:.3e}  "
                  f"Var(w)={named['Var(w)']:.3e}  Var(h)={named['Var(h)']:.3e}")

        self._calculate_global_prior(multi_ann)

        print(f"\n   Total class priors computed: {len(self.class_priors)}")
        return self.class_priors

    def _calculate_global_prior(self, multi_ann: Dict[str, List[List[Dict]]]) -> None:
        print("\n   Calculating global prior (fallback)...")
        all_sigmas = []
        total_clusters = 0

        for class_name, clusters in multi_ann.items():
            for cluster in clusters:
                boxes = np.array([ann["bbox"] for ann in cluster], dtype=float)
                # Unweighted, matching original Prior2DCalculator's global prior too.
                uniform_weights = np.ones(len(boxes))
                _, sigma_theta = weighted_mean_cov_theta(boxes, uniform_weights)
                all_sigmas.append(sigma_theta)
                total_clusters += 1

        if total_clusters == 0:
            self.has_global_fallback = False
            print("   Warning: Could not compute global prior")
            return

        avg_sigma = np.mean(all_sigmas, axis=0) + 1e-4 * np.eye(4)
        self.global_prior = {
            "sigma_theta": avg_sigma.tolist(),
            "n_samples": total_clusters,
        }
        self.has_global_fallback = True
        named = sigma_theta_to_named_dict(avg_sigma)
        print(f"   Global prior computed from {total_clusters} clusters")
        print(f"     Var(x)={named['Var(x)']:.3e}  Var(y)={named['Var(y)']:.3e}  "
              f"Var(w)={named['Var(w)']:.3e}  Var(h)={named['Var(h)']:.3e}")

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_prior_for_class(
        self, class_name: str, fallback_to_global: bool = True
    ) -> Optional[np.ndarray]:
        """Returns the 4x4 Sigma_theta prior for a class, or the global fallback."""
        if class_name in self.class_priors:
            return np.array(self.class_priors[class_name]["sigma_theta"])
        if fallback_to_global and self.has_global_fallback:
            return np.array(self.global_prior["sigma_theta"])
        return None

    # ------------------------------------------------------------------
    # Instance-level blending (replaces adjust_covariance, per-vertex version)
    # ------------------------------------------------------------------

    def adjust_covariance(
        self,
        instance_boxes: List[List[float]],
        class_name: str,
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Blend an instance's raw-box-derived Sigma_theta with the class prior,
        matching the original per-vertex logic exactly, but done once, on
        the true 4x4 object.

        Args:
            instance_boxes: list of [x, y, w, h] raw annotator boxes for
                this specific cluster.
            class_name: the cluster's dominant class.
            weights: optional per-annotator reliability weights (defaults
                to uniform weighting of 1.0 if not provided).

        Returns:
            4x4 adjusted Sigma_theta (np.ndarray).
        """
        n = len(instance_boxes)
        prior = self.get_prior_for_class(class_name, fallback_to_global=True)

        boxes = np.array(instance_boxes, dtype=float)
        w = np.array(weights, dtype=float) if weights is not None else np.ones(n)

        if prior is None:
            # No prior available anywhere -- fall back to a minimal-variance
            # identity, matching the original code's no-prior fallback.
            if n == 1:
                return 1e-4 * np.eye(4)
            _, instance_sigma = weighted_mean_cov_theta(boxes, w)
            return instance_sigma + 1e-4 * np.eye(4)

        if n == 1:
            # Matches original: single annotation uses the prior directly,
            # no blend, no extra regularisation.
            return prior

        _, instance_sigma = weighted_mean_cov_theta(boxes, w)
        adjusted = (n * instance_sigma + self.alpha * prior) / (n + self.alpha)
        adjusted = adjusted + self.reg_eps * np.eye(4)
        return adjusted

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_priors(self, path: str) -> None:
        out = {
            "class_priors": self.class_priors,
            "global_prior": self.global_prior,
            "alpha": self.alpha,
            "min_samples_per_class": self.min_samples_per_class,
            "reg_eps": self.reg_eps,
            "format": "theta_4x4",  # explicit marker distinguishing this from
                                    # the old per-vertex prior JSON format
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    def load_priors(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        if data.get("format") != "theta_4x4":
            raise ValueError(
                f"'{path}' does not look like a theta-space prior file "
                f"(expected format='theta_4x4'). This loader is not "
                f"compatible with the old per-vertex class_priors_2d.json."
            )
        self.class_priors = data["class_priors"]
        self.global_prior = data.get("global_prior")
        self.has_global_fallback = self.global_prior is not None
        self.alpha = data.get("alpha", self.alpha)
        self.min_samples_per_class = data.get("min_samples_per_class", self.min_samples_per_class)
        self.reg_eps = data.get("reg_eps", self.reg_eps)

    def print_statistics(self) -> None:
        print("\n" + "=" * 78)
        print("Theta-space Prior Calculation Statistics")
        print("=" * 78)
        print(f"Alpha (prior strength): {self.alpha}")
        print(f"Min samples per class:  {self.min_samples_per_class}")
        print(f"Regularisation floor:   {self.reg_eps}")
        print(f"Total classes with priors: {len(self.class_priors)}")
        print(f"Global fallback available: {self.has_global_fallback}")
        print("\nSample counts per class:")
        for cls, info in sorted(self.class_priors.items(),
                                key=lambda kv: -kv[1]["n_samples"]):
            print(f"  {cls}: {info['n_samples']} multi-annotated clusters")
        print("=" * 78)
