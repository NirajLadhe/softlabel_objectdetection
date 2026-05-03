"""
KαLOS-backed reliability calculator.

Uses kalos.iaa.core.calculate_iaa with collaboration_clusters=True
to compute pairwise Vision-Alpha scores between all rater pairs, then
derives a single per-rater score (mean pairwise) that matches the
column contract expected by the downstream GMM pipeline.

Reference: https://github.com/Madave94/kalos
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path
import copy

from kalos.correspondence.correspondence_algorithms import preprocess_data
from kalos.iaa.core import calculate_iaa

from softlabel_od.softlabel_generation.core.reliability.base import ReliabilityCalculator


class KalosReliabilityCalculator(ReliabilityCalculator):
    """
    Reliability calculator using the KαLOS 'Vision Alpha' metric.

    Matching method choices: 'greedy', 'shm', 'ahc'  (do NOT use 'mgm'
    unless pylibmgm is confirmed installed in your environment).
    """

    def __init__(
        self,
        method: str = "shm",
        threshold_func: str = "bbox_iou_similarity",
        cost_func: str = "negative_score",
    ):
        self.method = method
        self.threshold_func = threshold_func
        self.cost_func = cost_func

    def calculate(
        self,
        annotation_path: str,
        iou_threshold: float = 0.5,
        **kwargs,
    ) -> list[dict]:
        """
        Run KαLOS and return per-rater reliability dicts.

        The 'Avg K-alpha' value for each rater is the mean of their
        pairwise global_collaboration_matrix scores with all other raters.
        """
        with open(annotation_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        # KαLOS needs rater_id (not rater) — guaranteed by Step 1 of the roadmap.
        # Build categories dict: {id: name}
        categories = {
            cat["id"]: cat["name"]
            for cat in raw_data.get("categories", [])
        }

        # Collect all raters across the dataset
        all_raters = sorted({
            rater
            for img in raw_data.get("images", [])
            for rater in img.get("rater_list", [])
        })

        if not all_raters:
            return []

        img_dims = {img["id"]: (img["width"], img["height"])
                    for img in raw_data.get("images", [])}
        for ann in raw_data.get("annotations", []):
            bbox = ann.get("bbox", [])
            w, h = img_dims.get(ann["image_id"], (1, 1))

            if bbox == [0, 0, 1, 1]:
                ann["bbox"] = [0, 0, w, h]
            else:
                ann["bbox"] = [
                    bbox[0] * w,
                    bbox[1] * h,
                    bbox[2] * w,
                    bbox[3] * h,
                ]

        processed = preprocess_data(raw_data)

        (
            _mean_alpha,
            _global_alpha,
            _mean_vitalities,
            _global_vitalities,
            _mean_difficulties,
            _global_difficulties,
            _image_alphas,
            _mean_collab,
            global_collab,  # equivalent to pairwise_kalpha in the pairwise reliability calculator
        ) = calculate_iaa(
            processed_data=processed,
            categories=categories,
            method=self.method,
            threshold_func=self.threshold_func,
            cost_func=self.cost_func,
            similarity_threshold=iou_threshold,
            calculate_vitality=False,
            calculate_difficulty=False,
            collaboration_clusters=True,
            all_raters=all_raters,
        )

        # Build per-rater scores from the pairwise global collaboration matrix.
        # global_collab[r1][r2] = Vision-Alpha scalar for that pair.
        rater_scores = defaultdict(list)
        for r1 in all_raters:
            for r2 in all_raters:
                if r1 == r2:
                    continue
                # kalos stores only one direction: smaller index → larger index
                val = global_collab.get(r1, {}).get(r2, None)
                if val is None:
                    val = global_collab.get(r2, {}).get(r1, None)
                if val is not None and not np.isnan(val):
                    rater_scores[r1].append(val)

        scores = []
        for rater in all_raters:
            vals = rater_scores.get(rater, [])
            avg = float(np.mean(vals)) if vals else 0.0
            std = float(np.std(vals)) if len(vals) > 1 else 0.0
            scores.append({
                "Annotator": rater,
                "Avg K-alpha": avg,
                "K-alpha Std Dev": std,
                "Participation (Assigned Images)": len(vals),   # proxy
                "Consistency Factor (K-alpha)": 1.0 - std,
            })

        return sorted(scores, key=lambda x: x["Avg K-alpha"], reverse=True)