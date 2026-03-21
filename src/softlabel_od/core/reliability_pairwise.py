"""
Unified pairwise K-alpha reliability calculation.
Dataset adapters pass pre-loaded COCO data; single core function.
"""

from typing import Optional
import csv
import os
from collections import defaultdict
import numpy as np
import itertools
from tqdm import tqdm

from softlabel_od.core.reliability import pre_processing
from softlabel_od.core.reliability import reliability_data
from softlabel_od.core.reliability import krippendorff_alpha


def extract_annotators(data: dict) -> list:
    annotators_set = set()
    for img in data.get("images", []):
        if "rater_list" in img and isinstance(img["rater_list"], list):
            annotators_set.update(r for r in img["rater_list"] if r)
    for ann in data.get("annotations", []):
        if ann.get("rater"):
            annotators_set.add(ann["rater"])
    return sorted(annotators_set)


def calculate_pairwise_kalpha(
    data: dict,
    mode: str = "bbox",
    iou_threshold: float = 0.5,
    images_rater_key: str = "rater_list",
    annotations_rater_key: str = "rater",
) -> tuple:
    """Calculate pairwise K-alpha. Returns (matrix, counts, annotators)."""
    annotators = extract_annotators(data)
    num_annotators = len(annotators)
    if num_annotators == 0:
        return np.array([]), np.array([]), []

    pairwise_kalpha = np.full((num_annotators, num_annotators), np.nan)
    pair_total_assigned_images = np.zeros((num_annotators, num_annotators))
    kalpha_sum = defaultdict(float)
    kalpha_count = defaultdict(int)

    preprocess_data = pre_processing.Preprocess(data, images_rater_key, annotations_rater_key, silent=True)
    anns_by_img = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_img[ann["image_id"]].append(ann)
    fn_to_id = {img["file_name"]: img["id"] for img in data.get("images", [])}

    for img_fn in tqdm(preprocess_data.image_names, desc="Processing images"):
        img_id = fn_to_id.get(img_fn)
        if img_id is None:
            continue
        raters = preprocess_data.raters_per_image.get(img_fn, [])
        dims = preprocess_data.size_per_image.get(img_fn)
        if len(raters) < 2 or dims is None:
            continue

        img_anns = anns_by_img[img_id]
        for r1, r2 in itertools.combinations(raters, 2):
            if r1 not in annotators or r2 not in annotators:
                continue
            idx1, idx2 = annotators.index(r1), annotators.index(r2)
            pair_total_assigned_images[idx1, idx2] += 1
            pair_total_assigned_images[idx2, idx1] += 1

            r1_anns = [a for a in img_anns if a.get("rater") == r1]
            r2_anns = [a for a in img_anns if a.get("rater") == r2]
            if not r1_anns or not r2_anns:
                continue

            try:
                rel_data = reliability_data.ReliabilityData(img_fn, r1_anns + r2_anns, [r1, r2], dims)
                alpha = krippendorff_alpha.calculate_alpha(rel_data.run(mode, iou_threshold))
                if not np.isnan(alpha):
                    kalpha_sum[(r1, r2)] += alpha
                    kalpha_count[(r1, r2)] += 1
            except Exception:
                pass

    for r1, r2 in itertools.combinations(annotators, 2):
        idx1, idx2 = annotators.index(r1), annotators.index(r2)
        k_sum, k_cnt = kalpha_sum.get((r1, r2), 0.0), kalpha_count.get((r1, r2), 0)
        if k_cnt > 0:
            avg = k_sum / k_cnt
            pairwise_kalpha[idx1, idx2] = pairwise_kalpha[idx2, idx1] = avg

    for i in range(num_annotators):
        pairwise_kalpha[i, i] = 1.0
        pair_total_assigned_images[i, i] = sum(
            1 for fn in preprocess_data.raters_per_image
            if annotators[i] in preprocess_data.raters_per_image.get(fn, [])
        )

    return pairwise_kalpha, pair_total_assigned_images, annotators


def compute_reliability_scores(pkf, ptic, annotators) -> list:
    scores = []
    for i, r in enumerate(annotators):
        k_vals = [pkf[i, j] for j in range(len(annotators)) if i != j and not np.isnan(pkf[i, j])]
        avg_k = np.mean(k_vals) if k_vals else 0.0
        std_k = np.std(k_vals) if len(k_vals) > 1 else 0.0
        scores.append({
            "Annotator": r, "Avg K-alpha": avg_k, "K-alpha Std Dev": std_k,
            "Participation (Assigned Images)": int(ptic[i, i]),
            "Consistency Factor (K-alpha)": 1 - std_k,
        })
    return sorted(scores, key=lambda x: x["Avg K-alpha"], reverse=True)
