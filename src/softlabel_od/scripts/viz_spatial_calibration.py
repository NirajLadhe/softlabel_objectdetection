#!/usr/bin/env python3
"""
Spatial calibration curve: predicted localization std vs empirical corner error.

Compares soft-label model (learned σ) against hard-label baseline (class-prior heuristic σ).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

from softlabel_od.object_detection.adapters.datasets import get_dataset_class
from softlabel_od.object_detection.core.evaluation.metrics import (
    delta_stds_to_pixel_stds,
    match_predictions_to_gt,
)
from softlabel_od.object_detection.core.evaluation.pdq import (
    _pixel_corner_std_from_prior_block,
)
from softlabel_od.object_detection.core.evaluation.runner import collect_predictions, load_model
from softlabel_od.object_detection.utils.config import get_cfg_from_file
from softlabel_od.object_detection.utils.training_helpers import collate_fn
from softlabel_od.object_detection.utils.transforms import Compose, ToTensor

_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent

THEME = {
    "PRIMARY": "#336B87",
    "SECONDARY": "#4F85A0",
    "SUPPORT": "#3E464B",
    "PRIMARY_C": "#B5854A",
    "SECONDARY_C": "#F59927",
    "SUPPORT_C": "#75644D",
    "ACCENT": "#763626",
    "OKAY_ACCENT": "#A9FC6D",
}


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def load_priors(path: str) -> Dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def heuristic_pixel_std(
    class_idx: int,
    class_names: Dict[int, str],
    priors: Dict,
    img_h: float,
    img_w: float,
) -> float:
    """Class-prior heuristic σ for hard-label baseline (PDQ-style conversion)."""
    cls_name = class_names.get(int(class_idx))
    block = None
    if cls_name and cls_name in priors.get("class_priors", {}):
        block = priors["class_priors"][cls_name]
    elif "global_prior" in priors:
        block = priors["global_prior"]
    if block is None:
        return 10.0
    return _pixel_corner_std_from_prior_block(block, img_h, img_w)


def extract_calibration_pairs(
    predictions: List[Dict],
    ground_truths: List[Dict],
    *,
    class_names: Dict[int, str],
    iou_threshold: float = 0.5,
    use_heuristic_std: bool = False,
    priors: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each matched TP, return:
      pred_stds[i]   = Predicted Localization Standard Deviation (px)
      actual_err[i]  = mean |pred_box - gt_box| over corners (px)
    """
    pred_stds: List[float] = []
    actual_errors: List[float] = []

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = gt["boxes"]

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            continue

        img_h, img_w = gt.get("image_size", (2880, 2304))

        matches, _, _ = match_predictions_to_gt(
            pred_boxes, pred_scores, gt_boxes, iou_threshold
        )

        for pred_idx, gt_idx, _ in matches:
            pred_box = pred_boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]
            actual_err = float(np.mean(np.abs(pred_box - gt_box)))

            if use_heuristic_std:
                cls_idx = int(pred["labels"][pred_idx])
                pixel_std = heuristic_pixel_std(
                    cls_idx, class_names, priors, img_h, img_w
                )
            else:
                if "bbox_stds" not in pred:
                    continue
                std_delta = pred["bbox_stds"][pred_idx]
                pixel_std = float(
                    np.mean(
                        delta_stds_to_pixel_stds(
                            pred_box[None], std_delta[None]
                        )[0]
                    )
                )

            pred_stds.append(pixel_std)
            actual_errors.append(actual_err)

    return np.asarray(pred_stds), np.asarray(actual_errors)


def binned_curve(
    pred_stds: np.ndarray,
    actual_errors: np.ndarray,
    bin_edges: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin by predicted std using fixed edges.
    Returns centres, mean errors, stderr, counts.
    """
    centres, means, stderrs, counts = [], [], [], []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < len(bin_edges) - 2:
            mask = (pred_stds >= lo) & (pred_stds < hi)
        else:
            mask = (pred_stds >= lo) & (pred_stds <= hi)

        n = int(mask.sum())
        if n == 0:
            continue

        centres.append(float(pred_stds[mask].mean()))
        means.append(float(actual_errors[mask].mean()))
        stderrs.append(float(actual_errors[mask].std() / np.sqrt(n)))
        counts.append(n)

    return (
        np.asarray(centres),
        np.asarray(means),
        np.asarray(stderrs),
        np.asarray(counts),
    )


def build_bin_edges(reference_stds: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.quantile(reference_stds, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        raise ValueError("Not enough unique std values to form bins.")
    return edges


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_inference(
    *,
    config_path: str,
    weights_path: str,
    label_file: str,
    gmm_type: str,
    data_root: str,
    dataset: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict], Dict[int, str]]:
    cfg = get_cfg_from_file(config_path)
    is_hard = gmm_type.upper() == "HARD_LABEL"
    probabilistic = not is_hard
    label_mode = "hard" if is_hard else f"soft_{gmm_type.lower()}"

    dataset_cls = get_dataset_class(dataset)
    test_dataset = dataset_cls(
        root=data_root,
        label_file=label_file,
        transforms=Compose([ToTensor()]),
        split="test",
        test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
        val_split_ratio=cfg.DATA.VAL_SPLIT_RATIO,
        label_mode=label_mode,
    )

    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )

    model = load_model(
        weights_path=weights_path,
        num_classes=cfg.MODEL.NUM_CLASSES,
        model_type=gmm_type,
        backbone=cfg.MODEL.BACKBONE,
        probabilistic=probabilistic,
        device=device,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS,
    )

    predictions, ground_truths = collect_predictions(
        model, loader, device=device, is_probabilistic=probabilistic
    )
    return predictions, ground_truths, test_dataset.class_names


def maybe_load_cache(cache_path: Path):
    if cache_path.is_file():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    return None


def save_cache(cache_path: Path, payload) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_calibration(
    soft_std, soft_err,
    hard_std, hard_err,
    bin_edges,
    output_path: Path,
    n_bins: int,
):
    cx_s, my_s, se_s, cnt_s = binned_curve(soft_std, soft_err, bin_edges)
    cx_h, my_h, se_h, cnt_h = binned_curve(hard_std, hard_err, bin_edges)

    rho_s, _ = spearmanr(soft_std, soft_err)
    rho_h, _ = spearmanr(hard_std, hard_err)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.errorbar(
        cx_s, my_s, yerr=se_s,
        fmt="o-", color=THEME["PRIMARY"], ecolor=THEME["PRIMARY"],
        capsize=3, linewidth=2, markersize=6,
        label=f"S-R50-IoU (learned $\\sigma$, $\\rho$={rho_s:.2f})",
    )
    ax.errorbar(
        cx_h, my_h, yerr=se_h,
        fmt="s--", color=THEME["ACCENT"], ecolor=THEME["ACCENT"],
        capsize=3, linewidth=2, markersize=6,
        label=f"H-R50-IoU (prior heuristic $\\sigma$, $\\rho$={rho_h:.2f})",
    )

    xmax = max(
        np.nanmax(cx_s) if len(cx_s) else 0,
        np.nanmax(cx_h) if len(cx_h) else 0,
        np.nanmax(my_s) if len(my_s) else 0,
        np.nanmax(my_h) if len(my_h) else 0,
    ) * 1.05
    ax.plot([0, xmax], [0, xmax], color=THEME["SUPPORT"], linestyle=":", alpha=0.6,
            label="Identity reference ($y = x$)")

    ax.set_xlabel("Predicted Localization Standard Deviation (px)", fontsize=12, color=THEME["SUPPORT"])
    ax.set_ylabel("Empirical Mean Localization Error (px)", fontsize=12, color=THEME["SUPPORT"])
    ax.set_title("Localization Uncertainty Versus Empirical Error", fontsize=13, fontweight="bold",
                 color=THEME["SUPPORT"])
    ax.tick_params(colors=THEME["SUPPORT"])
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=True, fontsize=9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_path} and {output_path.with_suffix('.png')}")
    print(f"Soft pairs: {len(soft_std)} | Hard pairs: {len(hard_std)} | Bins: {n_bins}")
    print(f"Spearman ρ — soft: {rho_s:.3f}, hard: {rho_h:.3f}")
    print("Soft bin counts:", cnt_s.tolist())
    print("Hard bin counts:", cnt_h.tolist())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot spatial calibration curve (soft vs hard baseline)."
    )

    # Soft model defaults (S-R50-IoU)
    parser.add_argument("--soft-config", default="configs/vindrcxr_s-r50-iou.yaml")
    parser.add_argument(
        "--soft-weights",
        default="outputs/saved_models/vindrcxr/1d_gaussian_softlabels/vindrcxr_s-r50-iou_30Epoch.pth",
    )
    parser.add_argument(
        "--soft-label-file",
        default="outputs/vindrcxr/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json",
    )

    # Hard model defaults (H-R50-IoU)
    parser.add_argument("--hard-config", default="configs/vindrcxr_h-r50-iou.yaml")
    parser.add_argument(
        "--hard-weights",
        default="outputs/saved_models/vindrcxr/hardlabels/vindrcxr_h-r50-iou_30Epoch.pth",
    )
    parser.add_argument(
        "--hard-label-file",
        default="outputs/vindrcxr/hard_labels/HardLabels_MajorityVote_iou0_5.json",
    )

    parser.add_argument(
        "--priors-file",
        default="outputs/vindrcxr/1d_gaussian_softlabels/priors/class_priors_1d_iou0_5.json",
        help="Class priors used for hard-label heuristic σ",
    )
    parser.add_argument("--dataset", default="vindrcxr")
    parser.add_argument("--data-root", default="/media/klee_ro/public_datasets/vindr_cxr")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--cache-dir",
        default="outputs/vindrcxr/visualizations/spatial_calibration_cache",
        help="Cache inference outputs here as pickle files",
    )
    parser.add_argument(
        "--output",
        default="outputs/vindrcxr/visualizations/fig_spatial_calibration.pdf",
    )
    parser.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore cached predictions and re-run inference",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    cache_dir = Path(args.cache_dir)
    soft_cache = cache_dir / "soft_predictions.pkl"
    hard_cache = cache_dir / "hard_predictions.pkl"

    if not args.force_rerun:
        soft_cached = maybe_load_cache(soft_cache)
        hard_cached = maybe_load_cache(hard_cache)
    else:
        soft_cached = hard_cached = None

    if soft_cached is None:
        print("\nRunning soft-model inference (S-R50-IoU)...")
        soft_preds, soft_gts, class_names = run_inference(
            config_path=str(PROJECT_ROOT / args.soft_config),
            weights_path=str(PROJECT_ROOT / args.soft_weights),
            label_file=str(PROJECT_ROOT / args.soft_label_file),
            gmm_type="1D",
            data_root=args.data_root,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        save_cache(soft_cache, (soft_preds, soft_gts, class_names))
        print(f"Cached soft predictions -> {soft_cache}")
    else:
        soft_preds, soft_gts, class_names = soft_cached
        print(f"Loaded soft predictions from cache: {soft_cache}")

    if hard_cached is None:
        print("\nRunning hard-model inference (H-R50-IoU)...")
        hard_preds, hard_gts, _ = run_inference(
            config_path=str(PROJECT_ROOT / args.hard_config),
            weights_path=str(PROJECT_ROOT / args.hard_weights),
            label_file=str(PROJECT_ROOT / args.hard_label_file),
            gmm_type="HARD_LABEL",
            data_root=args.data_root,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        save_cache(hard_cache, (hard_preds, hard_gts, class_names))
        print(f"Cached hard predictions -> {hard_cache}")
    else:
        hard_preds, hard_gts, _ = hard_cached
        print(f"Loaded hard predictions from cache: {hard_cache}")

    priors = load_priors(str(PROJECT_ROOT / args.priors_file))

    print("\nExtracting calibration pairs...")
    soft_std, soft_err = extract_calibration_pairs(
        soft_preds, soft_gts,
        class_names=class_names,
        iou_threshold=args.iou_threshold,
        use_heuristic_std=False,
    )
    hard_std, hard_err = extract_calibration_pairs(
        hard_preds, hard_gts,
        class_names=class_names,
        iou_threshold=args.iou_threshold,
        use_heuristic_std=True,
        priors=priors,
    )

    if len(soft_std) < args.n_bins:
        raise ValueError(
            f"Only {len(soft_std)} soft matched detections; reduce --n-bins."
        )

    bin_edges = build_bin_edges(soft_std, args.n_bins)
    print(f"Bin edges (from soft-model std quantiles): {np.round(bin_edges, 2)}")

    plot_calibration(
        soft_std, soft_err,
        hard_std, hard_err,
        bin_edges,
        Path(PROJECT_ROOT / args.output),
        args.n_bins,
    )


if __name__ == "__main__":
    main()