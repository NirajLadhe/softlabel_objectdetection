"""
CLI entry point: softlabel-infer-kaggle

Run trained Faster R-CNN on Kaggle VinBigData test PNGs and write
submission.csv (image_id, PredictionString).
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from softlabel_od.object_detection.core.evaluation.runner import load_model
from softlabel_od.object_detection.utils.config import get_cfg_from_file
from softlabel_od.object_detection.utils.kaggle_submission import (
    MODEL_IDX_TO_KAGGLE,
    MODEL_NO_FINDING_IDX,
    KaggleTestImageDataset,
    build_submission_dataframe,
    inject_no_finding_if_empty,
    prediction_dict_to_string,
)
from softlabel_od.object_detection.utils.training_helpers import collate_fn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


def default_config_for_gmm_type(gmm_type: str) -> str:
    if gmm_type == "1D":
        return os.path.join(
            PROJECT_ROOT, "configs/vindrcxr_1d_softlabel_config_albers.yaml"
        )
    if gmm_type == "HARD_LABEL":
        return os.path.join(
            PROJECT_ROOT, "configs/vindrcxr_hardlabel_config_albers.yaml"
        )
    raise ValueError(f"Unsupported gmm_type: {gmm_type}")


def run_inference(args) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Device: {device}")

    cfg = get_cfg_from_file(args.config)
    is_hard = args.gmm_type == "HARD_LABEL"
    probabilistic = not is_hard
    num_classes = args.num_classes or cfg.MODEL.NUM_CLASSES
    backbone = args.backbone or cfg.MODEL.BACKBONE

    print(f"\nLoading test images from: {args.test_dir}")
    test_dataset = KaggleTestImageDataset(
        args.test_dir,
        image_extension=args.image_extension,
        recursive=args.recursive,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    print(f"Test set: {len(test_dataset)} images")

    print(f"\nLoading model from: {args.weights}")
    print(
        f"Model type: {args.gmm_type} "
        f"({'hard-label' if is_hard else '1D soft-label/probabilistic'})"
    )
    model = load_model(
        weights_path=args.weights,
        num_classes=num_classes,
        model_type=args.gmm_type,
        backbone=backbone,
        probabilistic=probabilistic,
        device=device,
        anchor_sizes=cfg.MODEL.ANCHOR_SIZES,
        anchor_aspect_ratios=cfg.MODEL.ANCHOR_ASPECT_RATIOS,
    )
    model.eval()

    rows = []
    pbar = tqdm(test_loader, desc="Kaggle inference")
    with torch.no_grad():
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for pred, target in zip(outputs, targets):
                image_id = target["image_id_str"]
                width = int(target["image_width"])
                height = int(target["image_height"])

                pred_string = prediction_dict_to_string(
                    pred,
                    width,
                    height,
                    score_threshold=args.score_threshold,
                    nf_huge_area_frac=args.nf_huge_area_frac,
                )

                if args.inject_no_finding:
                    pathology_scores = []
                    boxes = pred["boxes"].cpu().numpy()
                    scores = pred["scores"].cpu().numpy()
                    labels = pred["labels"].cpu().numpy()
                    for box, score, label in zip(boxes, scores, labels):
                        label = int(label)
                        if label == 0 or label == MODEL_NO_FINDING_IDX:
                            continue
                        if float(score) < args.score_threshold:
                            continue
                        if label not in MODEL_IDX_TO_KAGGLE:
                            continue
                        pathology_scores.append(float(score))

                    pred_string = inject_no_finding_if_empty(
                        pred_string,
                        pathology_scores,
                        width,
                        height,
                        full_image_box=args.nf_full_image_box,
                    )

                rows.append(
                    {"image_id": image_id, "PredictionString": pred_string}
                )

    df = build_submission_dataframe(
        rows,
        sample_submission_path=args.sample_submission,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote submission: {args.output} ({len(df)} rows)")
    nonempty = (df["PredictionString"].str.strip() != "").sum()
    print(f"  Images with >=1 detection token: {nonempty}/{len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="VinBigData Kaggle test inference -> submission.csv"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Model config (.yaml) used at training time. If omitted, a default "
            "VinDr-CXR config is selected from --gmm-type."
        ),
    )
    parser.add_argument(
        "--gmm-type",
        type=str,
        default=None,
        choices=["1D", "HARD_LABEL"],
        help=(
            "Checkpoint/model family to load, matching the evaluation script. "
            "If omitted with --config, inferred from DATA.GMM_TYPE; if both "
            "are omitted, defaults to HARD_LABEL."
        ),
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Checkpoint path (.pth)",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Directory containing test/*.png (or flat folder of PNGs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs/vindrcxr/submission.csv"),
        help="Output submission CSV path",
    )
    parser.add_argument(
        "--sample-submission",
        type=str,
        default=None,
        help="Optional sample_submission.csv to fix row order and ensure all IDs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of model classes including background; defaults to config",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet50", "resnet101", "mobilenet"],
        help="Backbone architecture; defaults to config",
    )
    parser.add_argument(
        "--image-extension",
        type=str,
        default=".png",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Glob PNGs recursively under test-dir",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Extra score filter at CSV time (model already uses score_thresh=0.4)",
    )
    parser.add_argument(
        "--nf-huge-area-frac",
        type=float,
        default=0.90,
        help="Treat No-finding boxes larger than this image-area fraction as full-image placeholders",
    )
    parser.add_argument(
        "--no-inject-no-finding",
        dest="inject_no_finding",
        action="store_false",
        help="Do not add class_id=14 No finding when no pathology boxes remain",
    )
    parser.set_defaults(inject_no_finding=True)

    parser.add_argument(
        "--nf-full-image-box",
        action="store_true",
        default=False,
        help="Use full-image box for injected No finding instead of Kaggle marker box",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
    )
    args = parser.parse_args()

    if args.config is None:
        args.gmm_type = args.gmm_type or "HARD_LABEL"
        args.config = default_config_for_gmm_type(args.gmm_type)
        print(f"Auto-selected config: {args.config}")
    elif args.gmm_type is None:
        cfg = get_cfg_from_file(args.config)
        args.gmm_type = getattr(cfg.DATA, "GMM_TYPE", "HARD_LABEL")
        if args.gmm_type not in ("1D", "HARD_LABEL"):
            raise ValueError(
                f"Unsupported DATA.GMM_TYPE in config: {args.gmm_type}"
            )
        print(f"Inferred model type from config: {args.gmm_type}")

    if args.nf_huge_area_frac is not None and args.nf_huge_area_frac < 0:
        args.nf_huge_area_frac = None

    run_inference(args)


if __name__ == "__main__":
    main()