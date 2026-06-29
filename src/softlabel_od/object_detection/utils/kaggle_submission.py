"""
VinBigData Kaggle submission helpers.

Maps model label indices (from Vin_Dr_CXR_SoftLabel_Dataset.build_class_to_idx)
to official competition class_id values (0-13 pathology, 14 = No finding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

# Model index -> Kaggle class_id (by class name, not by offset)
MODEL_IDX_TO_KAGGLE: dict[int, int] = {
    1: 14,   # No Finding
    2: 0,    # Aortic enlargement
    3: 1,    # Atelectasis
    4: 2,    # Calcification
    5: 3,    # Cardiomegaly
    6: 4,    # Consolidation
    7: 5,    # ILD
    8: 6,    # Infiltration
    9: 7,    # Lung Opacity
    10: 8,   # Nodule/Mass
    11: 9,   # Other lesion
    12: 10,  # Pleural effusion
    13: 11,  # Pleural thickening
    14: 12,  # Pneumothorax
    15: 13,  # Pulmonary fibrosis
}

KAGGLE_NO_FINDING_ID = 14
MODEL_NO_FINDING_IDX = 1
# Marker box for "healthy" / empty CXR in submission space
NF_SUBMISSION_BOX = (0, 0, 1, 1)
DEFAULT_NF_HUGE_AREA_FRAC = 0.90


def _to_numpy(pred: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = pred["boxes"]
    scores = pred["scores"]
    labels = pred["labels"]
    if hasattr(boxes, "cpu"):
        boxes = boxes.cpu().numpy()
    if hasattr(scores, "cpu"):
        scores = scores.cpu().numpy()
    if hasattr(labels, "cpu"):
        labels = labels.cpu().numpy()
    return boxes, scores, labels


def is_huge_nofinding_box(
    box: Sequence[float],
    image_width: int,
    image_height: int,
    *,
    huge_area_frac: float = DEFAULT_NF_HUGE_AREA_FRAC,
) -> bool:
    """True if this NF box looks like a full-image placeholder."""
    if image_width <= 0 or image_height <= 0:
        return True
    x1, y1, x2, y2 = clip_box_xyxy(box, image_width, image_height)
    area_frac = (x2 - x1) * (y2 - y1) / (image_width * image_height)
    return area_frac > huge_area_frac


def format_detection_token(
    kaggle_class_id: int,
    score: float,
    box_xyxy: tuple[int, int, int, int],
) -> str:
    x1, y1, x2, y2 = box_xyxy
    return f"{kaggle_class_id} {float(score):.4f} {x1} {y1} {x2} {y2}"


def resolve_nofinding_policy(
    pred: dict,
    image_width: int,
    image_height: int,
    *,
    score_threshold: float = 0.0,
    nf_huge_area_frac: float = DEFAULT_NF_HUGE_AREA_FRAC,
) -> List[str]:
    """
    Deterministic healthy vs pathological resolution for one image.

    - Huge NF boxes are never written as huge; if NF is kept, use (0,0,1,1).
    - If max(NF scores) >= max(pathology scores): single NF token, suppress pathology.
    - Else: pathology tokens only, drop NF.
    - NF-only images: single NF token at (0,0,1,1) with best NF score.
    """
    boxes, scores, labels = _to_numpy(pred)

    nf_scores: List[float] = []
    pathology: List[tuple[float, int, Sequence[float]]] = []

    for box, score, label in zip(boxes, scores, labels):
        label = int(label)
        s = float(score)
        if s < score_threshold or label == 0:
            continue
        if label == MODEL_NO_FINDING_IDX:
            # Only huge NF boxes participate (your training placeholder pattern)
            if is_huge_nofinding_box(
                box, image_width, image_height, huge_area_frac=nf_huge_area_frac
            ):
                nf_scores.append(s)
            continue
        if label not in MODEL_IDX_TO_KAGGLE:
            continue
        pathology.append((s, label, box))

    tokens: List[str] = []

    # --- Healthy branch: NF wins or NF-only ---
    if nf_scores and (not pathology or max(nf_scores) >= max(p[0] for p in pathology)):
        best_nf = max(nf_scores)
        tokens.append(
            format_detection_token(
                KAGGLE_NO_FINDING_ID, best_nf, NF_SUBMISSION_BOX
            )
        )
        return tokens

    # --- Pathological branch ---
    for s, label, box in pathology:
        kaggle_id = MODEL_IDX_TO_KAGGLE[label]
        xyxy = clip_box_xyxy(box, image_width, image_height)
        tokens.append(format_detection_token(kaggle_id, s, xyxy))

    return tokens


def prediction_dict_to_string(
    pred: dict,
    image_width: int,
    image_height: int,
    *,
    score_threshold: float = 0.0,
    nf_huge_area_frac: float = DEFAULT_NF_HUGE_AREA_FRAC,
    max_nf_box_area_frac: float | None = None,  # deprecated, ignored
) -> str:
    """Build PredictionString with NF vs pathology resolution."""
    parts = resolve_nofinding_policy(
        pred,
        image_width,
        image_height,
        score_threshold=score_threshold,
        nf_huge_area_frac=nf_huge_area_frac,
    )
    return " ".join(parts)

def clip_box_xyxy(
    box: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(max(0, min(round(x1), width - 1)))
    y1 = int(max(0, min(round(y1), height - 1)))
    x2 = int(max(0, min(round(x2), width - 1)))
    y2 = int(max(0, min(round(y2), height - 1)))
    if x2 <= x1:
        x2 = min(x1 + 1, width - 1)
    if y2 <= y1:
        y2 = min(y1 + 1, height - 1)
    return x1, y1, x2, y2


# def detection_to_tokens(
#     box: Sequence[float],
#     score: float,
#     model_label: int,
#     image_width: int,
#     image_height: int,
#     *,
#     skip_background: bool = True,
#     max_nf_box_area_frac: float | None = 0.95,
# ) -> str | None:
#     """
#     One detection -> 'class_id score x_min y_min x_max y_max'.
#     Returns None if the detection should be dropped.
#     """
#     label = int(model_label)
#     if skip_background and label == 0:
#         return None
#     if label not in MODEL_IDX_TO_KAGGLE:
#         return None

#     kaggle_id = MODEL_IDX_TO_KAGGLE[label]
#     x1, y1, x2, y2 = clip_box_xyxy(box, image_width, image_height)

#     if (
#         max_nf_box_area_frac is not None
#         and label == MODEL_NO_FINDING_IDX
#         and image_width > 0
#         and image_height > 0
#     ):
#         area_frac = (x2 - x1) * (y2 - y1) / (image_width * image_height)
#         if area_frac > max_nf_box_area_frac:
#             return None

#     return f"{kaggle_id} {float(score):.4f} {x1} {y1} {x2} {y2}"


# def prediction_dict_to_string(
#     pred: dict,
#     image_width: int,
#     image_height: int,
#     *,
#     score_threshold: float = 0.0,
#     max_nf_box_area_frac: float | None = 0.95,
# ) -> str:
#     boxes = pred["boxes"]
#     scores = pred["scores"]
#     labels = pred["labels"]

#     if hasattr(boxes, "cpu"):
#         boxes = boxes.cpu().numpy()
#     if hasattr(scores, "cpu"):
#         scores = scores.cpu().numpy()
#     if hasattr(labels, "cpu"):
#         labels = labels.cpu().numpy()

#     parts: List[str] = []
#     for box, score, label in zip(boxes, scores, labels):
#         if float(score) < score_threshold:
#             continue
#         token = detection_to_tokens(
#             box,
#             float(score),
#             int(label),
#             image_width,
#             image_height,
#             max_nf_box_area_frac=max_nf_box_area_frac,
#         )
#         if token:
#             parts.append(token)
#     return " ".join(parts)


# def inject_no_finding_if_empty(
#     prediction_string: str,
#     pathology_scores: Iterable[float],
#     image_width: int,
#     image_height: int,
#     *,
#     full_image_box: bool = True,
# ) -> str:
#     """
#     Competition-style fallback: if no pathology boxes remain, emit No finding (14).
#     Score = 1 - max(pathology confidence), or 1.0 if none were scored.
#     """
#     if prediction_string.strip():
#         return prediction_string

#     scores = list(pathology_scores)
#     nf_score = 1.0 - max(scores) if scores else 1.0
#     nf_score = float(np.clip(nf_score, 1e-4, 1.0))

#     if full_image_box and image_width > 0 and image_height > 0:
#         x1, y1, x2, y2 = 0, 0, image_width - 1, image_height - 1
#     else:
#         x1, y1, x2, y2 = 0, 0, 1, 1

#     return f"{KAGGLE_NO_FINDING_ID} {nf_score:.4f} {x1} {y1} {x2} {y2}"
def inject_no_finding_if_empty(
    prediction_string: str,
    pathology_scores: Iterable[float],
    image_width: int,
    image_height: int,
    *,
    full_image_box: bool = False,
) -> str:
    if prediction_string.strip():
        return prediction_string
    scores = list(pathology_scores)
    nf_score = 1.0 - max(scores) if scores else 1.0
    nf_score = float(np.clip(nf_score, 1e-4, 1.0))
    if full_image_box and image_width > 0 and image_height > 0:
        x1, y1, x2, y2 = 0, 0, image_width - 1, image_height - 1
    else:
        x1, y1, x2, y2 = NF_SUBMISSION_BOX
    return f"{KAGGLE_NO_FINDING_ID} {nf_score:.4f} {x1} {y1} {x2} {y2}"


def build_submission_dataframe(
    rows: List[dict],
    sample_submission_path: str | None = None,
) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No predictions collected; check test image directory.")

    df["image_id"] = df["image_id"].astype(str)
    if "PredictionString" not in df.columns:
        df["PredictionString"] = ""

    if sample_submission_path:
        sample = pd.read_csv(sample_submission_path)
        sample["image_id"] = sample["image_id"].astype(str)
        required_ids = sample["image_id"].tolist()
        by_id = df.set_index("image_id")["PredictionString"].to_dict()
        df = pd.DataFrame(
            {
                "image_id": required_ids,
                "PredictionString": [
                    by_id.get(i, "") for i in required_ids
                ],
            }
        )
        missing = set(required_ids) - set(by_id)
        if missing:
            print(
                f"Warning: {len(missing)} image_ids from sample_submission "
                f"had no predictions (empty PredictionString)."
            )

    return df[["image_id", "PredictionString"]]


class KaggleTestImageDataset(Dataset):
    """
    Image-only dataset for Kaggle test PNGs.
    Does not use canonical JSON labels or train/val splits.
    """

    def __init__(
        self,
        image_dir: str,
        *,
        image_extension: str = ".png",
        recursive: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.image_extension = image_extension
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Test image directory not found: {image_dir}")

        pattern = f"**/*{image_extension}" if recursive else f"*{image_extension}"
        paths = sorted(self.image_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"No {image_extension} files under {image_dir} "
                f"(recursive={recursive})"
            )
        self._paths = paths
        self._image_ids = [p.stem for p in paths]

    def __len__(self) -> int:
        return len(self._paths)

    @property
    def image_ids(self) -> List[str]:
        return list(self._image_ids)

    def __getitem__(self, idx: int):
        path = self._paths[idx]
        image_id = self._image_ids[idx]
        image = Image.open(path).convert("RGB")
        width, height = image.size

        target = {
            "image_id_str": image_id,
            "image_width": width,
            "image_height": height,
        }
        image_tensor = F.to_tensor(image)
        return image_tensor, target