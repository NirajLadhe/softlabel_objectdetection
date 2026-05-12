"""
COCO-LVIS preprocessing adapter.

Handles bbox coordinate normalization for the COCO-LVIS multi-annotator
real-world image dataset.

Raw LVIS/COCO bbox format:
    [x, y, width, height]

Internal pipeline bbox format:
    x_min, y_min, x_max, y_max normalized to [0, 1]

The adapter also standardizes:
    rater_id    -> rad_id
    category_id -> class_id
"""

from __future__ import annotations

import ast
from typing import List, Optional

import numpy as np
import pandas as pd

from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter


class LVISAdapter(DatasetPreprocessingAdapter):
    """
    Dataset adapter for COCO-LVIS.

    LVIS is a real-world object detection / instance annotation dataset.
    Unlike VinDr-CXR, it does not contain a medical "No Finding" class.

    Expected raw or intermediate dataframe columns can be either:

    Case 1: already expanded bbox columns:
        image_id, rater_id/rad_id, category_id/class_id,
        x_min, y_min, x_max, y_max, width, height

    Case 2: COCO bbox format:
        image_id, rater_id/rad_id, category_id/class_id,
        bbox, width, height

    where bbox is [x, y, width, height].
    """

    @property
    def no_finding_labels(self) -> List[str]:
        """
        LVIS does not have a VinDr-style 'No Finding' class.
        """
        return []

    @property
    def no_finding_id(self) -> Optional[int]:
        """
        Use -1 as a sentinel value because LVIS has no no-finding category.
        """
        return -1

    @property
    def image_file_extension(self) -> str:
        """
        Real COCO-LVIS images are JPEG files.

        If your old pipeline has hardcoded VinDr/DICOM assumptions,
        patch that pipeline instead of pretending LVIS images are DICOM.
        """
        return ".jpg"

    @property
    def supports_absent_rater_vote(self) -> bool:
        """
        LVIS-style multi-rater annotations can support absent-rater logic
        if rater information is available.
        """
        return True

    def _parse_bbox(self, bbox):
        """
        Parse bbox if it comes from CSV as a string.

        Accepts:
            [x, y, w, h]
            "[x, y, w, h]"
        """
        if isinstance(bbox, str):
            try:
                bbox = ast.literal_eval(bbox)
            except Exception:
                return None

        if not isinstance(bbox, (list, tuple, np.ndarray)):
            return None

        if len(bbox) != 4:
            return None

        return [float(v) for v in bbox]

    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize LVIS CSV for the soft-label pipeline.

        If bbox values are already in [0, 1], keep them unchanged.
        If bbox values are absolute pixels, normalize using width/height.
        """

        df = df.copy()

        required_cols = [
            "image_id",
            "rad_id",
            "class_id",
            "class_name",
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]

        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"LVIS CSV missing required columns: {missing}")

        df["image_id"] = df["image_id"].astype(str)
        df["rad_id"] = df["rad_id"].astype(str)
        df["class_id"] = df["class_id"].astype(int)
        df["class_name"] = df["class_name"].astype(str)

        bbox_cols = ["x_min", "y_min", "x_max", "y_max"]

        for col in bbox_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if df[bbox_cols].isna().any().any():
            raise ValueError("LVIS CSV contains invalid bbox values.")

        bbox_min = df[bbox_cols].min().min()
        bbox_max = df[bbox_cols].max().max()

        already_normalized = bbox_min >= 0.0 and bbox_max <= 1.0

        if already_normalized:
            print("LVIS bboxes already normalized. Skipping width/height normalization.")
        else:
            if "width" not in df.columns or "height" not in df.columns:
                raise ValueError(
                    "LVIS bboxes are not normalized, and width/height columns are missing."
                )

            df["width"] = pd.to_numeric(df["width"], errors="coerce")
            df["height"] = pd.to_numeric(df["height"], errors="coerce")

            if df[["width", "height"]].isna().any().any():
                raise ValueError("LVIS CSV contains invalid width/height values.")

            df["x_min"] = df["x_min"] / df["width"]
            df["x_max"] = df["x_max"] / df["width"]
            df["y_min"] = df["y_min"] / df["height"]
            df["y_max"] = df["y_max"] / df["height"]

        for col in bbox_cols:
            df[col] = df[col].clip(0.0, 1.0)

        valid = (
            (df["x_max"] > df["x_min"]) &
            (df["y_max"] > df["y_min"])
        )

        before = len(df)
        df = df.loc[valid].copy()
        after = len(df)

        print(f"LVIS valid boxes kept: {after}/{before}")

        return df