"""
NuCLS preprocessing adapter.

Handles bbox coordinate normalisation for the NuCLS multi-annotator
pathology nucleus dataset. Images are PNG crops from whole-slide images.
Every annotation is a real nucleus — there is no 'No Finding' category.
Raters annotate only their assigned FOV, so absent raters should NOT
vote NoFindingInCluster.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter


class NuCLSAdapter(DatasetPreprocessingAdapter):

    @property
    def no_finding_labels(self) -> List[str]:
        return []

    @property
    def no_finding_id(self) -> Optional[int]:
        return None

    @property
    def image_file_extension(self) -> str:
        return ".png"

    @property
    def supports_absent_rater_vote(self) -> bool:
        # Each rater sees only their own FOV crop.
        # Absence from a cluster does NOT mean "I saw it and found nothing".
        return False

    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize NuCLS bounding boxes from pixel coords to [0, 1].

        Every row is a real nucleus annotation — no No Finding special case.
        Inverted boxes are corrected; zero-area boxes are dropped.
        """
        df = df.copy()

        # canonical types
        df["image_id"] = df["image_id"].astype(str)
        df["rad_id"]   = df["rad_id"].astype(str)
        df["class_id"] = df["class_id"].astype(int)

        # drop rows with missing bbox
        bbox_cols = ["x_min", "y_min", "x_max", "y_max"]
        before = len(df)
        df = df.dropna(subset=bbox_cols).copy()
        dropped = before - len(df)
        if dropped:
            print(f"  [NuCLSAdapter] Dropped {dropped} rows with missing bbox.")

        # fix inverted boxes
        df["x_min"], df["x_max"] = (
            np.minimum(df["x_min"], df["x_max"]),
            np.maximum(df["x_min"], df["x_max"]),
        )
        df["y_min"], df["y_max"] = (
            np.minimum(df["y_min"], df["y_max"]),
            np.maximum(df["y_min"], df["y_max"]),
        )

        # drop zero-area boxes
        before = len(df)
        df = df[
            (df["x_max"] > df["x_min"]) & (df["y_max"] > df["y_min"])
        ].copy()
        dropped = before - len(df)
        if dropped:
            print(f"  [NuCLSAdapter] Dropped {dropped} zero-area boxes.")

        # normalize all coords to [0, 1]
        df["x_min"] = (df["x_min"] / df["width"]).clip(0.0, 1.0)
        df["y_min"] = (df["y_min"] / df["height"]).clip(0.0, 1.0)
        df["x_max"] = (df["x_max"] / df["width"]).clip(0.0, 1.0)
        df["y_max"] = (df["y_max"] / df["height"]).clip(0.0, 1.0)

        return df
