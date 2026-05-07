"""
VinDr-CXR preprocessing adapter.

Handles bbox coordinate normalisation for the VinDr-CXR multi-annotator
chest X-ray dataset. Images are stored as DICOM files; image IDs in the
CSV are bare hash strings without an extension.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter


class VinDrCXRAdapter(DatasetPreprocessingAdapter):

    @property
    def no_finding_labels(self) -> List[str]:
        return ["No finding", "No findings"]

    @property
    def no_finding_id(self) -> Optional[int]:
        return 14

    @property
    def image_file_extension(self) -> str:
        return ".dicom"

    @property
    def supports_absent_rater_vote(self) -> bool:
        return True

    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise VinDr-CXR bounding boxes from absolute pixel coords to [0, 1].

        'No Finding' rows receive placeholder coordinates (0, 0, 1, 1).
        All other rows are normalised by the image's pixel dimensions.
        """
        df = df.copy()

        # Ensure canonical column types
        df["image_id"] = df["image_id"].astype(str)
        df["rad_id"] = df["rad_id"].astype(str)

        # Fill missing bbox fields
        df["x_min"] = df["x_min"].fillna(0)
        df["y_min"] = df["y_min"].fillna(0)
        df["x_max"] = df["x_max"].fillna(1)
        df["y_max"] = df["y_max"].fillna(1)

        no_finding_ids = [self.no_finding_id] if self.no_finding_id is not None else []
        is_finding = ~df["class_id"].isin(no_finding_ids)

        df.loc[is_finding, "x_min"] = df.loc[is_finding, "x_min"] / df.loc[is_finding, "width"]
        df.loc[is_finding, "y_min"] = df.loc[is_finding, "y_min"] / df.loc[is_finding, "height"]
        df.loc[is_finding, "x_max"] = df.loc[is_finding, "x_max"] / df.loc[is_finding, "width"]
        df.loc[is_finding, "y_max"] = df.loc[is_finding, "y_max"] / df.loc[is_finding, "height"]

        return df