"""
TexBig preprocessing adapter.

 The prepare_texbig.py script first converts the COCO JSON
into the canonical CSV format. This adapter then handles the CSV normalisation
step that follows.

Key differences from VinDr-CXR:
  - No "No Finding" concept: every annotation is a real object (text region).
  - Boxes in the CSV are already in absolute pixel coordinates (x_min, y_min,
    x_max, y_max) after prepare_texbig.py runs the COCO → CSV conversion.
  - Annotator IDs are stored in the 'rater_id' column (populated by prepare_texbig.py).
  - Image IDs are string filenames, e.g. 'kunstgewerbe1890_0001.png'.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter


class TexBigAdapter(DatasetPreprocessingAdapter):
    """
    Adapter for the TexBig multi-annotator document layout dataset.

    TexBig has no background / "No Finding" class, so no_finding_id is None
    and no_finding_labels is empty. Bounding boxes are left in absolute pixel
    coordinates — the GMM pipeline works in normalised [0, 1] space, so we
    divide by image width and height here.
    """

    @property
    def no_finding_labels(self) -> List[str]:
        # TexBig has no "No Finding" or background-only class.
        return []

    @property
    def no_finding_id(self) -> Optional[int]:
        # No background class.
        return None

    @property
    def image_file_extension(self) -> str:
        return ".png"

    @property
    def supports_absent_rater_vote(self) -> bool:
        # Annotators only label what they see; absent votes are not recorded.
        return False

    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise TexBig bounding boxes from absolute pixel coords to [0, 1].

        After prepare_texbig.py, the CSV has columns:
            image_id, rater_id, class_id, class_name,
            x_min, y_min, x_max, y_max, width, height

        We normalise all bbox columns by dividing by image width/height.
        No special handling needed for a "No Finding" class.
        """
        df = df.copy()

        # Ensure canonical column types
        df["image_id"] = df["image_id"].astype(str)
        df["rater_id"] = df["rater_id"].astype(str)

        # Fill any missing bbox values with safe defaults
        df["x_min"] = df["x_min"].fillna(0)
        df["y_min"] = df["y_min"].fillna(0)
        df["x_max"] = df["x_max"].fillna(df["width"])
        df["y_max"] = df["y_max"].fillna(df["height"])

        # Normalise all rows to [0, 1] by image dimensions
        df["x_min"] = df["x_min"] / df["width"]
        df["y_min"] = df["y_min"] / df["height"]
        df["x_max"] = df["x_max"] / df["width"]
        df["y_max"] = df["y_max"] / df["height"]

        # Clamp to [0, 1] in case of floating-point drift
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            df[col] = df[col].clip(0.0, 1.0)

        return df
