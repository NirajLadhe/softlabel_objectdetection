"""
MultiOrg preprocessing adapter.

Handles bbox coordinate normalisation for the MultiOrg dataset. 
Images are stored as PNG files; image IDs in the
CSV are bare hash strings without an extension.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from softlabel_od.softlabel_generation.adapters.datasets import NullAdapter


class MultiOrgAdapter(NullAdapter):

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
        return False

    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize bounding boxes to [0, 1]."""
        df = df.copy()
        
        # Fill missing bbox fields
        df["x_min"] = df["x_min"].fillna(0)
        df["y_min"] = df["y_min"].fillna(0)
        df["x_max"] = df["x_max"].fillna(1)
        df["y_max"] = df["y_max"].fillna(1)
        
        # Check if already normalized (all coordinates between 0 and 1)
        if (df["x_min"] >= 0).all() and (df["x_min"] <= 1).all() and \
           (df["x_max"] >= 0).all() and (df["x_max"] <= 1).all() and \
           (df["y_min"] >= 0).all() and (df["y_min"] <= 1).all() and \
           (df["y_max"] >= 0).all() and (df["y_max"] <= 1).all():
            return df
        
        # Check for width/height columns
        if "width" not in df.columns or "height" not in df.columns:
            raise ValueError(
                "MultiOrgAdapter: Cannot normalize - missing 'width' and/or 'height' columns"
            )
        
        # Normalize
        df["x_min"] = df["x_min"] / df["width"]
        df["y_min"] = df["y_min"] / df["height"]
        df["x_max"] = df["x_max"] / df["width"]
        df["y_max"] = df["y_max"] / df["height"]
        
        # Clip to [0, 1] for safety
        for col in ["x_min", "y_min", "x_max", "y_max"]:
            df[col] = df[col].clip(0, 1)
        
        return df