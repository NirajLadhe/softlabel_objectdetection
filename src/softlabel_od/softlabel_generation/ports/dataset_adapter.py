"""
Abstract base for dataset-specific preprocessing adapters.

Implement DatasetPreprocessingAdapter to integrate a new multi-annotator dataset
into the soft-label pipeline without modifying any core pipeline files.

Canonical CSV contract
----------------------
After normalize_csv() returns, the DataFrame must have these column names:

    image_id    : str  — unique image identifier
    rad_id      : str  — annotator / rater identifier
    class_name  : str  — human-readable class label
    class_id    : int  — integer class index
    x_min       : float — left   edge, normalised to [0, 1]
    y_min       : float — top    edge, normalised to [0, 1]
    x_max       : float — right  edge, normalised to [0, 1]
    y_max       : float — bottom edge, normalised to [0, 1]
    width       : int  — original image width  in pixels
    height      : int  — original image height in pixels

For 'no-finding' rows, x_min/y_min/x_max/y_max may be 0/0/1/1 (placeholder).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import pandas as pd


class DatasetPreprocessingAdapter(ABC):
    """Dataset-specific preprocessing hooks for the soft-label pipeline."""

    @property
    @abstractmethod
    def no_finding_labels(self) -> List[str]:
        """Class names that represent 'No Finding' (matched case-insensitively)."""
        ...

    @property
    def no_finding_id(self) -> Optional[int]:
        """Integer class_id for 'No Finding' rows; None if dataset has no such class."""
        return None

    @property
    def image_file_extension(self) -> str:
        """
        Extension appended to image_id when writing file_name into COCO JSON.
        E.g. '.dicom' for VinDr-CXR, '.jpg' for most natural-image datasets.
        Return '' to use image_id as-is.
        """
        return ""

    @property
    def supports_absent_rater_vote(self) -> bool:
        """
        When True, raters who annotated an image but placed no box in a cluster
        cast a weighted implicit 'no-finding-in-cluster' vote.
        Set to False for datasets without a 'No Finding' category.
        """
        return True

    @abstractmethod
    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise a raw dataset CSV into the canonical column contract above.

        Typical responsibilities:
        - Rename dataset-specific column names to canonical names.
        - Normalise bounding box coordinates to [0, 1] using image width/height.
        - Cast image_id and rad_id to str.
        - Fill missing bbox values for 'No Finding' rows.

        Args:
            df: Raw DataFrame as loaded from the contributor's input CSV.

        Returns:
            DataFrame satisfying the canonical column contract.
        """
        ...