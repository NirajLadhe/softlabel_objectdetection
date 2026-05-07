"""
NullAdapter — passthrough adapter for datasets that are already normalised.

Use this when:
- Your dataset CSV already uses canonical column names.
- Bounding boxes are already in [0, 1] coordinates.
- There is no 'No Finding' category.

Contributors can subclass NullAdapter and override only the properties/methods
they need to customise rather than implementing the full ABC from scratch.
"""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter


class NullAdapter(DatasetPreprocessingAdapter):
    """No-op adapter: returns the DataFrame unchanged."""

    @property
    def no_finding_labels(self) -> List[str]:
        return []

    @property
    def no_finding_id(self) -> Optional[int]:
        return None

    @property
    def image_file_extension(self) -> str:
        return ""

    @property
    def supports_absent_rater_vote(self) -> bool:
        return False

    def normalize_csv(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["image_id"] = df["image_id"].astype(str)
        df["rad_id"] = df["rad_id"].astype(str)
        return df