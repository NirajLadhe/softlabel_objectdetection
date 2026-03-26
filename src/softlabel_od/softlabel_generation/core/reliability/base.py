"""
Abstract base class for annotator reliability calculation.

To add a new reliability backend:
1. Subclass ReliabilityCalculator
2. Implement calculate()
3. Register in calculate_reliability.py

The output contract (for downstream use in priors + GMM) is a CSV with
at minimum these columns:
    Annotator        — string rater ID matching the annotation JSON
    Avg K-alpha      — float reliability score (higher = more reliable)
"""

from abc import ABC, abstractmethod
from pathlib import Path


class ReliabilityCalculator(ABC):
    """
    Dataset-agnostic interface for computing per-rater reliability scores.
    """

    @abstractmethod
    def calculate(
        self,
        annotation_path: str,
        iou_threshold: float,
        **kwargs,
    ) -> list[dict]:
        """
        Compute per-rater reliability scores.

        Args:
            annotation_path: Path to a COCO-style JSON with rater_id on
                             each annotation and rater_list on each image.
            iou_threshold:   IoU (or similarity) threshold for matching.
            **kwargs:        Backend-specific options.

        Returns:
            List of dicts, each with at minimum:
                {"Annotator": str, "Avg K-alpha": float}
            Additional columns are allowed and will be written to the CSV.
        """
        ...