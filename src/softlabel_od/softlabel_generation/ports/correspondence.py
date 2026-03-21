"""
Correspondence solver port.
Abstract interface for annotation clustering; AHC adapter wraps current scipy logic.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class CorrespondenceSolver(ABC):
    """Abstract interface for annotation correspondence/clustering."""

    @abstractmethod
    def solve(
        self,
        annotations: List[Dict[str, Any]],
        distance_metric: str = "iou",
        threshold: float = 0.5,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Cluster annotations into correspondence groups.

        Args:
            annotations: List of annotation dicts with 'bbox', 'rater_id', 'class_name'
            distance_metric: 'iou', 'iomin', or 'iomin_center_gate'
            threshold: Distance threshold for clustering
            **kwargs: Metric-specific params (e.g. center_ratio_threshold)

        Returns:
            List of clusters, each cluster is a list of annotations
        """
        pass


class AHCAdapter(CorrespondenceSolver):
    """Adapter wrapping scipy hierarchical clustering (current implementation)."""

    def solve(
        self,
        annotations: List[Dict[str, Any]],
        distance_metric: str = "iou",
        threshold: float = 0.5,
        center_ratio_threshold: float = 0.5,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        from softlabel_od.softlabel_generation.utils.clustering import cluster_annotations

        method = distance_metric if distance_metric in ("iou", "iomin", "iomin_center_gate") else "iou"
        return cluster_annotations(
            annotations,
            method=method,
            iou_threshold=threshold,
            iomin_threshold=threshold,
            center_ratio_threshold=center_ratio_threshold,
        )
