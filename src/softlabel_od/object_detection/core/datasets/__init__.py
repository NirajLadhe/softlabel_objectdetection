"""
Shared dataset utilities for object detection.

This package contains reusable logic for canonical label-file consumers so that
new dataset adapters only need to define class mappings and image resolution.
"""

from softlabel_od.object_detection.core.datasets.base import CanonicalDetectionDataset

__all__ = ["CanonicalDetectionDataset"]
