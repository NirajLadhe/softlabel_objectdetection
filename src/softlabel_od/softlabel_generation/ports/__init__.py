"""
Ports (abstract interfaces) for the soft-label generation pipeline.

CorrespondenceSolver  — plug in any annotation clustering algorithm.
DatasetPreprocessingAdapter — plug in dataset-specific preprocessing.
"""

from softlabel_od.softlabel_generation.ports.correspondence import (
    CorrespondenceSolver,
    AHCAdapter,
)
from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter