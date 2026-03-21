"""
Inter-annotator reliability core.

Public API
----------
calculate_iaa_from_annotations  — compute Krippendorff's alpha from COCO JSON
ReliabilityData                 — multi-annotator matching matrix
"""

from softlabel_od.softlabel_generation.core.reliability.calculate_iaa import (
    calculate_iaa_from_annotations,
)
from softlabel_od.softlabel_generation.core.reliability.reliability_data import (
    ReliabilityData,
    MatrixEntry,
    EmptyEntry,
)
