"""
Inter-annotator reliability core.

Public API
----------
ReliabilityCalculator       — abstract base (subclass to add a new backend)
KalosReliabilityCalculator  — KαLOS Vision-Alpha backend (recommended)
"""

from softlabel_od.softlabel_generation.core.reliability.base import (ReliabilityCalculator)
from softlabel_od.softlabel_generation.core.reliability.kalos_reliability import (KalosReliabilityCalculator)

__all__ = ["ReliabilityCalculator", "KalosReliabilityCalculator"]