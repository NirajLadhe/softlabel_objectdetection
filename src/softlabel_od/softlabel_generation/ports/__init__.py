"""
Ports (abstract interfaces) for the soft label generation pipeline.

CorrespondenceSolver — plug in any annotation clustering / matching algorithm.
"""

from softlabel_od.softlabel_generation.ports.correspondence import (
    CorrespondenceSolver,
    AHCAdapter,
)
