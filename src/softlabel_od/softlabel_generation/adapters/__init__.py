"""
Adapters for the soft-label generation pipeline.

datasets/   — dataset-specific preprocessing adapters.
              Add new datasets here without touching core logic.
"""

from softlabel_od.softlabel_generation.adapters.datasets.registry import get_adapter, register_adapter