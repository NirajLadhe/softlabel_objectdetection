"""
Model factory and meta-architecture.

get_model() is the single entry point for creating any supported model variant.
New architectures (YOLO, mmDetection, etc.) should be added here as adapters
and registered in this factory.
"""

from softlabel_od.object_detection.core.model.factory import get_model
