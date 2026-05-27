"""
Dataset adapters.

VinDrCXRDataset — VinDr-CXR soft/hard label dataset (primary dataset).

Adding a new dataset
--------------------
1. Create a new adapter subclassing `CanonicalDetectionDataset` or
   `DetectionDatasetPort`.
2. Register it in the dataset registry.
"""

from softlabel_od.object_detection.adapters.datasets.registry import (
    build_detection_dataset,
    get_dataset_class,
    register_dataset,
)
from softlabel_od.object_detection.adapters.datasets.vindrcxr import (
    VinDrCXRDataset,
    Vin_Dr_CXR_SoftLabel_Dataset,
)

from softlabel_od.object_detection.adapters.datasets.nucls import NuCLSDataset

register_dataset("vindrcxr", Vin_Dr_CXR_SoftLabel_Dataset)
register_dataset("vindr-cxr", Vin_Dr_CXR_SoftLabel_Dataset)
register_dataset("nucls", NuCLSDataset)

__all__ = [
    "Vin_Dr_CXR_SoftLabel_Dataset",
    "VinDrCXRDataset",
    "build_detection_dataset",
    "get_dataset_class",
    "register_dataset",
    "NuCLSDataset",
]
