"""
Dataset adapters.

VinDrCXRDataset — VinDr-CXR soft/hard label dataset (primary dataset).

Adding a new dataset
--------------------
1. Create a new file, e.g. texbig.py, subclassing DetectionDatasetPort.
2. Register it here with an import.
"""

from softlabel_od.object_detection.adapters.datasets.vindrcxr import Vin_Dr_CXR_SoftLabel_Dataset
