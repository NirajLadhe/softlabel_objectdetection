"""
Dataset adapters for annotation preprocessing.

Each adapter normalises a dataset's raw CSV into the canonical column format
expected by the core pipeline. See ports/dataset_adapter.py for the contract.

Adding a new dataset
--------------------
1. Create <yourdata>.py in this directory, subclassing DatasetPreprocessingAdapter
   (or NullAdapter if only minor changes are needed).
2. Register it below with register_adapter().
3. Add a config at configs/softlabel_pipeline/<yourdata>.yaml.
"""

from softlabel_od.softlabel_generation.adapters.datasets.registry import register_adapter
from softlabel_od.softlabel_generation.adapters.datasets.vindrcxr import VinDrCXRAdapter
from softlabel_od.softlabel_generation.adapters.datasets.null_adapter import NullAdapter
from softlabel_od.softlabel_generation.adapters.datasets.multiorg import MultiOrgAdapter

register_adapter("vindrcxr", VinDrCXRAdapter)
register_adapter("vindr-cxr", VinDrCXRAdapter)   # alias
register_adapter("null", NullAdapter)
register_adapter("multiorg", MultiOrgAdapter)