"""
Dataset registry and factory for object detection adapters.

Contributors should only need to register a new dataset name here and supply a
YAML config selecting it. Training, evaluation, and data modules should not
import dataset adapters directly.
"""

from __future__ import annotations

from typing import Dict, Type

from softlabel_od.object_detection.utils.config import cfg_node_to_dict, resolve_label_file, resolve_label_mode

DATASET_REGISTRY: Dict[str, type] = {}


def normalize_dataset_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "")


def register_dataset(name: str, dataset_cls: Type) -> None:
    DATASET_REGISTRY[normalize_dataset_name(name)] = dataset_cls


def get_dataset_class(name: str):
    normalized_name = normalize_dataset_name(name)
    if normalized_name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY)) or "<none>"
        raise KeyError(f"Unknown dataset '{name}'. Available datasets: {available}")
    return DATASET_REGISTRY[normalized_name]


def build_detection_dataset(cfg, train: bool, transforms=None):
    dataset_name = getattr(cfg.DATA, "DATASET", "vindrcxr")
    dataset_cls = get_dataset_class(dataset_name)
    options = cfg_node_to_dict(getattr(cfg.DATA, "OPTIONS", {}))
    return dataset_cls(
        root=cfg.DATA.ROOT,
        label_file=resolve_label_file(cfg),
        transforms=transforms,
        train=train,
        test_split_ratio=cfg.DATA.TEST_SPLIT_RATIO,
        label_mode=resolve_label_mode(cfg),
        **options,
    )
