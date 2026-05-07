"""
Dataset adapter registry for the soft-label generation pipeline.

Contributors: register your adapter here with a dataset name.
run_pipeline.py and manipulate_csv.py look up adapters by name.
"""

from __future__ import annotations

from typing import Dict, Type

from softlabel_od.softlabel_generation.ports.dataset_adapter import DatasetPreprocessingAdapter

_ADAPTER_REGISTRY: Dict[str, Type[DatasetPreprocessingAdapter]] = {}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "").replace("_", "")


def register_adapter(name: str, adapter_cls: Type[DatasetPreprocessingAdapter]) -> None:
    _ADAPTER_REGISTRY[_normalize_name(name)] = adapter_cls


def get_adapter(name: str) -> DatasetPreprocessingAdapter:
    """Return an instantiated adapter for *name*, or raise KeyError."""
    key = _normalize_name(name)
    if key not in _ADAPTER_REGISTRY:
        available = ", ".join(sorted(_ADAPTER_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown dataset adapter '{name}'. "
            f"Registered adapters: {available}. "
            f"Add a new adapter in softlabel_generation/adapters/datasets/ and "
            f"register it in adapters/datasets/__init__.py."
        )
    return _ADAPTER_REGISTRY[key]()