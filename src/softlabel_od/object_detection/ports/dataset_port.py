"""
Abstract dataset port for the object detection pipeline.

All dataset adapters (VinDr-CXR, TexBig, new datasets) must subclass
DetectionDatasetPort so the training loop can treat them uniformly.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import torch


class DetectionDatasetPort(ABC, torch.utils.data.Dataset):
    """
    Unified interface for detection datasets.

    Subclasses must implement __getitem__ returning (image_tensor, target_dict)
    where target_dict contains at minimum:
        'boxes'   : FloatTensor [N, 4]  in [x1, y1, x2, y2] pixel format
        'labels'  : LongTensor  [N]
        'image_id': Tensor      [1]

    Soft-label datasets additionally include:
        'gt_means'          : FloatTensor [N, 4]  normalised [x, y, w, h]
        'gt_stds'           : FloatTensor [N, 4]
        'gt_class_probs'    : FloatTensor [N, C]
        'annotation_weight' : FloatTensor [N]
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return (image_tensor, target_dict) for sample at idx."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
