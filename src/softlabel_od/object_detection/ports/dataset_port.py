"""
Dataset port for the object detection pipeline.

The repository treats dataset adapters as plug-and-play components selected by
configuration. To keep training, evaluation, sampling, and future dataset
extensions aligned, adapters expose both the standard PyTorch Dataset protocol
and a small amount of image-level metadata used by shared utilities.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List, Set
import torch


class DetectionDatasetPort(ABC, torch.utils.data.Dataset):
    """
    Unified interface for detection datasets that consume canonical label files.

    Required `__getitem__` output:
        'boxes'   : FloatTensor [N, 4] in [x1, y1, x2, y2] pixel format
        'labels'  : LongTensor  [N]    with background reserved as 0
        'image_id': Tensor      [1]

    1D soft-label datasets additionally include:
        'gt_means'          : FloatTensor [N, 4] normalized [x, y, w, h]
        'gt_stds'           : FloatTensor [N, 4]
        'gt_class_probs'    : FloatTensor [N, C]
        'annotation_weight' : FloatTensor [N]

    2D soft-label datasets additionally include:
        'gt_vertices'       : FloatTensor [N, 8]
        'gt_covariances'    : list
        'gt_class_probs'    : FloatTensor [N, C]
    """

    @property
    @abstractmethod
    def class_to_idx(self) -> Dict[str, int]:
        """Mapping from canonical class names to foreground indices."""
    
    @property
    def class_names(self) -> Dict[int, str]:
        """Mapping from foreground index to class name. Derived from class_to_idx."""
        return {v: k for k, v in self.class_to_idx.items()}

    @property
    @abstractmethod
    def image_ids(self) -> List[str]:
        """Ordered image ids for the active split."""

    @property
    @abstractmethod
    def image_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Canonical label records grouped by `image_id`."""

    @property
    @abstractmethod
    def label_mode(self) -> str:
        """One of `hard`, `soft_1d`, or `soft_2d`."""

    @property
    @abstractmethod
    def dataset_type(self) -> str:
        """Legacy-compatible coarse type: `hard` or `soft`."""

    @abstractmethod
    def get_image_ids(self) -> List[str]:
        """Return active split image ids."""

    @abstractmethod
    def get_records_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        """Return canonical label records for one image."""

    @abstractmethod
    def get_presence_labels(self, image_id: str, threshold: float = 0.1) -> Set[int]:
        """Return foreground class ids present in an image for split/sampling."""

    @abstractmethod
    def get_dominant_class(
        self,
        labels_data: List[Dict[str, Any]] | None = None,
        image_id: str | None = None,
    ) -> int:
        """Return the dominant class id for a record list or image."""

    @abstractmethod
    def compute_roi_head_class_weights_per_cluster(self, **kwargs) -> torch.Tensor:
        """Compute class weights for the ROI head on the active split."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return `(image_tensor, target_dict)` for sample at idx."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of images in the active split."""
