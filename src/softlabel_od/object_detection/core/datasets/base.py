"""
Reusable canonical-label dataset implementation for object detection.

Dataset adapters subclass `CanonicalDetectionDataset` and provide only the
dataset-specific pieces that should differ across contributors:
class taxonomy, image-path resolution, and optional adapter defaults.
"""

from __future__ import annotations

import json
import os
from abc import abstractmethod
from collections import Counter
from typing import Any, Dict, List, Set, Literal

import numpy as np
import torch
from PIL import Image
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from softlabel_od.object_detection.ports.dataset_port import DetectionDatasetPort
from softlabel_od.object_detection.utils.training_helpers import load_rater_weights
from softlabel_od.object_detection.utils.config import get_cfg_defaults

SplitName = Literal["train", "val", "test"]

cfg = get_cfg_defaults()
NO_FINDING_KEYS = {
    "NoFindingInCluster",
    "No Finding",
    "No finding",
    "NoFinding",
    "no_finding",
}
BACKGROUND_KEYS = {"__background__"}


def normalize_label_mode(
    label_mode: str | None,
    dataset_type: str | None = None,
    gmm_type: str | None = None,
) -> str:
    """Normalize legacy `(dataset_type, gmm_type)` inputs to one label mode."""
    if label_mode:
        mode = str(label_mode).strip().lower()
        aliases = {
            "1d": "soft_1d",
            "soft1d": "soft_1d",
            "soft_1d": "soft_1d",
            "2d": "soft_2d",
            "soft2d": "soft_2d",
            "soft_2d": "soft_2d",
            "hard": "hard",
            "hard_label": "hard",
        }
        if mode not in aliases:
            raise ValueError(f"Unsupported label mode: {label_mode}")
        return aliases[mode]

    dataset_type = (dataset_type or "soft").strip().lower()
    if dataset_type == "hard":
        return "hard"

    gmm_type = (gmm_type or "1D").strip().upper()
    if gmm_type == "1D":
        return "soft_1d"
    if gmm_type == "2D":
        return "soft_2d"
    raise ValueError(f"Unsupported GMM type: {gmm_type}")


def label_mode_to_legacy_fields(label_mode: str) -> tuple[str, str | None]:
    """Return legacy `(dataset_type, gmm_type)` values used in older code paths."""
    if label_mode == "hard":
        return "hard", None
    if label_mode == "soft_1d":
        return "soft", "1D"
    if label_mode == "soft_2d":
        return "soft", "2D"
    raise ValueError(f"Unsupported label mode: {label_mode}")

class CanonicalDetectionDataset(DetectionDatasetPort):
    """
    Shared implementation for datasets that consume canonical hard/soft label
    JSON files and return Faster R-CNN compatible targets.
    """
    _SPLIT_CACHE: dict[tuple, tuple[list[str], list[str], list[str]]] = {}

    def __init__(
        self,
        root: str,
        label_file: str,
        transforms=None,
        split: SplitName = "train",
        train: bool | None = None,          # backward compat; prefer `split`
        test_split_ratio: float = 0.2,
        val_split_ratio: float = 0.2,
        label_mode: str | None = None,
        dataset_type: str | None = None,
        gmm_type: str | None = None,
        ignore_nofinding_clusters: bool = False,
        rater_weights_csv: str | None = None,
        placeholder_tolerance: float = 0.01,
    ):
        self.root = root
        self._label_file = os.path.abspath(label_file)
        self.transforms = transforms
        if train is not None:
            split = "train" if train else "val"
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        self.split = split
        self.test_split_ratio = test_split_ratio
        self.val_split_ratio = val_split_ratio
        self.ignore_nofinding_clusters = ignore_nofinding_clusters
        self.placeholder_tolerance = placeholder_tolerance

        self._label_mode = normalize_label_mode(
            label_mode=label_mode,
            dataset_type=dataset_type,
            gmm_type=gmm_type,
        )
        self._dataset_type, self.gmm_type = label_mode_to_legacy_fields(self._label_mode)
        self._class_to_idx = self.build_class_to_idx()

        if rater_weights_csv:
            self.rater_weights = load_rater_weights(rater_weights_csv)
        else:
            self.rater_weights = None

        print(f"Loading {self._label_mode} labels from: {label_file}")
        with open(label_file, "r", encoding="utf-8") as handle:
            raw_labels = json.load(handle)
        print(f"Loaded {len(raw_labels)} label entries")

        self._image_data = self.group_records_by_image(raw_labels)
        train_ids, val_ids, test_ids = self.create_stratified_splits()
        split_map = {"train": train_ids, "val": val_ids, "test": test_ids}
        self._image_ids = split_map[self.split]
        split_name = {"train": "Training", "val": "Validation", "test": "Test"}[self.split]
        print(f"{split_name} set: {len(self._image_ids)} images")

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return self._class_to_idx

    @property
    def image_ids(self) -> List[str]:
        return self._image_ids

    @image_ids.setter
    def image_ids(self, value: List[str]) -> None:
        self._image_ids = value

    @property
    def image_data(self) -> Dict[str, List[Dict[str, Any]]]:
        return self._image_data

    @property
    def label_mode(self) -> str:
        return self._label_mode

    @property
    def dataset_type(self) -> str:
        return self._dataset_type

    @abstractmethod
    def build_class_to_idx(self) -> Dict[str, int]:
        """Return canonical foreground class mapping for the dataset."""

    @abstractmethod
    def resolve_image_path(self, image_id: str) -> str:
        """Resolve an image id to an on-disk image path."""

    def get_image_ids(self) -> List[str]:
        return list(self.image_ids)

    def get_records_for_image(self, image_id: str) -> List[Dict[str, Any]]:
        return self.image_data[image_id]

    def get_num_classes(self) -> int:
        return len(self.class_to_idx) + 1

    def group_records_by_image(self, raw_labels: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        image_data: Dict[str, List[Dict[str, Any]]] = {}
        for record in raw_labels:
            image_id = record["image_id"]
            image_data.setdefault(image_id, []).append(record)
        return image_data

    def is_background_key(self, class_name: str) -> bool:
        return class_name in BACKGROUND_KEYS
    
    def is_nofinding_key(self, class_name: str) -> bool:
        return class_name in NO_FINDING_KEYS

    def resolve_class_name_to_idx(self, class_name: str) -> int | None:
        """Map a label string to a model class index (0=background, 1=No Finding, 2+=pathology)."""
        if class_name in self.class_to_idx:
            return self.class_to_idx[class_name]
        if self.is_nofinding_key(class_name) and "No Finding" in self.class_to_idx:
            return self.class_to_idx["No Finding"]
        if class_name in BACKGROUND_KEYS:
            return 0
        return None

    def is_placeholder_bbox(self, bbox: List[float] | tuple[float, ...] | None) -> bool:
        if not bbox or len(bbox) != 4:
            return False
        x, y, w, h = bbox
        tol = self.placeholder_tolerance
        return x <= tol and y <= tol and w >= 1.0 - tol and h >= 1.0 - tol

    def is_placeholder_mean_box(self, mean_box: List[float] | tuple[float, ...] | None) -> bool:
        return self.is_placeholder_bbox(mean_box)

    def is_placeholder_vertices(self, vertices: Dict[str, Dict[str, List[float]]] | None) -> bool:
        if not vertices:
            return False

        tol = self.placeholder_tolerance
        expected = {
            "top_left": (0.0, 0.0),
            "top_right": (1.0, 0.0),
            "bottom_right": (1.0, 1.0),
            "bottom_left": (0.0, 1.0),
        }
        for vertex_name, expected_xy in expected.items():
            entry = vertices.get(vertex_name) or {}
            mean = entry.get("mean")
            if not mean or len(mean) != 2:
                return False
            if abs(mean[0] - expected_xy[0]) > tol or abs(mean[1] - expected_xy[1]) > tol:
                return False
        return True

    def record_is_placeholder(self, record: Dict[str, Any]) -> bool:
        if self.label_mode == "hard":
            return self.is_placeholder_bbox(record.get("bbox"))
        if self.label_mode == "soft_1d":
            spatial = record.get("spatial") or {}
            return self.is_placeholder_mean_box(spatial.get("mean"))
        if self.label_mode == "soft_2d":
            return self.is_placeholder_vertices(record.get("vertices"))
        return False

    def build_class_prob_vector(self, class_probs: Dict[str, float] | None) -> torch.Tensor:
        prob_vector = torch.zeros(self.get_num_classes(), dtype=torch.float32)
        no_finding_idx = self.class_to_idx.get("No Finding", 1)

        for class_name, prob in (class_probs or {}).items():
            prob = float(prob)

            if class_name in BACKGROUND_KEYS:
                prob_vector[0] += prob
            elif class_name == "NoFindingInCluster" or self.is_nofinding_key(class_name):
                prob_vector[no_finding_idx] += prob
            elif class_name in self.class_to_idx:
                cls_idx = self.class_to_idx[class_name]
                prob_vector[cls_idx] += prob

        prob_sum = prob_vector.sum()
        if prob_sum > 0:
            prob_vector = prob_vector / prob_sum
        else:
            prob_vector[0] = 1.0

        return prob_vector

    def get_presence_labels(self, image_id: str, threshold: float = 0.1) -> Set[int]:
        labels = self.get_records_for_image(image_id)
        present: Set[int] = set()

        if self.dataset_type == "soft":
            for record in labels:
                if self.record_is_placeholder(record):
                    continue
                class_probs = record.get("class_probs", {}) or {}
                for class_name, prob in class_probs.items():
                    if class_name in self.class_to_idx and prob >= threshold:
                        present.add(self.class_to_idx[class_name])
        else:
            for finding in labels:
                dominant = self.get_dominant_class(labels_data=[finding])
                if dominant >= 1:
                    present.add(dominant)
        return present

    def get_dominant_class(
        self,
        labels_data: List[Dict[str, Any]] | None = None,
        image_id: str | None = None,
    ) -> int:
        if image_id is not None:
            labels_data = self.get_records_for_image(image_id)
        labels_data = labels_data or []
        if not labels_data:
            return 0

        class_vote_counts: Dict[int, float] = {}
        for label in labels_data:
            if self.dataset_type == "hard":
                best_class = None
                best_prob = -1.0

                renorm_probs = label.get("renormalized_class_probs") or {}
                for class_name, prob in renorm_probs.items():
                    cls = self.resolve_class_name_to_idx(class_name)
                    if cls is not None and float(prob) > best_prob:
                        best_prob = float(prob)
                        best_class = cls

                if best_class is None and "class" in label:
                    cls = self.resolve_class_name_to_idx(label["class"])
                    if cls is not None:
                        best_class = cls
                        best_prob = 1.0

                if best_class is not None:
                    class_vote_counts[best_class] = class_vote_counts.get(best_class, 0.0) + best_prob
            else:
                best_class = None
                best_prob = 0.0
                for class_name, prob in (label.get("class_probs") or {}).items():
                    if class_name in self.class_to_idx and prob > best_prob:
                        best_prob = float(prob)
                        best_class = self.class_to_idx[class_name]
                if best_class is not None:
                    class_vote_counts[best_class] = class_vote_counts.get(best_class, 0.0) + best_prob

        if not class_vote_counts:
            return 0
        return max(class_vote_counts.items(), key=lambda item: item[1])[0]

    def get_cluster_dominant_overall(self, cluster_label: Dict[str, Any]) -> int:
        probs = (cluster_label or {}).get("class_probs", {}) or {}
        best_class = 0
        best_prob = -1.0
        for class_name, prob in probs.items():
            cls = self.resolve_class_name_to_idx(class_name)
            if cls is None:
                continue
            if prob > best_prob:
                best_prob = float(prob)
                best_class = cls
        if best_class == 0 and self.dataset_type == "hard":
            return self.get_dominant_class(labels_data=[cluster_label])
        return best_class
    
    def has_pathology_clusters(self, image_id: str) -> bool:
        """
        Check if image has any pathology clusters (not just no-finding/background).
        
        Returns True if at least one cluster has a disease class as dominant,
        False if all clusters are no-finding (healthy CXR).
        """
        labels = self.get_records_for_image(image_id)
        for label in labels:
            # Skip placeholder records
            if self.record_is_placeholder(label):
                continue
            # Check if this cluster has a pathology (non-background) dominant class
            if self.get_cluster_dominant_overall(label) >= 2:
                return True
        return False
    
    def is_pure_nofinding_image(self, image_id: str) -> bool:
        """
        Check if image is pure no-finding (healthy CXR with no pathology clusters).
        
        Returns True if ALL clusters are no-finding/background (no disease present),
        False if any cluster has a pathology.
        """
        return not self.has_pathology_clusters(image_id)

    def get_cluster_dominant_disease(self, cluster_label: Dict[str, Any]) -> int | None:
        probs = (cluster_label or {}).get("class_probs", {}) or {}
        best_class = None
        best_prob = -1.0
        for class_name, prob in probs.items():
            if class_name not in self.class_to_idx:
                continue
            cls = self.class_to_idx[class_name]
            if cls <= 1: # added check for No Finding (class 1)
                continue
            if prob > best_prob:
                best_prob = float(prob)
                best_class = cls
        return best_class

    def compute_roi_head_class_weights_per_cluster(
        self,
        num_classes: int | None = None,
        presence_prob_thresh: float = 0.0,
        normalize: bool = False,
        eps: float = 1e-12,
        freq_term: str = cfg.MODEL.ROI_HEAD_BUFFER_FREQ_TERM,
        max_weight: float = cfg.MODEL.ROI_HEAD_BUFFER_MAX_WEIGHT,
        verbose: bool = True,
    ) -> torch.Tensor:
        counts = Counter()
        total = 0
        num_classes = num_classes or self.get_num_classes()

        for image_id in self.image_ids:
            labels = self.image_data[image_id]
            if self.dataset_type == "soft":
                for record in labels:
                    if self.record_is_placeholder(record):
                        continue
                    probs = record.get("class_probs", {}) or {}
                    best_class = None
                    best_prob = -1.0
                    for class_name, prob in probs.items():
                        if class_name in self.class_to_idx and prob > best_prob:
                            best_prob = float(prob)
                            best_class = self.class_to_idx[class_name]
                    if best_class is None or best_prob < presence_prob_thresh:
                        continue
                    counts[best_class] += 1
                    total += 1
            else:
                for finding in labels:
                    best_class = None
                    best_prob = -1.0
                    for class_name, prob in (finding.get("renormalized_class_probs") or {}).items():
                        cls = self.resolve_class_name_to_idx(class_name)
                        if cls is not None and float(prob) > best_prob:
                            best_prob = float(prob)
                            best_class = cls
                    if best_class is None and "class" in finding:
                        best_class = self.resolve_class_name_to_idx(finding["class"])
                    if best_class is None or best_class <= 1:
                        continue
                    counts[best_class] += 1
                    total += 1

        weights = torch.ones(num_classes, dtype=torch.float32)
        disease_classes = list(range(2, num_classes))
        total = max(1, total)
        num_disease_classes = max(1, len(disease_classes))

        for class_idx in disease_classes:
            count = counts.get(class_idx, 0)
            if count > 0:
                if freq_term == "inverse":
                    weight = total / (num_disease_classes * max(count, eps))
                elif freq_term == "inverse_sqrt":
                    weight = np.sqrt(total / (num_disease_classes * max(count, eps)))
                else:
                    raise ValueError(f"Unsupported freq_term: {freq_term}")
                weights[class_idx] = float(weight)
            else:
                weights[class_idx] = float(max_weight)

        if normalize and disease_classes:
            mean_weight = weights[disease_classes].mean().clamp_min(1e-6)
            weights[disease_classes] = weights[disease_classes] / mean_weight
        weights = weights.clamp(max=max_weight)
        weights[0] = 1.0
        # Explicitly set No Finding weight to 1.0
        if "No Finding" in self.class_to_idx:
            weights[self.class_to_idx["No Finding"]] = 0.1

        if verbose:
            reverse_mapping = {idx: name for name, idx in self.class_to_idx.items()}
            print("\n[RoI head weights] Per-cluster dominant disease counts:")
            for class_idx in sorted(counts.keys()):
                print(f"  {reverse_mapping.get(class_idx, class_idx)} ({class_idx}): {counts[class_idx]}")
            print(f"[RoI head weights] Total clusters counted: {total}")
            print(f"[RoI head weights] Mode: {freq_term}")
            print(f"[RoI head weights] Weights: {weights}")
        return weights

    def create_stratified_splits(
        self,
        presence_prob_thresh: float = 0.1,
        random_state: int = 42,
    ) -> tuple[list[str], list[str], list[str]]:
        cache_key = (
            self._label_file,
            self.test_split_ratio,
            self.val_split_ratio,
            random_state,
        )
        if cache_key in CanonicalDetectionDataset._SPLIT_CACHE:
            return CanonicalDetectionDataset._SPLIT_CACHE[cache_key]

        all_image_ids = list(self.image_data.keys())
        n_images = len(all_image_ids)
        num_classes = self.get_num_classes()

        y = np.zeros((n_images, num_classes), dtype=np.int32)
        for row_idx, image_id in enumerate(all_image_ids):
            present = self.get_presence_labels(image_id, threshold=presence_prob_thresh)
            if not present:
                y[row_idx, 0] = 1
            else:
                for class_idx in present:
                    y[row_idx, class_idx] = 1

        x_dummy = np.zeros((n_images, 1), dtype=np.int32)

        # Stage 1: lock away test set (same seed/ratio as the old 80/20 split)
        dev_idx, test_idx = next(
            MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.test_split_ratio,
                random_state=random_state,
            ).split(x_dummy, y)
        )

        # Stage 2: split development pool into train / val
        dev_y = y[dev_idx]
        dev_x = x_dummy[dev_idx]
        train_rel_idx, val_rel_idx = next(
            MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=self.val_split_ratio,
                random_state=random_state,
            ).split(dev_x, dev_y)
        )

        train_idx = dev_idx[train_rel_idx]
        val_idx = dev_idx[val_rel_idx]

        train_ids = [all_image_ids[i] for i in train_idx]
        val_ids = [all_image_ids[i] for i in val_idx]
        test_ids = [all_image_ids[i] for i in test_idx]

        CanonicalDetectionDataset._SPLIT_CACHE[cache_key] = (train_ids, val_ids, test_ids)
        return train_ids, val_ids, test_ids

    def build_image_id_tensor(self, image_id: str) -> torch.Tensor:
        return torch.tensor([hash(image_id) % (2**31)])

    def get_record_image_size(self, records: List[Dict[str, Any]], image) -> tuple[int, int]:
        if records and "image_width" in records[0] and "image_height" in records[0]:
            return int(records[0]["image_width"]), int(records[0]["image_height"])
        return image.size

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        image_path = self.resolve_image_path(image_id)
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Image not found: {image_path}. Check dataset root and adapter path rules."
            ) from exc

        labels = self.image_data[image_id]
        if self.label_mode == "hard":
            return self._process_hard_item(image, labels, image_id)
        if self.label_mode == "soft_1d":
            return self._process_1d_labels(image, labels, image_id)
        if self.label_mode == "soft_2d":
            return self._process_2d_labels(image, labels, image_id)
        raise ValueError(f"Unsupported label mode: {self.label_mode}")

    def _process_hard_item(self, image, findings, image_id: str):
        orig_w, orig_h = self.get_record_image_size(findings, image)
        boxes = []
        labels_tensor = []
        confidences = []

        for finding in findings:
            dominant_class = self.get_dominant_class(labels_data=[finding])
            if dominant_class == 0:
                continue
            # 0=background, 1=No Finding; skip both when ignoring NF clusters
            if self.ignore_nofinding_clusters and dominant_class <= 1:
                continue

            bbox = finding["bbox"]
            # Full-image NF placeholder only when the whole image is pure no-finding
            if (
                dominant_class == 1
                and self.is_placeholder_bbox(bbox)
                and not self.is_pure_nofinding_image(image_id)
            ):
                continue

            norm_x, norm_y, norm_w, norm_h = bbox
            x = norm_x * orig_w
            y = norm_y * orig_h
            w = norm_w * orig_w
            h = norm_h * orig_h
            x2 = x + w
            y2 = y + h

            x = max(0, min(x, orig_w - 1))
            y = max(0, min(y, orig_h - 1))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            if x2 <= x or y2 <= y:
                continue

            boxes.append([x, y, x2, y2])
            labels_tensor.append(dominant_class)
            confidences.append(float(finding.get("confidence", 1.0)))

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            confidences = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_tensor, dtype=torch.int64)
            confidences = torch.as_tensor(confidences, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels_tensor,
            "confidences": confidences,
            "image_id": self.build_image_id_tensor(image_id),
        }
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _process_1d_labels(self, image, labels, image_id: str):
        boxes = []
        gt_means = []
        gt_stds = []
        class_probs = []
        labels_tensor = []
        annotation_weights_list = []
        orig_w, orig_h = self.get_record_image_size(labels, image)

        for label in labels:
            spatial = label.get("spatial") or {}
            mean_box = spatial.get("mean")
            std_box = spatial.get("std")
            # if not mean_box or not std_box or self.is_placeholder_mean_box(mean_box):
            #     continue
            # REMOVED: self.is_placeholder_mean_box(mean_box)
            if not mean_box or not std_box:
                continue

            if self.ignore_nofinding_clusters and self.get_cluster_dominant_overall(label) <= 1:
                continue

            x_mean = mean_box[0] * orig_w
            y_mean = mean_box[1] * orig_h
            w_mean = mean_box[2] * orig_w
            h_mean = mean_box[3] * orig_h
            boxes.append([x_mean, y_mean, x_mean + w_mean, y_mean + h_mean])

            gt_means.append([mean_box[0], mean_box[1], mean_box[2], mean_box[3]])
            gt_stds.append([std_box[0], std_box[1], std_box[2], std_box[3]])

            prob_vector = self.build_class_prob_vector(label.get("class_probs"))
            class_probs.append(prob_vector)

            original_annotations = label.get("original_annotations", [])
            if original_annotations and self.rater_weights is not None:
                rater_ids = [ann["rater_id"] for ann in original_annotations if "rater_id" in ann]
                if rater_ids:
                    weight = float(sum(self.rater_weights.get(rater_id, 0.5) for rater_id in rater_ids) / len(rater_ids))
                else:
                    bg_prob = prob_vector[0].item() + (prob_vector[self.class_to_idx["No Finding"]].item() if "No Finding" in self.class_to_idx else 0)
                    weight = max(float(1.0 - bg_prob), 0.1)
            else:
                bg_prob = prob_vector[0].item() + (prob_vector[self.class_to_idx["No Finding"]].item() if "No Finding" in self.class_to_idx else 0)
                weight = max(float(1.0 - bg_prob), 0.1)

            # NEW: Downweight if No Finding is the majority (handles singletons and full-image placeholders)
            if "No Finding" in self.class_to_idx:
                nf_idx = self.class_to_idx["No Finding"]
                if prob_vector[nf_idx].item() > 0.5:
                    weight = weight * max(float(1.0 - prob_vector[nf_idx].item()), 0.1)

            annotation_weights_list.append(weight)

            dominant_disease = self.get_cluster_dominant_disease(label)
            if dominant_disease is not None:
                labels_tensor.append(dominant_disease)
            else:
                labels_tensor.append(1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            gt_means = torch.zeros((0, 4), dtype=torch.float32)
            gt_stds = torch.zeros((0, 4), dtype=torch.float32)
            class_probs = torch.zeros((0, self.get_num_classes()), dtype=torch.float32)
            annotation_weight = torch.zeros((0,), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_tensor, dtype=torch.int64)
            gt_means = torch.as_tensor(gt_means, dtype=torch.float32)
            gt_stds = torch.as_tensor(gt_stds, dtype=torch.float32)
            class_probs = torch.stack(class_probs)
            annotation_weight = torch.tensor(annotation_weights_list, dtype=torch.float32)

        target = {
            "boxes": boxes,
            "labels": labels_tensor,
            "gt_means": gt_means,
            "gt_stds": gt_stds,
            "gt_class_probs": class_probs,
            "image_id": self.build_image_id_tensor(image_id),
            "annotation_weight": annotation_weight,
        }
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def _process_2d_labels(self, image, labels, image_id: str):
        boxes = []
        gt_vertices = []
        gt_covariances = []
        class_probs = []
        labels_tensor = []
        vertex_keys = ["top_left", "top_right", "bottom_right", "bottom_left"]

        for label in labels:
            vertices = label.get("vertices")
            if not vertices or self.is_placeholder_vertices(vertices):
                continue

            top_left = vertices["top_left"]["mean"]
            bottom_right = vertices["bottom_right"]["mean"]
            x1, y1 = top_left[0], top_left[1]
            x2, y2 = bottom_right[0], bottom_right[1]
            boxes.append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])

            vertex_data = []
            covariance_data = []
            for vertex_key in vertex_keys:
                vertex_data.extend(vertices[vertex_key]["mean"])
                covariance_data.append(vertices[vertex_key]["covariance"])
            gt_vertices.append(vertex_data)
            gt_covariances.append(covariance_data)

            prob_vector = self.build_class_prob_vector(label.get("class_probs"))
            class_probs.append(prob_vector)
            labels_tensor.append(self.get_cluster_dominant_disease(label) or 1)

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            gt_vertices = torch.zeros((0, 8), dtype=torch.float32)
            gt_covariances = []
            class_probs = torch.zeros((0, self.get_num_classes()), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_tensor, dtype=torch.int64)
            gt_vertices = torch.as_tensor(gt_vertices, dtype=torch.float32)
            class_probs = torch.stack(class_probs)

        target = {
            "boxes": boxes,
            "labels": labels_tensor,
            "gt_vertices": gt_vertices,
            "gt_covariances": gt_covariances,
            "gt_class_probs": class_probs,
            "image_id": self.build_image_id_tensor(image_id),
        }
        if self.transforms:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)
