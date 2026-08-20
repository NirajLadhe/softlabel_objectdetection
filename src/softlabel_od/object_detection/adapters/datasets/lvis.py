import json
import os
from typing import Any, Dict, List, Optional, Set

from softlabel_od.object_detection.core.datasets.base import (
    BACKGROUND_KEYS,
    CanonicalDetectionDataset,
)

project_root_ext = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)


class LVIS_SoftLabel_Dataset(CanonicalDetectionDataset):
    """
    COCO-LVIS adapter over the shared canonical-label dataset workflow.

    The underlying label files are expected to follow the repository's canonical
    hard / soft 1D / soft 2D schemas consumed by `CanonicalDetectionDataset`.
    """

    def __init__(
        self,
        root,
        label_file,
        transforms=None,
        train=True,
        test_split_ratio=0.2,
        label_mode=None,
        gmm_type="1D",
        dataset_type="soft",
        ignore_nofinding_clusters=False,
        rater_weights_csv=os.path.join(
            project_root_ext,
            "outputs/raters_reliability/annotator_reliability_lvis/"
            "annotator_reliability_scores_lvis_iou0_5.csv",
        ),
        image_subdir="",
        image_extension=".jpg",
        image_id_prefix="lvis_",
        coco_images_json: Optional[str] = None,
        categories_json: Optional[str] = None,
        placeholder_tolerance=0.01,
        exclude_singleton_val: bool = False,
        exclude_singleton_all: bool = False,
    ):
        self.image_subdir = image_subdir
        self.image_extension = image_extension
        self.image_id_prefix = image_id_prefix
        self._image_id_to_filename = self._load_image_filename_map(coco_images_json)
        self._class_names = self._resolve_class_names(label_file, categories_json)
        super().__init__(
            root=root,
            label_file=label_file,
            transforms=transforms,
            train=train,
            test_split_ratio=test_split_ratio,
            label_mode=label_mode,
            dataset_type=dataset_type,
            gmm_type=gmm_type,
            ignore_nofinding_clusters=ignore_nofinding_clusters,
            rater_weights_csv=rater_weights_csv,
            placeholder_tolerance=placeholder_tolerance,
            exclude_singleton_val=exclude_singleton_val,
            exclude_singleton_all=exclude_singleton_all,
        )

    @staticmethod
    def _collect_class_names_from_records(records: List[Dict[str, Any]]) -> List[str]:
        names: Set[str] = set()
        for record in records:
            for class_name in (record.get("class_probs") or {}):
                if class_name not in BACKGROUND_KEYS:
                    names.add(class_name)

            if "class" in record and record["class"] not in BACKGROUND_KEYS:
                names.add(record["class"])

            for class_name in (record.get("renormalized_class_probs") or {}):
                if class_name not in BACKGROUND_KEYS:
                    names.add(class_name)
        return sorted(names)

    @staticmethod
    def _load_class_names_from_categories(categories_json: str) -> List[str]:
        with open(categories_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        categories = payload.get("categories", payload)
        if not isinstance(categories, list):
            raise ValueError(
                f"Expected a categories list in {categories_json}, got {type(categories)}"
            )

        ordered = sorted(categories, key=lambda category: int(category["id"]))
        return [str(category["name"]) for category in ordered]

    def _resolve_class_names(
        self,
        label_file: str,
        categories_json: Optional[str],
    ) -> List[str]:
        if categories_json and os.path.exists(categories_json):
            return self._load_class_names_from_categories(categories_json)

        with open(label_file, "r", encoding="utf-8") as handle:
            raw_labels = json.load(handle)
        return self._collect_class_names_from_records(raw_labels)

    def _load_image_filename_map(
        self,
        coco_images_json: Optional[str],
    ) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        if not coco_images_json or not os.path.exists(coco_images_json):
            return mapping

        with open(coco_images_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for image_entry in payload.get("images", []):
            file_name = image_entry.get("file_name")
            if not file_name:
                continue

            image_id = str(image_entry.get("id"))
            stem = os.path.splitext(file_name)[0]
            mapping[image_id] = file_name
            mapping[stem] = file_name
            if self.image_id_prefix:
                mapping[f"{self.image_id_prefix}{image_id}"] = file_name
        return mapping

    def build_class_to_idx(self) -> Dict[str, int]:
        return {class_name: index + 1 for index, class_name in enumerate(self._class_names)}

    def resolve_image_path(self, image_id: str) -> str:
        file_name = self._image_id_to_filename.get(image_id)
        if file_name is None and self.image_id_prefix and image_id.startswith(self.image_id_prefix):
            numeric_id = image_id[len(self.image_id_prefix):]
            try:
                file_name = f"{int(numeric_id):012d}{self.image_extension}"
            except ValueError:
                file_name = f"{numeric_id}{self.image_extension}"
        elif file_name is None:
            file_name = f"{image_id}{self.image_extension}"

        return os.path.join(self.root, self.image_subdir, file_name)


LVISDataset = LVIS_SoftLabel_Dataset
