import os

from softlabel_od.object_detection.core.datasets.base import CanonicalDetectionDataset
from softlabel_od.object_detection.utils.transforms import Compose, RandomHorizontalFlip, ToTensor

project_root_ext = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))


class Vin_Dr_CXR_SoftLabel_Dataset(CanonicalDetectionDataset):
    """
    VinDr-CXR adapter over the shared canonical-label dataset workflow.

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
            "outputs/raters_reliability/annotator_reliability_vindrcxr/annotator_reliability_scores_vindrcxr_iou0_5.csv",
        ),
        image_subdir="train",
        image_extension=".png",
        placeholder_tolerance=0.01,
    ):
        self.image_subdir = image_subdir
        self.image_extension = image_extension
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
        )

    def build_class_to_idx(self):
        return {
            "Aortic enlargement": 1,
            "Atelectasis": 2,
            "Calcification": 3,
            "Cardiomegaly": 4,
            "Consolidation": 5,
            "ILD": 6,
            "Infiltration": 7,
            "Lung Opacity": 8,
            "Nodule/Mass": 9,
            "Other lesion": 10,
            "Pleural effusion": 11,
            "Pleural thickening": 12,
            "Pneumothorax": 13,
            "Pulmonary fibrosis": 14,
        }

    def resolve_image_path(self, image_id: str) -> str:
        return os.path.join(self.root, self.image_subdir, image_id + self.image_extension)


VinDrCXRDataset = Vin_Dr_CXR_SoftLabel_Dataset