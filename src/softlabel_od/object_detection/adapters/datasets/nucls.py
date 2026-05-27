import os

from softlabel_od.object_detection.core.datasets.base import CanonicalDetectionDataset

project_root_ext = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))


class NuCLSDataset(CanonicalDetectionDataset):
    """
    NuCLS adapter over the shared canonical-label dataset workflow.

    Nucleus-level pathology dataset. No 'No Finding' class — foreground
    classes start at index 2, index 1 is "No Finding" class just for framework convention.
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
            "outputs/nucls/reliability/annotator_reliability_scores_nucls_iou0_25.csv",
        ),
        image_extension=".png",
        placeholder_tolerance=0.01,
    ):
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

    def build_class_to_idx(self) -> dict:
        return {
            "tumor":                2,
            "mitotic_figure":       3,
            "fibroblast":           4,
            "vascular_endothelium": 5,
            "macrophage":           6,
            "lymphocyte":           7,
            "plasma_cell":          8,
            "neutrophil":           9,
            "eosinophil":           10,
            "myoepithelium":        11,
            "apoptotic_body":       12,
            "unlabeled":            13,
        }

    def get_num_classes(self) -> int:
    # Indices run 0=background, 1=vacant, 2-13=nucleus types
    # max index is 13 → need 14 slots
        return max(self.class_to_idx.values()) + 1
    
    def resolve_image_path(self, image_id: str) -> str:
        return os.path.join(self.root, "images", image_id + self.image_extension)

    

NuCLSDataset = NuCLSDataset