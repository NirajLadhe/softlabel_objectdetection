"""
Model factory — single entry point for all model variants.

Supported model_type values
---------------------------
'1d'         — Faster R-CNN with probabilistic GMM head (1D independent Gaussians)
'2d'         — Faster R-CNN with vertex-based 2D GMM head
'hard'       — Faster R-CNN with standard cross-entropy head

Adding a new architecture
--------------------------
1. Implement the model in adapters/model_components/ (or a new file here).
2. Add a branch in get_model() below.
3. Register any new backbone in the backbone section of the relevant model file.
"""


def get_model(
    num_classes,
    model_type="1d",
    backbone_name="resnet50",
    pretrained=True,
    probabilistic=True,
    anchor_sizes=None,
    anchor_aspect_ratios=None,
    architecture=None,
):
    """
    Factory function to get the appropriate detection model.

    Args:
        num_classes:    Number of classes including background (e.g. 15 for VinDr-CXR).
        model_type:     '1d', '2d', 'hard', or 'lvis_1d'.
        backbone_name:  'resnet50', 'resnet101', or 'mobilenet'.
        pretrained:     Use ImageNet-pretrained backbone weights.
        probabilistic:  Use probabilistic GMM head (only relevant for model_type='1d').
        anchor_sizes:   Anchor sizes (list of lists).
        anchor_aspect_ratios: Anchor aspect ratios (list of lists).
        architecture:   Optional override ('lvis_1d', 'vindr_1d', ...). When set, takes
                        precedence over model_type for backbone/head selection.
    Returns:
        A torch.nn.Module ready for training.
    """
    arch = (architecture or model_type or "").strip().lower().replace("-", "_")

    if arch in ("lvis_1d", "lvis"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d_lvis import (
            get_model_lvis_1d,
        )
        return get_model_lvis_1d(
            num_classes,
            backbone_name,
            pretrained,
            probabilistic,
            anchor_sizes,
            anchor_aspect_ratios,
        )

    if model_type in ("1d", "1D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic, anchor_sizes, anchor_aspect_ratios)

    elif model_type in ("2d", "2D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_2d import get_model_2d
        return get_model_2d(num_classes, backbone_name, pretrained)

    elif model_type in ("hard", "HARD_LABEL"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic=False, anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_aspect_ratios)

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Supported values: '1d', '2d', 'hard', 'lvis_1d'."
        )
