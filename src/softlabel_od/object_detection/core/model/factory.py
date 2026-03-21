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


def get_model(num_classes, model_type="1d", backbone_name="resnet50", pretrained=True, probabilistic=True):
    """
    Factory function to get the appropriate detection model.

    Args:
        num_classes:    Number of classes including background (e.g. 15 for VinDr-CXR).
        model_type:     '1d', '2d', or 'hard'.
        backbone_name:  'resnet50', 'resnet101', or 'mobilenet'.
        pretrained:     Use ImageNet-pretrained backbone weights.
        probabilistic:  Use probabilistic GMM head (only relevant for model_type='1d').

    Returns:
        A torch.nn.Module ready for training.
    """
    if model_type in ("1d", "1D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic)

    elif model_type in ("2d", "2D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_2d import get_model_2d
        return get_model_2d(num_classes, backbone_name, pretrained)

    elif model_type in ("hard", "HARD_LABEL"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic=False)

    else:
        raise ValueError(
            f"Unknown model_type: '{model_type}'. "
            "Supported values: '1d', '2d', 'hard'."
        )
