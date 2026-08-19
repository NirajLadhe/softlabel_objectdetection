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


def get_model(num_classes, model_type="1d", backbone_name="resnet50", pretrained=True,
             probabilistic=True, anchor_sizes=None, anchor_aspect_ratios=None,
             lambda_theta_mean=0.3, lambda_theta_cov=0.02, lambda_theta_kl=0.0,
             pos_cls_weight=1.2, neg_cls_weight=0.8):
    """
    ... (existing docstring, add:)
    lambda_theta_mean, lambda_theta_cov, lambda_theta_kl, pos_cls_weight, neg_cls_weight:
        only used by model_type='theta_2d'; harmlessly ignored otherwise.
    """
    if model_type in ("1d", "1D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic, anchor_sizes, anchor_aspect_ratios)

    elif model_type in ("2d", "2D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_2d import get_model_2d
        return get_model_2d(num_classes, backbone_name, pretrained)

    elif model_type in ("theta_2d", "2d_theta", "THETA_2D"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_theta_2d import get_model_theta_2d
        return get_model_theta_2d(
            num_classes, backbone_name, pretrained, anchor_sizes, anchor_aspect_ratios,
            lambda_theta_mean=lambda_theta_mean, lambda_theta_cov=lambda_theta_cov,
            lambda_theta_kl=lambda_theta_kl, pos_cls_weight=pos_cls_weight, neg_cls_weight=neg_cls_weight,
        )

    elif model_type in ("hard", "HARD_LABEL"):
        from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic=False, anchor_sizes=anchor_sizes, anchor_aspect_ratios=anchor_aspect_ratios)

    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Supported values: '1d', '2d', 'theta_2d', 'hard'.")