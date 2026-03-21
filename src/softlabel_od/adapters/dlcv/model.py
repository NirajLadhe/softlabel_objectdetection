"""
model.py - Model factory for different training types

Provides a unified interface to get models for:
- 1D GMM (probabilistic with independent uncertainties)
- 2D GMM (probabilistic with correlated vertices)
- Hard labels (deterministic, standard Faster R-CNN)
"""

def get_model(num_classes, model_type='1d', backbone_name='resnet50', pretrained=True, probabilistic=True):
    """
    Factory function to get the appropriate model.
    
    Args:
        num_classes: Number of classes (including background)
        model_type: '1d', '2d', or 'hard'
        backbone_name: 'resnet50', 'resnet101', or 'mobilenet'
        pretrained: Use pretrained backbone weights
        probabilistic: Use probabilistic head (only for model_type='1d')
    
    Returns:
        model: Appropriate Faster R-CNN variant
    """
    if model_type == '1d' or model_type == '1D':
        from softlabel_od.adapters.dlcv.model_1d import get_model_1d
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic)
    
    elif model_type == '2d' or model_type == '2D':
        from softlabel_od.adapters.dlcv.model_2d import get_model_2d
        return get_model_2d(num_classes, backbone_name, pretrained)
    
    elif model_type == 'hard' or model_type == 'HARD_LABEL':
        from softlabel_od.adapters.dlcv.model_1d import get_model_1d
        # Use standard (non-probabilistic) head for hard labels
        return get_model_1d(num_classes, backbone_name, pretrained, probabilistic=False)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose '1d', '2d', or 'hard'")