import numpy as np
import torch
from tqdm import tqdm

from softlabel_od.object_detection.core.model.factory import get_model
def load_model(weights_path, num_classes, model_type, backbone, probabilistic, device, anchor_sizes=None, anchor_aspect_ratios=None):
    """Load model from checkpoint."""
    model = get_model(
        num_classes=num_classes,
        model_type=model_type,
        backbone_name=backbone,
        pretrained=False,
        probabilistic=probabilistic,
        anchor_sizes=anchor_sizes,
        anchor_aspect_ratios=anchor_aspect_ratios,
    )

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter out unexpected keys (e.g. class_weights buffer from fine-tune runs)
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        print(f"Note: {len(missing)} keys not in checkpoint (expected for buffers added post-training)")

    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()
    return model


def collect_predictions(model, data_loader, device, is_probabilistic):
    """Run inference on the full validation set and collect predictions + GT."""
    all_predictions = []
    all_ground_truths = []

    pbar = tqdm(data_loader, desc="Running inference")
    for images, targets in pbar:
        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for pred, target, img in zip(outputs, targets, images):
            img_h, img_w = img.shape[1], img.shape[2]

            pred_entry = {
                'boxes': pred['boxes'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy(),
                'scores': pred['scores'].cpu().numpy(),
            }
            if 'bbox_stds' in pred:
                pred_entry['bbox_stds'] = pred['bbox_stds'].cpu().numpy()
                pred_entry['bbox_means'] = pred['boxes'].cpu().numpy()
            if 'class_probs' in pred:
                probs_np = pred['class_probs'].cpu().numpy()        # [N, 15]
                pred_entry['class_probs'] = probs_np
                # Class confidence for ECE: P(predicted class) from the full distribution
                pred_labels_np = pred['labels'].cpu().numpy()       # [N]
                pred_entry['class_scores'] = probs_np[np.arange(len(pred_labels_np)), pred_labels_np]         # [N]

            gt_entry = {
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy(),
                'image_size': (img_h, img_w),
            }
            if 'gt_class_probs' in target:
                gt_entry['gt_class_probs'] = target['gt_class_probs'].cpu().numpy()
            if 'gt_stds' in target:
                gt_entry['gt_stds'] = target['gt_stds'].cpu().numpy()

            all_predictions.append(pred_entry)
            all_ground_truths.append(gt_entry)

    return all_predictions, all_ground_truths