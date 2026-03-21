import os
import csv
import json
import random
from PIL import Image
import collections
import time
import datetime
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler
from collections import Counter
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import yaml
from yacs.config import CfgNode as CN
from softlabel_od.adapters.dlcv.config import get_cfg_defaults
from softlabel_od.adapters.dlcv.dataset import *

def load_rater_weights(csv_path: str) -> dict:
    """Normalised K-alpha weights keyed by annotator ID string e.g. 'R1'."""
    import pandas as pd
    df = pd.read_csv(csv_path).set_index('Annotator')
    raw = df['Avg K-alpha']
    return (raw / raw.max()).to_dict()  # values in (0, 1]

def create_balanced_sampler(dataset, rfs_threshold=0.01, presence_prob_thresh=0.1, ignore_nofinding=False, alpha=0.5, nofinding_weight=0.1):
    """
    Repeat Factor Sampling (RFS) style sampler for long-tailed detection.
    Citation: LVIS paper
    Samples *images* with weights derived from rare class presence.

    - rfs_threshold (t): typical values 0.001–0.01
    - presence_prob_thresh: soft-label disease presence threshold
    - ignore_nofinding: if True, pure no-finding images get weight 0.1 (minimally sampled)
    """
    import numpy as np
    from collections import Counter

    img_ids = dataset.image_ids
    N = len(img_ids)

    img_present = []
    cat_img_count = Counter()

    for img_id in img_ids:
        labels = dataset.image_data[img_id]
        present = set()

        if dataset.dataset_type == "soft":
            for cl in labels:
                mean_box = (cl.get("spatial") or {}).get("mean")
                if mean_box and mean_box[2] >= 0.99 and mean_box[3] >= 0.99:
                    continue

                probs = cl.get("class_probs", {}) or {}
                for name, p in probs.items():
                    if name in dataset.class_to_idx and p >= presence_prob_thresh:
                        present.add(dataset.class_to_idx[name])  # 1..14

        else:  # hard
            for finding in labels:
                if "renormalized_class_probs" in finding:
                    for name, p in (finding.get("renormalized_class_probs") or {}).items():
                        if name in dataset.class_to_idx and p >= presence_prob_thresh:
                            present.add(dataset.class_to_idx[name])
                elif "class" in finding and finding["class"] in dataset.class_to_idx:
                    present.add(dataset.class_to_idx[finding["class"]])

        img_present.append(present)
        for c in present:
            cat_img_count[c] += 1
    print("Creating balanced sampler (RFS)...")
    print(f"  N images: {N}")
    # print(f"  dataset_type: {dataset.dataset_type}")
    # print(f"  presence_prob_thresh: {presence_prob_thresh}")
    # print(f"  rfs_threshold: {rfs_threshold}")
    # print(f"  ignore_nofinding: {ignore_nofinding}")

    num_pure_nofinding = sum(1 for s in img_present if not s)
    print(f"  Pure NoFinding images (no disease present): {num_pure_nofinding} ({num_pure_nofinding/max(N,1)*100:.1f}%)")

    # Class presence distribution (by images)
    total_present = sum(cat_img_count.values())  # note: this double-counts multi-label images
    print("\nClass presence in training set (image-level):")
    for class_idx in sorted(cat_img_count.keys()):
        count = cat_img_count[class_idx]
        class_name = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(class_idx)]
        print(f"  {class_name} ({class_idx}): {count} images ({count/max(N,1)*100:.2f}%)")
    if total_present == 0:
        print("  (no disease classes present under current threshold)")

    # category repeat factors r_c = max(1, sqrt(t / f_c))
    r_c = {}
    for c, cnt in cat_img_count.items():
        f_c = cnt / max(1, N)
        ratio = rfs_threshold / max(f_c, 1e-12)
        r_c[c] = max(1.0, float(ratio ** (alpha / 2.0)))
    
    print("\nRFS repeat factors (category-level):")
    for class_idx in sorted(r_c.keys()):
        cnt = cat_img_count[class_idx]
        f_c = cnt / max(1, N)
        class_name = list(dataset.class_to_idx.keys())[list(dataset.class_to_idx.values()).index(class_idx)]
        print(f"  {class_name} ({class_idx}): f_c={f_c:.6f}, r_c={r_c[class_idx]:.3f}")

    # image repeat factor r_i = max_{c in present(i)} r_c, else 1
    weights = []
    for present in img_present:
        if not present:
            w = nofinding_weight if ignore_nofinding else 1.0
        else:
            w = max(r_c[c] for c in present)
        weights.append(w)
    
    weights_np = np.array(weights, dtype=np.float32)
    nonzero = int((weights_np > 0).sum())
    print("\nImage sampling weights summary:")
    print(f"  nonzero-weight images: {nonzero}/{N}")
    print(f"  min/mean/max: {weights_np.min():.3f} / {weights_np.mean():.3f} / {weights_np.max():.3f}")

    # optional: show top-K heaviest images
    topk = min(10, N)
    if topk > 0:
        top_idx = weights_np.argsort()[-topk:][::-1]
        print(f"\nTop-{topk} sampled images (weight, num_classes_present):")
        for j in top_idx:
            print(f"  {img_ids[j]}: w={weights_np[j]:.3f}, |C|={len(img_present[j])}")

    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

def gaussian_nll_loss(pred_means, pred_log_vars, gt_means, gt_stds):
    """
    Computes the Gaussian Negative Log-Likelihood loss.
    pred_log_vars: The predicted log of the variance (log(sigma^2)).
    gt_stds: The ground truth standard deviation (sigma).
    """
    # Ensure gt_stds are non-zero to avoid division by zero in variance calculation
    gt_vars = (gt_stds.pow(2)) + 1e-6
    
    log_vars_pred = pred_log_vars
    vars_pred = torch.exp(log_vars_pred)
    
    # The NLL loss formula
    loss = 0.5 * (log_vars_pred + (gt_means - pred_means).pow(2) / vars_pred)
    return loss.sum()

def gaussian_kl_loss(pred_means, pred_log_vars, gt_means, gt_stds):
    """
    KL Divergence between predicted Gaussian and target Gaussian.
    KL(N(gt_mean, gt_std) || N(pred_mean, pred_std))
    
    Args:
        pred_means: [N, 4] predicted bbox means
        pred_log_vars: [N, 4] predicted log-variances (log(σ²))
        gt_means: [N, 4] ground truth bbox means  
        gt_stds: [N, 4] ground truth standard deviations
    """
    # Clamp log_vars to prevent numerical issues
    pred_log_vars = torch.clamp(pred_log_vars, min=-10, max=10)
    
    # Convert to standard deviations
    pred_stds = torch.exp(0.5 * pred_log_vars)  # σ_pred = exp(0.5 * log(σ²))
    gt_stds = gt_stds.clamp(min=1e-6)  # Avoid division by zero
    
    # KL divergence: KL(N(μ_gt, σ_gt) || N(μ_pred, σ_pred))
    # = log(σ_pred/σ_gt) + (σ_gt² + (μ_gt - μ_pred)²) / (2σ_pred²) - 0.5
    
    log_ratio = torch.log(pred_stds / gt_stds)
    mean_diff_sq = (gt_means - pred_means).pow(2)
    gt_var = gt_stds.pow(2)
    pred_var = pred_stds.pow(2)
    
    kl = log_ratio + (gt_var + mean_diff_sq) / (2 * pred_var) - 0.5
    
    # Average over all dimensions and samples
    return kl.mean()

def js_divergence_categorical(p, q, eps=1e-8):
    """
    Jensen–Shannon divergence for categorical distributions.
    p, q: [N, C] probs that sum to 1
    returns scalar
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    js = 0.5 * (q * (q / m).log()).sum(dim=-1) + 0.5 * (p * (p / m).log()).sum(dim=-1)
    return js.mean()

def variance_only_kl(gt_std, pred_std, eps=1e-8):
    """
    KL(N(0, gt_std^2) || N(0, pred_std^2)) averaged.
    Stable because it removes the mean-mismatch term.
    """
    gt_std = gt_std.clamp(min=eps)
    pred_std = pred_std.clamp(min=eps)
    return (torch.log(pred_std / gt_std) + (gt_std.pow(2)) / (2.0 * pred_std.pow(2)) - 0.5).mean()

def variance_only_sym_kl(gt_std, pred_std, eps=1e-8):
    """
    0.5*(KL(gt||pred)+KL(pred||gt)) for zero-mean Gaussians.
    More symmetric, still stable with a reasonable std floor.
    """
    gt_std = gt_std.clamp(min=eps)
    pred_std = pred_std.clamp(min=eps)
    kl_gt_pred = torch.log(pred_std / gt_std) + (gt_std.pow(2)) / (2.0 * pred_std.pow(2)) - 0.5
    kl_pred_gt = torch.log(gt_std / pred_std) + (pred_std.pow(2)) / (2.0 * gt_std.pow(2)) - 0.5
    return 0.5 * (kl_gt_pred + kl_pred_gt).mean()

def huber_log_std_loss(pred_log_vars, gt_stds, beta=1.0, eps=1e-8):
    """
    Huber loss for log-standard deviations.
    Stable because it removes the mean-mismatch term.
    """
    pred_stds = torch.exp(0.5 * pred_log_vars)
    gt_stds = gt_stds.clamp(min=eps)
    return F.smooth_l1_loss(pred_stds, gt_stds, beta=beta, reduction='mean')
    
def soft_cross_entropy_loss(logits, soft_targets, class_weights=None, reduction='mean'):
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [N, C]
    if class_weights is not None:
        w = class_weights.to(logits.device).unsqueeze(0)          # [1, C]
        per_sample = (-soft_targets * log_probs * w).sum(dim=-1)
    else:
        per_sample = (-soft_targets * log_probs).sum(dim=-1)      # [N]

    if reduction == 'none':   return per_sample
    elif reduction == 'sum':  return per_sample.sum()
    else:                     return per_sample.mean()

def soft_focal_loss(pred_logits, gt_labels, alpha=0.25, gamma=2.0, num_classes=None):
    """
    Focal Loss that works with both soft and hard labels.
    
    Args:
        pred_logits: [N, num_classes] - raw model outputs
        gt_labels: [N, num_classes] (soft) OR [N] (hard integers)
        alpha: focusing parameter for hard examples
        gamma: modulation exponent
        num_classes: required if gt_labels are hard (integers)
        
    Returns:
        Scalar loss
    """
    # Auto-detect if labels are hard or soft
    if gt_labels.dim() == 1:
        # Hard labels - convert to one-hot
        if num_classes is None:
            num_classes = pred_logits.shape[-1]
        gt_class_probs = F.one_hot(gt_labels, num_classes=num_classes).float()
    else:
        # Soft labels - use directly
        gt_class_probs = gt_labels
    
    # Compute focal loss
    pred_probs = torch.nn.functional.softmax(pred_logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(pred_logits, dim=-1)
    ce_loss = -gt_class_probs * log_probs
    
    p_t = torch.sum(pred_probs * gt_class_probs, dim=-1, keepdim=True)
    modulation = (1 - p_t) ** gamma
    if isinstance(alpha, torch.Tensor) and alpha.dim() >= 1:
        focal_weight = alpha.unsqueeze(0) * modulation  # [1, C] * [N, 1]
    else:
        focal_weight = alpha * modulation
    focal_loss = focal_weight * ce_loss
    
    return focal_loss.sum(dim=-1).mean()

def vertex_nll_loss_2d(pred_vertices, pred_covs, gt_vertices, gt_covs):
    """
    Computes the Negative Log-Likelihood for 2D vertex predictions.
    
    Each vertex follows a 2D Gaussian: v ~ N(μ, Σ)
    NLL = -log p(v|μ,Σ) = 0.5 * (log|Σ| + (v-μ)^T Σ^{-1} (v-μ) + 2*log(2π))
    
    Args:
        pred_vertices: [N, 8] predicted vertex means
        pred_covs: [N, 4, 2, 2] predicted covariance matrices
        gt_vertices: [N, 8] ground truth vertices
        gt_covs: [N, 4, 2, 2] ground truth covariances
    
    Returns:
        loss: scalar
    """
    N = pred_vertices.shape[0]
    total_loss = 0.0
    
    for v_idx in range(4):
        # Extract vertex v_idx for all samples
        pred_v = pred_vertices[:, v_idx*2:(v_idx+1)*2]  # [N, 2]
        gt_v = gt_vertices[:, v_idx*2:(v_idx+1)*2]      # [N, 2]
        
        pred_cov = pred_covs[:, v_idx]  # [N, 2, 2]
        gt_cov = gt_covs[:, v_idx]      # [N, 2, 2]
        
        # Compute difference
        diff = gt_v - pred_v  # [N, 2]
        
        # Add small regularization for numerical stability
        pred_cov = pred_cov + 1e-6 * torch.eye(2, device=pred_cov.device).unsqueeze(0)
        
        # Compute log determinant
        log_det = torch.logdet(pred_cov)  # [N]
        
        # Compute inverse
        pred_cov_inv = torch.inverse(pred_cov)  # [N, 2, 2]
        
        # Mahalanobis distance: (v-μ)^T Σ^{-1} (v-μ)
        # Using einsum for batch matrix operations
        mahalanobis = torch.einsum('ni,nij,nj->n', diff, pred_cov_inv, diff)  # [N]
        
        # NLL formula (ignoring constant term 2*log(2π))
        vertex_loss = 0.5 * (log_det + mahalanobis)
        
        total_loss += vertex_loss.sum()
    
    # Average over all vertices and samples
    return total_loss / (N * 4)

def giou_loss(pred_boxes, target_boxes):
    """
    Generalized IoU Loss
    pred_boxes, target_boxes: [N, 4] in [x1, y1, x2, y2] format
    """
    # Compute IoU
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    
    inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)
    
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-6)
    
    # Compute enclosing box
    x1_enclose = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_enclose = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_enclose = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_enclose = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    
    enclose_area = (x2_enclose - x1_enclose) * (y2_enclose - y1_enclose)
    
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    return (1 - giou).mean()

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return tuple(zip(*batch))

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_np(v) for v in obj]
    return obj


def cfg_node_to_dict(cfg_node):
    """Convert a yacs CfgNode to a dictionary."""
    if not isinstance(cfg_node, CN):
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = cfg_node_to_dict(v)
        return cfg_dict

def create_config(run_name, backbone, base_lr, batch_size, num_epochs,
                   horizontal_flip_prob, rotation_degrees, milestones, gamma, early_stopping,
                    pretrained_weights= '',
                    root='/kaggle/input/construction-industry-steel-ordering-lists-cisol/cisol_TD-TSR',
                    config_dir='/kaggle/working/create_config'):
    # Get default configuration
    cfg = get_cfg_defaults()

    # Update the configuration with provided arguments
    cfg.DATA.ROOT = root
    cfg.MISC.RUN_NAME = run_name
    cfg.MODEL.BACKBONE = backbone
    cfg.TRAIN.BASE_LR = base_lr
    cfg.TRAIN.BATCH_SIZE = batch_size
    cfg.TRAIN.NUM_EPOCHS = num_epochs
    cfg.AUGMENTATION.HORIZONTAL_FLIP_PROB = horizontal_flip_prob
    cfg.AUGMENTATION.ROTATION_DEGREES = rotation_degrees
    cfg.TRAIN.MILESTONES = milestones
    cfg.TRAIN.GAMMA = gamma
    cfg.MISC.PRETRAINED_WEIGHTS = pretrained_weights
    cfg.TRAIN.EARLY_STOPPING = early_stopping

    # Ensure the config directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Define the path for the new config file
    config_file_path = os.path.join(config_dir, f'{run_name}.yaml')
    #os.makedirs(config_file_path, exist_ok=True)

     # Convert the config object to a dictionary
    cfg_dict = cfg_node_to_dict(cfg)
    
    # Save the updated configuration to a YAML file
    with open(config_file_path, 'w') as config_file:
        yaml.dump(cfg_dict, config_file, default_flow_style=False)

    print(f"Config file saved at: {config_file_path}")
    with open(config_file_path, 'r') as file:
        print(file.read())
    return config_file_path




def load_pretrained_weights(network, weights_path, device):
    # Load the state_dict from the saved file
    state_dict = torch.load(weights_path, map_location=device)
    
    # Load the state_dict into the network
    network.load_state_dict(state_dict)
    
    return network
    pass # ToDo


def freeze_layers(network, frozen_layers):
    for name, param in network.named_parameters():
        if any(layer_name in name for layer_name in frozen_layers):
            param.requires_grad = False

def save_model(model, path):
    # Ensure device independence
    model_state = model.to(torch.device('cpu')).state_dict()
    torch.save(model_state, path + ".pth")

def get_stratified_param_groups(network, base_lr=0.001, stratification_rates=None):
    param_groups = []
    for name, param in network.named_parameters():
        for layer_name, lr in stratification_rates.items():
            if name.startswith(layer_name):
                param_groups.append({'params': param, 'lr': lr})
                break
        else:
            param_groups.append({'params': param, 'lr': base_lr})
    return param_groups


def write_results_to_csv(file_path, train_losses, test_losses, test_accuracies):
    
    with open(file_path + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Test Loss', 'Test Accuracy'])
        for epoch in range(len(train_losses)):
            writer.writerow([epoch + 1, train_losses[epoch], test_losses[epoch], test_accuracies[epoch]])



def visualize_inference_results(model, dataset, coco, device, num_images=5, output_dir='/kaggle/working/inference_images'):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.ToTensor()
    dataset_size = len(dataset)
    random_indices = random.sample(range(dataset_size), num_images)

    for idx in random_indices:
        img, _ = dataset[idx]
        img_id = coco.getImgIds(imgIds=coco.dataset['images'][idx]['id'])[0]
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(dataset.root, 'TD-TSR/images/val', img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # Convert image tensor to PIL image for visualization
        img = transforms.ToPILImage()(img_tensor.squeeze().cpu())

        # Plot image and bounding boxes
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img)

        for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            if score >= 0.5:  # Filter out low confidence detections
                xmin, ymin, xmax, ymax = box.cpu().numpy()
                width, height = xmax - xmin, ymax - ymin
                rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(xmin, ymin, f"{label.item()} {score:.2f}", bbox=dict(facecolor='yellow', alpha=0.5))

        plt.title(f"Inference Result for Image ID {img_id}")
        save_path = os.path.join(output_dir, f'inference_{img_id}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"Inference result saved for image {img_id} at {save_path}")


def generate_coco_results(model, test_image_dir, output_json_path, device):
    model = model.to(device)
    model.eval()
    transform = transforms.ToTensor()
    results = []
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    for image_name in os.listdir(test_image_dir):
        if image_name.endswith('.png'):
            img_path = os.path.join(test_image_dir, image_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(img_tensor)[0]

            for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
                if score >= 0.5:  # Use a threshold to filter low confidence detections
                    xmin, ymin, xmax, ymax = box.cpu().numpy()
                    width, height = xmax - xmin, ymax - ymin
                    results.append({
                        "file_name": image_name,
                        "category_id": int(label.item()),
                        "bbox": [float(xmin), float(ymin), float(width), float(height)],
                        "score": float(score.item())
                    })

    with open(output_json_path, 'w') as f:
        json.dump(results, f)

    print(f"COCO results saved to {output_json_path}")


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            value=self.value,
        )

class MetricLogger:
    def __init__(self, delimiter="\t"):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [header, "[{0" + space_fmt + "}/{1}]", "eta: {eta}", "{meters}", "time: {time}", "data: {data}"]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable), eta=eta_string, meters=str(self), time=str(iter_time), data=str(data_time)
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")



def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

# You will also need these helpers for the evaluation
def get_coco_api_from_dataset(dataset):
    # This function is needed for the CocoEvaluator
    # It assumes your dataset has a 'get_coco' method or similar
    # For now, we can create a placeholder if it doesn't exist
    if hasattr(dataset, 'coco'):
        return dataset.coco
    
    # If not, you might need to build a COCO object on the fly from your annotations
    # This is a simplified placeholder
    return COCO(dataset.annotation_file) # Assuming your dataset has this attribute

def reduce_dict(input_dict, average=True):
    # For multi-gpu support, not essential for single GPU but good practice
    return input_dict

class CocoEvaluator:
    # A placeholder for the full CocoEvaluator class. In a real scenario,
    # you would copy the full class from torchvision's references.
    def __init__(self, coco, iou_types):
        self.coco_gt = coco
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(self.coco_gt, iou_type) for iou_type in iou_types}
        self.results = []

    def update(self, predictions):
        self.results.extend(predictions)

    def synchronize_between_processes(self):
        pass # For multi-gpu

    def accumulate(self):
        if not self.results:
            return
        coco_dt = self.coco_gt.loadRes(self.results)
        for iou_type in self.iou_types:
            self.coco_eval[iou_type].cocoDt = coco_dt
            self.coco_eval[iou_type].evaluate()
            self.coco_eval[iou_type].accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"IoU metric: {iou_type}")
            coco_eval.summarize()


class OverfitDataLoader:
    """
    Caches N batches from a DataLoader for overfitting tests.
    
    Usage:
        normal_loader = DataLoader(dataset, sampler=sampler, ...)
        overfit_loader = OverfitDataLoader(normal_loader, num_batches=5)
        
        for epoch in range(100):
            for batch in overfit_loader:  # Same 5 batches every epoch
                train_step(batch)
    """
    
    def __init__(self, data_loader, num_batches, move_to_device=None):
        """
        Args:
            data_loader: Source DataLoader (sampler runs once during caching)
            num_batches: int or float (fraction of total batches)
            move_to_device: Optional device to pre-move tensors to
        """
        self.cached_batches = []
        self.device = move_to_device
        
        total_batches = len(data_loader)
        
        # Handle float (fraction) or int
        if isinstance(num_batches, float) and 0 < num_batches < 1:
            num_batches = max(1, int(total_batches * num_batches))
        
        num_batches = int(min(num_batches, total_batches))
        
        print(f"🔥 OVERFIT MODE: Caching {num_batches}/{total_batches} batches...")
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            # Optionally pre-move to GPU to avoid repeated transfers
            if self.device is not None:
                images, targets = batch
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} 
                          for t in targets]
                batch = (images, targets)
            
            self.cached_batches.append(batch)
        
        # Report what was cached
        self._report_cached_distribution()
    
    def _report_cached_distribution(self):
        """Print class distribution of cached batches"""
        from collections import Counter
        all_labels = []
        
        for images, targets in self.cached_batches:
            for t in targets:
                all_labels.extend(t['labels'].cpu().tolist())
        
        dist = Counter(all_labels)
        print(f"📊 Cached batch class distribution: {dict(sorted(dist.items()))}")
        print(f"📊 Total samples in cache: {len(all_labels)}")
    
    def __iter__(self):
        return iter(self.cached_batches)
    
    def __len__(self):
        return len(self.cached_batches)