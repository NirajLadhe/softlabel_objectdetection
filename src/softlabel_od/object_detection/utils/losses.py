"""
Loss functions for probabilistic object detection.

All functions operate on PyTorch tensors and return scalar losses unless
noted otherwise.
"""

import torch
import torch.nn.functional as F


def gaussian_nll_loss(pred_means, pred_log_vars, gt_means, gt_stds):
    """Gaussian Negative Log-Likelihood loss."""
    gt_vars = gt_stds.pow(2) + 1e-6
    vars_pred = torch.exp(pred_log_vars)
    loss = 0.5 * (pred_log_vars + (gt_means - pred_means).pow(2) / vars_pred)
    return loss.sum()


def gaussian_kl_loss(pred_means, pred_log_vars, gt_means, gt_stds):
    """KL divergence KL(N(gt_mean, gt_std) || N(pred_mean, pred_std))."""
    pred_log_vars = torch.clamp(pred_log_vars, min=-10, max=10)
    pred_stds = torch.exp(0.5 * pred_log_vars)
    gt_stds = gt_stds.clamp(min=1e-6)
    log_ratio = torch.log(pred_stds / gt_stds)
    mean_diff_sq = (gt_means - pred_means).pow(2)
    gt_var = gt_stds.pow(2)
    pred_var = pred_stds.pow(2)
    kl = log_ratio + (gt_var + mean_diff_sq) / (2 * pred_var) - 0.5
    return kl.mean()


def js_divergence_categorical(p, q, eps=1e-8):
    """Jensen-Shannon divergence for categorical distributions. p, q: [N, C]."""
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    m = 0.5 * (p + q)
    js = 0.5 * (q * (q / m).log()).sum(dim=-1) + 0.5 * (p * (p / m).log()).sum(dim=-1)
    return js.mean()


def variance_only_kl(gt_std, pred_std, eps=1e-8):
    """KL(N(0, gt_std^2) || N(0, pred_std^2)) — removes mean-mismatch term."""
    gt_std = gt_std.clamp(min=eps)
    pred_std = pred_std.clamp(min=eps)
    return (torch.log(pred_std / gt_std) + (gt_std.pow(2)) / (2.0 * pred_std.pow(2)) - 0.5).mean()


def variance_only_sym_kl(gt_std, pred_std, eps=1e-8):
    """Symmetric KL for zero-mean Gaussians."""
    gt_std = gt_std.clamp(min=eps)
    pred_std = pred_std.clamp(min=eps)
    kl_gt_pred = torch.log(pred_std / gt_std) + (gt_std.pow(2)) / (2.0 * pred_std.pow(2)) - 0.5
    kl_pred_gt = torch.log(gt_std / pred_std) + (pred_std.pow(2)) / (2.0 * gt_std.pow(2)) - 0.5
    return 0.5 * (kl_gt_pred + kl_pred_gt).mean()


def huber_log_std_loss(pred_log_vars, gt_stds, beta=1.0, eps=1e-8):
    """Huber loss on log-standard deviations."""
    pred_stds = torch.exp(0.5 * pred_log_vars)
    gt_stds = gt_stds.clamp(min=eps)
    return F.smooth_l1_loss(pred_stds, gt_stds, beta=beta, reduction="mean")


def soft_cross_entropy_loss(logits, soft_targets, class_weights=None, reduction="mean"):
    """
    Cross-entropy loss that accepts soft (probability) targets.

    Args:
        logits:        [N, C] raw model outputs.
        soft_targets:  [N, C] target probability distributions.
        class_weights: [C] optional per-class weights.
        reduction:     'mean', 'sum', or 'none'.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    if class_weights is not None:
        w = class_weights.to(logits.device).unsqueeze(0)
        per_sample = (-soft_targets * log_probs * w).sum(dim=-1)
    else:
        per_sample = (-soft_targets * log_probs).sum(dim=-1)

    if reduction == "none":
        return per_sample
    elif reduction == "sum":
        return per_sample.sum()
    return per_sample.mean()


def soft_focal_loss(pred_logits, gt_labels, alpha=0.25, gamma=2.0, num_classes=None):
    """
    Focal loss supporting both soft (probability) and hard (integer) targets.

    Args:
        pred_logits: [N, C] raw model outputs.
        gt_labels:   [N, C] soft labels OR [N] hard integer labels.
        alpha:       Focusing parameter.
        gamma:       Modulation exponent.
        num_classes: Required when gt_labels are hard integers.
    """
    if gt_labels.dim() == 1:
        if num_classes is None:
            num_classes = pred_logits.shape[-1]
        gt_class_probs = F.one_hot(gt_labels, num_classes=num_classes).float()
    else:
        gt_class_probs = gt_labels

    pred_probs = F.softmax(pred_logits, dim=-1)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    ce_loss = -gt_class_probs * log_probs

    p_t = torch.sum(pred_probs * gt_class_probs, dim=-1, keepdim=True)
    modulation = (1 - p_t) ** gamma
    if isinstance(alpha, torch.Tensor) and alpha.dim() >= 1:
        focal_weight = alpha.unsqueeze(0) * modulation
    else:
        focal_weight = alpha * modulation

    return (focal_weight * ce_loss).sum(dim=-1).mean()


def vertex_nll_loss_2d(pred_vertices, pred_covs, gt_vertices, gt_covs):
    """
    Negative Log-Likelihood for 2D vertex predictions.

    Each vertex follows a 2D Gaussian: v ~ N(μ, Σ).
    NLL = 0.5 * (log|Σ| + (v-μ)^T Σ^{-1} (v-μ))

    Args:
        pred_vertices: [N, 8] predicted vertex means.
        pred_covs:     [N, 4, 2, 2] predicted covariance matrices.
        gt_vertices:   [N, 8] ground truth vertices.
        gt_covs:       [N, 4, 2, 2] ground truth covariances.

    Returns:
        Scalar loss.
    """
    N = pred_vertices.shape[0]
    total_loss = 0.0

    for v_idx in range(4):
        pred_v = pred_vertices[:, v_idx * 2 : (v_idx + 1) * 2]
        gt_v = gt_vertices[:, v_idx * 2 : (v_idx + 1) * 2]
        pred_cov = pred_covs[:, v_idx]

        pred_cov = pred_cov + 1e-6 * torch.eye(2, device=pred_cov.device).unsqueeze(0)
        log_det = torch.logdet(pred_cov)
        pred_cov_inv = torch.inverse(pred_cov)

        diff = gt_v - pred_v
        mahalanobis = torch.einsum("ni,nij,nj->n", diff, pred_cov_inv, diff)
        total_loss += 0.5 * (log_det + mahalanobis).sum()

    return total_loss / (N * 4)


def giou_loss(pred_boxes, target_boxes):
    """
    Generalised IoU Loss.

    Args:
        pred_boxes:   [N, 4] in [x1, y1, x2, y2] format.
        target_boxes: [N, 4] in [x1, y1, x2, y2] format.
    """
    x1_inter = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    y1_inter = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    x2_inter = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    y2_inter = torch.min(pred_boxes[:, 3], target_boxes[:, 3])

    inter_area = (x2_inter - x1_inter).clamp(min=0) * (y2_inter - y1_inter).clamp(min=0)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
    union_area = pred_area + target_area - inter_area
    iou = inter_area / (union_area + 1e-6)

    x1_enc = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    y1_enc = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    x2_enc = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    y2_enc = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    enclose_area = (x2_enc - x1_enc) * (y2_enc - y1_enc)

    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    return (1 - giou).mean()
