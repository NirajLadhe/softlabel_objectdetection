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

# ══════════════════════════════════════════════════════════════════════════
# Theta-space (4x4 Sigma_theta over x, y, w, h) losses.
#
# Companion to vertex_nll_loss_2d above, for the canonical Sigma_theta
# representation (see project specification document). Sigma_theta is
# built via a Cholesky factor L (Sigma_theta = L @ L^T), which guarantees
# validity for any raw network output -- no clipping or rejection needed.
# The parameterisation below was hardened through adversarial testing (see
# build_cholesky_L's docstring); an earlier, simpler version is documented
# there as a cautionary note, since the failure mode it had is easy to
# reintroduce by well-intentioned refactoring.
# ══════════════════════════════════════════════════════════════════════════
 
 
def build_cholesky_L(
    raw_params,
    log_diag_clamp_min: float = -8.0,
    log_diag_clamp_max: float = 1.0,
    max_corr_ratio: float = 0.9,
    diag_floor: float = 1e-4,
):
    """
    Build a batch of lower-triangular Cholesky factors L from raw network
    output, guaranteeing Sigma_theta = L @ L^T is always a valid (positive
    definite) covariance matrix, well-conditioned enough to produce a
    trainable (not just technically finite) loss.
 
    IMPORTANT -- correction from an earlier version of this function:
    An earlier version clamped the off-diagonal entries to a fixed
    ABSOLUTE range (e.g. +-5), independent of the diagonal. This is wrong:
    for a Cholesky factor, L[i,0] = Cov(x_i, x_0) / L[0,0] exactly, so a
    FIXED-size off-diagonal entry next to a SMALL diagonal entry implies an
    enormous, invalid correlation (confirmed empirically: ratio -11482 in
    one adversarial test) -- pushing Sigma_theta to near-singular, and
    solving a near-singular system (the Mahalanobis term) is exactly what
    produced the earlier "finite but training-destroying" loss/gradients
    (~1e15 / ~1e9), not a true edge case but a direct, reproducible
    consequence of bounding the wrong quantity.
 
    The fix: build the diagonal first, then parameterise EACH off-diagonal
    entry as a bounded FRACTION of its own row's diagonal, via tanh. This
    directly bounds the implied correlation to [-max_corr_ratio,
    +max_corr_ratio] regardless of how small or large the diagonal is --
    the off-diagonal now scales down automatically whenever the diagonal
    does, which is what actually prevents the cascading blow-up.
 
    Args:
        raw_params: [N, 10] raw network output, ordered:
            [z1, z2, z3, z4,            <- 4 diagonal entries (pre-exp)
             z5, z6, z7, z8, z9, z10]   <- 6 off-diagonal entries (free sign)
 
        log_diag_clamp_min / _max: bounds on raw diagonal input before
            exp(). theta is normalised to [0,1], so std has no legitimate
            reason to exceed ~e^1=2.7 (log_diag_clamp_max=1.0); real
            verified ground truth runs std in [0.003, 0.03] (log in
            [-5.8,-3.5]); -8.0 gives headroom below that.
 
        max_corr_ratio: caps each off-diagonal entry at this fraction of
            its row's diagonal value. 0.9 keeps every implied pairwise
            correlation safely below +-1 (a correlation of exactly +-1
            means perfectly degenerate, singular), while still allowing
            strong, meaningful correlation when the data supports it.
 
        diag_floor: small additive floor on the diagonal after exp(), a
            second independent safety line, consistent with the +1e-4
            floors already used elsewhere in this project's prior
            calculation code (Prior2DCalculator / PriorThetaCalculator).
 
    Returns:
        L: [N, 4, 4] lower-triangular, positive diagonal, off-diagonal
           bounded relative to the diagonal.
    """
    N = raw_params.shape[0]
    device, dtype = raw_params.device, raw_params.dtype
 
    diag_raw = raw_params[:, :4].clamp(min=log_diag_clamp_min, max=log_diag_clamp_max)
    diag_vals = torch.exp(diag_raw) + diag_floor           # [N, 4], > 0, bounded
 
    off_raw = torch.tanh(raw_params[:, 4:])                 # [N, 6], bounded to (-1, 1)
 
    L = torch.zeros(N, 4, 4, device=device, dtype=dtype)
    L[:, 0, 0] = diag_vals[:, 0]
 
    L[:, 1, 0] = off_raw[:, 0] * max_corr_ratio * diag_vals[:, 1]
    L[:, 1, 1] = diag_vals[:, 1]
 
    L[:, 2, 0] = off_raw[:, 1] * max_corr_ratio * diag_vals[:, 2]
    L[:, 2, 1] = off_raw[:, 2] * max_corr_ratio * diag_vals[:, 2]
    L[:, 2, 2] = diag_vals[:, 2]
 
    L[:, 3, 0] = off_raw[:, 3] * max_corr_ratio * diag_vals[:, 3]
    L[:, 3, 1] = off_raw[:, 4] * max_corr_ratio * diag_vals[:, 3]
    L[:, 3, 2] = off_raw[:, 5] * max_corr_ratio * diag_vals[:, 3]
    L[:, 3, 3] = diag_vals[:, 3]
 
    return L
 
 
def theta_mean_loss(pred_means, gt_means, beta: float = 1.0, reduction: str = "sum"):
    """
    Direct Smooth L1 loss on (x, y, w, h) -- the theta-space analogue of
    loss_box_mean in the 1D pipeline. Gives the mean prediction a clean,
    undistorted localisation gradient, independent of covariance quality.
 
    Args:
        pred_means: [N, 4] predicted (x, y, w, h)
        gt_means:   [N, 4] ground truth theta.mean, read directly from
                    the soft label JSON
    """
    return F.smooth_l1_loss(pred_means, gt_means, beta=beta, reduction=reduction)
 
 
def sigma_theta_nll_loss(
    pred_means,
    pred_L_raw,
    gt_theta,
    detach_mean: bool = True,
    eps: float = 1e-6,
):
    """
    Multivariate Gaussian negative log-likelihood over theta = (x, y, w, h),
    using a Cholesky-parameterised covariance. Theta-space generalisation
    of gaussian_nll_loss above (the 1D scalar case is the 1x1 special case
    of this same construction).
 
        NLL = 0.5 * ( log|Sigma_theta| + (gt - mean)^T . Sigma_theta^-1 . (gt - mean) )
 
    Computed via the Cholesky factor directly, never forming
    Sigma_theta^-1 or calling logdet() explicitly:
        log|Sigma_theta| = 2 * sum(log of L's diagonal)   -- cheap, stable
        Mahalanobis term = ||y||^2, where L @ y = diff     -- triangular solve,
                                                                not a general inverse
 
    Args:
        pred_means:  [N, 4] predicted (x, y, w, h). Only used to form the
                     residual; see detach_mean.
        pred_L_raw:  [N, 10] raw network output for the Cholesky factor
                     (see build_cholesky_L for the exact layout).
        gt_theta:    [N, 4] ground truth theta.mean from the soft label JSON.
        detach_mean: if True (default), gradients from this loss do NOT
                     flow into pred_means -- localisation is left entirely
                     to theta_mean_loss, and this loss only calibrates the
                     covariance around wherever the mean loss has already
                     placed the box. Prevents the two losses from fighting
                     over the mean with different effective gradient scales.
        eps:         numerical floor to keep log() well-defined.
 
    Returns:
        Scalar loss (summed over the batch; divide by N at the call site
        if a mean-reduction is preferred, matching how loss_box_mean is
        normalised in the existing 1D code).
    """
    means_for_residual = pred_means.detach() if detach_mean else pred_means
 
    L = build_cholesky_L(pred_L_raw)                        # [N, 4, 4]
 
    diag = torch.diagonal(L, dim1=-2, dim2=-1)               # [N, 4]
    log_det_sigma = 2.0 * torch.log(diag.clamp(min=eps)).sum(dim=1)   # [N]
 
    diff = (gt_theta - means_for_residual).unsqueeze(-1)     # [N, 4, 1]
    y = torch.linalg.solve_triangular(L, diff, upper=False)   # [N, 4, 1]
    mahalanobis = (y.squeeze(-1) ** 2).sum(dim=1)             # [N]
 
    loss = 0.5 * (log_det_sigma + mahalanobis)
    return loss.sum()
 
 
def theta_spatial_loss(
    pred_means,
    pred_L_raw,
    gt_means,
    gt_theta,
    lambda_mean: float = 0.3,
    lambda_cov: float = 1.0,
    detach_mean_in_cov: bool = True,
    num_normalizer: float = None,
):
    """
    Combined theta-space spatial loss -- the direct analogue of computing
    loss_box_mean and loss_box_var together in the existing 1D RoI head.
    Returns both components separately (not pre-summed) so they can be
    logged and monitored independently, matching how train/loss_box_mean
    and train/loss_box_var are logged separately today.
 
    Args:
        pred_means:  [N, 4] predicted (x, y, w, h)
        pred_L_raw:  [N, 10] raw Cholesky-factor network output
        gt_means:    [N, 4] ground truth mean (kept as a separate arg for
                     API symmetry with the existing loss_box_mean call site)
        gt_theta:    [N, 4] ground truth theta mean (used inside the NLL)
        lambda_mean: weight on the mean loss
        lambda_cov:  weight on the covariance NLL loss
        detach_mean_in_cov: see sigma_theta_nll_loss
        num_normalizer: if provided, both losses are divided by this
                     (typically all_labels.numel(), matching how
                     loss_box_mean is normalised in the 1D code); if
                     None, losses are returned as raw sums.
 
    Returns:
        {"loss_theta_mean": scalar, "loss_theta_cov": scalar}
    """
    loss_mean = theta_mean_loss(pred_means, gt_means, reduction="sum")
    loss_cov = sigma_theta_nll_loss(
        pred_means, pred_L_raw, gt_theta, detach_mean=detach_mean_in_cov
    )
 
    if num_normalizer is not None:
        loss_mean = loss_mean / num_normalizer
        loss_cov = loss_cov / num_normalizer
 
    return {
        "loss_theta_mean": loss_mean * lambda_mean,
        "loss_theta_cov": loss_cov * lambda_cov,
    }
 