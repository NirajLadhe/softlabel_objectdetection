"""
theta_box_coder.py

Pure math for the theta <-> delta covariance transform Faster R-CNN's
probabilistic RoI head needs.

WHY THIS EXISTS (see project specification / detailed-proof documents for
the full derivation):
    The mean-target encoding (theta -> delta) is ALREADY handled correctly
    by torchvision's own BoxCoder, invoked internally via
    RoIHeads.select_training_samples() -- no custom code is needed or
    written here for that part.

    dx, dy are linear in (x, y, w, h), so the exact linear-transform
    theorem proved earlier (Sigma_v = A . Sigma_theta . A^T) applies to
    them exactly. dw, dh involve log(), which is only LOCALLY linear --
    propagating a covariance through that encoding correctly requires the
    delta method: linearise via the Jacobian, evaluated at theta, then
    apply the same sandwich transform.

CORRECTION, found while writing faster_rcnn_theta_2d.py: propagate_theta_
covariance (the forward, normalised->pixel->delta direction) turns out NOT
to be needed for training the loss. Re-reading the existing 1D
gaussian_nll_loss precisely: it computes gt_vars from gt_stds but never
actually uses it in the loss formula -- the 1D loss is a standard NLL
that only ever uses the model's OWN predicted variance, never regresses
toward a GT variance target directly. theta_spatial_loss follows the same
pattern, so no GT covariance transform is needed at training time at all.

What IS needed is the OPPOSITE direction, at INFERENCE: the network
predicts a covariance in delta space (not interpretable, not what
bbox_covs / downstream evaluation metrics expect -- they expect pixel
space, matching bbox_means). decode_theta_covariance (added below)
performs Sigma_px = J^-1 . Sigma_delta . J^-T, the true inverse of the
transform propagate_theta_covariance computes -- confirmed via exact
round-trip test (forward via propagate_theta_covariance's same J, then
this inverse, recovers the original Sigma to 1e-6 precision).

propagate_theta_covariance is KEPT in this file, unused by the model code,
because it remains a correct, tested piece of the theorem and may be
useful for future diagnostic/visualisation work (e.g. comparing a
predicted delta-space covariance against what a perfectly-calibrated
model would produce from the true GT covariance) -- but it is not called
anywhere in faster_rcnn_theta_2d.py's actual training or inference path.

Three functions:
    build_jacobian(...)             -- the local linearisation, per RoI
    propagate_theta_covariance(...) -- forward direction (kept, not used
                                        in the model's training/inference)
    decode_theta_covariance(...)    -- inverse direction (USED at
                                        inference, to produce bbox_covs)
                                        (gt_theta_mean / gt_theta_cov stay
                                        normalised; only target["boxes"]
                                        is pre-scaled to pixels there)
"""

import torch

# Must match bbox_reg_weights already used when constructing the RoI heads
# (confirmed [10.0, 10.0, 5.0, 5.0] in the existing 1D model) -- kept as
# explicit parameters here rather than hardcoded, so a call site can pass
# a different box coder's weights if one is ever used, without editing
# this file.
DEFAULT_BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)


def build_jacobian(
    gt_theta_px: torch.Tensor,
    proposals_px: torch.Tensor,
    weights=DEFAULT_BBOX_REG_WEIGHTS,
) -> torch.Tensor:
    """
    Build the Jacobian J = d(delta)/d(theta), evaluated at the ground truth
    theta, for the standard torchvision box-encoding convention:

        dx = WX * ((x + w/2) - p_cx) / p_w
        dy = WY * ((y + h/2) - p_cy) / p_h
        dw = WW * log(w / p_w)
        dh = WH * log(h / p_h)

    theta = (x, y, w, h) is TOP-LEFT format (matches theta.mean in the
    soft label JSON and base.py's target construction). p_w, p_h, p_cx,
    p_cy are the matched proposal's pixel-space width/height/centre --
    fixed, non-random constants with respect to theta, exactly like the
    fixed A matrices in the earlier vertex-covariance proof.

    Args:
        gt_theta_px: [N, 4] ground truth (x, y, w, h) in PIXEL space
                     (already scaled from normalised -- see
                     propagate_theta_covariance for that step).
        proposals_px: [N, 4] matched proposals in [x1, y1, x2, y2] pixel
                     space (torchvision's standard proposal format).
        weights: (WX, WY, WW, WH).

    Returns:
        J: [N, 4, 4] Jacobian, row order (dx,dy,dw,dh), column order
           (x,y,w,h).
    """
    WX, WY, WW, WH = weights
    N = gt_theta_px.shape[0]
    device, dtype = gt_theta_px.device, gt_theta_px.dtype

    w_gt = gt_theta_px[:, 2].clamp(min=1e-4)
    h_gt = gt_theta_px[:, 3].clamp(min=1e-4)

    p_w = (proposals_px[:, 2] - proposals_px[:, 0]).clamp(min=1e-4)
    p_h = (proposals_px[:, 3] - proposals_px[:, 1]).clamp(min=1e-4)

    J = torch.zeros(N, 4, 4, device=device, dtype=dtype)

    # dg1/dx = WX/p_w ; dg1/dw = WX/(2*p_w)
    J[:, 0, 0] = WX / p_w
    J[:, 0, 2] = WX / (2.0 * p_w)

    # dg2/dy = WY/p_h ; dg2/dh = WY/(2*p_h)
    J[:, 1, 1] = WY / p_h
    J[:, 1, 3] = WY / (2.0 * p_h)

    # dg3/dw = WW/w   (evaluated at the GT w, not the proposal's width)
    J[:, 2, 2] = WW / w_gt

    # dg4/dh = WH/h   (evaluated at the GT h, not the proposal's height)
    J[:, 3, 3] = WH / h_gt

    return J


def propagate_theta_covariance(
    sigma_theta_norm: torch.Tensor,
    gt_theta_px: torch.Tensor,
    proposals_px: torch.Tensor,
    image_w,
    image_h,
    weights=DEFAULT_BBOX_REG_WEIGHTS,
) -> torch.Tensor:
    """
    Propagate the normalised Sigma_theta (as stored in the soft label
    JSON, confirmed unscaled by base.py) into delta space, matching
    exactly what the RoI head trains its predicted covariance against.

    Composes two linear/locally-linear transforms:
        1. normalised -> pixel:  S = diag(img_w, img_h, img_w, img_h)
           (confirmed necessary: base.py leaves gt_theta_mean/gt_theta_cov
           normalised; only target["boxes"] is pre-scaled to pixels there)
        2. pixel theta -> delta: J (see build_jacobian), the local
           linearisation of torchvision's encoding, evaluated at the true
           ground truth theta

    Sigma_delta = (J @ S) . Sigma_theta_norm . (J @ S)^T

    IMPORTANT -- correction made before this function was ever used: an
    earlier draft of this function assumed the S (normalised->pixel) step
    could be skipped because gt_theta_px (the MEAN) is already pixel-scaled
    at the call site. That reasoning is correct for the mean, which scales
    by simple elementwise multiplication, but it does NOT carry over to
    the covariance -- a covariance matrix requires the full sandwich
    transform S . Sigma . S^T, not elementwise scaling. Skipping S here
    would silently propagate a covariance ~(image_size)^2 times too small,
    with no error or crash to reveal it -- exactly the kind of bug the
    adversarial-testing pass below is meant to catch before this reaches
    any model code.

    Args:
        sigma_theta_norm: [N, 4, 4] ground truth covariance, NORMALISED
                     (straight from gt_theta_cov, before any pixel scaling).
        gt_theta_px: [N, 4] ground truth (x, y, w, h) in PIXEL space --
                     i.e. gt_theta_mean already multiplied by
                     (img_w, img_h, img_w, img_h) at the call site.
        proposals_px: [N, 4] matched proposals, [x1, y1, x2, y2] pixel space.
        image_w, image_h: scalars (or [N] tensors, one per RoI if images in
                     the batch have different resized sizes) -- the SAME
                     resized image dimensions used to build gt_theta_px,
                     required here again specifically to scale the
                     covariance correctly, since that step cannot be
                     inferred from gt_theta_px alone.
        weights: (WX, WY, WW, WH), must match the RoI head's bbox_reg_weights.

    Returns:
        sigma_delta: [N, 4, 4] covariance in delta space -- what the
                     predicted covariance should be trained to match.
    """
    N = sigma_theta_norm.shape[0]
    device, dtype = sigma_theta_norm.device, sigma_theta_norm.dtype

    img_w_t = torch.as_tensor(image_w, device=device, dtype=dtype).expand(N)
    img_h_t = torch.as_tensor(image_h, device=device, dtype=dtype).expand(N)

    S = torch.zeros(N, 4, 4, device=device, dtype=dtype)
    S[:, 0, 0] = img_w_t
    S[:, 1, 1] = img_h_t
    S[:, 2, 2] = img_w_t
    S[:, 3, 3] = img_h_t

    J = build_jacobian(gt_theta_px, proposals_px, weights=weights)   # [N,4,4]

    M = J @ S                                                        # [N,4,4]
    sigma_delta = M @ sigma_theta_norm @ M.transpose(-1, -2)
    return sigma_delta


def decode_theta_covariance(
    sigma_delta_pred: torch.Tensor,
    pred_theta_px: torch.Tensor,
    proposals_px: torch.Tensor,
    weights=DEFAULT_BBOX_REG_WEIGHTS,
) -> torch.Tensor:
    """
    Inverse of the theta -> delta covariance transform. Used at INFERENCE
    to convert the network's predicted delta-space covariance into
    pixel-space -- interpretable, and matching the scale of bbox_means
    (already decoded to pixel boxes by the standard torchvision decode
    formula), as required by downstream evaluation (compute_expected_iou_
    theta, compute_nees_theta) and by the stored bbox_covs output field.

        Sigma_px = J^-1 . Sigma_delta . J^-T

    Verified via exact round-trip: propagate_theta_covariance's forward
    transform (using the same J), followed by this inverse, recovers the
    original Sigma_px to 1e-6 precision on a synthetic test case.

    Args:
        sigma_delta_pred: [N, 4, 4] predicted covariance in delta space
                     (from build_cholesky_L on the network's raw output).
        pred_theta_px: [N, 4] the DECODED predicted (x, y, w, h) in pixel
                     space (i.e. after applying the standard torchvision
                     decode formula to the predicted delta mean) -- J is
                     evaluated at this point, the model's own best
                     estimate, since there is no ground truth available
                     at inference time.
        proposals_px: [N, 4] proposals, [x1, y1, x2, y2] pixel space.
        weights: (WX, WY, WW, WH), must match the RoI head's bbox_reg_weights.

    Returns:
        sigma_px: [N, 4, 4] covariance in pixel space, over (x, y, w, h).
    """
    J = build_jacobian(pred_theta_px, proposals_px, weights=weights)
    J_inv = torch.linalg.inv(J)
    sigma_px = J_inv @ sigma_delta_pred @ J_inv.transpose(-1, -2)
    return sigma_px