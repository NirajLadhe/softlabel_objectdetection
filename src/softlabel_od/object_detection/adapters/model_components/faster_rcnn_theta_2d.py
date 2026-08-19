"""
faster_rcnn_theta_2d.py

Probabilistic Faster R-CNN with a 4x4 Sigma_theta (over x, y, w, h) head.
Structurally mirrors faster_rcnn_1d.py EXACTLY for every piece that does
not depend on the spatial-uncertainty representation (FPN backbone,
anchors, RPN thresholds, RoI matching thresholds, box_reg_weights) --
confirmed against the real faster_rcnn_1d.py, not recalled from memory.
Only the predictor head, the box loss computation, and the inference
covariance decode differ from that file.

IMPORTANT DESIGN NOTE (found while writing this file, not assumed going
in): the existing 1D loss (gaussian_nll_loss) computes gt_vars from
gt_stds but never actually uses it in the loss formula -- it is a
standard NLL that only ever uses the model's OWN predicted variance.
theta_spatial_loss (see losses.py) follows the same pattern. This means
NO ground-truth covariance transform is needed at training time --
propagate_theta_covariance (theta_box_coder.py) is correct and tested,
but genuinely unused here. What IS needed, and IS used below, is the
INVERSE direction at inference (decode_theta_covariance), to convert the
network's predicted delta-space covariance into interpretable pixel
space for the bbox_covs output field.

Same POS_CLS_WEIGHT / NEG_CLS_WEIGHT classification weighting mechanism
as the 1D head, unchanged -- confirmed this does not exclude background
from training, only reweights it (see conversation). Box regression
(loss_theta_mean, loss_theta_cov) scoped to pathology_positive_indices
only (excludes "No Finding"), matching the 1D box regression scope
exactly, per explicit instruction to keep this consistent rather than
diverge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from softlabel_od.object_detection.adapters.model_components.theta_box_coder import (
    build_jacobian,
    decode_theta_covariance,
    propagate_theta_covariance,
    DEFAULT_BBOX_REG_WEIGHTS,
)
from softlabel_od.object_detection.utils.losses import (
    theta_spatial_loss,
    soft_cross_entropy_loss,
    build_cholesky_L,
)


# ══════════════════════════════════════════════════════════════════════
# Predictor head
# ══════════════════════════════════════════════════════════════════════

class ProbabilisticThetaPredictor(nn.Module):
    """
    Outputs, per RoI, per class: 4 mean values (x, y, w, h delta) + 10 raw
    Cholesky-factor values (see build_cholesky_L in losses.py) = 14 total.
    Structurally the direct extension of ProbabilisticFastRCNNPredictor's
    num_classes*8 (4 means + 4 log-vars) -> num_classes*14.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.gmm_params = nn.Linear(in_channels, num_classes * 14)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.constant_(self.cls_score.bias, 0)

        nn.init.normal_(self.gmm_params.weight, std=0.001)
        nn.init.constant_(self.gmm_params.bias, 0)

        # Bias-initialise the 4 diagonal Cholesky entries (indices 4:8 of
        # each class's 14-slot block) to -4.0, matching the 1D predictor's
        # log_var_bias philosophy exactly: start with small, non-degenerate
        # uncertainty rather than a random value. Off-diagonal entries
        # (indices 8:14) stay at the default 0 bias -- no correlation
        # assumed at initialisation, a neutral starting point.
        with torch.no_grad():
            bias_view = self.gmm_params.bias.view(num_classes, 14)
            bias_view[:, 4:8] = -4.0

    def forward(self, x):
        if x.dim() == 4:
            x = x.flatten(start_dim=1)
        class_logits = self.cls_score(x)
        gmm_params = self.gmm_params(x)
        return class_logits, gmm_params


# ══════════════════════════════════════════════════════════════════════
# RoI Heads
# ══════════════════════════════════════════════════════════════════════

class ProbabilisticThetaRoIHeads(RoIHeads):
    """
    Theta-space counterpart of ProbabilisticRoIHeads. Same structural
    pattern (matching, positive-index scoping, classification loss) --
    only box regression's loss computation and inference decode differ.
    """

    def __init__(self, *args, pos_cls_weight=1.2, neg_cls_weight=0.8,
                lambda_theta_mean=0.3, lambda_theta_cov=1.0, lambda_theta_kl=0.0,
                score_thresh=0.10, nms_thresh=0.17, detections_per_img=200,
                bbox_reg_weights=DEFAULT_BBOX_REG_WEIGHTS, **kwargs):
        super().__init__(
            *args,
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            bbox_reg_weights=bbox_reg_weights,
            **kwargs,
        )
        self.pos_cls_weight = pos_cls_weight
        self.neg_cls_weight = neg_cls_weight
        self.lambda_theta_mean = lambda_theta_mean
        self.lambda_theta_cov = lambda_theta_cov
        self.lambda_theta_kl = lambda_theta_kl
        self.bbox_reg_weights_tuple = bbox_reg_weights

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, gmm_params = self.box_predictor(box_features)

        result = []
        losses = {}

        if self.training:
            assert labels is not None and regression_targets is not None

            all_labels = torch.cat(labels, dim=0)
            all_reg_targets = torch.cat(regression_targets, dim=0)

            # Soft cross-entropy classification -- same structure as the
            # 1D head, POS/NEG_CLS_WEIGHT unchanged, background included
            # (reweighted, not excluded -- confirmed).
            gt_class_probs_list = []
            for i, t in enumerate(targets):
                m_idx = matched_idxs[i]
                pos_mask_i = labels[i] > 0
                gt_class_probs_list.append(
                    t["gt_class_probs"][m_idx[pos_mask_i]]
                )
            gt_class_probs_pos = torch.cat(gt_class_probs_list, dim=0)

            cls_targets = torch.zeros_like(class_logits)
            pos_mask = all_labels > 0
            neg_mask = all_labels == 0
            cls_targets[neg_mask, 0] = 1.0
            cls_targets[pos_mask] = gt_class_probs_pos

            # class_weights buffer: registered externally by
            # BaseDetectionLightningModule.on_fit_start() (confirmed --
            # inherited unchanged by SoftLabel2DThetaLightningModule, no
            # override needed), via train_dataset.compute_roi_head_class_
            # weights_per_cluster(...) + register_buffer("class_weights", ...)
            # directly on self (roi_heads). getattr with None fallback
            # handles inference-only usage where on_fit_start() never ran
            # (e.g. loading a checkpoint without going through Trainer.fit()).
            class_weights = getattr(self, "class_weights", None)
            per_sample_ce = soft_cross_entropy_loss(
                class_logits, cls_targets, class_weights=class_weights, reduction="none"
            )
            pos_loss = per_sample_ce[pos_mask].mean() if pos_mask.any() else per_sample_ce.sum() * 0.0
            neg_loss = per_sample_ce[neg_mask].mean() if neg_mask.any() else per_sample_ce.sum() * 0.0
            loss_cls = self.pos_cls_weight * pos_loss + self.neg_cls_weight * neg_loss

            # Box regression: scoped to pathology_positive_indices (labels > 1,
            # excludes "No Finding") -- matches the 1D head's box regression
            # scope exactly, per explicit instruction to keep consistent.
            pathology_mask = all_labels > 1
            num_classes = class_logits.shape[1]
            gmm_params_reshaped = gmm_params.view(-1, num_classes, 14)

            # Gather gt_theta_px, proposals_px, gt_cov (normalised), and
            # per-proposal image size -- in the SAME concatenation order
            # as all_labels/all_reg_targets -- so they can be sliced by
            # pathology_mask alongside pred_means/pred_L_raw/pos_reg_targets
            # with guaranteed index alignment. Needed to transform gt_cov
            # from normalised space into delta space via
            # propagate_theta_covariance before it reaches the KL loss --
            # confirmed necessary (see conversation): Sigma_pred lives in
            # delta space, raw theta.covariance does not, and comparing
            # them directly without this transform would be the same class
            # of unit-mismatch bug already caught once in this project.
            all_gt_theta_px, all_proposals_px, all_gt_cov_norm = [], [], []
            all_img_w, all_img_h = [], []
            for i, t in enumerate(targets):
                m_idx = matched_idxs[i]
                props_i = proposals[i]
                gt_boxes_i = t["boxes"][m_idx]  # [n_props_i, 4] x1y1x2y2, pixel space
                gt_theta_px_i = torch.stack([
                    gt_boxes_i[:, 0], gt_boxes_i[:, 1],
                    gt_boxes_i[:, 2] - gt_boxes_i[:, 0], gt_boxes_i[:, 3] - gt_boxes_i[:, 1],
                ], dim=1)
                gt_cov_i = t["gt_theta_cov"][m_idx]  # [n_props_i, 4, 4], NORMALISED
                h_i, w_i = image_shapes[i]
                all_gt_theta_px.append(gt_theta_px_i)
                all_proposals_px.append(props_i)
                all_gt_cov_norm.append(gt_cov_i)
                all_img_w.append(torch.full((props_i.shape[0],), float(w_i), device=props_i.device))
                all_img_h.append(torch.full((props_i.shape[0],), float(h_i), device=props_i.device))

            all_gt_theta_px = torch.cat(all_gt_theta_px, dim=0)
            all_proposals_px = torch.cat(all_proposals_px, dim=0)
            all_gt_cov_norm = torch.cat(all_gt_cov_norm, dim=0)
            all_img_w = torch.cat(all_img_w, dim=0)
            all_img_h = torch.cat(all_img_h, dim=0)

            if pathology_mask.any():
                pos_labels = all_labels[pathology_mask]
                pos_gmm = gmm_params_reshaped[pathology_mask]
                idx = torch.arange(pos_gmm.shape[0], device=pos_gmm.device)
                pos_gmm_selected = pos_gmm[idx, pos_labels]   # [N_pos, 14]

                pred_means = pos_gmm_selected[:, :4]
                pred_L_raw = pos_gmm_selected[:, 4:]
                pos_reg_targets = all_reg_targets[pathology_mask]

                # Slice the new parallel tensors by the SAME mask, then
                # transform gt_cov to delta space before the loss ever sees it.
                gt_theta_px_pos = all_gt_theta_px[pathology_mask]
                proposals_px_pos = all_proposals_px[pathology_mask]
                gt_cov_norm_pos = all_gt_cov_norm[pathology_mask]
                img_w_pos = all_img_w[pathology_mask]
                img_h_pos = all_img_h[pathology_mask]

                gt_cov_delta_pos = propagate_theta_covariance(
                    gt_cov_norm_pos, gt_theta_px_pos, proposals_px_pos,
                    img_w_pos, img_h_pos, weights=self.bbox_reg_weights_tuple,
                )

                theta_losses_dict = theta_spatial_loss(
                    pred_means, pred_L_raw, pos_reg_targets, pos_reg_targets,
                    gt_cov=gt_cov_delta_pos,
                    lambda_mean=self.lambda_theta_mean,
                    lambda_cov=self.lambda_theta_cov,
                    lambda_kl=self.lambda_theta_kl,
                    detach_mean_in_cov=True,
                    # num_normalizer=float(all_labels.numel()),
                    num_normalizer=float(pathology_mask.sum()),
                )
                with torch.no_grad():
                    L_pred_diag = build_cholesky_L(pred_L_raw)
                    eye_diag = torch.eye(4, device=pred_L_raw.device, dtype=pred_L_raw.dtype).expand(
                        pred_L_raw.shape[0], 4, 4
                    )
                    Sigma_pred_inv_diag = torch.cholesky_solve(eye_diag, L_pred_diag)
                    diff_diag = (pos_reg_targets - pred_means).unsqueeze(-1)
                    nees_delta_per_sample = (
                        diff_diag.transpose(-1, -2) @ Sigma_pred_inv_diag @ diff_diag
                    ).squeeze(-1).squeeze(-1)
                    theta_losses_dict["nees_delta"] = nees_delta_per_sample.mean().detach()

                    theta_losses_dict["diag_raw_max"] = pred_L_raw[:, :4].max().detach()
                    theta_losses_dict["diag_clamp_saturation_frac"] = (
                        (pred_L_raw[:, :4] >= 0.99).float().mean().detach()
                    )
                    Sigma_pred_diag_only = L_pred_diag @ L_pred_diag.transpose(-1, -2)
                    diag_Sigma = torch.diagonal(Sigma_pred_diag_only, dim1=-2, dim2=-1)
                    diff_flat = diff_diag.squeeze(-1)
                    per_axis_z2 = (diff_flat ** 2) / diag_Sigma.clamp(min=1e-8)
                    per_axis_mean = per_axis_z2.mean(dim=0)
                    theta_losses_dict["nees_delta_x"] = per_axis_mean[0].detach()
                    theta_losses_dict["nees_delta_y"] = per_axis_mean[1].detach()
                    theta_losses_dict["nees_delta_w"] = per_axis_mean[2].detach()
                    theta_losses_dict["nees_delta_h"] = per_axis_mean[3].detach()
            else:
                zero = gmm_params.sum() * 0.0
                theta_losses_dict = {
                    "loss_theta_mean": zero,
                    "loss_theta_cov": zero,
                    "loss_theta_kl": zero,
                    "loss_theta_mean_raw": zero.detach(),
                    "loss_theta_cov_raw": zero.detach(),
                    "loss_theta_kl_raw": zero.detach(),
                }

            losses = {
                "loss_probabilistic_classifier": loss_cls,
                **theta_losses_dict,
            }

        else:
            # ── Inference: decode means + covariance to pixel space ──
            boxes_per_image = [p.shape[0] for p in proposals]
            pred_scores = F.softmax(class_logits, dim=-1)
            num_classes = class_logits.shape[1]
            gmm_params_reshaped = gmm_params.view(-1, num_classes, 14)

            proposals_cat = torch.cat(proposals, dim=0)
            pred_labels_all = pred_scores.argmax(dim=1)
            idx = torch.arange(gmm_params_reshaped.shape[0], device=gmm_params_reshaped.device)
            selected_gmm = gmm_params_reshaped[idx, pred_labels_all]   # [N, 14]

            pred_means_delta = selected_gmm[:, :4]
            pred_L_raw = selected_gmm[:, 4:]

            # Decode means: standard torchvision delta -> pixel formula,
            # same WX,WY,WW,WH convention as build_jacobian uses.
            WX, WY, WW, WH = self.bbox_reg_weights_tuple
            p_w = (proposals_cat[:, 2] - proposals_cat[:, 0]).clamp(min=1e-4)
            p_h = (proposals_cat[:, 3] - proposals_cat[:, 1]).clamp(min=1e-4)
            p_cx = proposals_cat[:, 0] + 0.5 * p_w
            p_cy = proposals_cat[:, 1] + 0.5 * p_h

            dx, dy, dw, dh = (
                pred_means_delta[:, 0], pred_means_delta[:, 1],
                pred_means_delta[:, 2], pred_means_delta[:, 3],
            )
            pred_cx = (dx / WX) * p_w + p_cx
            pred_cy = (dy / WY) * p_h + p_cy
            pred_w = torch.exp(dw / WW) * p_w
            pred_h = torch.exp(dh / WH) * p_h

            pred_x = pred_cx - 0.5 * pred_w
            pred_y = pred_cy - 0.5 * pred_h
            pred_theta_px = torch.stack([pred_x, pred_y, pred_w, pred_h], dim=1)
            pred_boxes_xyxy = torch.stack(
                [pred_x, pred_y, pred_x + pred_w, pred_y + pred_h], dim=1
            )

            # Decode covariance: delta -> pixel, via the confirmed,
            # round-trip-tested inverse Jacobian transform.
            sigma_delta_pred = build_cholesky_L(pred_L_raw)
            sigma_delta_pred = sigma_delta_pred @ sigma_delta_pred.transpose(-1, -2)
            sigma_px_pred = decode_theta_covariance(
                sigma_delta_pred, pred_theta_px, proposals_cat,
                weights=self.bbox_reg_weights_tuple,
            )
            sigma_px_pred = 0.5 * (sigma_px_pred + sigma_px_pred.transpose(-1, -2))
            sigma_px_pred = sigma_px_pred + 1e-6 * torch.eye(4, device=sigma_px_pred.device, dtype=sigma_px_pred.dtype)

            pred_scores_max, pred_labels_final = pred_scores.max(dim=1)

            boxes_split = pred_boxes_xyxy.split(boxes_per_image, dim=0)
            scores_split = pred_scores_max.split(boxes_per_image, dim=0)
            labels_split = pred_labels_final.split(boxes_per_image, dim=0)
            covs_split = sigma_px_pred.split(boxes_per_image, dim=0)
            covs_delta_split = sigma_delta_pred.split(boxes_per_image, dim=0)
            class_probs_split = pred_scores.split(boxes_per_image, dim=0)
            proposals_split = proposals_cat.split(boxes_per_image, dim=0)
            pred_means_delta_split = pred_means_delta.split(boxes_per_image, dim=0)

            for boxes_i, scores_i, labels_i, covs_i, covs_delta_i, proposals_i, pred_delta_i, cprobs_i in zip(
                boxes_split, scores_split, labels_split, covs_split, covs_delta_split,
                proposals_split, pred_means_delta_split, class_probs_split
            ):
                keep = scores_i > self.score_thresh
                boxes_i, scores_i = boxes_i[keep], scores_i[keep]
                labels_i, covs_i, covs_delta_i, proposals_i, pred_delta_i, cprobs_i = (
                    labels_i[keep], covs_i[keep], covs_delta_i[keep],
                    proposals_i[keep], pred_delta_i[keep], cprobs_i[keep]
                )

                keep_nms = torchvision.ops.nms(boxes_i, scores_i, self.nms_thresh)
                keep_nms = keep_nms[: self.detections_per_img]

                final_boxes = boxes_i[keep_nms]
                final_covs_px = covs_i[keep_nms]
                final_covs_delta = covs_delta_i[keep_nms]
                final_proposals = proposals_i[keep_nms]

                if final_boxes.shape[0] > 0:
                    final_theta_px = torch.stack([
                        final_boxes[:, 0], final_boxes[:, 1],
                        final_boxes[:, 2] - final_boxes[:, 0],
                        final_boxes[:, 3] - final_boxes[:, 1],
                    ], dim=1)
                    sigma_px_check = decode_theta_covariance(
                        final_covs_delta, final_theta_px, final_proposals,
                        weights=self.bbox_reg_weights_tuple,
                    )
                    max_diff = (sigma_px_check - final_covs_px).abs().max().item()
                    is_close = torch.allclose(sigma_px_check, final_covs_px, atol=1e-3, rtol=1e-3)
                    if not is_close:
                        print(f"WARNING: post-filtering self-consistency check FAILED, max_diff={max_diff}")
                    assert is_close, \
                        f"Decoded covariance inconsistent AFTER filtering! max_diff={max_diff}"

                result.append({
                    "boxes": final_boxes,
                    "scores": scores_i[keep_nms],
                    "labels": labels_i[keep_nms],
                    "bbox_covs": final_covs_px,
                    "bbox_covs_delta": final_covs_delta,
                    "proposal_boxes": final_proposals,
                    "pred_means_delta": pred_delta_i[keep_nms],
                    "class_probs": cprobs_i[keep_nms],
                })

        return result, losses


# ══════════════════════════════════════════════════════════════════════
# Factory function
# ══════════════════════════════════════════════════════════════════════

def get_model_theta_2d(
    num_classes, backbone_name="resnet50", pretrained=True,
    anchor_sizes=None, anchor_aspect_ratios=None,
    lambda_theta_mean=0.3, lambda_theta_cov=0.02, lambda_theta_kl=0.0,
    pos_cls_weight=1.2, neg_cls_weight=0.8,
):
    """
    Build a probabilistic Faster R-CNN with the 4x4 Sigma_theta head.
    Every structural value below (FPN, anchors default, resolution, RPN/
    RoI thresholds, bbox_reg_weights) copied EXACTLY from the confirmed,
    real faster_rcnn_1d.py -- not recalled, verified line-by-line against
    the actual file before writing this.

    lambda_theta_mean, lambda_theta_cov, pos_cls_weight, neg_cls_weight:
    now accepted and forwarded to ProbabilisticThetaRoIHeads -- previously
    these were silently hardcoded to the class defaults regardless of
    what a config specified (flagged explicitly in
    nucls_theta_2d_softlabel_config.yaml's TRAIN section comments; that
    flag is resolved by this change). factory.py's "theta_2d" branch and
    the caller (base_module.py, via get_model()) still need to actually
    pass these through from cfg.TRAIN.LAMBDA_THETA_MEAN / LAMBDA_THETA_COV
    -- see note at the bottom of this function.
    """
    backbone = resnet_fpn_backbone(backbone_name, pretrained=pretrained, trainable_layers=3)

    # torchvision's AnchorGenerator strictly requires Tuple[Tuple[int]] /
    # Tuple[Tuple[float]] and asserts on type -- NOT just matching length.
    # Config-sourced anchor_sizes/anchor_aspect_ratios arrive as lists of
    # lists (YAML has no native tuple type), which fail this assert even
    # though the VALUES are correct. The `or` fallback below only produces
    # the correct type for the None-default case; anything passed in from
    # config needs explicit conversion too. Missed in earlier testing,
    # which only ever exercised the None-default path or manually-built
    # tuples -- never a real YAML-sourced list-of-lists.
    if anchor_sizes is None:
        anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
    else:
        anchor_sizes = tuple(tuple(s) for s in anchor_sizes)

    if anchor_aspect_ratios is None:
        aspect_ratios = ((0.5, 0.75, 1.0, 1.5, 2.0),) * len(anchor_sizes)
    else:
        aspect_ratios = tuple(tuple(a) for a in anchor_aspect_ratios)
        if len(aspect_ratios) == 1 and len(anchor_sizes) > 1:
            # Matches the 1D config's convention: a single aspect-ratio
            # tuple in the YAML (one row) applied to every FPN level.
            aspect_ratios = aspect_ratios * len(anchor_sizes)

    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = MultiScaleRoIAlign(
        featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=512,
        max_size=600,
        rpn_fg_iou_thresh=0.4,
        rpn_bg_iou_thresh=0.25,
        rpn_batch_size_per_image=512,
        box_score_thresh=0.10,
        box_nms_thresh=0.17,
        box_detections_per_img=200,
        box_fg_iou_thresh=0.5,
        box_bg_iou_thresh=0.4,
        box_batch_size_per_image=1024,
        box_positive_fraction=0.5,
        box_reg_weights=(10.0, 10.0, 5.0, 5.0),
    )

    out_channels = model.backbone.out_channels
    resolution = roi_pooler.output_size[0]
    representation_size = 1024

    box_head = torchvision.models.detection.faster_rcnn.TwoMLPHead(
        out_channels * resolution ** 2, representation_size
    )
    box_predictor = ProbabilisticThetaPredictor(representation_size, num_classes)

    model.roi_heads = ProbabilisticThetaRoIHeads(
        box_roi_pool=roi_pooler,
        box_head=box_head,
        box_predictor=box_predictor,
        fg_iou_thresh=0.5,
        bg_iou_thresh=0.4,
        batch_size_per_image=1024,
        positive_fraction=0.5,
        bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
        score_thresh=0.10,
        nms_thresh=0.17,
        detections_per_img=200,
        pos_cls_weight=pos_cls_weight,
        neg_cls_weight=neg_cls_weight,
        lambda_theta_mean=lambda_theta_mean,
        lambda_theta_cov=lambda_theta_cov,
        lambda_theta_kl=lambda_theta_kl,
    )

    return model
