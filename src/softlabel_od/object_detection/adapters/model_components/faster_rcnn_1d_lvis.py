"""
LVIS-tuned probabilistic Faster R-CNN (1D GMM head).

Derived from faster_rcnn_1d.py (VinDr-CXR baseline) with architecture parameters
adjusted for COCO-LVIS natural images:

  - Smaller anchor scales (many LVIS instances are small at 800px resize)
  - Broader aspect ratios (objects are less elongated than chest X-ray findings)
  - Lower classification loss weight (long-tailed ~163-class taxonomy)
  - Stronger mean-box regression anchor (stabilizes early training)
  - Lower score threshold + higher detections_per_img at inference

Anchor configuration is data-driven from Assignment 2 soft labels:
  - bbox area p10=0.0003, p50=0.0042, p90=0.0677 (normalized)
  - aspect ratio p10=0.22, p50=0.75, p90=2.07
  At 800px resize: objects span ~14px (p10) to ~208px (p90).

Re-run anchor_analysis.py on outputs/lvis/*_normalized.csv to refine
LVIS_DEFAULT_ANCHOR_SIZES and LVIS_DEFAULT_ANCHOR_ASPECT_RATIOS before
large training runs.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from softlabel_od.object_detection.adapters.model_components.faster_rcnn_1d import (
    HardLabelRoIHeads,
    ProbabilisticFastRCNNPredictor,
    ProbabilisticRoIHeads,
)
from softlabel_od.object_detection.utils.losses import soft_cross_entropy_loss
from softlabel_od.object_detection.utils.training_helpers import create_anchor_generator


class LVISProbabilisticRoIHeads(ProbabilisticRoIHeads):
    """LVIS-specific probabilistic RoI head with a self-contained ``forward``.

    Decoupled from the base ``ProbabilisticRoIHeads`` (faster_rcnn_1d.py), whose
    inference path is now specialised for VinDr-CXR (it gates on a "No Finding"
    class at index 1 vs "pathology" classes at index >= 2 — meaningless for LVIS,
    where index 1 is the real object class ``aerosol_can``).

    Training loss mirrors the base file's
    ``fix: extend cls loss to all proposals (positives+negatives) to suppress FPs``:
    classification loss is computed over **every** sampled proposal, not just the
    positives. Negatives target hard background ``[1, 0, ...]`` so the model is
    penalised for predicting an object class on empty regions — the direct cause of
    LVIS's huge false-positive count. Inference uses generic, class-agnostic
    foreground-confidence gating (``fg = 1 - P(background)``).
    """

    def __init__(self, *args, lambda_cls: float = 1.0, lambda_mean: float = 2.0,
                 lambda_var: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_cls = lambda_cls
        self.lambda_mean = lambda_mean
        self.lambda_var = lambda_var

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, gmm_params = self.box_predictor(box_features)

        result: list = []
        losses: dict = {}

        if self.training:
            # ---- 1) Positive proposals ----
            all_labels = torch.cat(labels, dim=0)
            positive_proposals_indices = torch.where(all_labels > 0)[0]
            num_pos = positive_proposals_indices.shape[0]
            num_classes = class_logits.shape[1]

            if num_pos == 0:
                device = class_logits.device
                losses = {
                    "loss_probabilistic_classifier": torch.tensor(0.0, device=device, requires_grad=True),
                    "loss_box_var": torch.tensor(0.0, device=device, requires_grad=True),
                    "loss_box_mean": torch.tensor(0.0, device=device, requires_grad=True),
                }
                return result, losses

            pos_labels = all_labels[positive_proposals_indices]
            pos_gmm_params = gmm_params[positive_proposals_indices]

            reg_targets = torch.cat(regression_targets, dim=0)
            pos_reg_targets = reg_targets[positive_proposals_indices]

            # ---- 2) Predicted means/log-vars (delta space) ----
            pos_gmm_params = pos_gmm_params.reshape(num_pos, num_classes, 8)
            pred_means_delta = pos_gmm_params[torch.arange(num_pos, device=pos_gmm_params.device), pos_labels, :4]
            raw_log_vars = pos_gmm_params[torch.arange(num_pos), pos_labels, 4:]
            pred_log_vars_delta = torch.clamp(raw_log_vars, min=-10.0, max=10.0)

            # ---- 3) GT stds (pixels -> delta space) ----
            gt_stds_delta_list = []
            for i in range(len(targets)):
                pos_mask = labels[i] > 0
                if pos_mask.sum() == 0:
                    continue

                props_i = proposals[i][pos_mask]
                gt_idx_i = matched_idxs[i][pos_mask]
                gt_boxes_i = targets[i]["boxes"][gt_idx_i]

                gt_stds_norm_i = targets[i]["gt_stds"][gt_idx_i]
                h_i, w_i = image_shapes[i]
                scale_i = gt_stds_norm_i.new_tensor([w_i, h_i, w_i, h_i])
                gt_stds_px_i = (gt_stds_norm_i * scale_i).clamp(min=1.0)

                pw = (props_i[:, 2] - props_i[:, 0]).clamp(min=1.0)
                ph = (props_i[:, 3] - props_i[:, 1]).clamp(min=1.0)
                gt_w = (gt_boxes_i[:, 2] - gt_boxes_i[:, 0]).clamp(min=1.0)
                gt_h = (gt_boxes_i[:, 3] - gt_boxes_i[:, 1]).clamp(min=1.0)

                sigma_x, sigma_y, sigma_w, sigma_h = (
                    gt_stds_px_i[:, 0], gt_stds_px_i[:, 1], gt_stds_px_i[:, 2], gt_stds_px_i[:, 3]
                )
                sigma_cx = torch.sqrt(sigma_x.pow(2) + 0.25 * sigma_w.pow(2))
                sigma_cy = torch.sqrt(sigma_y.pow(2) + 0.25 * sigma_h.pow(2))

                gt_stds_delta_i = torch.stack(
                    [sigma_cx / pw, sigma_cy / ph, sigma_w / gt_w, sigma_h / gt_h], dim=1
                ).clamp(min=1e-3, max=10.0)
                gt_stds_delta_list.append(gt_stds_delta_i)

            gt_stds_delta = torch.cat(gt_stds_delta_list, dim=0)
            REG_WEIGHTS = gt_stds_delta.new_tensor([10.0, 10.0, 5.0, 5.0])
            gt_stds_delta_scaled = gt_stds_delta * REG_WEIGHTS

            # ---- 4) Smooth-L1 loss on means ----
            LAMBDA_MEAN = getattr(self, "lambda_mean", 2.0)
            loss_box_mean = torch.nn.functional.smooth_l1_loss(
                pred_means_delta, pos_reg_targets, beta=1.0, reduction="sum",
            ) / all_labels.numel() * LAMBDA_MEAN

            # ---- 5) Smooth-L1 loss on log-stds ----
            gt_stds_delta_safe = gt_stds_delta_scaled.clamp(min=1e-3, max=50.0)
            pred_log_stds = 0.5 * pred_log_vars_delta
            gt_log_stds = torch.log(gt_stds_delta_safe)
            LAMBDA_VAR = getattr(self, "lambda_var", 1.0)
            loss_box_var = torch.nn.functional.smooth_l1_loss(
                pred_log_stds, gt_log_stds, beta=0.5, reduction="mean",
            ) * LAMBDA_VAR

            # ---- 6) Classification loss over ALL proposals (positives + negatives) ----
            # fix: extend cls loss to all proposals (positives+negatives) to suppress FPs.
            # Computing CE only on positives gave the background classifier zero gradient,
            # so empty regions were freely labelled as objects (the source of LVIS's huge
            # FP count). Build a full [N_total, C] target: every row starts as hard
            # background [1, 0, ...]; positive rows are overwritten with their soft GMM
            # targets. Annotator weights stay 1.0 for negatives.
            num_total = all_labels.shape[0]
            cls_targets = class_logits.new_zeros((num_total, num_classes))
            cls_targets[:, 0] = 1.0
            ann_weights = class_logits.new_ones(num_total)

            if num_pos > 0:
                pos_gt_class_probs_list = []
                pos_ann_weights_list = []
                for i in range(len(targets)):
                    pos_mask = labels[i] > 0
                    if pos_mask.sum() == 0:
                        continue
                    gt_idx_i = matched_idxs[i][pos_mask]
                    pos_gt_class_probs_list.append(targets[i]["gt_class_probs"][gt_idx_i])
                    pos_ann_weights_list.append(targets[i]["annotation_weight"][gt_idx_i])

                pos_gt_class_probs = torch.cat(pos_gt_class_probs_list, dim=0)
                pos_ann_weights = torch.cat(pos_ann_weights_list, dim=0)
                cls_targets[positive_proposals_indices] = pos_gt_class_probs
                ann_weights[positive_proposals_indices] = pos_ann_weights

            LAMBDA_CLS = getattr(self, "lambda_cls", 1.0)
            per_sample_ce = soft_cross_entropy_loss(
                class_logits, cls_targets, class_weights=self.class_weights, reduction="none",
            )
            loss_classifier = (per_sample_ce * ann_weights).mean() * LAMBDA_CLS

            losses = {
                "loss_probabilistic_classifier": loss_classifier,
                "loss_box_var": loss_box_var,
                "loss_box_mean": loss_box_mean,
            }
            return result, losses

        # ----------------------- LVIS inference -----------------------
        pred_scores = torch.nn.functional.softmax(class_logits, -1)
        num_boxes = gmm_params.shape[0]
        num_classes = class_logits.shape[1]

        def _empty():
            return {
                "boxes": torch.empty((0, 4), device=gmm_params.device),
                "labels": torch.empty((0,), dtype=torch.int64, device=gmm_params.device),
                "scores": torch.empty((0,), device=gmm_params.device),
                "bbox_stds": torch.empty((0, 4), device=gmm_params.device),
                "class_probs": torch.empty((0, num_classes), device=gmm_params.device),
            }

        if num_boxes == 0:
            for _ in range(len(image_shapes)):
                result.append(_empty())
            return result, losses

        gmm_params = gmm_params.view(num_boxes, num_classes, 8)
        pred_means = gmm_params[:, :, :4]
        pred_log_vars = gmm_params[:, :, 4:]
        pred_stds_scaled = torch.exp(0.5 * pred_log_vars)
        REG_WEIGHTS = pred_stds_scaled.new_tensor([10.0, 10.0, 5.0, 5.0])
        pred_stds = pred_stds_scaled / REG_WEIGHTS.view(1, 1, 4)

        boxes_per_image = [len(p) for p in proposals]
        pred_means_list = pred_means.split(boxes_per_image, dim=0)
        pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
        pred_stds_list = pred_stds.split(boxes_per_image, dim=0)

        for means_i, scores_i, stds_i, props_i, img_shape in zip(
            pred_means_list, pred_scores_list, pred_stds_list, proposals, image_shapes
        ):
            if len(props_i) == 0:
                result.append(_empty())
                continue

            prop_widths = (props_i[:, 2] - props_i[:, 0]).unsqueeze(1)
            prop_heights = (props_i[:, 3] - props_i[:, 1]).unsqueeze(1)
            prop_ctr_x = (props_i[:, 0] + 0.5 * (props_i[:, 2] - props_i[:, 0])).unsqueeze(1)
            prop_ctr_y = (props_i[:, 1] + 0.5 * (props_i[:, 3] - props_i[:, 1])).unsqueeze(1)

            WX, WY, WW, WH = 10.0, 10.0, 5.0, 5.0
            dx = means_i[:, :, 0] / WX
            dy = means_i[:, :, 1] / WY
            dw = means_i[:, :, 2] / WW
            dh = means_i[:, :, 3] / WH

            pred_ctr_x = dx * prop_widths + prop_ctr_x
            pred_ctr_y = dy * prop_heights + prop_ctr_y
            pred_w = torch.exp(dw) * prop_widths
            pred_h = torch.exp(dh) * prop_heights

            pred_boxes_i = torch.stack(
                [pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                 pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h],
                dim=2,
            )
            pred_boxes_i = self.box_ops.clip_boxes_to_image(
                pred_boxes_i.reshape(-1, 4), img_shape
            ).view(-1, num_classes, 4)

            boxes_i = pred_boxes_i[:, 1:]
            scores_i_full = scores_i.clone()
            scores_i = scores_i[:, 1:]
            stds_i = stds_i[:, 1:]

            # Stage 1: generic, class-agnostic foreground gating.
            fg_confidence = 1.0 - scores_i_full[:, 0]
            fg_mask = fg_confidence > self.score_thresh
            if fg_mask.sum() == 0:
                result.append(_empty())
                continue

            boxes_fg = boxes_i[fg_mask]
            stds_fg = stds_i[fg_mask]
            scores_fg = scores_i[fg_mask]
            fg_conf_fg = fg_confidence[fg_mask]
            full_probs_fg = scores_i_full[fg_mask]

            # Stage 2: MAP class over foreground classes (1-indexed output).
            map_class_idx = scores_fg.argmax(dim=-1)
            labels_out = map_class_idx + 1
            idx = map_class_idx.unsqueeze(-1).unsqueeze(-1)
            boxes_out = boxes_fg.gather(1, idx.expand(-1, -1, 4)).squeeze(1)
            stds_out = stds_fg.gather(1, idx.expand(-1, -1, 4)).squeeze(1)

            # Stage 3: NMS on foreground confidence, then cap.
            keep = torchvision.ops.nms(boxes_out, fg_conf_fg, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            result.append({
                "boxes": boxes_out[keep],
                "labels": labels_out[keep],
                "scores": fg_conf_fg[keep],
                "bbox_stds": stds_out[keep],
                "class_probs": full_probs_fg[keep],
            })

        return result, losses


# ---------------------------------------------------------------------------
# Anchor defaults tuned for COCO-LVIS @ Faster R-CNN resize (min_size=800).
# One size per FPN level (P2–P6); ratios are shared across all levels.
# ---------------------------------------------------------------------------
LVIS_DEFAULT_ANCHOR_SIZES: List[List[int]] = [
    [16],
    [32],
    [64],
    [128],
    [256],
]
LVIS_DEFAULT_ANCHOR_ASPECT_RATIOS: List[float] = [
    0.33,
    0.5,
    0.67,
    1.0,
    1.33,
    1.5,
    2.0,
]

# Loss balance: soft CE vs 1D Gaussian localization.
# LVIS has a long-tailed 163-class taxonomy; lower lambda_cls reduces
# gradient dominance from frequent categories.
# Stronger lambda_mean helps anchor box convergence early in training.
LVIS_LAMBDA_CLS = 1
LVIS_LAMBDA_MEAN = 3.0
LVIS_LAMBDA_VAR = 0.75

# Log-variance bias: controls initial bbox uncertainty width.
# Raised -3.5 -> -2.0: v3 eval showed predicted std ~2.6px vs actual error ~13px
# (5x overconfident), giving PDQ spatial quality of only 0.028. Wider initial
# Gaussians let the variance head calibrate toward the true error scale.
LVIS_LOG_VAR_BIAS = -2.0

# Post-NMS inference: LVIS images are object-dense; LVIS evaluation
# protocols use low score thresholds.
LVIS_SCORE_THRESH = 0.05
LVIS_NMS_THRESH = 0.5
LVIS_DETECTIONS_PER_IMG = 300

# Weight decay stored on the model for optimizers that read model.lvis_weight_decay.
LVIS_WEIGHT_DECAY = 0.0001


def _expand_aspect_ratios(
    aspect_ratios: Sequence[float],
    num_levels: int,
) -> Tuple[Tuple[float, ...], ...]:
    ratios = tuple(float(r) for r in aspect_ratios)
    return tuple(ratios for _ in range(num_levels))


def get_model_lvis_1d(
    num_classes: int,
    backbone_name: str = "resnet50",
    pretrained: bool = True,
    probabilistic: bool = True,
    anchor_sizes: Optional[List[List[int]]] = None,
    anchor_aspect_ratios: Optional[Sequence[float]] = None,
    lambda_cls: float = LVIS_LAMBDA_CLS,
    lambda_mean: float = LVIS_LAMBDA_MEAN,
    lambda_var: float = LVIS_LAMBDA_VAR,
    log_var_bias: float = LVIS_LOG_VAR_BIAS,
    score_thresh: float = LVIS_SCORE_THRESH,
    nms_thresh: float = LVIS_NMS_THRESH,
    detections_per_img: int = LVIS_DETECTIONS_PER_IMG,
    weight_decay: float = LVIS_WEIGHT_DECAY,
):
    """
    Build a Faster R-CNN model with LVIS-specific anchors, losses, and inference settings.

    Args:
        num_classes: Foreground classes + background (164 for Assignment 2 LVIS subset).
        backbone_name: 'resnet50' or 'resnet101'.
        pretrained: Use ImageNet backbone weights.
        probabilistic: Use 1D GMM RoI head; if False, standard hard-label head.
        anchor_sizes: Override FPN anchor sizes (one list per FPN level).
        anchor_aspect_ratios: Shared aspect ratios replicated across FPN levels.
        lambda_cls / lambda_mean / lambda_var: RoI-head loss balance.
        log_var_bias: Initial bias for log-variance channels in the GMM head.
        score_thresh / nms_thresh / detections_per_img: Post-NMS inference knobs.
        weight_decay: Stored on the model for optimizers that read it.

    Returns:
        torch.nn.Module ready for training.
    """
    if backbone_name in ("resnet50", "resnet101"):
        weights = (
            "ResNet50_Weights.IMAGENET1K_V1"
            if backbone_name == "resnet50"
            else "ResNet101_Weights.IMAGENET1K_V1"
        )
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights=weights,
            trainable_layers=5,
        )
    elif backbone_name == "mobilenet":
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
        backbone.out_channels = 960
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")

    sizes = anchor_sizes or LVIS_DEFAULT_ANCHOR_SIZES
    ratios = anchor_aspect_ratios or LVIS_DEFAULT_ANCHOR_ASPECT_RATIOS
    # Config delivers ratios as list-of-lists [[0.5, 1.0, 2.0]] (torchvision format).
    # _expand_aspect_ratios expects a flat list [0.5, 1.0, 2.0]; flatten if needed.
    if ratios and isinstance(ratios[0], (list, tuple)):
        ratios = list(ratios[0])
    aspect_ratios_per_level = _expand_aspect_ratios(ratios, len(sizes))
    anchor_generator = create_anchor_generator(sizes, aspect_ratios_per_level)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        # fg_iou 0.5->0.4: treat more proposals as positive so the RPN learns to
        # propose more candidates. v3 recall ceiling was 27% even at 300 outputs,
        # so the proposal stage (not the post-NMS cap) is the recall limiter.
        rpn_fg_iou_thresh=0.4,
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_pre_nms_top_n_train=3000,   # more proposals for dense LVIS scenes
        rpn_post_nms_top_n_train=2500,
        rpn_pre_nms_top_n_test=3000,    # 2000->3000: more candidates reach the head
        rpn_post_nms_top_n_test=4000,   # 2000->4000: raise the recall ceiling
        box_detections_per_img=detections_per_img,
        # Scale jitter (paper Table 3c, +1.2 AP): torchvision samples one min_size
        # per image during training and uses the last value (800) at eval.
        min_size=(640, 672, 704, 736, 768, 800),
        max_size=1333,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    if probabilistic:
        print("Using LVIS Probabilistic RoI Head (1D GMM).")
        model.roi_heads.box_predictor = ProbabilisticFastRCNNPredictor(
            in_features,
            num_classes,
            log_var_bias=log_var_bias,
        )
        model.roi_heads = LVISProbabilisticRoIHeads(
            box_roi_pool=model.roi_heads.box_roi_pool,
            box_head=model.roi_heads.box_head,
            box_predictor=model.roi_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.5,
            bbox_reg_weights=[10, 10, 5, 5],
            score_thresh=score_thresh,
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
            lambda_cls=lambda_cls,
            lambda_mean=lambda_mean,
            lambda_var=lambda_var,
        )
    else:
        print("Using LVIS Standard FastRCNNPredictor Head.")
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features,
            num_classes,
        )
        model.roi_heads = HardLabelRoIHeads(
            box_roi_pool=model.roi_heads.box_roi_pool,
            box_head=model.roi_heads.box_head,
            box_predictor=model.roi_heads.box_predictor,
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.5,
            bbox_reg_weights=None,
            score_thresh=max(score_thresh, 0.1),
            nms_thresh=nms_thresh,
            detections_per_img=detections_per_img,
        )

    model.lvis_weight_decay = float(weight_decay)
    return model
