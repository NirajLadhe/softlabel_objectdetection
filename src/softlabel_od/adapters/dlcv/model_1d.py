import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops
import math

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from softlabel_od.adapters.dlcv.utils import *

class ProbabilisticFastRCNNPredictor(nn.Module):
    """
    Custom prediction head for Faster R-CNN to predict Gaussian parameters.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, num_classes):
        super(ProbabilisticFastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        # The regression head now outputs 8 values per class: 4 for mean, 4 for log-variance
        self.bbox_pred_gmm = nn.Linear(in_channels, num_classes * 8)
        # Initialize GMM params
        nn.init.normal_(self.bbox_pred_gmm.weight, std=0.01)
        nn.init.constant_(self.bbox_pred_gmm.bias, 0)
        # Manually set the bias for the log-variance channels (indices 4-7) to a negative value
        # (-5.0 => exp(-5) approx 0.006 std)
        # Reshape bias to [num_classes, 8] to index easily
        with torch.no_grad():
            bias = self.bbox_pred_gmm.bias.view(num_classes, 8)
            bias[:, 4:] = -4.0 
            self.bbox_pred_gmm.bias.copy_(bias.view(-1))

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        
        scores = self.cls_score(x)
        gmm_params = self.bbox_pred_gmm(x)

        return scores, gmm_params
    
class ProbabilisticRoIHeads(RoIHeads):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.box_ops = torchvision.ops.boxes

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Overrides the forward pass of the RoIHeads to use our custom loss.
        """
        # --- Standard RoI Heads Logic ---
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, gmm_params = self.box_predictor(box_features)

        result = []
        losses = {}
        # --- Custom Loss Calculation in Training ---
        if self.training:
            # -----------------------------
            # 1) Targets for positives
            # -----------------------------
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

            pos_labels = all_labels[positive_proposals_indices]  # [N_pos]
            pos_class_logits = class_logits[positive_proposals_indices]  # [N_pos, C]
            pos_gmm_params = gmm_params[positive_proposals_indices]      # [N_pos, C*8]

            # torchvision regression targets (already in [dx,dy,dw,dh] box-coder space)
            reg_targets = torch.cat(regression_targets, dim=0)          # [N_total, 4]
            pos_reg_targets = reg_targets[positive_proposals_indices]    # [N_pos, 4]

            # -----------------------------
            # 2) Predicted means/vars (in delta space!)
            # -----------------------------
            pos_gmm_params = pos_gmm_params.reshape(num_pos, num_classes, 8)
            pred_means_delta = pos_gmm_params[torch.arange(num_pos, device=pos_gmm_params.device), pos_labels, :4]
            raw_log_vars = pos_gmm_params[torch.arange(num_pos), pos_labels, 4:]
            pred_log_vars_delta = torch.clamp(raw_log_vars, min=-10.0, max=10.0)

            # -----------------------------
            # 4) Build GT std in DELTA space from GT std in PIXELS
            # -----------------------------
            # dataset stores target stds normalized in [x,y,w,h] ∈ [0,1]
            pos_proposals_list = []
            gt_boxes_list = []
            gt_stds_delta_list = []

            for i in range(len(targets)):
                pos_mask = labels[i] > 0
                if pos_mask.sum() == 0:
                    continue

                props_i = proposals[i][pos_mask]  # [N_pos_i,4] in resized pixels
                gt_idx_i = matched_idxs[i][pos_mask]
                gt_boxes_i = targets[i]["boxes"][gt_idx_i]  # [N_pos_i,4] in resized pixels

                gt_stds_norm_i = targets[i]["gt_stds"][gt_idx_i]  # [N_pos_i,4] normalized stds (x,y,w,h)
                h_i, w_i = image_shapes[i]  # (H, W) of resized image
                scale_i = gt_stds_norm_i.new_tensor([w_i, h_i, w_i, h_i])
                gt_stds_px_i = gt_stds_norm_i * scale_i

                # Pixel noise floor
                STD_PX_FLOOR = 1.0
                gt_stds_px_i = gt_stds_px_i.clamp(min=STD_PX_FLOOR)

                # Proposal sizes
                pw = (props_i[:, 2] - props_i[:, 0]).clamp(min=1.0)
                ph = (props_i[:, 3] - props_i[:, 1]).clamp(min=1.0)

                # GT sizes
                gt_w = (gt_boxes_i[:, 2] - gt_boxes_i[:, 0]).clamp(min=1.0)
                gt_h = (gt_boxes_i[:, 3] - gt_boxes_i[:, 1]).clamp(min=1.0)

                # Interpret gt_stds_px_i as std for [x_tl, y_tl, w, h] in pixels.
                # Convert to std for center coords (approx, assuming independence):
                sigma_x = gt_stds_px_i[:, 0]
                sigma_y = gt_stds_px_i[:, 1]
                sigma_w = gt_stds_px_i[:, 2]
                sigma_h = gt_stds_px_i[:, 3]
                sigma_cx = torch.sqrt(sigma_x.pow(2) + 0.25 * sigma_w.pow(2))
                sigma_cy = torch.sqrt(sigma_y.pow(2) + 0.25 * sigma_h.pow(2))

                # Convert to std in box-coder delta space:
                # dx,dy are normalized by proposal width/height; dw,dh are approx relative width/height
                sigma_dx = sigma_cx / pw
                sigma_dy = sigma_cy / ph
                sigma_dw = sigma_w / gt_w
                sigma_dh = sigma_h / gt_h

                gt_stds_delta_i = torch.stack([sigma_dx, sigma_dy, sigma_dw, sigma_dh], dim=1)

                # Delta-space floor (small, but not ultra-tiny)
                gt_stds_delta_i = gt_stds_delta_i.clamp(min=1e-3, max=10.0)

                pos_proposals_list.append(props_i)
                gt_boxes_list.append(gt_boxes_i)
                gt_stds_delta_list.append(gt_stds_delta_i)

            gt_stds_delta = torch.cat(gt_stds_delta_list, dim=0)  # [N_pos,4]
            # Scale GT stds into the same space the model predicts in (BoxCoder weights)
            REG_WEIGHTS = gt_stds_delta.new_tensor([10.0, 10.0, 5.0, 5.0])
            gt_stds_delta_scaled = gt_stds_delta * REG_WEIGHTS          # [N_pos, 4]

            # -----------------------------
            # 5a) Smooth L1 loss on means (direct localization anchor)
            # -----------------------------
            LAMBDA_MEAN = 2.0
            loss_box_mean = torch.nn.functional.smooth_l1_loss(
                pred_means_delta,
                pos_reg_targets,
                beta=1.0,
                reduction="sum",
            ) / all_labels.numel()
            loss_box_mean = loss_box_mean * LAMBDA_MEAN

            # -----------------------------
            # 5b) Smooth L1 loss on variances
            # -----------------------------
            # We enforce a small floor for stability
            GT_STD_DELTA_FLOOR = 1e-3
            gt_stds_delta_safe = gt_stds_delta_scaled.clamp(min=GT_STD_DELTA_FLOOR, max=50.0)

            pred_log_stds = 0.5 * pred_log_vars_delta                       # [N_pos, 4]
            gt_log_stds = torch.log(gt_stds_delta_safe)                      # [N_pos, 4]
            
            LAMBDA_VAR = 1.0
            loss_box_var = torch.nn.functional.smooth_l1_loss(
                pred_log_stds,
                gt_log_stds,
                beta=0.5,
                reduction="mean",
            ) * LAMBDA_VAR
            # -----------------------------
            # 6) Classification loss = soft cross entropy
            # -----------------------------
            gt_class_probs_list = []
            for i in range(len(targets)):
                pos_mask = labels[i] > 0
                if pos_mask.sum() == 0:
                    continue
                gt_idx_i = matched_idxs[i][pos_mask]
                gt_class_probs_list.append(targets[i]["gt_class_probs"][gt_idx_i])

            gt_class_probs = torch.cat(gt_class_probs_list, dim=0)  # [N_pos, C]

            pos_ann_weights_list = []
            for i in range(len(targets)):
                pos_mask = labels[i] > 0
                if pos_mask.sum() == 0:
                    continue
                gt_idx_i = matched_idxs[i][pos_mask]
                pos_ann_weights_list.append(targets[i]['annotation_weight'][gt_idx_i])

            pos_ann_weights = torch.cat(pos_ann_weights_list, dim=0)  # [N_pos]

            LAMBDA_CLS = 1.0
            # loss_classifier = soft_cross_entropy_loss(pos_class_logits, gt_class_probs, class_weights=self.class_weights)
            per_sample_ce = soft_cross_entropy_loss(
                pos_class_logits, gt_class_probs,
                class_weights=self.class_weights,
                reduction='none'        # [N_pos]
            )
            loss_classifier = (per_sample_ce * pos_ann_weights).mean() * LAMBDA_CLS

            # -----------------------------
            # 7) Final losses
            # -----------------------------

            losses = {
                "loss_probabilistic_classifier": loss_classifier,
                "loss_box_var": loss_box_var, 
                "loss_box_mean": loss_box_mean,
            }
        else:
            # Get class scores
            pred_scores = torch.nn.functional.softmax(class_logits, -1)
            
            # Reshape GMM params: [N, num_classes * 8] -> [N, num_classes, 8]
            num_boxes = gmm_params.shape[0]
            num_classes = class_logits.shape[1]
            
            # Handle empty proposals case
            if num_boxes == 0:
                num_images = len(image_shapes)
                for _ in range(num_images):
                    result.append({
                        "boxes": torch.empty((0, 4), device=gmm_params.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=gmm_params.device),
                        "scores": torch.empty((0,), device=gmm_params.device),
                        "bbox_stds": torch.empty((0, 4), device=gmm_params.device),
                        "class_probs": torch.empty((0, num_classes), device=gmm_params.device),
                    })
                return result, losses
            
            gmm_params = gmm_params.view(num_boxes, num_classes, 8)
            
            # Extract means (box deltas) and log-variances
            pred_means = gmm_params[:, :, :4]      # [N, num_classes, 4]
            pred_log_vars = gmm_params[:, :, 4:]   # [N, num_classes, 4]
            pred_stds_scaled = torch.exp(0.5 * pred_log_vars)                  # scaled space
            REG_WEIGHTS = pred_stds_scaled.new_tensor([10.0, 10.0, 5.0, 5.0])
            pred_stds = pred_stds_scaled / REG_WEIGHTS.view(1, 1, 4)
            
            # Process per image
            boxes_per_image = [len(p) for p in proposals]
            
            # Split tensors by image
            pred_means_list = pred_means.split(boxes_per_image, dim=0)
            pred_scores_list = pred_scores.split(boxes_per_image, dim=0)
            pred_stds_list = pred_stds.split(boxes_per_image, dim=0)
            
            for img_idx, (means_i, scores_i, stds_i, props_i, img_shape) in enumerate(
                zip(pred_means_list, pred_scores_list, pred_stds_list, proposals, image_shapes)
            ):
                
                if len(props_i) == 0:
                    result.append({
                        "boxes": torch.empty((0, 4), device=gmm_params.device),
                        "labels": torch.empty((0,), dtype=torch.int64, device=gmm_params.device),
                        "scores": torch.empty((0,), device=gmm_params.device),
                        "bbox_stds": torch.empty((0, 4), device=gmm_params.device),
                        "class_probs": torch.empty((0, num_classes), device=gmm_params.device),
                    })
                    continue
                
                # Convert proposals to [cx, cy, w, h]
                prop_widths = props_i[:, 2] - props_i[:, 0]
                prop_heights = props_i[:, 3] - props_i[:, 1]
                prop_ctr_x = props_i[:, 0] + 0.5 * prop_widths
                prop_ctr_y = props_i[:, 1] + 0.5 * prop_heights
                
                # Expand to [num_proposals, 1] for broadcasting with [num_proposals, num_classes, 4]
                prop_widths = prop_widths.unsqueeze(1)
                prop_heights = prop_heights.unsqueeze(1)
                prop_ctr_x = prop_ctr_x.unsqueeze(1)
                prop_ctr_y = prop_ctr_y.unsqueeze(1)
                
                # Scale deltas by weights
                WX, WY, WW, WH = 10.0, 10.0, 5.0, 5.0

                dx = means_i[:, :, 0] / WX
                dy = means_i[:, :, 1] / WY
                dw = means_i[:, :, 2] / WW
                dh = means_i[:, :, 3] / WH

                # Apply deltas (standard box parameterization)
                pred_ctr_x = dx * prop_widths + prop_ctr_x
                pred_ctr_y = dy * prop_heights + prop_ctr_y
                pred_w = torch.exp(dw) * prop_widths
                pred_h = torch.exp(dh) * prop_heights
                
                # Convert back to [x1, y1, x2, y2]
                pred_x1 = pred_ctr_x - 0.5 * pred_w
                pred_y1 = pred_ctr_y - 0.5 * pred_h
                pred_x2 = pred_ctr_x + 0.5 * pred_w
                pred_y2 = pred_ctr_y + 0.5 * pred_h
                
                # Stack to [num_proposals, num_classes, 4]
                pred_boxes_i = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=2)
                
                # Clip boxes to image bounds
                pred_boxes_i = self.box_ops.clip_boxes_to_image(pred_boxes_i.reshape(-1, 4), img_shape)
                pred_boxes_i = pred_boxes_i.view(-1, num_classes, 4)
                
                # Remove background class predictions (class 0)
                boxes_i = pred_boxes_i[:, 1:]      # [num_proposals, num_classes-1, 4]
                scores_i_full = scores_i.clone()   # [num_proposals, num_classes] - KEEP FULL PROBABILITIES
                scores_i = scores_i[:, 1:]         # [num_proposals, num_classes-1]
                stds_i = stds_i[:, 1:]             # [num_proposals, num_classes-1, 4]
                
                # Flatten for NMS processing
                boxes_flat = boxes_i.reshape(-1, 4)
                scores_flat = scores_i.reshape(-1)
                stds_flat = stds_i.reshape(-1, 4)

                num_proposals = scores_i_full.shape[0]
                num_fg_classes = num_classes - 1  # excluding background
                probs_flat = scores_i_full.unsqueeze(1).expand(-1, num_fg_classes, -1).reshape(-1, num_classes)
                
                # Create labels (1-indexed, since we removed background)
                labels_flat = torch.arange(1, num_classes, device=scores_i.device)
                labels_flat = labels_flat.view(1, -1).expand(boxes_i.shape[0], -1).reshape(-1)
                
                
                # Stage 1: gate on foreground confidence — one test per proposal
                fg_confidence = 1.0 - scores_i_full[:, 0]       # [num_proposals]
                fg_mask = fg_confidence > self.score_thresh      # score_thresh=0.5

                if fg_mask.sum() == 0:
                    result.append({
                        "boxes":       torch.empty((0, 4), device=gmm_params.device),
                        "labels":      torch.empty((0,), dtype=torch.int64, device=gmm_params.device),
                        "scores":      torch.empty((0,), device=gmm_params.device),
                        "bbox_stds":   torch.empty((0, 4), device=gmm_params.device),
                        "class_probs": torch.empty((0, num_classes), device=gmm_params.device),
                    })
                    continue

                # Filter all per-proposal tensors
                boxes_fg    = boxes_i[fg_mask]           # [M, C-1, 4]
                stds_fg     = stds_i[fg_mask]            # [M, C-1, 4]
                scores_fg   = scores_i[fg_mask]          # [M, C-1]
                fg_conf_fg  = fg_confidence[fg_mask]     # [M]
                full_probs_fg = scores_i_full[fg_mask]   # [M, num_classes]

                # Stage 2: pick MAP class label per surviving proposal
                map_class_idx = scores_fg.argmax(dim=-1)         # [M], 0-indexed over fg classes
                labels_out    = map_class_idx + 1                # shift to 1-indexed

                M   = boxes_fg.shape[0]
                idx = map_class_idx.unsqueeze(-1).unsqueeze(-1)  # [M, 1, 1]
                boxes_out = boxes_fg.gather(1, idx.expand(-1, -1, 4)).squeeze(1)   # [M, 4]
                stds_out  = stds_fg.gather(1, idx.expand(-1, -1, 4)).squeeze(1)    # [M, 4]

                # Stage 3: NMS on foreground confidence
                keep = torchvision.ops.nms(boxes_out, fg_conf_fg, self.nms_thresh)
                keep = keep[:self.detections_per_img]

                result.append({
                    "boxes":       boxes_out[keep],
                    "labels":      labels_out[keep],
                    "scores":      fg_conf_fg[keep],
                    "bbox_stds":   stds_out[keep],
                    "class_probs": full_probs_fg[keep],
                })
            
        return result, losses
    
class ProbabilisticFasterRCNN(FasterRCNN):
    def __init__(self, backbone, num_classes, **kwargs):
        super().__init__(backbone, num_classes=num_classes, **kwargs)

        # Get the number of input features for the RoI head
        in_channels = self.roi_heads.box_predictor.cls_score.in_features
        
        # Create an instance of our custom probabilistic head
        probabilistic_predictor = ProbabilisticFastRCNNPredictor(in_channels, num_classes)
        
        # Create an instance of our custom RoI Heads, passing in the new predictor
        probabilistic_roi_heads = ProbabilisticRoIHeads(
            box_roi_pool=self.roi_heads.box_roi_pool,
            box_head=self.roi_heads.box_head,
            box_predictor=probabilistic_predictor,
            fg_iou_thresh=self.roi_heads.proposal_matcher.high_threshold,
            bg_iou_thresh=self.roi_heads.proposal_matcher.low_threshold,
            batch_size_per_image=self.roi_heads.fg_bg_sampler.batch_size_per_image,
            positive_fraction=self.roi_heads.fg_bg_sampler.positive_fraction,
            bbox_reg_weights=None,
            score_thresh=self.roi_heads.score_thresh,
            nms_thresh=self.roi_heads.nms_thresh,
            detections_per_img=self.roi_heads.detections_per_img
        )
        
        # Replace the standard RoI heads with our custom one
        self.roi_heads = probabilistic_roi_heads

class HardLabelRoIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Custom RoIHeads for Hard Labels that uses Focal Loss.
        """
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        
        if self.training:
            labels = torch.cat(labels, dim=0)
            regression_targets = torch.cat(regression_targets, dim=0)
            
            # Build one-hot targets for soft_focal_loss
            gt_class_probs = torch.nn.functional.one_hot(labels.clamp_min(0), num_classes=class_logits.shape[1]).float()
            
            # Classification Loss (Focal Loss) - Gamma=2.0 for hard mining
            # loss_classifier = soft_focal_loss(class_logits, gt_class_probs, alpha=0.25, gamma=2.0)
            LAMBDA_CLS = 10.0
            loss_classifier = torch.nn.functional.cross_entropy(class_logits, labels) * LAMBDA_CLS
            
            # Regression Loss (Standard Smooth L1)
            # We must manually compute this to ensure we match the specific class regressor
            # Logic adapted from standard RoIHeads
            sampled_pos_inds_subset = torch.where(labels > 0)[0]
            labels_pos = labels[sampled_pos_inds_subset]
            N = class_logits.shape[0]
            box_regression = box_regression.reshape(N, -1, 4)
            
            LAMBDA_BOX_REG = 5.0
            loss_box_reg = torch.nn.functional.smooth_l1_loss(
                box_regression[sampled_pos_inds_subset, labels_pos],
                regression_targets[sampled_pos_inds_subset],
                beta=1.0,
                reduction="sum"
            )
            loss_box_reg = loss_box_reg / labels.numel()
            loss_box_reg = loss_box_reg * LAMBDA_BOX_REG

            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg,
            }
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                    }
                )

        return result, losses

        
def get_model_1d(num_classes, backbone_name='resnet50', pretrained=True, probabilistic=True):
    if backbone_name in ('resnet50', 'resnet101'):
        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights='ResNet50_Weights.IMAGENET1K_V1' if backbone_name == 'resnet50' else 'ResNet101_Weights.IMAGENET1K_V1',
            trainable_layers=3
        )
        # FPN always outputs 256 channels regardless of backbone depth
        # out_channels is set automatically, no need to set manually

    elif backbone_name == 'mobilenet':
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
        backbone.out_channels = 960
        
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")
    
    # anchor_sizes = ((32, 64, 96, 192, 384),)
    # aspect_ratios = ((0.5,  0.75, 1.0, 2.0, 3.0),)*len(anchor_sizes)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.33, 0.5, 0.75, 1.0, 2.0, 3.0),) * 5
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    # Create the model using the specified backbone
    model = FasterRCNN(
        backbone, 
        num_classes=num_classes, 
        rpn_anchor_generator=anchor_generator, 
        # weights=weights,
        rpn_fg_iou_thresh=0.5,  # Lowered from 0.7
        rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256,
        rpn_pre_nms_top_n_train=2000, # Keep more proposals
        rpn_post_nms_top_n_train=2000,
        box_detections_per_img=100,
        min_size=800, # Explicitly set resize
        max_size=1333
    )

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    if probabilistic:
        print("Using Probabilistic GMM Head for the model.")
        # Replace the pre-trained head with our new probabilistic one
        model.roi_heads.box_predictor = ProbabilisticFastRCNNPredictor(in_features, num_classes)

        model.roi_heads = ProbabilisticRoIHeads(
            box_roi_pool=model.roi_heads.box_roi_pool,
            box_head=model.roi_heads.box_head,
            box_predictor=model.roi_heads.box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=[10, 10, 5, 5], # [dx, dy, dw, dh] scaled at inference time
            score_thresh=0.6, nms_thresh=0.4, detections_per_img=200    # nms_thresh=0.3-0.5
        )

    else:
        print("Using Standard FastRCNNPredictor Head.")
        # The standard head
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        model.roi_heads = HardLabelRoIHeads(
            box_roi_pool=model.roi_heads.box_roi_pool,
            box_head=model.roi_heads.box_head,
            box_predictor=model.roi_heads.box_predictor,
            fg_iou_thresh=0.5, bg_iou_thresh=0.5,
            batch_size_per_image=512, positive_fraction=0.25,
            bbox_reg_weights=None,
            score_thresh=0.4, nms_thresh=0.5, detections_per_img=100
        )

    return model