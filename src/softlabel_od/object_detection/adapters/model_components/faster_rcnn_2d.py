"""
model_2d.py - 2D GMM Vertex-based Probabilistic Object Detection

This module implements a custom Faster R-CNN that predicts bounding boxes
as 4 vertices, each with a 2D Gaussian distribution (mean + covariance).

Key differences from 1D GMM:
- Predicts 4 vertices instead of [x, y, w, h]
- Each vertex has a 2x2 covariance matrix (captures x-y correlation)
- More expressive for irregular annotation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads

from softlabel_od.object_detection.utils.losses import soft_cross_entropy_loss, vertex_nll_loss_2d


class VertexFastRCNNPredictor(nn.Module):
    """
    Custom prediction head for Faster R-CNN to predict vertex-based Gaussian parameters.
    
    For each detection, predicts:
    - 4 vertices (8 coordinates: x1,y1,x2,y2,x3,y3,x4,y4)
    - 4 covariance matrices (2x2 each, 3 unique values per matrix)
    
    Total per class: 8 (means) + 12 (covariance params) = 20 values
    
    Args:
        in_channels (int): number of input channels from RoI head
        num_classes (int): number of output classes (including background)
    """
    def __init__(self, in_channels, num_classes):
        super(VertexFastRCNNPredictor, self).__init__()
        self.num_classes = num_classes
        
        # Classification head
        self.cls_score = nn.Linear(in_channels, num_classes)
        
        # Vertex prediction head
        # Outputs 20 values per class:
        # - 8 for vertex means [x1,y1,x2,y2,x3,y3,x4,y4]
        # - 12 for covariances (3 params per 2x2 matrix × 4 vertices)
        self.vertex_pred_gmm = nn.Linear(in_channels, num_classes * 20)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        
        scores = self.cls_score(x)
        vertex_params = self.vertex_pred_gmm(x)
        
        return scores, vertex_params


class VertexRoIHeads(RoIHeads):
    """
    Custom RoI Heads for vertex-based 2D GMM predictions.
    """
    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Overrides the forward pass to use vertex-based 2D GMM loss.
        """
        # Standard RoI Heads Logic
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(
                proposals, targets
            )
        
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, vertex_params = self.box_predictor(box_features)

        result = []
        losses = {}
        
        # Training mode: compute custom losses
        if self.training:
            # Filter for positive proposals
            positive_proposals_mask = [
                lbl is not None and len(lbl) > 0 for lbl in labels
            ]
            positive_proposals_indices = torch.where(
                torch.cat([torch.tensor(m).repeat(len(labels[i])) 
                          for i, m in enumerate(positive_proposals_mask)])
            )[0]
            
            # Get positive samples
            pos_class_logits = class_logits[positive_proposals_indices]
            pos_vertex_params = vertex_params[positive_proposals_indices]
            
            # Gather ground truth data
            gt_class_probs_list = []
            gt_vertices_list = []
            gt_covariances_list = []
            
            for i in range(len(targets)):
                matched_gt_idxs_for_image = matched_idxs[i][labels[i] > 0]
                
                gt_class_probs_list.append(
                    targets[i]['gt_class_probs'][matched_gt_idxs_for_image]
                )
                gt_vertices_list.append(
                    targets[i]['gt_vertices'][matched_gt_idxs_for_image]
                )
                # Covariances are stored as list of lists
                gt_covariances_list.extend([
                    targets[i]['gt_covariances'][idx] 
                    for idx in matched_gt_idxs_for_image
                ])
            
            gt_class_probs = torch.cat(gt_class_probs_list, dim=0)
            gt_vertices = torch.cat(gt_vertices_list, dim=0)
            
            # Convert covariances to tensor [N, 4, 2, 2]
            gt_covs = self._covariances_to_tensor(gt_covariances_list, gt_vertices.device)
            
            # Calculate losses
            loss_classifier = soft_cross_entropy_loss(pos_class_logits, gt_class_probs)
            
            # Vertex localization loss
            num_pos_proposals = pos_vertex_params.shape[0]
            pos_vertex_params = pos_vertex_params.reshape(num_pos_proposals, -1, 20)
            
            # Get params for the predicted class
            pos_labels = torch.cat(labels, dim=0)[positive_proposals_indices]
            
            # Extract vertex means (first 8 values)
            pred_vertices = pos_vertex_params[
                torch.arange(num_pos_proposals), pos_labels, :8
            ]
            
            # Extract covariance parameters (remaining 12 values)
            pred_cov_params = pos_vertex_params[
                torch.arange(num_pos_proposals), pos_labels, 8:
            ]
            
            # Reconstruct covariance matrices from parameters
            pred_covs = self._params_to_covariances(pred_cov_params)
            
            # Compute vertex-based NLL loss
            loss_vertex_reg = vertex_nll_loss_2d(
                pred_vertices, pred_covs, gt_vertices, gt_covs
            )
            
            losses = {
                "loss_probabilistic_classifier": loss_classifier,
                "loss_probabilistic_vertex_reg": loss_vertex_reg,
            }
            
        else:
            # Inference mode
            pred_scores = F.softmax(class_logits, -1)
            num_boxes = vertex_params.shape[0]
            vertex_params = vertex_params.view(num_boxes, -1, 20)
            
            # Extract predictions for each proposal
            boxes_all = []
            scores_all = []
            labels_all = []
            vertices_all = []
            covariances_all = []
            
            for i in range(num_boxes):
                # Find the class with highest score (excluding background)
                class_id = torch.argmax(pred_scores[i, 1:]) + 1
                score = pred_scores[i, class_id]
                
                # Get vertex predictions
                vertex_means = vertex_params[i, class_id, :8]
                cov_params = vertex_params[i, class_id, 8:]
                
                # Convert vertices to box format [x1, y1, x2, y2]
                box = self._vertices_to_box(vertex_means)
                
                boxes_all.append(box)
                scores_all.append(score)
                labels_all.append(class_id)
                vertices_all.append(vertex_means)
                covariances_all.append(cov_params)
            
            # Create dummy box regression for compatibility
            boxes = torch.stack(boxes_all, dim=0) if boxes_all else torch.empty((0, 4))
            box_regression = torch.zeros_like(boxes)
            
            # Use standard postprocessing
            result = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes
            )
            
            # Store vertex predictions in results for later analysis
            # (You'll need to extract these separately for probabilistic evaluation)
            
        return result, losses
    
    def _vertices_to_box(self, vertices):
        """
        Convert 4 vertices to axis-aligned bounding box.
        
        Args:
            vertices: [8] tensor [x1,y1,x2,y2,x3,y3,x4,y4]
        
        Returns:
            box: [4] tensor [x_min, y_min, x_max, y_max]
        """
        xs = vertices[0::2]  # Extract x coordinates
        ys = vertices[1::2]  # Extract y coordinates
        
        x_min = torch.min(xs)
        x_max = torch.max(xs)
        y_min = torch.min(ys)
        y_max = torch.max(ys)
        
        return torch.tensor([x_min, y_min, x_max, y_max], device=vertices.device)
    
    def _params_to_covariances(self, cov_params):
        """
        Convert 12 parameters to 4 covariance matrices (2x2 each).
        
        Uses Cholesky decomposition for positive-definite guarantees:
        Σ = L L^T where L is lower triangular
        
        Args:
            cov_params: [N, 12] parameters
        
        Returns:
            covs: [N, 4, 2, 2] covariance matrices
        """
        N = cov_params.shape[0]
        covs = []
        
        for v_idx in range(4):
            # Extract 3 parameters for this vertex's covariance
            params = cov_params[:, v_idx*3:(v_idx+1)*3]
            
            # Build lower triangular matrix L
            # L = [[exp(a),    0   ],
            #      [  b,    exp(c)]]
            # This ensures positive diagonal elements
            a = params[:, 0]
            b = params[:, 1]
            c = params[:, 2]
            
            L_00 = torch.exp(a) + 1e-6  # Ensure positive
            L_10 = b
            L_11 = torch.exp(c) + 1e-6
            
            # Σ = L L^T
            cov_00 = L_00 * L_00
            cov_01 = L_00 * L_10
            cov_10 = L_10 * L_00
            cov_11 = L_10 * L_10 + L_11 * L_11
            
            # Stack into 2x2 matrix
            cov = torch.stack([
                torch.stack([cov_00, cov_01], dim=1),
                torch.stack([cov_10, cov_11], dim=1)
            ], dim=1)
            
            covs.append(cov)
        
        # Stack all 4 covariance matrices
        return torch.stack(covs, dim=1)  # [N, 4, 2, 2]
    
    def _covariances_to_tensor(self, cov_list, device):
        """
        Convert list of covariance matrices to tensor.
        
        Args:
            cov_list: List of N elements, each containing 4 covariance matrices (2x2)
        
        Returns:
            tensor: [N, 4, 2, 2]
        """
        tensors = []
        for covs in cov_list:
            # covs is a list of 4 (2x2) matrices
            cov_tensor = torch.tensor(covs, dtype=torch.float32, device=device)
            tensors.append(cov_tensor)
        
        return torch.stack(tensors, dim=0)



def get_model_2d(num_classes, backbone_name='resnet50', pretrained=True):
    """
    Creates a Faster R-CNN model with 2D GMM vertex-based predictions.
    
    Args:
        num_classes: Number of classes (including background)
        backbone_name: 'resnet50', 'resnet101', or 'mobilenet'
        pretrained: Whether to use pretrained backbone weights
    
    Returns:
        model: VertexFasterRCNN model
    """
    # Define backbones
    if backbone_name == 'resnet50':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone = torchvision.models.resnet50(pretrained=pretrained)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        
    elif backbone_name == 'resnet101':
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone = torchvision.models.resnet101(pretrained=pretrained)
        backbone = torch.nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        
    elif backbone_name == 'mobilenet':
        weights = torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
        backbone.out_channels = 960
        
    else:
        raise ValueError(f"Backbone '{backbone_name}' is not supported.")
    
    # Anchor generator
    anchor_sizes = ((32, 64, 128, 256, 512),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    
    # Create base model
    model = FasterRCNN(
        backbone, 
        num_classes=num_classes, 
        rpn_anchor_generator=anchor_generator,
        weights=weights
    )
    
    # Get input features for the RoI head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace with vertex predictor
    print("Using 2D GMM Vertex-based Prediction Head")
    vertex_predictor = VertexFastRCNNPredictor(in_features, num_classes)
    
    # Create custom RoI heads
    vertex_roi_heads = VertexRoIHeads(
        box_roi_pool=model.roi_heads.box_roi_pool,
        box_head=model.roi_heads.box_head,
        box_predictor=vertex_predictor,
        fg_iou_thresh=model.roi_heads.proposal_matcher.high_threshold,
        bg_iou_thresh=model.roi_heads.proposal_matcher.low_threshold,
        batch_size_per_image=model.roi_heads.fg_bg_sampler.batch_size_per_image,
        positive_fraction=model.roi_heads.fg_bg_sampler.positive_fraction,
        bbox_reg_weights=None,
        score_thresh=model.roi_heads.score_thresh,
        nms_thresh=model.roi_heads.nms_thresh,
        detections_per_img=model.roi_heads.detections_per_img
    )
    
    # Replace RoI heads
    model.roi_heads = vertex_roi_heads
    
    return model