import numpy as np
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from softlabel_od.adapters.dlcv.model import get_model


def compute_iou_simple(box1, box2):
    # box: [x1, y1, x2, y2]
    x1_min, y1_min, x1_max, y1_max = box1[:4]
    x2_min, y2_min, x2_max, y2_max = box2[:4]

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


class HardLabelLightningModule(pl.LightningModule):
    """
    LightningModule for hard-label Faster R-CNN training.

    Logs:
    - train/loss + train/<loss_key>
    - val/loss + val/class_loss + val/reg_loss + val/rpn_objectness + val/rpn_box_reg
    - val/avg_iou and val/class_accuracy (quick sanity metrics)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model(
            num_classes=cfg.MODEL.NUM_CLASSES,
            model_type="hard",
            backbone_name=cfg.MODEL.BACKBONE,
            pretrained=True,
            probabilistic=False,
        )

    def forward(self, images, targets=None):
        if targets is None:
            return self.model(images)
        return self.model(images, targets)

    @staticmethod
    def _unwrap_loss_dict(out):
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out[1]
        if isinstance(out, dict):
            return out
        raise TypeError(f"Unexpected loss output type: {type(out)}")

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        total_loss = sum(loss_dict.values())

        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True)
        for k, v in loss_dict.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=True, on_epoch=True)

        return total_loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = [img for img in images]
        targets = [{k: v for k, v in t.items()} for t in targets]

        assert not torch.is_grad_enabled()

        # Compute losses (torchvision returns losses only in train mode)
        was_training = self.model.training
        self.model.train()
        loss_out = self(images, targets)
        loss_dict = self._unwrap_loss_dict(loss_out)

        # Log raw components
        for k, v in loss_dict.items():
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False)

        val_total_loss = sum(loss_dict.values()) if len(loss_dict) else torch.tensor(0.0, device=images[0].device)

        # Map to your summary fields
        val_class_loss = loss_dict.get("loss_classifier", torch.tensor(0.0, device=images[0].device))
        val_reg_loss = loss_dict.get("loss_box_reg", torch.tensor(0.0, device=images[0].device))
        val_rpn_obj = loss_dict.get("loss_objectness", torch.tensor(0.0, device=images[0].device))
        val_rpn_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0.0, device=images[0].device))

        self.log("val/loss", val_total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/class_loss", val_class_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/reg_loss", val_reg_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/rpn_objectness", val_rpn_obj, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/rpn_box_reg", val_rpn_reg, prog_bar=False, on_step=False, on_epoch=True)

        # Restore + inference for quick metrics
        self.model.train(was_training)
        self.model.eval()
        preds = self(images)

        # Quick IoU and classification accuracy
        ious = []
        correct = 0
        total = 0
        for pred, target in zip(preds, targets):
            if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                pred_boxes = pred["boxes"].detach().cpu().numpy()
                pred_scores = pred["scores"].detach().cpu().numpy()
                gt_boxes = target["boxes"].detach().cpu().numpy()
                
                matches, _, _ = match_predictions_to_gt(
                    pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
                )
                
                for match in matches:
                    ious.append(match[2])

        avg_iou = float(np.mean(ious)) if len(ious) else 0.0

        # match-by-best-iou accuracy (quick, same spirit as training_hard_label.py)
        if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
            pred_boxes = pred["boxes"].detach().cpu().numpy()
            pred_labels = pred["labels"].detach().cpu().numpy()
            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy()

            for p_box, p_label in zip(pred_boxes, pred_labels):
                best = 0.0
                best_idx = -1
                for j, g_box in enumerate(gt_boxes):
                    v = compute_iou_simple(p_box, g_box)
                    if v > best:
                        best = v
                        best_idx = j
                if best_idx >= 0 and best > 0.5:
                    total += 1
                    if int(p_label) == int(gt_labels[best_idx]):
                        correct += 1       
        self.log("val/avg_iou", float(np.mean(ious)) if len(ious) else 0.0, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val/class_accuracy", float(correct / total) if total > 0 else 0.0, prog_bar=False, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        bbox_params = []
        class_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "box_predictor.bbox_pred" in name:
                bbox_params.append(param)
            elif "box_predictor.cls_score" in name:
                class_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.SGD(
            [
                {"params": bbox_params, "lr": self.cfg.TRAIN.BASE_LR_BBOX},
                {"params": class_params, "lr": self.cfg.TRAIN.BASE_LR_CLASS},
                {"params": other_params, "lr": self.cfg.TRAIN.BASE_LR},
            ],
            momentum=0.9,
            weight_decay=0.0005,
        )

        scheduler = MultiStepLR(
            optimizer,
            milestones=list(self.cfg.TRAIN.MILESTONES),
            gamma=self.cfg.TRAIN.GAMMA,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }