"""
training_hard_label.py - Training utilities for hard label (deterministic) models

This trains a standard Faster R-CNN on deterministic labels extracted from
soft label data (mean positions + argmax classes).

Tracks both Phase 1 (training monitoring) and Phase 2 (comprehensive evaluation)
metrics for fair comparison with probabilistic models.
"""

import torch
import os
import json
import numpy as np
from softlabel_od.object_detection.core.training.train_epoch import train_one_epoch
from softlabel_od.object_detection.utils.training_helpers import (
    convert_np,
    save_model,
    MetricLogger,
    SmoothedValue,
)
from tqdm import tqdm
from softlabel_od.object_detection.core.evaluation.metrics import match_predictions_to_gt

@torch.no_grad()
def validate_hard_labels(model, data_loader, device, comprehensive=False, epoch=0, num_epochs=1):
    """
    Validation for hard label (deterministic) models.
    
    Args:
        model: Faster R-CNN model
        data_loader: Validation data loader
        device: torch device
        comprehensive: If True, compute Phase 2 metrics (slower but complete)
    
    Returns:
        dict of validation metrics
    """
    model.eval()
    
    # Loss tracking
    total_loss = 0.0
    total_class_loss = 0.0
    total_reg_loss = 0.0
    total_rpn_obj = 0.0
    total_rpn_reg = 0.0
    num_batches = 0
    
    # Prediction storage for metrics
    all_predictions = []
    all_ground_truths = []
    
    # Quick metrics
    all_ious = []
    all_class_correct = 0
    all_class_total = 0
    
    print(f"Starting Hard Label validation (comprehensive={comprehensive})...")
    
    eval_type = "Comprehensive" if comprehensive else "Quick"
    pbar = tqdm(data_loader, 
                desc=f'🔍 {eval_type} Validation', 
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # ===== LOSS COMPUTATION =====
        model.train()
        loss_outputs = model(images, targets)
        
        if isinstance(loss_outputs, tuple):
            _, loss_dict = loss_outputs
        else:
            loss_dict = loss_outputs
        
        if isinstance(loss_dict, dict):
            batch_loss = sum(loss for loss in loss_dict.values()).item()
            total_loss += batch_loss
            
            # Standard Faster R-CNN losses
            if 'loss_classifier' in loss_dict:
                total_class_loss += loss_dict['loss_classifier'].item()
            if 'loss_box_reg' in loss_dict:
                total_reg_loss += loss_dict['loss_box_reg'].item()
            if 'loss_objectness' in loss_dict:
                total_rpn_obj += loss_dict['loss_objectness'].item()
            if 'loss_rpn_box_reg' in loss_dict:
                total_rpn_reg += loss_dict['loss_rpn_box_reg'].item()
        
        # ===== PREDICTION EXTRACTION =====
        model.eval()
        predictions = model(images)

        # Track batch metrics for progress bar
        batch_ious = []
        batch_correct = 0
        batch_total = 0
        
        for pred, target in zip(predictions, targets):
            # Quick IoU check
            if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()
                
                matches, _, _ = match_predictions_to_gt(
                    pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5
                )
                
                for match in matches:
                    all_ious.append(match[2])
                # batch_ious.append(matches[0][2]) 
            
            # Classification accuracy
            pred_labels = pred['labels'].cpu().numpy()
            gt_labels = target['labels'].cpu().numpy()
            
            # Match by IoU for accuracy computation
            if len(pred['boxes']) > 0 and len(target['boxes']) > 0:
                pred_boxes = pred['boxes'].cpu().numpy()
                gt_boxes = target['boxes'].cpu().numpy()
                
                for p_box, p_label in zip(pred_boxes, pred_labels):
                    # Find best matching GT
                    ious = [compute_iou_simple(p_box, g_box) for g_box in gt_boxes]
                    best_idx = np.argmax(ious)
                    if ious[best_idx] > 0.5:
                        is_correct = (p_label == gt_labels[best_idx])
                        all_class_correct += is_correct
                        batch_correct += is_correct
                        all_class_total += 1
                        batch_total += 1
            
            # Store for comprehensive evaluation
            if comprehensive:
                pred_entry = {
                    'boxes': pred['boxes'].cpu().numpy(),
                    'labels': pred['labels'].cpu().numpy(),
                    'scores': pred['scores'].cpu().numpy(),
                }
                
                gt_entry = {
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy(),
                }
                
                # Include soft GT if available (for comparison)
                if 'gt_class_probs' in target:
                    gt_entry['gt_class_probs'] = target['gt_class_probs'].cpu().numpy()
                
                all_predictions.append(pred_entry)
                all_ground_truths.append(gt_entry)
        
        # Update progress bar with current batch metrics
        current_avg_loss = total_loss / (num_batches + 1)
        current_avg_iou = np.mean(all_ious) if all_ious else 0.0
        current_accuracy = all_class_correct / all_class_total if all_class_total > 0 else 0.0
        
        pbar.set_postfix({
            'loss': f'{current_avg_loss:.4f}',
            'IoU': f'{current_avg_iou:.3f}',
            'acc': f'{current_accuracy:.2%}'
        })
        
        num_batches += 1
        torch.cuda.empty_cache()
    
    model.eval()
    
    # ===== COMPUTE METRICS =====
    metrics = {
        'avg_loss': total_loss / num_batches if num_batches > 0 else 0.0,
        'avg_class_loss': total_class_loss / num_batches if num_batches > 0 else 0.0,
        'avg_reg_loss': total_reg_loss / num_batches if num_batches > 0 else 0.0,
        'avg_rpn_obj': total_rpn_obj / num_batches if num_batches > 0 else 0.0,
        'avg_rpn_reg': total_rpn_reg / num_batches if num_batches > 0 else 0.0,
        'avg_iou': np.mean(all_ious) if len(all_ious) > 0 else 0.0,
        'class_accuracy': all_class_correct / all_class_total if all_class_total > 0 else 0.0,
    }
    
    # ===== COMPREHENSIVE METRICS (PHASE 2) =====
    if comprehensive and len(all_predictions) > 0:
       from softlabel_od.object_detection.core.evaluation.metrics import (
            compute_map,
            compute_calibration_error,
        )
        
        print("  Computing comprehensive metrics...")
        
        # Standard detection metrics
        try:
            map_metrics = compute_map(all_predictions, all_ground_truths)
            metrics['mAP'] = map_metrics['mAP']
            metrics['mAP_50'] = map_metrics['mAP_50']
            metrics['mAP_75'] = map_metrics['mAP_75']
        except Exception as e:
            print(f"  Warning: Could not compute mAP: {e}")
            metrics['mAP'] = 0.0
            metrics['mAP_50'] = 0.0
            metrics['mAP_75'] = 0.0
        
        # Calibration (for confidence scores)
        try:
            metrics['calibration_error'] = compute_calibration_error(
                all_predictions, all_ground_truths
            )
        except Exception as e:
            print(f"  Warning: Could not compute calibration: {e}")
            metrics['calibration_error'] = 0.0
    
    # ===== PRINT SUMMARY =====
    print(f"\n Hard Label Validation:")
    print(f"  Losses: Total={metrics['avg_loss']:.4f}, "
          f"  Class={metrics['avg_class_loss']:.4f}, Reg={metrics['avg_reg_loss']:.4f}")
    print(f"  RPN: Obj={metrics['avg_rpn_obj']:.4f}, Reg={metrics['avg_rpn_reg']:.4f}")
    print(f"  Quick: IoU={metrics['avg_iou']:.3f}, Accuracy={metrics['class_accuracy']:.2%}")
    
    if comprehensive and 'mAP' in metrics:
        print(f"  Detection: mAP={metrics['mAP']:.3f}, "
              f"mAP@50={metrics['mAP_50']:.3f}, mAP@75={metrics['mAP_75']:.3f}")
        print(f"  Calibration: ECE={metrics['calibration_error']:.3f}")
    
    # === DEBUG: PRINT LAST BATCH DETAILS (FINAL EPOCH ONLY) ===
    if epoch == num_epochs - 1:
        print("\n" + "="*80)
        print("🔍 DETAILED INSPECTION (FINAL EPOCH - HARD LABELS)")
        print("="*80)
        
        if len(all_predictions) > 0:
            # Show last 4 samples
            num_show = min(4, len(all_predictions))
            # Use negative indexing to get the very last processed batch
            for i in range(1, num_show + 1):
                pred = all_predictions[-i]
                gt = all_ground_truths[-i]
                
                print(f"\n📸 Image #{i}")
                
                print("  [Prediction]")
                if len(pred['boxes']) > 0:
                    # Print top 3 boxes
                    for k in range(min(3, len(pred['boxes']))):
                        p_box = pred['boxes'][k]
                        p_score = pred['scores'][k]
                        p_label = pred['labels'][k]
                        print(f"    Box {k+1}: [{p_box[0]:.1f}, {p_box[1]:.1f}, {p_box[2]:.1f}, {p_box[3]:.1f}] | Score: {p_score:.4f} | Label: {p_label}")
                else:
                    print("    [NO BOXES PREDICTED]")
                    
                print("  [Ground Truth]")
                if len(gt['boxes']) > 0:
                    for k in range(len(gt['boxes'])):
                        g_box = gt['boxes'][k]
                        g_label = gt['labels'][k]
                        print(f"    Box {k+1}: [{g_box[0]:.1f}, {g_box[1]:.1f}, {g_box[2]:.1f}, {g_box[3]:.1f}] | Label: {g_label}")
                else:
                    print("    [NO FINDING]")
        
        print("="*80 + "\n")
        
    return metrics


def compute_iou_simple(box1, box2):
    """Quick IoU computation"""
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


def train_and_evaluate_model_hard(model, train_loader, val_loader, optimizer, num_epochs, device,
                                    early_stopping=False, scheduler=None, config_name="default_run",
                                    comprehensive_eval_freq=5):
    """
    Training loop for hard label (deterministic) models.
    
    Tracks Phase 1 metrics every epoch + Phase 2 metrics every N epochs.
    
    Args:
        comprehensive_eval_freq: Run comprehensive evaluation every N epochs
                                 (slower but provides full metrics)
    """
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    
    # Metrics tracking
    metrics_history = {
        # Core losses
        "train_loss": [],
        "val_loss": [],
        "val_class_loss": [],
        "val_box_reg_loss": [],
        "val_rpn_obj": [],
        "val_rpn_reg": [],
        
        # Quick metrics (every epoch)
        "val_avg_iou": [],
        "val_class_accuracy": [],
        "learning_rate": [],
        
        # Comprehensive metrics (every N epochs)
        "val_mAP": [],
        "val_mAP_50": [],
        "val_mAP_75": [],
        "val_calibration_error": [],
    }

    results_dir = f"{cwd}/results/{config_name}"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, "training_metrics.json")
    saved_model_dir = cwd + f"/saved_models/{config_name}"
    saved_model_path = os.path.join(saved_model_dir, "models")
    os.makedirs(saved_model_path, exist_ok=True)

    print(f"Starting Hard Label Training")
    print(f"Results: {results_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Mode: Deterministic (Standard Faster R-CNN)")
    print(f"Comprehensive eval: Every {comprehensive_eval_freq} epochs")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"📅 Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        
        if scheduler is not None:
            scheduler.step()

        # Validation (comprehensive every N epochs)
        run_comprehensive = (epoch + 1) % comprehensive_eval_freq == 0 or (epoch + 1) == num_epochs
        val_metrics = validate_hard_labels(model, val_loader, device, comprehensive=run_comprehensive, epoch=epoch, num_epochs=num_epochs)
        
        # Log metrics
        train_loss = train_logger.meters['loss'].global_avg
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_metrics['avg_loss'])
        metrics_history["val_class_loss"].append(val_metrics['avg_class_loss'])
        metrics_history["val_box_reg_loss"].append(val_metrics['avg_reg_loss'])
        metrics_history["val_rpn_obj"].append(val_metrics['avg_rpn_obj'])
        metrics_history["val_rpn_reg"].append(val_metrics['avg_rpn_reg'])
        metrics_history["val_avg_iou"].append(val_metrics['avg_iou'])
        metrics_history["val_class_accuracy"].append(val_metrics['class_accuracy'])
        metrics_history["learning_rate"].append(current_lr)
        
        # Add comprehensive metrics if available
        if run_comprehensive:
            metrics_history["val_mAP"].append(val_metrics.get('mAP', 0.0))
            metrics_history["val_mAP_50"].append(val_metrics.get('mAP_50', 0.0))
            metrics_history["val_mAP_75"].append(val_metrics.get('mAP_75', 0.0))
            metrics_history["val_calibration_error"].append(val_metrics.get('calibration_error', 0.0))

        # Summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['avg_loss']:.4f} "
              f"  (Class: {val_metrics['avg_class_loss']:.4f}, Reg: {val_metrics['avg_reg_loss']:.4f})")
        print(f"  Quick Metrics: IoU={val_metrics['avg_iou']:.4f}, "
              f"  Accuracy={val_metrics['class_accuracy']:.2%}")
        print(f"  LR: {current_lr:.6f}")
        
        if run_comprehensive and 'mAP' in val_metrics:
            print(f"Comprehensive: mAP={val_metrics['mAP']:.3f}, "
                  f"Calibration={val_metrics['calibration_error']:.3f}")
        
        # Early stopping
        if early_stopping:
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                consecutive_no_improvement = 0
                best_model_path = os.path.join(saved_model_path, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✅ New best! Saved to {best_model_path}")
            else:
                consecutive_no_improvement += 1
                print(f"  ⚠️  No improvement for {consecutive_no_improvement} epoch(s)")

            if consecutive_no_improvement >= 5:
                print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                break

        # Save metrics
        with open(metrics_file_path, 'w') as f:
            json.dump(convert_np(metrics_history), f, indent=2)

        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Metrics saved to: {metrics_file_path}")
    print(f"{'='*60}")
    
    return metrics_history