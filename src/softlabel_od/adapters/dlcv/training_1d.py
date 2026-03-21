"""
training_1d.py - Lightweight training monitoring
"""

import torch
import os
import json
import numpy as np
from softlabel_od.adapters.dlcv.training import train_one_epoch
import softlabel_od.adapters.dlcv.utils as utils
from tqdm import tqdm
from softlabel_od.adapters.dlcv.config import get_cfg_defaults, get_cfg_from_file
cfg = get_cfg_defaults()
from softlabel_od.adapters.dlcv.evaluation_metrics import *
from softlabel_od.adapters.dlcv.utils import convert_np

@torch.no_grad()
def validate_soft_labels_1d_quick(model, data_loader, device, print_debug=False):
    """
    Quick validation for 1D GMM probabilistic model.
    Uses distribution-aware metrics.
    """
    model.eval()
    
    # Loss tracking (still compute losses for monitoring)
    total_loss = 0.0
    total_class_loss = 0.0
    total_var_loss = 0.0
    total_rpn_cls_loss = 0.0 
    total_rpn_box_loss = 0.0 
    total_mean_loss = 0.0
    num_batches = 0
    
    # Collect predictions and GTs for metric computation
    all_predictions = []
    all_ground_truths = []
    
    pbar = tqdm(data_loader, desc='🔍 1D GMM Validation', ncols=100)
    
    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # === LOSS COMPUTATION (training mode) ===
        model.train()
        try:
            loss_outputs = model(images, targets)
            if isinstance(loss_outputs, tuple):
                _, loss_dict = loss_outputs
            else:
                loss_dict = loss_outputs
            
            if isinstance(loss_dict, dict) and len(loss_dict) > 0:
                batch_loss = 0.0
                for loss_name, loss_val in loss_dict.items():
                    if isinstance(loss_val, torch.Tensor):
                        if not torch.isnan(loss_val) and not torch.isinf(loss_val):
                            batch_loss += loss_val.item()
                            
                            # Track individual losses
                            if 'classifier' in loss_name:
                                total_class_loss += loss_val.item()
                            if 'box_var' in loss_name:
                                total_var_loss += loss_val.item()
                            if 'box_mean' in loss_name:
                                total_mean_loss += loss_val.item()
                            if 'loss_objectness' in loss_name:
                                total_rpn_cls_loss += loss_val.item()
                            if 'loss_rpn_box_reg' in loss_name:
                                total_rpn_box_loss += loss_val.item()
                
                total_loss += batch_loss
                
        except Exception as e:
            # Only print occasionally to avoid spam
            if num_batches % 100 == 0:
                print(f"Loss computation warning: {e}")
        
        # === INFERENCE (eval mode) ===
        model.eval()
        predictions = model(images)
        debug_printed = 0
        # Collect for metric computation
        for idx, (pred, target, img) in enumerate(zip(predictions, targets, images)):
            img_h, img_w = img.shape[1], img.shape[2]
            
            pred_entry = {
                'boxes': pred['boxes'].cpu().numpy(),
                'labels': pred['labels'].cpu().numpy(),
                'scores': pred['scores'].cpu().numpy(),
            }
            if 'bbox_stds' in pred:
                pred_entry['bbox_stds'] = pred['bbox_stds'].cpu().numpy()
                pred_entry['bbox_means'] = pred['boxes'].cpu().numpy() 
            if 'class_probs' in pred:
                pred_entry['class_probs'] = pred['class_probs'].cpu().numpy()
            
            
            gt_entry = {
                'boxes': target['boxes'].cpu().numpy(),
                'labels': target['labels'].cpu().numpy(),
                'image_size': (img_h, img_w)
            }

            if 'gt_class_probs' in target:
                gt_entry['gt_class_probs'] = target['gt_class_probs'].cpu().numpy()
            # Add GT bbox stds if available (you need to pass these through your dataset)
            if 'gt_stds' in target:
                gt_entry['gt_stds'] = target['gt_stds'].cpu().numpy()
            # Add GT soft labels if available
            if 'gt_class_probs' in target:
                gt_entry['gt_class_probs'] = target['gt_class_probs'].cpu().numpy()
            
            all_predictions.append(pred_entry)
            all_ground_truths.append(gt_entry)
        
        num_batches += 1
        pbar.set_postfix({
            'loss': f'{total_loss/(num_batches):.4f}',
        })
    
    model.eval()
    
        # === COMPUTE PROBABILISTIC METRICS ===
    avg_loss = total_loss / num_batches if num_batches > 0 else float('nan')
    avg_class_loss = total_class_loss / num_batches if num_batches > 0 else float('nan')
    avg_var_loss = total_var_loss / num_batches if num_batches > 0 else float('nan')
    avg_mean_loss = total_mean_loss / num_batches if num_batches > 0 else float('nan')
    # --- ADD THESE ---
    avg_rpn_cls = total_rpn_cls_loss / num_batches if num_batches > 0 else float('nan')
    avg_rpn_box = total_rpn_box_loss / num_batches if num_batches > 0 else float('nan')
    
    # Standard metrics
    ece = compute_calibration_error(all_predictions, all_ground_truths)
    uw_map = compute_uncertainty_weighted_map(all_predictions, all_ground_truths)
    
    # Soft label metrics (NEW)
    soft_metrics = compute_soft_detection_metrics(all_predictions, all_ground_truths)
    dist_metrics = compute_class_distribution_metrics(all_predictions, all_ground_truths)
    
    # IoU calculation (keep existing)
    total_ious = []
    for pred, gt in zip(all_predictions, all_ground_truths):
        if len(pred['boxes']) == 0 or len(gt['boxes']) == 0:
            continue
        matches, _, _ = match_predictions_to_gt(
            pred['boxes'], pred['scores'], gt['boxes'], iou_threshold=0.5
        )
        for match in matches:
            total_ious.append(match[2])
    
    avg_iou = np.mean(total_ious) if total_ious else 0.0
    # Bbox uncertainty metrics (NEW)
    expected_iou = compute_expected_iou(all_predictions, all_ground_truths, num_samples=50)
    bbox_calibration_error = compute_bbox_uncertainty_calibration(all_predictions, all_ground_truths)
    
    print(f"\nValidation Results:")
    print(f"  Losses: Total={avg_loss:.4f}")
    print(f"  Box Variance: {avg_var_loss:.4f}")
    print(f"  Box Mean (SmoothL1 anchor): {avg_mean_loss:.4f}")
    print(f"  Class Distribution (SoftCE): {avg_class_loss:.4f}")
    print(f"  RPN: Objectness={avg_rpn_cls:.4f}, BoxReg={avg_rpn_box:.4f}")
    print(f"  Stats: Top1 Acc={soft_metrics['top1_accuracy']:.2%}, JS Div={dist_metrics['js_divergence']:.3f}")

    if print_debug:
        print("\n" + "="*80)
        print("🔍 FINAL INFERENCE RESULTS (Overfit Run - Matched Boxes Only)")
        print("="*80)
        
        if len(all_predictions) > 0:
            for img_idx, (pred, gt) in enumerate(zip(all_predictions, all_ground_truths)):
                print(f"\n{'─'*60}")
                print(f"📸 Image ID: {img_idx}")
                print(f"{'─'*60}")
                
                # Get image dimensions for converting GT stds to pixel space
                img_h, img_w = gt.get('image_size', (1, 1))
                
                # Get matched predictions using the same matching function
                if len(pred['boxes']) > 0 and len(gt['boxes']) > 0:
                    matches, unmatched_preds, unmatched_gts = match_predictions_to_gt(
                        pred['boxes'], pred['scores'], gt['boxes'], iou_threshold=0.5
                    )
                else:
                    matches = []
                    unmatched_gts = list(range(len(gt['boxes'])))
                
                print(f"\n  Matched: {len(matches)} | Unmatched GT: {len(unmatched_gts)} | Total Pred: {len(pred['boxes'])}")
                
                # Print each matched pair
                for match_idx, (pred_idx, gt_idx, iou) in enumerate(matches):
                    print(f"\n  ── Match {match_idx} (IoU: {iou:.3f}) ──")
                    
                    # GT Box
                    g_box = gt['boxes'][gt_idx]
                    print(f"    GT Box [{gt_idx}]:   [{g_box[0]:.1f}, {g_box[1]:.1f}, {g_box[2]:.1f}, {g_box[3]:.1f}]")
                    
                    # Pred Box
                    p_box = pred['boxes'][pred_idx]
                    print(f"    Pred Box [{pred_idx}]: [{p_box[0]:.1f}, {p_box[1]:.1f}, {p_box[2]:.1f}, {p_box[3]:.1f}]")
                    
                    # GT STD (converted to pixel space)
                    if 'gt_stds' in gt and gt_idx < len(gt['gt_stds']):
                        g_std_norm = gt['gt_stds'][gt_idx]
                        g_box = gt['boxes'][gt_idx]
                        gt_w = max(g_box[2] - g_box[0], 1.0)
                        gt_h = max(g_box[3] - g_box[1], 1.0)
                        
                        # Pixel std
                        g_std_px = [g_std_norm[i] * (img_w if i % 2 == 0 else img_h) for i in range(4)]
                        
                        # Center std (propagation)
                        sigma_cx = (g_std_px[0]**2 + 0.25 * g_std_px[2]**2)**0.5
                        sigma_cy = (g_std_px[1]**2 + 0.25 * g_std_px[3]**2)**0.5
                        
                        g_std_delta = [sigma_cx/gt_w, sigma_cy/gt_h, g_std_px[2]/gt_w, g_std_px[3]/gt_h]
                        print(f"    GT Std (delta):   [{g_std_delta[0]:.4f}, {g_std_delta[1]:.4f}, {g_std_delta[2]:.4f}, {g_std_delta[3]:.4f}]")
                    else:
                        print(f"    GT Std (px):  N/A")
                    
                    # Pred STD
                    if 'bbox_stds' in pred and pred_idx < len(pred['bbox_stds']):
                        p_std = pred['bbox_stds'][pred_idx]
                        print(f"    Pred Std (delta):     [{p_std[0]:.4f}, {p_std[1]:.4f}, {p_std[2]:.4f}, {p_std[3]:.4f}]")
                    else:
                        print(f"    Pred Std:     N/A")
                    
                    # GT Class Prob Distribution
                    if 'gt_class_probs' in gt and gt_idx < len(gt['gt_class_probs']):
                        g_dist = gt['gt_class_probs'][gt_idx]
                        g_str = "[" + ", ".join([f"{x:.3f}" for x in g_dist]) + "]"
                        print(f"    GT Class Dist:   {g_str}")
                    else:
                        print(f"    GT Class:        Label {gt['labels'][gt_idx]}")
                    
                    # Pred Class Prob Distribution
                    if 'class_probs' in pred and pred_idx < len(pred['class_probs']):
                        p_dist = pred['class_probs'][pred_idx]
                        p_str = "[" + ", ".join([f"{x:.3f}" for x in p_dist]) + "]"
                        print(f"    Pred Class Dist: {p_str}")
                    else:
                        print(f"    Pred Class:      Label {pred['labels'][pred_idx]}, Score {pred['scores'][pred_idx]:.3f}")
                
                # Show unmatched GT boxes (missed detections)
                if len(unmatched_gts) > 0:
                    print(f"\n  ── Unmatched GT Boxes (Missed) ──")
                    for gt_idx in unmatched_gts:
                        g_box = gt['boxes'][gt_idx]
                        print(f"    GT Box [{gt_idx}]: [{g_box[0]:.1f}, {g_box[1]:.1f}, {g_box[2]:.1f}, {g_box[3]:.1f}]")
        
        print("\n" + "="*80)
    
    return {
        'avg_loss': avg_loss,
        'avg_class_loss': avg_class_loss,
        'avg_var_loss': avg_var_loss,
        'avg_mean_loss': avg_mean_loss,
        'avg_rpn_cls_loss': avg_rpn_cls,
        'avg_rpn_box_loss': avg_rpn_box,
        'avg_iou': avg_iou,
        'expected_iou': expected_iou,
        'top1_accuracy': soft_metrics['top1_accuracy'],
        'top3_accuracy': soft_metrics['top3_accuracy'],
        'js_divergence': dist_metrics['js_divergence'],
    }


def compute_iou_simple(box1, box2):
    """Quick IoU computation for sanity check"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_early_stopping_score(val_metrics):
    """
    Composite score combining the metrics that matter most.
    Higher is better.
    """
    # Normalize each metric to a [0,1]-ish scale based on expected ranges
    # Weight them by research priority
    
    iou_score = val_metrics['avg_iou']                    # already 0-1, higher better
    js_score = 1.0 - val_metrics['js_divergence']        # invert: lower JS = better
    top1_score = val_metrics['top1_accuracy']              # already 0-1
    var_score = 1.0 - min(val_metrics.get('avg_var_loss', 0.1), 1.0)  # invert loss

    # Weights: classification matters most for mAP, then localization, then uncertainty
    score = (
        0.40 * top1_score   +   # classification quality → directly drives mAP
        0.35 * iou_score    +   # localization quality
        0.15 * js_score     +   # distribution matching → your novelty metric
        0.10 * var_score        # uncertainty calibration
    )
    return score

def train_and_evaluate_model_1d(model, train_loader, val_loader, optimizer, num_epochs, device, 
                                  early_stopping=False, scheduler=None, config_name="default_run"):

    best_composite_score = -float('inf')
    best_val_loss = float('inf')
    consecutive_no_improvement = 0

    # Per-metric stagnation counters (independent of composite)
    stagnation_counters = {
        'avg_iou': 0,
        'js_divergence': 0,
        'top1_accuracy': 0,
    }
    stagnation_best = {
        'avg_iou': -float('inf'),
        'js_divergence': float('inf'),   # lower is better
        'top1_accuracy': -float('inf'),
    }
    MIN_EPOCHS_BEFORE_STOPPING = 15  # don't stop before this regardless
    STAGNATION_PATIENCE = 12   # epochs without improvement on any single metric
    COMPOSITE_PATIENCE  = 7   # epochs without composite improvement → stop
    ema_metrics = {}
    EMA_ALPHA = 0.3

    # Minimum improvement thresholds (ignore noise)
    MIN_DELTA = {
        'composite':     0.001,
        'avg_iou':       0.003,
        'js_divergence': 0.003,
        'top1_accuracy': 0.005,
    }

    
    metrics_history = {
        # Core losses ()
        "train_loss": [],
        "val_loss": [],
        "val_box_var_loss": [],       
        "val_box_mean_loss": [],
        "val_class_loss": [], 
        "val_rpn_cls_loss": [],
        "val_rpn_box_loss": [],

        # Position & Shape ()
        "val_avg_iou": [],           
        "val_expected_iou": [],      
        
        # Classification ()
        "val_js_divergence": [],     
        "val_top1_accuracy": [],     
        "val_top3_accuracy": [],   
        
        # Learning rate
        "learning_rate": [],
    }

    results_dir = cfg.MISC.RESULTS_CSV + f"/{config_name}"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, "training_metrics_phase_1.json")

    print(f" Starting 1D GMM Training")
    print(f" Results: {results_dir}")
    print(f" Epochs: {num_epochs}")
    print(f" Mode: Quick monitoring (Phase 1)")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"📅 Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        
        if scheduler is not None:
            scheduler.step()

        is_overfit_run = cfg.TRAIN.OVERFIT_BATCHES > 0
        is_final_epoch = (epoch == num_epochs - 1)
        print_inference_debug = is_overfit_run and is_final_epoch
        # Quick validation
        val_metrics = validate_soft_labels_1d_quick(model, val_loader, device, print_debug=print_inference_debug)
        for key in ['avg_iou', 'js_divergence', 'top1_accuracy']:
            raw = val_metrics[key]
            if key not in ema_metrics:
                ema_metrics[key] = raw
            else:
                ema_metrics[key] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * ema_metrics[key]
        
        # Log metrics
        train_loss = train_logger.meters['loss'].global_avg
        current_lr = optimizer.param_groups[0]['lr']
        
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_metrics['avg_loss'])

        metrics_history["val_box_var_loss"].append(val_metrics.get('avg_var_loss')) 
        metrics_history["val_class_loss"].append(val_metrics['avg_class_loss'])
        metrics_history["val_box_mean_loss"].append(val_metrics['avg_mean_loss'])
        
        metrics_history["val_avg_iou"].append(val_metrics['avg_iou'])
        metrics_history["val_expected_iou"].append(val_metrics['expected_iou'])

        metrics_history["val_rpn_cls_loss"].append(val_metrics.get('avg_rpn_cls_loss', 0))
        metrics_history["val_rpn_box_loss"].append(val_metrics.get('avg_rpn_box_loss', 0))
        
        metrics_history["val_js_divergence"].append(val_metrics['js_divergence'])
        metrics_history["val_top1_accuracy"].append(val_metrics['top1_accuracy'])
        metrics_history["val_top3_accuracy"].append(val_metrics['top3_accuracy'])
        
        metrics_history["learning_rate"].append(current_lr)


       # Summary
        print(f"\n Epoch Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['avg_loss']:.4f}")
        print(f"  Box Quality: IoU={val_metrics['avg_iou']:.3f}, Expected_IoU={val_metrics['expected_iou']:.3f} (Goal: High)")
        print(f"  Class Quality: Top1={val_metrics['top1_accuracy']:.1%}, Top3={val_metrics['top3_accuracy']:.1%}, JS_Div={val_metrics['js_divergence']:.3f} (Goal: Low)")
        print(f"  RPN Loss: Obj={val_metrics.get('avg_rpn_cls_loss',0):.4f}, Box={val_metrics.get('avg_rpn_box_loss',0):.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # ── Composite score ───────────────────────────────────────────────
        composite = compute_early_stopping_score(val_metrics)

        # ── Save best model (on composite, not val_loss) ──────────────────
        if composite > best_composite_score + MIN_DELTA['composite']:
            best_composite_score = composite
            consecutive_no_improvement = 0
            best_model_path = os.path.join(results_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best composite={composite:.4f} — saved checkpoint")
        else:
            consecutive_no_improvement += 1

        # ── Per-metric stagnation tracking ───────────────────────────────
        metric_map = {
            'avg_iou':       (val_metrics['avg_iou'],      '>'),
            'js_divergence': (val_metrics['js_divergence'], '<'),   # lower better
            'top1_accuracy': (val_metrics['top1_accuracy'], '>'),
        }
        
        any_stagnated = False
        for metric_name, (value, direction) in metric_map.items():
            best = stagnation_best[metric_name]
            if direction == '>':
                improved = value > best + MIN_DELTA[metric_name]
            else:
                improved = value < best - MIN_DELTA[metric_name]

            if improved:
                stagnation_best[metric_name] = value
                stagnation_counters[metric_name] = 0
            else:
                stagnation_counters[metric_name] += 1
                if stagnation_counters[metric_name] >= STAGNATION_PATIENCE:
                    any_stagnated = True
                    print(f"  Stagnation: {metric_name} no improvement for "
                        f"{STAGNATION_PATIENCE} epochs")

        # ── Stop conditions ───────────────────────────────────────────────
        if epoch + 1 >= MIN_EPOCHS_BEFORE_STOPPING:
            if consecutive_no_improvement >= COMPOSITE_PATIENCE:
                print(f"  Early stopping: composite score stagnated for "
                    f"{COMPOSITE_PATIENCE} epochs")
                break

            if any_stagnated:
                print(f"  Early stopping: per-metric stagnation detected")
                break

            if val_metrics['avg_loss'] > best_val_loss * 1.15:
                print(f"  Early stopping: val_loss diverged >15% above best")
                break

        if val_metrics['avg_loss'] < best_val_loss:
            best_val_loss = val_metrics['avg_loss']

        # Save metrics
        with open(metrics_file_path, 'w') as f:
            json.dump(convert_np(metrics_history), f, indent=2)

        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f" Training Complete!")
    print(f" Best validation loss: {best_val_loss:.4f}")
    print(f" Metrics saved to: {metrics_file_path}")
    print(f"{'='*60}")
    
    return metrics_history