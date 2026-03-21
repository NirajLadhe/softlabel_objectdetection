"""
training_2d.py - Training utilities specific to 2D GMM vertex models
"""

import torch
import os
import json
from softlabel_od.adapters.dlcv.training import train_one_epoch
import softlabel_od.adapters.dlcv.utils as utils


@torch.no_grad()
def validate_soft_labels_2d(model, data_loader, device):
    """Validation specifically for 2D GMM vertex soft label models"""
    model.eval()
    total_loss = 0.0
    total_class_loss = 0.0
    total_vertex_loss = 0.0
    num_batches = 0
    
    print("Starting 2D GMM validation...")
    
    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Get predictions and losses
        model.train()  # Need training mode for loss computation
        outputs = model(images, targets)
        
        # Handle model output format
        if isinstance(outputs, dict):
            loss_dict = outputs
        elif isinstance(outputs, tuple) and len(outputs) == 2:
            _, loss_dict = outputs
        else:
            loss_dict = outputs
        
        if isinstance(loss_dict, dict):
            batch_loss = sum(loss for loss in loss_dict.values()).item()
            total_loss += batch_loss
            
            if 'loss_probabilistic_classifier' in loss_dict:
                total_class_loss += loss_dict['loss_probabilistic_classifier'].item()
            if 'loss_probabilistic_vertex_reg' in loss_dict:  # 2D-specific
                total_vertex_loss += loss_dict['loss_probabilistic_vertex_reg'].item()
        
        num_batches += 1
        torch.cuda.empty_cache()
    
    model.eval()
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_class_loss = total_class_loss / num_batches if num_batches > 0 else 0.0
    avg_vertex_loss = total_vertex_loss / num_batches if num_batches > 0 else 0.0
    
    print(f"2D GMM Validation - Total Loss: {avg_loss:.4f}, Class Loss: {avg_class_loss:.4f}, Vertex Reg Loss: {avg_vertex_loss:.4f}")
    
    return avg_loss, avg_class_loss, avg_vertex_loss


def train_and_evaluate_model_2d(model, train_loader, val_loader, optimizer, num_epochs, device, 
                                  early_stopping=False, scheduler=None, config_name="default_run"):
    """Training loop optimized for 2D GMM vertex soft label models"""
    best_val_loss = float('inf')
    consecutive_no_improvement = 0
    metrics_history = {
        "train_loss": [], 
        "val_loss": [], 
        "val_class_loss": [], 
        "val_vertex_reg_loss": []  # 2D-specific name
    }

    results_dir = f"/kaggle/working/results/{config_name}"
    os.makedirs(results_dir, exist_ok=True)
    metrics_file_path = os.path.join(results_dir, "metrics_history.json")

    print(f"Starting 2D GMM vertex training for {num_epochs} epochs...")
    print(f"Results will be saved to: {results_dir}")

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*50}")
        
        # Training
        train_logger = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=50)
        
        if scheduler is not None:
            scheduler.step()

        # Validation with soft labels
        val_loss, val_class_loss, val_vertex_loss = validate_soft_labels_2d(model, val_loader, device)
        
        # Log metrics
        train_loss = train_logger.meters['loss'].global_avg
        metrics_history["train_loss"].append(train_loss)
        metrics_history["val_loss"].append(val_loss)
        metrics_history["val_class_loss"].append(val_class_loss)
        metrics_history["val_vertex_reg_loss"].append(val_vertex_loss)

        print(f"\nEpoch [{epoch + 1}/{num_epochs}] Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f} (Class: {val_class_loss:.4f}, Vertex Reg: {val_vertex_loss:.4f})")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                consecutive_no_improvement = 0
                best_model_path = os.path.join(results_dir, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print(f"  New best model saved! (Val Loss: {val_loss:.4f})")
            else:
                consecutive_no_improvement += 1
                print(f"  No improvement for {consecutive_no_improvement} epoch(s)")

            if consecutive_no_improvement >= 5:
                print(f"\n Early stopping triggered!")
                break

        # Save metrics
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)

        torch.cuda.empty_cache()

    print(f"\n Training finished!")
    print(f" Best validation loss: {best_val_loss:.4f}")
    print(f" Results saved to: {results_dir}")
    
    return metrics_history