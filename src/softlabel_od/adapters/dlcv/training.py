import torch
from tqdm import tqdm
from softlabel_od.adapters.dlcv.utils import warmup_lr_scheduler

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Generic training function with tqdm progress bar.
    """
    model.train()
    
    # Setup epoch metrics
    epoch_loss = 0.0
    epoch_cls_loss = 0.0
    epoch_reg_loss = 0.0
    
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    
    # Create progress bar
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}', 
                ncols=100, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for images, targets in pbar:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        if 'loss_box_reg' in loss_dict:
            loss_dict['loss_box_reg'] = loss_dict['loss_box_reg'] * 10.0
        if 'loss_box_mean' in loss_dict:
            loss_dict['loss_box_mean'] = loss_dict['loss_box_mean'] * 1.0
        if 'loss_box_var' in loss_dict:
            loss_dict['loss_box_var'] = loss_dict['loss_box_var'] * 1.0
            
        losses = sum(loss for loss in loss_dict.values())
        
        loss_value = losses.item()
        epoch_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping to prevent large KL gradients from destabilizing training
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Update progress bar with current loss
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    # Return summary
    class Summary:
        def __init__(self):
            self.meters = {
                'loss': type('obj', (object,), {'global_avg': epoch_loss / len(data_loader)})()
            }
    
    return Summary()