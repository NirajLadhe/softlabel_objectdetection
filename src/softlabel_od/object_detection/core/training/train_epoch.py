"""Generic per-epoch training loop (vanilla PyTorch)."""

import torch
from tqdm import tqdm

from softlabel_od.object_detection.utils.schedulers import warmup_lr_scheduler


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    Train the model for one epoch.

    Args:
        model:        Detection model (must return a loss dict in training mode).
        optimizer:    PyTorch optimizer.
        data_loader:  DataLoader yielding (images, targets) batches.
        device:       torch.device.
        epoch:        Current epoch index (0-based; triggers warmup on epoch 0).
        print_freq:   Unused — kept for API compatibility.

    Returns:
        Summary object with a ``meters`` dict containing ``'loss'``.
    """
    model.train()

    epoch_loss = 0.0

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    pbar = tqdm(
        data_loader,
        desc=f"Epoch {epoch}",
        ncols=100,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )

    for images, targets in pbar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        epoch_loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        pbar.set_postfix(
            loss=f"{loss_value:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.6f}",
        )

    class _Summary:
        def __init__(self):
            self.meters = {
                "loss": type("_Meter", (), {"global_avg": epoch_loss / max(len(data_loader), 1)})()
            }

    return _Summary()
