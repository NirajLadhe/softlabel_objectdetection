"""Learning rate scheduler utilities."""

import torch


def warmup_lr_scheduler(optimizer, warmup_iters: int, warmup_factor: float):
    """
    Linear warmup scheduler.

    Scales the LR from ``warmup_factor * base_lr`` up to ``base_lr`` over
    ``warmup_iters`` steps, then holds at 1.0 (no further scaling).
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
