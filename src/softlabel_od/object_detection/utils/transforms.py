"""
Shared transforms for object-detection datasets.

These are intentionally lightweight and operate on the `(image, target)` pairs
used across hard and soft label training/evaluation.
"""

import random

from torchvision.transforms import functional as F


class ToTensor:
    """Convert a PIL image to a tensor while preserving the target."""

    def __call__(self, image, target):
        return F.to_tensor(image), target


class RandomHorizontalFlip:
    """Horizontally flip an image and all supported target fields."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                _, _, width = image.shape
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox

                if "gt_means" in target:
                    gt_means = target["gt_means"]
                    gt_means[:, [0, 2]] = width - gt_means[:, [2, 0]]
                    target["gt_means"] = gt_means
        return image, target


class Compose:
    """Compose transforms that may or may not require a target dict."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        if target is None:
            for transform in self.transforms:
                image = transform(image)
            return image

        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target
