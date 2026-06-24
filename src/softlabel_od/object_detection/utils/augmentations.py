"""
augmentations.py - Class-conditional augmentation for imbalanced datasets

Changes from original:
  - horizontal_flip_prob made configurable (was hardcoded 0.5)
  - Added: vertical flip with correct box handling
  - Added: hue/saturation adjustment for H&E stain variation
  - Added: gaussian blur for focus variation simulation
  - Added: random crop + resize with correct box/label handling
  - Fixed: rotation now uses discrete 90° steps with correct box transformation
           (original arbitrary-angle rotation did NOT update boxes — incorrect for detection)
  - Fixed: flip box update now also updates gt_means for soft label compatibility
"""

import torch
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image, ImageFilter


class ClassConditionalAugmentation:
    """
    Apply different augmentation strategies based on class rarity.

    Rare pathology classes get aggressive augmentation to increase diversity.
    Common classes get minimal augmentation.
    """

    def __init__(
        self,
        rare_class_indices,
        rare_augment_prob=0.7,
        common_augment_prob=0.2,
        horizontal_flip_prob=0.5,
        vertical_flip_prob=0.5,
        rotation_degrees=90,
        brightness_range=(0.8, 1.2),
        contrast_range=(0.8, 1.2),
        hue_delta=0.05,
        saturation_range=(0.8, 1.2),
        gaussian_blur_prob=0.3,
        crop_prob=0.5,
        crop_scale=(0.7, 1.0),
        noise_std=0.03,
    ):
        self.rare_class_indices = set(rare_class_indices)
        self.rare_augment_prob = rare_augment_prob
        self.common_augment_prob = common_augment_prob
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        # rotation_degrees used only for determining augmentation intensity
        # Actual rotation is always 0/90/180/270 for correct box handling
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.hue_delta = hue_delta
        self.saturation_range = saturation_range
        self.gaussian_blur_prob = gaussian_blur_prob
        self.crop_prob = crop_prob
        self.crop_scale = crop_scale
        self.noise_std = noise_std

    def __call__(self, img, target):
        """
        Args:
            img: PIL Image (before ToTensor)
            target: dict with 'boxes', 'labels', and optionally
                    'gt_class_probs', 'gt_stds', 'annotation_weight', 'gt_means'

        Returns:
            img (tensor), target (dict with updated boxes)
        """
        labels = target.get('labels', torch.tensor([14]))
        is_rare = (
            len(labels) > 0 and
            any(label.item() in self.rare_class_indices for label in labels)
        )
        augment_prob = self.rare_augment_prob if is_rare else self.common_augment_prob

        # Convert PIL to tensor
        img = TF.to_tensor(img)

        # Random crop + resize — applied before flips/rotations so boxes are correct
        if random.random() < self.crop_prob:
            img, target = self._random_crop_resize(img, target)

        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            img, target = self._hflip(img, target)

        # Vertical flip — safe for nuclei (no anatomical orientation)
        if random.random() < self.vertical_flip_prob:
            img, target = self._vflip(img, target)

        # Rotation + intensity augmentations with class-conditional probability
        if random.random() < augment_prob:
            img, target = self._apply_augmentations(img, target, aggressive=is_rare)

        return img, target

    # ─────────────────────────────────────────────────────────────
    # Spatial transforms (update boxes)
    # ─────────────────────────────────────────────────────────────

    def _hflip(self, img, target):
        """Horizontal flip with correct box and gt_means update."""
        img = TF.hflip(img)
        W = img.shape[2]
        if 'boxes' in target and len(target['boxes']) > 0:
            b = target['boxes'].clone()
            b[:, [0, 2]] = W - b[:, [2, 0]]
            target['boxes'] = b
        if 'gt_means' in target and len(target['gt_means']) > 0:
            m = target['gt_means'].clone()
            m[:, [0, 2]] = W - m[:, [2, 0]]
            target['gt_means'] = m
        return img, target

    def _vflip(self, img, target):
        """Vertical flip with correct box and gt_means update."""
        img = TF.vflip(img)
        H = img.shape[1]
        if 'boxes' in target and len(target['boxes']) > 0:
            b = target['boxes'].clone()
            b[:, [1, 3]] = H - b[:, [3, 1]]
            target['boxes'] = b
        if 'gt_means' in target and len(target['gt_means']) > 0:
            m = target['gt_means'].clone()
            m[:, [1, 3]] = H - m[:, [3, 1]]
            target['gt_means'] = m
        return img, target

    def _rotate_90(self, img, target, k):
        """
        Rotate image by k * 90° (k in {1, 2, 3}) with correct box update.

        Box transformation rules (xyxy format):
          k=1 (90° CCW):  new_x1=y1, new_y1=W-x2, new_x2=y2, new_y2=W-x1
                          image shape: (C, W, H)
          k=2 (180°):     new_x1=W-x2, new_y1=H-y2, new_x2=W-x1, new_y2=H-y1
                          image shape: (C, H, W) — same as input
          k=3 (270° CCW): new_x1=H-y2, new_y1=x1, new_x2=H-y1, new_y2=x2
                          image shape: (C, W, H)
        """
        _, H, W = img.shape
        img = torch.rot90(img, k, dims=[1, 2])

        def _rotate_boxes(boxes, k, H, W):
            b = boxes.clone().float()
            x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
            if k == 1:
                new = torch.stack([y1, W - x2, y2, W - x1], dim=1)
            elif k == 2:
                new = torch.stack([W - x2, H - y2, W - x1, H - y1], dim=1)
            else:  # k == 3
                new = torch.stack([H - y2, x1, H - y1, x2], dim=1)
            return new

        if 'boxes' in target and len(target['boxes']) > 0:
            target['boxes'] = _rotate_boxes(target['boxes'], k, H, W)
        if 'gt_means' in target and len(target['gt_means']) > 0:
            target['gt_means'] = _rotate_boxes(target['gt_means'], k, H, W)
        return img, target

    def _random_crop_resize(self, img, target):
        """
        Random crop at scale (0.7–1.0) then resize back to original size.

        - Adjusts and clips all bounding boxes to crop region
        - Drops boxes whose area after clipping is < 25% of original area
        - Scales boxes back after resize
        - Updates all per-box tensors: boxes, labels, gt_class_probs,
          gt_stds, annotation_weight, gt_means
        """
        _, H, W = img.shape
        scale = random.uniform(*self.crop_scale)
        crop_h = max(int(H * scale), 1)
        crop_w = max(int(W * scale), 1)

        top  = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        img = img[:, top:top + crop_h, left:left + crop_w]

        if 'boxes' in target and len(target['boxes']) > 0:
            boxes_orig = target['boxes'].clone().float()

            b = boxes_orig.clone()
            b[:, [0, 2]] -= left
            b[:, [1, 3]] -= top
            b[:, [0, 2]] = b[:, [0, 2]].clamp(0, crop_w)
            b[:, [1, 3]] = b[:, [1, 3]].clamp(0, crop_h)

            # Keep boxes that retain ≥ 25% of original area
            orig_areas = (
                (boxes_orig[:, 2] - boxes_orig[:, 0]).clamp(min=1) *
                (boxes_orig[:, 3] - boxes_orig[:, 1]).clamp(min=1)
            )
            new_areas = (
                (b[:, 2] - b[:, 0]).clamp(min=0) *
                (b[:, 3] - b[:, 1]).clamp(min=0)
            )
            keep = (new_areas / orig_areas) >= 0.25

            if keep.sum() > 0:
                target['boxes'] = b[keep]
                per_box_keys = [
                    'labels', 'gt_class_probs', 'gt_stds',
                    'annotation_weight', 'gt_means',
                ]
                for key in per_box_keys:
                    if key in target and len(target[key]) == len(keep):
                        target[key] = target[key][keep]

        # Resize back to original H × W
        img = TF.resize(img, [H, W],
                        interpolation=TF.InterpolationMode.BILINEAR,
                        antialias=True)

        # Scale boxes
        if 'boxes' in target and len(target['boxes']) > 0:
            sx = W / crop_w
            sy = H / crop_h
            target['boxes'][:, [0, 2]] *= sx
            target['boxes'][:, [1, 3]] *= sy
        if 'gt_means' in target and len(target['gt_means']) > 0:
            sx = W / crop_w
            sy = H / crop_h
            target['gt_means'][:, [0, 2]] *= sx
            target['gt_means'][:, [1, 3]] *= sy

        return img, target

    # ─────────────────────────────────────────────────────────────
    # Colour / appearance transforms (no box changes)
    # ─────────────────────────────────────────────────────────────

    def _apply_augmentations(self, img, target, aggressive=True):
        """Apply appearance augmentations. No box updates needed here."""

        # Discrete 90° rotation — only when rotation_degrees == 90
        if aggressive and self.rotation_degrees >= 90 and random.random() < 0.7:
            k = random.choice([1, 2, 3])
            img, target = self._rotate_90(img, target, k)

        # Brightness
        if aggressive and random.random() < 0.6:
            img = TF.adjust_brightness(img, random.uniform(*self.brightness_range))

        # Contrast
        if aggressive and random.random() < 0.6:
            img = TF.adjust_contrast(img, random.uniform(*self.contrast_range))

        # Hue — H&E stain variation (small shift ±0.05)
        if aggressive and random.random() < 0.5:
            img = TF.adjust_hue(img, random.uniform(-self.hue_delta, self.hue_delta))

        # Saturation — stain intensity variation
        if aggressive and random.random() < 0.5:
            img = TF.adjust_saturation(img, random.uniform(*self.saturation_range))

        # Gaussian blur — focus variation
        if random.random() < self.gaussian_blur_prob:
            # kernel_size must be odd
            k = random.choice([3, 5, 7])
            sigma = random.uniform(0.1, 2.0)
            img = TF.gaussian_blur(img, kernel_size=k, sigma=sigma)

        # Gaussian noise
        if aggressive and random.random() < 0.5:
            img = torch.clamp(img + torch.randn_like(img) * self.noise_std, 0, 1)

        # Random erasing — simulates occlusion between adjacent nuclei
        if aggressive and random.random() < 0.3:
            img = self._random_erasing(img, scale=(0.02, 0.1))

        return img, target

    def _random_erasing(self, img, scale=(0.02, 0.1)):
        area = img.shape[1] * img.shape[2]
        erase_area = random.uniform(*scale) * area
        aspect = random.uniform(0.3, 3.0)
        h = int(round(np.sqrt(erase_area * aspect)))
        w = int(round(np.sqrt(erase_area / aspect)))
        if h < img.shape[1] and w < img.shape[2]:
            top  = random.randint(0, img.shape[1] - h)
            left = random.randint(0, img.shape[2] - w)
            img[:, top:top + h, left:left + w] = 0
        return img


class MinimalAugmentation:
    """Minimal augmentation for validation / testing."""

    def __call__(self, img, target):
        return TF.to_tensor(img), target