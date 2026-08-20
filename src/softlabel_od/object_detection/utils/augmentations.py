"""
augmentations.py - Class-conditional augmentation for imbalanced datasets
"""

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor, Compose
import random
import numpy as np
from PIL import Image

class ClassConditionalAugmentation:
    """
    Apply different augmentation strategies based on class rarity.
    
    Rare pathology classes get aggressive augmentation to increase diversity.
    Common "No Finding" class gets minimal augmentation.
    """
    
    def __init__(self, rare_class_indices,
                 rare_augment_prob=0.8,
                 common_augment_prob=0.3,
                 rotation_degrees=15,
                 rotation_prob=0.7,
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 noise_std=0.02):
        """
        Args:
            rare_class_indices: List of class indices considered rare (e.g., [0,1,2,...,13])
            rare_augment_prob: Probability of applying augmentation to rare classes
            common_augment_prob: Probability for common classes (like "No Finding")
            rotation_degrees: Max rotation angle for rare classes
            rotation_prob: Probability of applying rotation (when augmenting a rare class)
            brightness_range: (min, max) brightness factor
            contrast_range: (min, max) contrast factor
            noise_std: Standard deviation for Gaussian noise
        """
        self.rare_class_indices = set(rare_class_indices)
        self.rare_augment_prob = rare_augment_prob
        self.common_augment_prob = common_augment_prob
        self.rotation_degrees = rotation_degrees
        self.rotation_prob = rotation_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        # Albumentations transform that rotates the image AND its boxes together.
        # Built lazily so albumentations is only required when rotation is enabled.
        self._albu_rotate = self._build_rotation_transform()
    
    def __call__(self, img, target):
        """
        Args:
            img: PIL Image
            target: Dict with 'boxes', 'labels', etc.
        
        Returns:
            Augmented img (tensor) and target
        """
        # Determine if this is a rare class image
        labels = target.get('labels', torch.tensor([14]))  # Default to "No Finding"
        
        if len(labels) == 0:
            # No labels - treat as common
            is_rare = False
        else:
            # Check if ANY label is a rare class
            is_rare = any(label.item() in self.rare_class_indices for label in labels)
        
        # Choose augmentation probability based on class rarity
        augment_prob = self.rare_augment_prob if is_rare else self.common_augment_prob
        
        # Convert to tensor first
        img = TF.to_tensor(img)

        # Rotation is evaluated independently of the augment-probability gate
        # below so it reaches EVERY image containing a rare class — with
        # ROTATION_PROB=1.0 all rare_class_indices images get rotated. Image +
        # boxes are rotated together via Albumentations. Common-only images
        # (is_rare=False) are never rotated.
        if is_rare and self._albu_rotate is not None and random.random() < self.rotation_prob:
            img, target = self._rotate_with_boxes(img, target)

        # Remaining (pixel-level) augmentations stay probabilistic.
        if random.random() < augment_prob:
            img, target = self._apply_augmentations(img, target, aggressive=is_rare)

        return img, target
    
    def _apply_augmentations(self, img, target, aggressive=True):
        """Apply actual augmentation transformations."""
        
        # 1. Horizontal Flip (safe for medical images, symmetric anatomy)
        if random.random() < 0.5:
            img = TF.hflip(img)
            # Flip bounding boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                width = img.shape[2]  # Image width
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Flip x coordinates
                target['boxes'] = boxes
        
        # 2. Rotation is applied in __call__ (decoupled from this gate) so that
        # every rare-class image is rotated; nothing to do here.

        # 3. Brightness adjustment (aggressive for rare classes)
        if aggressive and random.random() < 0.6:
            brightness = random.uniform(*self.brightness_range)
            img = TF.adjust_brightness(img, brightness)
        
        # 4. Contrast adjustment (aggressive for rare classes)
        if aggressive and random.random() < 0.6:
            contrast = random.uniform(*self.contrast_range)
            img = TF.adjust_contrast(img, contrast)
        
        # 5. Gaussian noise (only for rare classes)
        if aggressive and random.random() < 0.5:
            noise = torch.randn_like(img) * self.noise_std
            img = torch.clamp(img + noise, 0, 1)
        
        # 6. Random erasing (simulates occlusions, aggressive for rare)
        if aggressive and random.random() < 0.3:
            img = self._random_erasing(img, scale=(0.02, 0.1))
        
        return img, target
    
    def _build_rotation_transform(self):
        """Build the Albumentations rotation pipeline (rotates image + boxes).

        Returns None when rotation is disabled (degrees == 0 or prob <= 0) so the
        albumentations dependency is only required when rotation is actually used.
        Boxes are pascal_voc ([x_min, y_min, x_max, y_max] in pixels), matching the
        dataset adapters. ``box_idx`` is carried as a label field so the remaining
        per-box target arrays can be re-aligned after boxes that rotate out of frame
        are dropped.
        """
        if not self.rotation_degrees or self.rotation_prob <= 0:
            return None
        import albumentations as A
        return A.Compose(
            [A.Rotate(limit=self.rotation_degrees, p=1.0, border_mode=0)],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["box_idx"],
                min_visibility=0.0,
                min_area=0.0,
            ),
        )

    def _rotate_with_boxes(self, img, target):
        """Rotate a CHW float image and its boxes by a random angle in
        [-rotation_degrees, +rotation_degrees], keeping every per-box target field
        (labels, gt_class_probs, gt_stds, annotation_weight, ...) aligned."""
        boxes = target.get("boxes")
        n = 0 if boxes is None else len(boxes)

        np_img = img.permute(1, 2, 0).contiguous().cpu().numpy()
        bboxes = boxes.cpu().numpy().tolist() if n > 0 else []
        out = self._albu_rotate(image=np_img, bboxes=bboxes, box_idx=list(range(n)))

        img = torch.from_numpy(out["image"]).permute(2, 0, 1).contiguous().to(img.dtype)

        if n > 0:
            kept = torch.as_tensor(out["box_idx"], dtype=torch.long)
            target["boxes"] = torch.as_tensor(out["bboxes"], dtype=boxes.dtype).reshape(-1, 4)
            # Re-index the remaining per-box fields so they stay aligned with boxes.
            for key, val in list(target.items()):
                if key in ("boxes", "image_id"):
                    continue
                if torch.is_tensor(val) and val.shape[:1] == torch.Size([n]):
                    target[key] = val[kept]

        return img, target

    def _random_erasing(self, img, scale=(0.02, 0.15)):
        """Randomly erase a rectangular region."""
        area = img.shape[1] * img.shape[2]
        erase_area = random.uniform(*scale) * area
        aspect_ratio = random.uniform(0.3, 3.0)
        
        h = int(round(np.sqrt(erase_area * aspect_ratio)))
        w = int(round(np.sqrt(erase_area / aspect_ratio)))
        
        if h < img.shape[1] and w < img.shape[2]:
            top = random.randint(0, img.shape[1] - h)
            left = random.randint(0, img.shape[2] - w)
            img[:, top:top+h, left:left+w] = 0
        
        return img


class MinimalAugmentation:
    """Minimal augmentation for validation/testing."""
    
    def __call__(self, img, target):
        img = TF.to_tensor(img)
        return img, target