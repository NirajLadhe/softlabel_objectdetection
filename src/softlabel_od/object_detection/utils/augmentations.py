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
                 brightness_range=(0.8, 1.2),
                 contrast_range=(0.8, 1.2),
                 noise_std=0.02):
        """
        Args:
            rare_class_indices: List of class indices considered rare (e.g., [0,1,2,...,13])
            rare_augment_prob: Probability of applying augmentation to rare classes
            common_augment_prob: Probability for common classes (like "No Finding")
            rotation_degrees: Max rotation angle for rare classes
            brightness_range: (min, max) brightness factor
            contrast_range: (min, max) contrast factor
            noise_std: Standard deviation for Gaussian noise
        """
        self.rare_class_indices = set(rare_class_indices)
        self.rare_augment_prob = rare_augment_prob
        self.common_augment_prob = common_augment_prob
        self.rotation_degrees = rotation_degrees
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
    
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
        
        # Apply augmentations with appropriate probability
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
        
        # 2. Rotation (only for rare classes, more aggressive)
        if aggressive and random.random() < 0.7:
            angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
            img = TF.rotate(img, angle, fill=0)
            # Note: Rotating bboxes is complex, simplified here
            # For production, use proper bbox rotation transformation
        
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