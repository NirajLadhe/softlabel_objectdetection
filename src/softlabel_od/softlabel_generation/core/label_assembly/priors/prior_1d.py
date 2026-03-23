"""
1D Prior Calculator for computing class-specific variance priors.

This module calculates variance priors for each bbox coordinate (x, y, w, h) independently,
then uses these priors to adjust instance-specific variance estimates using a Bayesian approach.
"""

import numpy as np
from typing import Dict, List, Any
from .prior_calculator import PriorCalculator


class Prior1DCalculator(PriorCalculator):
    """
    Calculate 1D variance priors for each coordinate (x, y, w, h) per class.
    
    For each class, we:
    1. Collect all multi-annotated clusters (n >= 2 annotations)
    2. Compute variance for each coordinate per cluster
    3. Average these variances to get class-specific prior variance
    
    Then for each instance:
    - If n = 1: use prior variance
    - If n >= 2: blend instance variance with prior using weighted average
    """
    
    def calculate_class_priors(
        self, 
        clusters_by_class: Dict[str, List[List[Dict]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate variance priors for each class.
        
        Args:
            clusters_by_class: Dictionary mapping class names to lists of clusters.
                              Each cluster is a list of annotation dicts with 'bbox' key.
                              bbox format: [x, y, w, h]
        
        Returns:
            Dictionary: {
                class_name: {
                    'x_var': float,  # Variance of x coordinate
                    'y_var': float,  # Variance of y coordinate
                    'w_var': float,  # Variance of width
                    'h_var': float,  # Variance of height
                    'n_samples': int # Number of multi-annotated clusters
                }
            }
        """
        print("\n[1D Prior Calculation] Processing clusters by class...")
        
        # Filter to multi-annotated clusters only
        multi_ann_clusters = self.filter_multi_annotated_clusters(clusters_by_class)
        
        if not multi_ann_clusters:
            print("⚠️  Warning: No multi-annotated clusters found!")
            return {}
        
        print(f"   Found {len(multi_ann_clusters)} classes with multi-annotated data")
        
        # Calculate prior for each class
        for class_name, clusters in multi_ann_clusters.items():
            if len(clusters) < self.min_samples_per_class:
                print(f"   Skipping {class_name}: only {len(clusters)} clusters "
                      f"(min required: {self.min_samples_per_class})")
                continue
            
            # Collect variances for each coordinate
            x_vars, y_vars, w_vars, h_vars = [], [], [], []
            
            for cluster in clusters:
                # Extract bboxes from annotations
                bboxes = np.array([ann['bbox'] for ann in cluster])  # shape: (n, 4)
                
                # Compute variance for each coordinate
                # Note: Using np.var with default ddof=0 (population variance)
                x_vars.append(np.var(bboxes[:, 0]))
                y_vars.append(np.var(bboxes[:, 1]))
                w_vars.append(np.var(bboxes[:, 2]))
                h_vars.append(np.var(bboxes[:, 3]))
            
            # Average variances across all clusters for this class
            self.class_priors[class_name] = {
                'x_var': float(np.mean(x_vars)),
                'y_var': float(np.mean(y_vars)),
                'w_var': float(np.mean(w_vars)),
                'h_var': float(np.mean(h_vars)),
                'n_samples': len(clusters)
            }
            
            print(f"   ✓ {class_name}: {len(clusters)} clusters, "
                  f"avg x_var={np.mean(x_vars):.2f}")
        
        # Calculate global prior as fallback
        if multi_ann_clusters:
            self._calculate_global_prior_1d(multi_ann_clusters)
        
        print(f"\n   Total class priors computed: {len(self.class_priors)}")
        return self.class_priors
    
    def _calculate_global_prior_1d(
        self, 
        multi_ann_clusters: Dict[str, List[List[Dict]]]
    ):
        """
        Calculate a global prior by pooling all multi-annotated clusters.
        Used as fallback for classes with insufficient data.
        
        Args:
            multi_ann_clusters: Filtered clusters (only multi-annotated)
        """
        print("\n   Calculating global prior (fallback)...")
        
        all_x_vars, all_y_vars, all_w_vars, all_h_vars = [], [], [], []
        total_clusters = 0
        
        for class_name, clusters in multi_ann_clusters.items():
            for cluster in clusters:
                bboxes = np.array([ann['bbox'] for ann in cluster])
                all_x_vars.append(np.var(bboxes[:, 0]))
                all_y_vars.append(np.var(bboxes[:, 1]))
                all_w_vars.append(np.var(bboxes[:, 2]))
                all_h_vars.append(np.var(bboxes[:, 3]))
                total_clusters += 1
        
        if all_x_vars:
            self.global_prior = {
                'x_var': float(np.mean(all_x_vars)),
                'y_var': float(np.mean(all_y_vars)),
                'w_var': float(np.mean(all_w_vars)),
                'h_var': float(np.mean(all_h_vars)),
                'n_samples': total_clusters
            }
            self.has_global_fallback = True
            print(f"   ✓ Global prior computed from {total_clusters} clusters")
            print(f"     x_var={self.global_prior['x_var']:.2f}, "
                  f"y_var={self.global_prior['y_var']:.2f}, "
                  f"w_var={self.global_prior['w_var']:.2f}, "
                  f"h_var={self.global_prior['h_var']:.2f}")
        else:
            self.has_global_fallback = False
            print("   ⚠️  Could not compute global prior")
    
    def adjust_variance(
        self, 
        instance_coords: List[float], 
        class_name: str, 
        coordinate: str,
        weights: List[float] = None
    ) -> float:
        """
        Adjust instance variance with class prior using Bayesian weighted average.
        
        Formula: var_adjusted = (n * var_instance + alpha * var_prior) / (n + alpha)
        
        Args:
            instance_coords: List of coordinate values (e.g., [x1, x2, x3])
            class_name: Class of the instance (e.g., 'Cardiomegaly')
            coordinate: Which coordinate ('x', 'y', 'w', or 'h')
            weights: Optional rater reliability weights for weighted variance
        
        Returns:
            Adjusted variance (float)
        """
        n = len(instance_coords)
        
        # Get prior variance for this class and coordinate
        prior = self.get_prior_for_class(class_name, fallback_to_global=True)
        
        if prior is None:
            # No prior available, use simple variance with regularization
            if n == 1:
                return 1e-6  # Minimal variance for single annotation
            else:
                if weights is not None:
                    # Weighted variance
                    instance_var = self._weighted_variance(instance_coords, weights)
                else:
                    instance_var = np.var(instance_coords)
                return max(instance_var, 1e-6)  # Ensure non-zero
        
        prior_var = prior[f'{coordinate}_var']
        
        if n == 1:
            # Single annotation: use prior variance directly
            return prior_var + 1e-6  # Small regularization
        else:
            # Multiple annotations: blend instance and prior
            if weights is not None:
                instance_var = self._weighted_variance(instance_coords, weights)
            else:
                instance_var = np.var(instance_coords)
            
            # Bayesian adjustment
            adjusted_var = (n * instance_var + self.alpha * prior_var) / (n + self.alpha)
            
            # Add small regularization for numerical stability
            return adjusted_var + 1e-6
    
    def _weighted_variance(self, values: List[float], weights: List[float]) -> float:
        """
        Calculate weighted variance.
        
        Args:
            values: List of values
            weights: List of weights (should sum to something meaningful)
        
        Returns:
            Weighted variance
        """
        values = np.array(values)
        weights = np.array(weights)
        
        if np.sum(weights) == 0:
            return np.var(values)  # Fallback to unweighted
        
        # Weighted mean
        weighted_mean = np.average(values, weights=weights)
        
        # Weighted variance
        weighted_var = np.average((values - weighted_mean)**2, weights=weights)
        
        return weighted_var
    
    def adjust_all_coordinates(
        self,
        bbox_annotations: List[List[float]],
        class_name: str,
        weights: List[float] = None
    ) -> Dict[str, float]:
        """
        Adjust variances for all coordinates of a bbox at once.
        
        Args:
            bbox_annotations: List of bboxes [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
            class_name: Class name
            weights: Optional rater weights
        
        Returns:
            Dictionary with adjusted variances: {'x_var': ..., 'y_var': ..., etc.}
        """
        bboxes = np.array(bbox_annotations)  # shape: (n, 4)
        
        return {
            'x_var': self.adjust_variance(bboxes[:, 0].tolist(), class_name, 'x', weights),
            'y_var': self.adjust_variance(bboxes[:, 1].tolist(), class_name, 'y', weights),
            'w_var': self.adjust_variance(bboxes[:, 2].tolist(), class_name, 'w', weights),
            'h_var': self.adjust_variance(bboxes[:, 3].tolist(), class_name, 'h', weights)
        }

