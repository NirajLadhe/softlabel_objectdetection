"""
2D Prior Calculator for computing class-specific covariance priors.

This module calculates covariance priors for each bbox vertex (top_left, top_right, etc.)
independently, then uses these priors to adjust instance-specific covariance estimates.
"""

import numpy as np
from typing import Dict, List, Any
from .prior_calculator import PriorCalculator


class Prior2DCalculator(PriorCalculator):
    """
    Calculate 2D covariance priors for each vertex per class.
    
    For each class, we:
    1. Collect all multi-annotated clusters (n >= 2 annotations)
    2. Convert bboxes to vertex coordinates
    3. Compute 2x2 covariance matrix for each vertex per cluster
    4. Average these covariances to get class-specific prior covariance
    
    Then for each instance:
    - If n = 1: use prior covariance
    - If n >= 2: blend instance covariance with prior using weighted average
    """
    
    VERTEX_NAMES = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
    
    def calculate_class_priors(
        self, 
        clusters_by_class: Dict[str, List[List[Dict]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate covariance priors for each class.
        
        Args:
            clusters_by_class: Dictionary mapping class names to lists of clusters.
                              Each cluster is a list of annotation dicts with 'bbox' key.
                              bbox format: [x, y, w, h]
        
        Returns:
            Dictionary: {
                class_name: {
                    'top_left': [[cov_xx, cov_xy], [cov_yx, cov_yy]],
                    'top_right': [[cov_xx, cov_xy], [cov_yx, cov_yy]],
                    'bottom_right': [[cov_xx, cov_xy], [cov_yx, cov_yy]],
                    'bottom_left': [[cov_xx, cov_xy], [cov_yx, cov_yy]],
                    'n_samples': int
                }
            }
        """
        print("\n[2D Prior Calculation] Processing clusters by class...")
        
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
            
            # Store covariances per vertex across all clusters
            vertex_covs = {v_name: [] for v_name in self.VERTEX_NAMES}
            
            for cluster in clusters:
                # Convert bboxes to vertex coordinates
                vertices = self._bboxes_to_vertices(cluster)
                
                # Compute covariance for each vertex
                for v_name in self.VERTEX_NAMES:
                    v_coords = np.array(vertices[v_name])  # shape: (n, 2)
                    if len(v_coords) >= 2:
                        cov = np.cov(v_coords.T)  # shape: (2, 2)
                        # Add small regularization to ensure positive definite
                        cov += 1e-6 * np.eye(2)
                        vertex_covs[v_name].append(cov)
            
            # Average covariances per vertex
            self.class_priors[class_name] = {
                'n_samples': len(clusters)
            }
            
            for v_name in self.VERTEX_NAMES:
                if vertex_covs[v_name]:
                    # Average covariance matrices element-wise
                    avg_cov = np.mean(vertex_covs[v_name], axis=0)
                    # Additional regularization for numerical stability
                    avg_cov += 1e-4 * np.eye(2)
                    self.class_priors[class_name][v_name] = avg_cov.tolist()
                else:
                    # Fallback: identity matrix with small scale
                    self.class_priors[class_name][v_name] = [[1e-3, 0.0], [0.0, 1e-3]]
            
            # Print sample statistics
            sample_cov = np.array(self.class_priors[class_name]['top_left'])
            eigenvals = np.linalg.eigvals(sample_cov)
            print(f"   ✓ {class_name}: {len(clusters)} clusters, "
                  f"top_left eigenvals=[{eigenvals[0]:.3f}, {eigenvals[1]:.3f}]")
        
        # Calculate global prior as fallback
        if multi_ann_clusters:
            self._calculate_global_prior_2d(multi_ann_clusters)
        
        print(f"\n   Total class priors computed: {len(self.class_priors)}")
        return self.class_priors
    
    def _bboxes_to_vertices(self, cluster: List[Dict]) -> Dict[str, List[List[float]]]:
        """
        Convert bbox annotations to vertex coordinates.
        
        Args:
            cluster: List of annotation dictionaries with 'bbox' key
        
        Returns:
            Dictionary mapping vertex names to lists of [x, y] coordinates
        """
        vertices = {v_name: [] for v_name in self.VERTEX_NAMES}
        
        for ann in cluster:
            x, y, w, h = ann['bbox']
            vertices['top_left'].append([x, y])
            vertices['top_right'].append([x + w, y])
            vertices['bottom_right'].append([x + w, y + h])
            vertices['bottom_left'].append([x, y + h])
        
        return vertices
    
    def _calculate_global_prior_2d(
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
        
        # Collect all covariances across all classes
        all_vertex_covs = {v_name: [] for v_name in self.VERTEX_NAMES}
        total_clusters = 0
        
        for class_name, clusters in multi_ann_clusters.items():
            for cluster in clusters:
                vertices = self._bboxes_to_vertices(cluster)
                
                for v_name in self.VERTEX_NAMES:
                    v_coords = np.array(vertices[v_name])
                    if len(v_coords) >= 2:
                        cov = np.cov(v_coords.T)
                        cov += 1e-6 * np.eye(2)
                        all_vertex_covs[v_name].append(cov)
                
                total_clusters += 1
        
        if total_clusters > 0:
            self.global_prior = {'n_samples': total_clusters}
            
            for v_name in self.VERTEX_NAMES:
                if all_vertex_covs[v_name]:
                    avg_cov = np.mean(all_vertex_covs[v_name], axis=0)
                    avg_cov += 1e-4 * np.eye(2)
                    self.global_prior[v_name] = avg_cov.tolist()
                else:
                    self.global_prior[v_name] = [[1e-3, 0.0], [0.0, 1e-3]]
            
            self.has_global_fallback = True
            
            # Print sample statistics
            sample_cov = np.array(self.global_prior['top_left'])
            eigenvals = np.linalg.eigvals(sample_cov)
            print(f"   ✓ Global prior computed from {total_clusters} clusters")
            print(f"     top_left eigenvals=[{eigenvals[0]:.3f}, {eigenvals[1]:.3f}]")
        else:
            self.has_global_fallback = False
            print("   ⚠️  Could not compute global prior")
    
    def adjust_covariance(
        self, 
        instance_coords: List[List[float]], 
        class_name: str, 
        vertex_name: str,
        weights: List[float] = None
    ) -> np.ndarray:
        """
        Adjust instance covariance with class prior using Bayesian weighted average.
        
        Formula: cov_adjusted = (n * cov_instance + alpha * cov_prior) / (n + alpha)
        
        Args:
            instance_coords: List of [x, y] coordinates for a vertex
            class_name: Class of the instance (e.g., 'Cardiomegaly')
            vertex_name: Which vertex ('top_left', 'top_right', etc.)
            weights: Optional rater reliability weights for weighted covariance
        
        Returns:
            Adjusted 2x2 covariance matrix (numpy array)
        """
        n = len(instance_coords)
        
        # Get prior covariance for this class and vertex
        prior = self.get_prior_for_class(class_name, fallback_to_global=True)
        
        if prior is None:
            # No prior available, use simple covariance with regularization
            if n == 1:
                return np.array([[1e-4, 0.0], [0.0, 1e-4]])  # Minimal covariance
            else:
                coords_array = np.array(instance_coords)
                if weights is not None:
                    instance_cov = self._weighted_covariance(coords_array, weights)
                else:
                    instance_cov = np.cov(coords_array.T)
                
                # Regularization
                instance_cov += 1e-4 * np.eye(2)
                return instance_cov
        
        prior_cov = np.array(prior[vertex_name])
        
        if n == 1:
            # Single annotation: use prior covariance directly
            return prior_cov
        else:
            # Multiple annotations: blend instance and prior
            coords_array = np.array(instance_coords)
            
            if weights is not None:
                instance_cov = self._weighted_covariance(coords_array, weights)
            else:
                instance_cov = np.cov(coords_array.T)
            
            # Bayesian adjustment
            adjusted_cov = (n * instance_cov + self.alpha * prior_cov) / (n + self.alpha)
            
            # Add regularization for numerical stability
            adjusted_cov += 1e-4 * np.eye(2)
            
            return adjusted_cov
    
    def _weighted_covariance(
        self, 
        coords: np.ndarray, 
        weights: List[float]
    ) -> np.ndarray:
        """
        Calculate weighted covariance matrix.
        
        Args:
            coords: Array of shape (n, 2) with [x, y] coordinates
            weights: List of weights (should sum to something meaningful)
        
        Returns:
            Weighted 2x2 covariance matrix
        """
        weights = np.array(weights)
        
        if np.sum(weights) == 0:
            return np.cov(coords.T)  # Fallback to unweighted
        
        # Weighted mean
        weighted_mean = np.average(coords, axis=0, weights=weights)
        
        # Weighted covariance
        diff = coords - weighted_mean
        weighted_cov = np.dot((diff * weights[:, None]).T, diff) / np.sum(weights)
        
        return weighted_cov
    
    def adjust_all_vertices(
        self,
        bbox_annotations: List[List[float]],
        class_name: str,
        weights: List[float] = None
    ) -> Dict[str, List[List[float]]]:
        """
        Adjust covariances for all vertices of a bbox at once.
        
        Args:
            bbox_annotations: List of bboxes [[x1, y1, w1, h1], [x2, y2, w2, h2], ...]
            class_name: Class name
            weights: Optional rater weights
        
        Returns:
            Dictionary with adjusted covariances: {
                'top_left': [[cov_xx, cov_xy], [cov_yx, cov_yy]], 
                ...
            }
        """
        # Convert bboxes to vertices
        vertices = self._bboxes_to_vertices([{'bbox': bbox} for bbox in bbox_annotations])
        
        result = {}
        for v_name in self.VERTEX_NAMES:
            adjusted_cov = self.adjust_covariance(
                vertices[v_name], class_name, v_name, weights
            )
            result[v_name] = adjusted_cov.tolist()
        
        return result
    
    def get_vertex_uncertainty_ellipse(
        self, 
        mean: List[float], 
        covariance: List[List[float]], 
        confidence: float = 0.95
    ) -> Dict[str, float]:
        """
        Get ellipse parameters for visualizing 2D uncertainty.
        
        Args:
            mean: [x, y] mean coordinates
            covariance: 2x2 covariance matrix
            confidence: Confidence level (0.95 = 95%)
        
        Returns:
            Dictionary with ellipse parameters: {
                'center_x': float,
                'center_y': float,
                'width': float,    # Semi-major axis length
                'height': float,   # Semi-minor axis length
                'angle': float     # Rotation angle in degrees
            }
        """
        cov = np.array(covariance)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (largest first)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]
        
        # Chi-squared value for confidence level
        from scipy.stats import chi2
        chi2_val = chi2.ppf(confidence, df=2)
        
        # Ellipse parameters
        width = 2 * np.sqrt(chi2_val * eigenvals[0])   # Semi-major axis
        height = 2 * np.sqrt(chi2_val * eigenvals[1])  # Semi-minor axis
        
        # Rotation angle (in degrees)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        
        return {
            'center_x': float(mean[0]),
            'center_y': float(mean[1]),
            'width': float(width),
            'height': float(height),
            'angle': float(angle)
        }
