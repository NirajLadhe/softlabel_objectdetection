"""
Base class for calculating and managing class-specific priors for GMM soft labels.

This module provides the foundation for computing priors from multi-annotated instances
and applying them to adjust covariance/variance estimates in soft label generation.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Optional

from softlabel_od.shared.utils.serialization import NpEncoder as NumpyEncoder


class PriorCalculator:
    """
    Base class for calculating class-specific priors from multi-annotated data.
    
    This implements a Bayesian approach where:
    - Prior = class-specific covariance/variance from multi-annotated instances
    - Likelihood = instance-specific covariance/variance
    - Posterior = weighted combination using alpha parameter
    
    Note: VinDr-CXR has max 3 raters per image, so clusters typically have 1-3 annotations.
    """
    
    def __init__(self, alpha: float = 1.0, min_samples_per_class: int = 50):
        """
        Initialize the prior calculator.
        
        Args:
            alpha: Prior strength (pseudo-observation count). Controls the weight of the prior
                   in the final estimate. Higher values give more weight to the prior.
                   - alpha = 0: No prior (baseline)
                   - alpha = 1: Prior equals one pseudo-observation (recommended)
                   - alpha = 5: Strong prior influence
            min_samples_per_class: Minimum number of multi-annotated instances required
                                   to compute a reliable class-specific prior.
        """
        self.alpha = alpha
        self.min_samples_per_class = min_samples_per_class
        self.class_priors = {}
        self.class_sample_counts = {}
        self.global_prior = None
        self.has_global_fallback = False
    
    def filter_multi_annotated_clusters(
        self, 
        clusters_by_class: Dict[str, List[List[Dict]]]
    ) -> Dict[str, List[List[Dict]]]:
        """
        Filter clusters to include only those with multiple annotations (n >= 2).
        
        Single-annotation clusters have zero variance and cannot contribute to prior calculation.
        
        Args:
            clusters_by_class: Dictionary mapping class names to lists of clusters.
                              Each cluster is a list of annotation dictionaries.
        
        Returns:
            Dictionary with same structure but only multi-annotated clusters.
        """
        filtered = {}
        for class_name, clusters in clusters_by_class.items():
            multi_ann = [cluster for cluster in clusters if len(cluster) >= 2]
            if multi_ann:
                filtered[class_name] = multi_ann
                self.class_sample_counts[class_name] = len(multi_ann)
        
        return filtered
    
    def get_prior_for_class(
        self, 
        class_name: str, 
        fallback_to_global: bool = True
    ) -> Optional[Dict]:
        """
        Retrieve the prior for a specific class.
        
        Args:
            class_name: Name of the class (e.g., 'Cardiomegaly', 'Lung Opacity')
            fallback_to_global: If True and class prior unavailable, use global prior
        
        Returns:
            Dictionary containing the prior parameters, or None if unavailable
        """
        # First, try to get class-specific prior
        if class_name in self.class_priors:
            return self.class_priors[class_name]
        
        # Fallback to global prior if available and requested
        if fallback_to_global and self.has_global_fallback:
            return self.global_prior
        
        # No prior available
        return None
    
    def save_priors(self, output_path: str):
        """
        Save computed priors to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
        """
        output_data = {
            'metadata': {
                'alpha': self.alpha,
                'min_samples_per_class': self.min_samples_per_class,
                'total_classes': len(self.class_priors),
                'has_global_fallback': self.has_global_fallback
            },
            'class_priors': self.class_priors,
            'class_sample_counts': self.class_sample_counts,
            'global_prior': self.global_prior
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, cls=NumpyEncoder)
        
        print(f"✅ Priors saved to: {output_path}")
        print(f"   - {len(self.class_priors)} class-specific priors")
        print(f"   - Global fallback: {self.has_global_fallback}")
    
    def load_priors(self, input_path: str):
        """
        Load pre-computed priors from a JSON file.
        
        Args:
            input_path: Path to the JSON file containing priors
        """
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Restore metadata
        metadata = data.get('metadata', {})
        self.alpha = metadata.get('alpha', self.alpha)
        self.min_samples_per_class = metadata.get('min_samples_per_class', self.min_samples_per_class)
        self.has_global_fallback = metadata.get('has_global_fallback', False)
        
        # Restore priors
        self.class_priors = data.get('class_priors', {})
        self.class_sample_counts = data.get('class_sample_counts', {})
        self.global_prior = data.get('global_prior')
        
        print(f"✅ Priors loaded from: {input_path}")
        print(f"   - {len(self.class_priors)} class-specific priors")
        print(f"   - Alpha: {self.alpha}")
        print(f"   - Global fallback: {self.has_global_fallback}")
    
    def print_statistics(self):
        """Print statistics about the computed priors."""
        print("\n" + "="*80)
        print("Prior Calculation Statistics")
        print("="*80)
        print(f"Alpha (prior strength): {self.alpha}")
        print(f"Min samples per class: {self.min_samples_per_class}")
        print(f"Total classes with priors: {len(self.class_priors)}")
        print(f"Global fallback available: {self.has_global_fallback}")
        
        if self.class_sample_counts:
            print("\nSample counts per class:")
            sorted_classes = sorted(
                self.class_sample_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for class_name, count in sorted_classes[:10]:  # Top 10
                print(f"  {class_name}: {count} multi-annotated clusters")
            
            if len(sorted_classes) > 10:
                print(f"  ... and {len(sorted_classes) - 10} more classes")
        
        print("="*80 + "\n")


