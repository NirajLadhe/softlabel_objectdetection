import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class TrainingMetricsCombiner:
    """Combine and visualize training metrics from multiple JSON files."""
    
    def __init__(self, save_dir: str = "src/dlcv/results/SoftLabel_overfitRun"):
        self.save_dir = Path(save_dir)
        
    def load_json(self, filepath: str) -> Dict[str, Any]:
        """Load a single JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def combine_metrics(self, file_paths: List[str]) -> Dict[str, List]:
        """
        Combine metrics from multiple JSON files.
        
        Args:
            file_paths: List of JSON file paths in chronological order
            
        Returns:
            Dictionary with combined metrics
        """
        combined = {}
        
        for file_path in file_paths:
            data = self.load_json(file_path)
            
            for key, values in data.items():
                if key not in combined:
                    combined[key] = []
                combined[key].extend(values)
        
        return combined
    
    def plot_metrics(self, metrics: Dict[str, List], title_suffix: str = ""):
        """
        Create comprehensive plots of all training metrics.
        
        Args:
            metrics: Dictionary containing training metrics
            title_suffix: Additional text to add to plot titles
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        epochs = range(1, len(metrics['train_loss']) + 1)
        
        # 1. Loss plots
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(epochs, metrics['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
        ax1.plot(epochs, metrics['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=4)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Classification Loss
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(epochs, metrics['val_class_loss'], 'g-o', label='Val Class Loss', linewidth=2, markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title(f'Validation Classification Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box Regression Loss
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(epochs, metrics['val_box_reg_loss'], 'orange', marker='o', label='Val Box Reg Loss', linewidth=2, markersize=4)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title(f'Validation Box Regression Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Accuracy
        ax4 = plt.subplot(3, 3, 4)
        ax4.plot(epochs, metrics['top1_accuracy'], 'purple', marker='s', label='Top 1 Accuracy', linewidth=2, markersize=4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title(f'Top1 Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Top3 Accuracy
        ax5 = plt.subplot(3, 3, 5)
        ax5.plot(epochs, metrics['top3_accuracy'], 'brown', marker='d', label='Val RPN Reg', linewidth=2, markersize=4)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title(f'Top 3 Accuracy')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. KL Divergence
        ax6 = plt.subplot(3, 3, 6)
        ax6.plot(epochs, metrics['kl_divergence'], 'c-o', label='KL Divergence', linewidth=2, markersize=4)
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('IoU')
        ax6.set_title(f'KL Div')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Classification Accuracy
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(epochs, metrics['js_divergence'], 'm-s', label='JS Divergence', linewidth=2, markersize=4)
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Accuracy')
        ax7.set_title(f'JS Divergence')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Cosine Similarity
        ax8 = plt.subplot(3, 3, 8)
        ax8.plot(epochs, metrics['cosine_similarity'], 'k-o', label='Cosine Similarity', linewidth=2, markersize=4)
        ax8.set_xlabel('Epoch')
        ax8.set_ylabel('Cosine Similarity')
        ax8.set_title(f'Cosine Similarity')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Val_Std_loss
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(epochs, metrics['val_box_std_loss'], 'k-o', label='Std Loss', linewidth=2, markersize=4)
        ax9.set_xlabel('Epoch')
        ax9.set_ylabel('Localization Std Loss')
        ax9.set_title(f'Localization Std Loss')
        ax9.legend()
        ax9.grid(True, alpha=0.3)
    
    def save_combined_metrics(self, metrics: Dict[str, List], output_filename: str):
        """Save combined metrics to a new JSON file."""
        output_path = self.save_dir / output_filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Combined metrics saved to: {output_path}")
    
    def print_summary(self, metrics: Dict[str, List]):
        """Print summary statistics of the training."""

def main():
    """
    Main function to combine and visualize training metrics.
    Edit the file_groups dictionary to define which files to combine.
    """
    combiner = TrainingMetricsCombiner()
    
    # Define groups of files to combine
    # Format: 'output_name': ['file1.json', 'file2.json', ...]
    # Files should be in chronological order (earliest training first)
    file_groups = {
        'SoftLabel_1D_Overfit1_100Epoch_IgnoreNoFinding_NoBalancedSampler': [
            'training_metrics_phase_1.json'
        ]
    }
    
    # Process each group
    for group_name, files in file_groups.items():
        print(f"\n{'='*60}")
        print(f"Processing: {group_name}")
        print(f"{'='*60}")
        print(f"Combining files: {', '.join(files)}")
        
        # Convert to full paths
        file_paths = [str(combiner.save_dir / f) for f in files]
        
        # Check if files exist
        missing_files = [f for f in file_paths if not Path(f).exists()]
        if missing_files:
            print(f"WARNING: Skipping {group_name} - Missing files:")
            for mf in missing_files:
                print(f"  - {mf}")
            continue
        
        # Combine metrics
        combined_metrics = combiner.combine_metrics(file_paths)
        
        # Print summary
        combiner.print_summary(combined_metrics)
        
        # Plot metrics
        fig = combiner.plot_metrics(combined_metrics, f" - {group_name}")
        
        # Save plot
        plot_filename = f"{group_name}_training_plots4.png"
        plt.savefig(combiner.save_dir / plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {combiner.save_dir / plot_filename}")
        
        # Save combined metrics
        output_json = f"{group_name}.json"
        combiner.save_combined_metrics(combined_metrics, output_json)
        
        plt.show()
    
    print(f"\n{'='*60}")
    print("All processing complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()