"""
check_prob_vectors.py - Find malformed probability vectors in soft label JSON

Usage:
    python -m dlcv.check_prob_vectors /path/to/your/soft_labels.json
"""

import json
import sys
from collections import defaultdict

def check_probability_vectors(json_path, tolerance=0.01):
    """
    Check soft label JSON for malformed probability vectors.
    
    Args:
        json_path: Path to the soft labels JSON file
        tolerance: How far from 1.0 is considered malformed (default 0.01)
    """
    print(f"Loading: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total entries: {len(data)}")
    print(f"Tolerance: ±{tolerance} from 1.0")
    print("=" * 60)
    
    # Track issues
    issues = {
        'sum_too_high': [],      # sum > 1 + tolerance
        'sum_too_low': [],       # sum < 1 - tolerance
        'missing_class_probs': [],
        'empty_class_probs': [],
        'negative_probs': [],
    }
    
    for idx, entry in enumerate(data):
        image_id = entry.get('image_id', f'unknown_{idx}')
        finding_id = entry.get('finding_id', 'N/A')
        
        # Check if class_probs exists
        if 'class_probs' not in entry:
            issues['missing_class_probs'].append({
                'image_id': image_id,
                'finding_id': finding_id,
            })
            continue
        
        class_probs = entry['class_probs']
        
        # Check if empty
        if not class_probs:
            issues['empty_class_probs'].append({
                'image_id': image_id,
                'finding_id': finding_id,
            })
            continue
        
        # Calculate sum
        prob_sum = sum(class_probs.values())
        
        # Check for negative probabilities
        negative_classes = {k: v for k, v in class_probs.items() if v < 0}
        if negative_classes:
            issues['negative_probs'].append({
                'image_id': image_id,
                'finding_id': finding_id,
                'negative_classes': negative_classes,
            })
        
        # Check sum
        if prob_sum > 1.0 + tolerance:
            issues['sum_too_high'].append({
                'image_id': image_id,
                'finding_id': finding_id,
                'sum': prob_sum,
                'class_probs': class_probs,
            })
        elif prob_sum < 1.0 - tolerance:
            issues['sum_too_low'].append({
                'image_id': image_id,
                'finding_id': finding_id,
                'sum': prob_sum,
                'class_probs': class_probs,
            })
    
    # Report results
    print("\n📊 SUMMARY")
    print("=" * 60)
    
    total_issues = sum(len(v) for v in issues.values())
    print(f"Total entries checked: {len(data)}")
    print(f"Total issues found: {total_issues}")
    print()
    
    # Report each issue type
    for issue_type, entries in issues.items():
        if entries:
            print(f"\n❌ {issue_type.upper()}: {len(entries)} entries")
            print("-" * 40)
            
            # Show first 10 examples
            for i, entry in enumerate(entries[:10]):
                print(f"\n  [{i+1}] Image: {entry['image_id']}, Finding: {entry['finding_id']}")
                if 'sum' in entry:
                    print(f"      Sum: {entry['sum']:.4f}")
                if 'class_probs' in entry:
                    print(f"      Probs: {entry['class_probs']}")
                if 'negative_classes' in entry:
                    print(f"      Negative: {entry['negative_classes']}")
            
            if len(entries) > 10:
                print(f"\n  ... and {len(entries) - 10} more")
    
    if total_issues == 0:
        print("✅ All probability vectors are valid!")
    
    # Return for programmatic use
    return issues


def export_problematic_ids(issues, output_path):
    """Export list of problematic image IDs to a file."""
    all_ids = set()
    for issue_list in issues.values():
        for entry in issue_list:
            all_ids.add(entry['image_id'])
    
    with open(output_path, 'w') as f:
        for img_id in sorted(all_ids):
            f.write(f"{img_id}\n")
    
    print(f"\n💾 Exported {len(all_ids)} problematic image IDs to: {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m dlcv.check_prob_vectors <path_to_json> [output_file]")
        print("\nExample:")
        print("  python -m dlcv.check_prob_vectors data/soft_labels_1d.json")
        print("  python -m dlcv.check_prob_vectors data/soft_labels_1d.json problematic_ids.txt")
        sys.exit(1)
    
    json_path = sys.argv[1]
    issues = check_probability_vectors(json_path)
    
    # Optionally export problematic IDs
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
        export_problematic_ids(issues, output_path)