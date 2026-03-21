"""
CLI entry point: softlabel-evaluate

Evaluates a trained model on the validation split.

Usage:
    softlabel-evaluate --weights outputs/saved_models/run.pth --gmm_type 1D
    softlabel-evaluate --weights outputs/saved_models/run.pth --gmm_type HARD_LABEL
"""

import argparse
import os

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


def main():
    from softlabel_od.object_detection.adapters.evaluators.evaluate_model import evaluate

    parser = argparse.ArgumentParser(description="Evaluate a trained detection model")
    parser.add_argument("--weights", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--gmm_type", type=str, required=True, choices=["1D", "HARD_LABEL"],
                        help="Model type: 1D (soft label) or HARD_LABEL")
    parser.add_argument("--label_file", type=str, default=None,
                        help="Path to label JSON (auto-selected if not provided)")
    parser.add_argument("--data_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data/vinbigdata"),
                        help="Dataset root directory")
    parser.add_argument("--output", type=str,
                        default=os.path.join(PROJECT_ROOT, "outputs/evaluation_results"),
                        help="Output JSON path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=15)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--test_split_ratio", type=float, default=0.20)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()

    if args.label_file is None:
        if args.gmm_type == "1D":
            args.label_file = os.path.join(PROJECT_ROOT, "outputs/1DGMM_softLabels_iou0_5_with_priors.json")
        else:
            args.label_file = os.path.join(PROJECT_ROOT, "outputs/1DGMM_HardLabels_Majority.json")
        print(f"Auto-selected label file: {args.label_file}")

    evaluate(args)


if __name__ == "__main__":
    main()
