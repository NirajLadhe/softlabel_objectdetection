#!/usr/bin/env python3
"""
Main Pipeline: VinDr-CXR Soft Label Generation

Orchestrates the full pipeline from raw data to soft labels.
Steps: 1) Normalize CSV, 2) Convert to COCO JSON, 3) Reliability (K-alpha),
4) Priors, 5) GMM Soft Labels.
"""

import argparse
from email import parser
from html import parser
import subprocess
import sys
import os
import time
from datetime import timedelta

# Project root: scripts/ -> softlabel_od/ -> src/ -> project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..'))

STEP_NAMES = {
    1: "Normalize CSV",
    2: "Convert CSV to COCO JSON",
    3: "Calculate Annotator Reliability (K-alpha)",
    4: "Calculate Class-Specific Priors",
    5: "Generate GMM Soft Labels",
}
ALL_STEPS = list(STEP_NAMES.keys())


def safe_threshold_str(value: float) -> str:
    return str(value).replace('.', '_')


def run_step(step_num: int, step_name: str, command: list, dry_run: bool = False):
    print(f"\n{'='*80}")
    print(f"  STEP {step_num}/{len(STEP_NAMES)}: {step_name}")
    print(f"{'='*80}")
    print(f"  Command: {' '.join(command)}")
    if dry_run:
        print(f"  [DRY RUN] Skipping execution.")
        return True
    print()
    start_time = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = str(timedelta(seconds=int(time.time() - start_time)))
    if result.returncode != 0:
        print(f"\n  FAILED: Step '{step_name}' exited with code {result.returncode}")
        return False
    print(f"\n  Step '{step_name}' completed successfully. ({elapsed})")
    return True


def build_clustering_suffix(args) -> str:
    if args.clustering_method == 'iou':
        return f"iou{safe_threshold_str(args.iou_threshold)}"
    elif args.clustering_method == 'iomin':
        return f"iomin{safe_threshold_str(args.iomin_threshold)}"
    elif args.clustering_method == 'iomin_center_gate':
        return f"iomin_cg{safe_threshold_str(args.iomin_threshold)}_{safe_threshold_str(args.center_ratio_threshold)}"
    return args.clustering_method


def build_step1_command(args) -> list:
    return [
        sys.executable, '-m', 'softlabel_od.softlabel_generation.core.preprocessing.manipulate_csv',
        '--input-csv', args.input_csv, '--output-csv', args.normalized_csv,
    ]

#================== lvis-specific step 1 command builder (can be ignored for VinDr-CXR) =======================================================
def build_step1_command(args) -> list:
    if args.dataset == "lvis":
        cmd = [
            sys.executable, "-m",
            "softlabel_od.softlabel_generation.core.preprocessing.preprocess_lvis",
            "--input-json", args.raw_lvis_json,
            "--output-dir", args.lvis_output_dir,
        ]
        if args.max_images is not None:
            cmd.extend(["--max-images", str(args.max_images)])
        return cmd

    return [
        sys.executable, "-m", "softlabel_od.softlabel_generation.core.preprocessing.manipulate_csv",
        "--input-csv", args.input_csv,
        "--output-csv", args.normalized_csv,
    ]

def build_step2_command(args) -> list:
    return [
        sys.executable, '-m', 'softlabel_od.softlabel_generation.core.preprocessing.convert_csv_to_json',
        '--input-csv', args.normalized_csv, '--output-json', args.coco_json,
        '--no-finding-id', str(args.no_finding_id),
    ]

#================== lvis-specific step 2 command builder (can be ignored for VinDr-CXR) =======================================================
def build_step2_command(args) -> list:
    if args.dataset == "lvis":
        return [
            sys.executable, "-c",
            f"import os; print('LVIS COCO JSON already created: {args.coco_json}'); assert os.path.exists('{args.coco_json}')"
        ]

    return [
        sys.executable, "-m", "softlabel_od.softlabel_generation.core.preprocessing.convert_csv_to_json",
        "--input-csv", args.normalized_csv,
        "--output-json", args.coco_json,
        "--no-finding-id", str(args.no_finding_id),
    ]


def build_step3_command(args) -> list:
    return [
        sys.executable, '-m',
        'softlabel_od.softlabel_generation.core.reliability.calculate_reliability',
        '--iou', str(args.iou_reliability),
        '--no_finding_id', str(args.no_finding_id),
        '--annotation-path', args.coco_json,
        '--output-dir', args.reliability_output_dir,
        '--backend', args.reliability_backend,
        '--method', args.reliability_method,
    ]


def build_step4_command(args) -> list:
    if args.gmm_mode == '1d':
        module = 'softlabel_od.softlabel_generation.core.label_assembly.priors.calculate_priors_1d'
    elif args.gmm_mode == '2d':
        module = 'softlabel_od.softlabel_generation.core.label_assembly.priors.calculate_priors_2d'
    else:
        raise ValueError(f"Invalid GMM mode: {args.gmm_mode}")
    cmd = [
        sys.executable, '-m', module,
        '--train-csv', args.normalized_csv, '--raters-json', args.coco_json,
        '--weights-csv', args.weights_csv, '--output', args.priors_path,
        '--alpha', str(args.prior_alpha), '--min-samples', str(args.min_samples),
        '--clustering-method', args.clustering_method,
        '--iou-threshold', str(args.iou_threshold),
        '--iomin-threshold', str(args.iomin_threshold),
        '--center-ratio-threshold', str(args.center_ratio_threshold),
        '--semantic-map', args.semantic_map,
        '--iou-reliability', str(args.iou_reliability),
    ]
    if args.max_images is not None:
        cmd.extend(['--max-images', str(args.max_images)])
    return cmd


def build_step5_command(args) -> list:
    if args.gmm_mode == '1d':
        module = 'softlabel_od.softlabel_generation.core.label_assembly.label_constructor.gmm_1d'
    elif args.gmm_mode == '2d':
        module = 'softlabel_od.softlabel_generation.core.label_assembly.label_constructor.gmm_2d'
    else:
        raise ValueError(f"Invalid GMM mode: {args.gmm_mode}")
    return [
        sys.executable, '-m', module,
        '--train-csv', args.normalized_csv, '--raters-json', args.coco_json,
        '--weights-csv', args.weights_csv, '--priors', args.priors_path,
        '--output', args.output_json, '--alpha', str(args.prior_alpha),
        '--clustering-method', args.clustering_method,
        '--iou-threshold', str(args.iou_threshold),
        '--iomin-threshold', str(args.iomin_threshold),
        '--center-ratio-threshold', str(args.center_ratio_threshold),
        '--semantic-map', args.semantic_map,
        '--iou-reliability', str(args.iou_reliability),
    ]


def main():
    parser = argparse.ArgumentParser(description="VinDr-CXR Soft Label Generation Pipeline")
    parser.add_argument('--steps', type=int, nargs='+', default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--gmm-mode', type=str, default='1d', choices=['1d', '2d'])
    parser.add_argument('--input-csv', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/vindrcxr/train_enriched.csv'))
    parser.add_argument('--normalized-csv', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/vindrcxr/train_enriched_normalized.csv'))
    parser.add_argument('--coco-json', type=str, default=os.path.join(PROJECT_ROOT, 'outputs/vindrcxr/train_vindrcxr_all.json'))
    parser.add_argument('--iou-reliability', type=float, default=0.5)
    parser.add_argument('--no-finding-id', type=int, default=14)
    parser.add_argument('--clustering-method', type=str, default='iou', choices=['iou', 'iomin', 'iomin_center_gate'])
    parser.add_argument('--semantic-map', type=str, default='vindrcxr', choices=['vindrcxr'])
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--iomin-threshold', type=float, default=0.5)
    parser.add_argument('--center-ratio-threshold', type=float, default=0.5)
    parser.add_argument('--prior-alpha', type=float, default=1.0)
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--reliability-backend', type=str, default='kalos', choices=['kalos', 'legacy'], help='Reliability calculation backend.')
    parser.add_argument('--reliability-method', type=str, default='shm', choices=['greedy', 'shm', 'ahc'], help='Matching algorithm (kalos backend only).')
#======= for LVIS-specific pipeline arguments (can be ignored for VinDr-CXR) =======================================================
    parser.add_argument("--dataset", type=str, default="vindrcxr", choices=["vindrcxr", "lvis"])
    parser.add_argument("--raw-lvis-json", type=str, default="/media/klee_ro/public_datasets/coco_lvis/lvis_standardized_lvis_val.json")
    parser.add_argument("--lvis-output-dir", type=str, default=os.path.join(PROJECT_ROOT, "outputs/lvis"))
    args = parser.parse_args()
    dataset_name = args.dataset.lower().replace("-", "_")

    if args.dataset == "lvis":
        args.input_csv = os.path.join(args.lvis_output_dir, "lvis_train_enriched_normalized.csv")
        args.normalized_csv = args.input_csv
        args.coco_json = os.path.join(args.lvis_output_dir, "lvis_train_all.json")
        args.semantic_map = "lvis"
        args.no_finding_id = -1

#==================changes based on dataset aware names===========
    iou_rel_str = safe_threshold_str(args.iou_reliability)
    clustering_suffix = build_clustering_suffix(args)
    args.reliability_output_dir = os.path.join(PROJECT_ROOT, "outputs","raters_reliability",f"annotator_reliability_{dataset_name}")
    args.weights_csv = os.path.join(args.reliability_output_dir, f"annotator_reliability_scores_{dataset_name}_iou{iou_rel_str}.csv")
    args.priors_path = os.path.join(PROJECT_ROOT, f"outputs/{dataset_name}/1d_gaussian_softlabels/priors", f"class_priors_{args.gmm_mode}_{clustering_suffix}.json")
    args.output_json = os.path.join(PROJECT_ROOT, f"outputs/{dataset_name}/1d_gaussian_softlabels", f"{args.gmm_mode.upper()}GMM_softLabels_{clustering_suffix}_with_priors.json")

    steps_to_run = args.steps if args.steps else ALL_STEPS
    invalid = [s for s in steps_to_run if s not in ALL_STEPS]
    if invalid:
        print(f"Error: Invalid steps {invalid}. Valid: {ALL_STEPS}")
        sys.exit(1)

    print("\n" + "="*80)
    print("  VinDr-CXR Soft Label Generation Pipeline")
    print("="*80)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  GMM mode: {args.gmm_mode}")
    print(f"  Output: {args.output_json}")
    print("="*80)

    step_builders = {1: build_step1_command, 2: build_step2_command, 3: build_step3_command,
                     4: build_step4_command, 5: build_step5_command}
    failed_step = None
    for step_num in sorted(steps_to_run):
        cmd = step_builders[step_num](args)
        if not run_step(step_num, STEP_NAMES[step_num], cmd, dry_run=args.dry_run):
            failed_step = step_num
            break

    print("\n" + "="*80)
    if failed_step:
        print(f"  Pipeline FAILED at step {failed_step}")
    elif args.dry_run:
        print("  Dry run complete.")
    else:
        print("  Pipeline completed successfully!")
    print("="*80 + "\n")
    sys.exit(1 if failed_step else 0)


if __name__ == "__main__":
    main()
