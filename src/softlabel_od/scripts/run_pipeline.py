#!/usr/bin/env python3
"""
Soft Label Generation Pipeline — dataset-agnostic orchestrator.

Usage:
    python -m softlabel_od.scripts.run_pipeline --config configs/softlabel_pipeline/vindrcxr.yaml
    python -m softlabel_od.scripts.run_pipeline --config configs/softlabel_pipeline/texbig.yaml --steps 3 4 5
    python -m softlabel_od.scripts.run_pipeline --config configs/softlabel_pipeline/vindrcxr.yaml --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
import yaml
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace


# ──────────────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────────────

def _to_ns(obj):
    """Recursively convert dicts to SimpleNamespace for attribute-style access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(i) for i in obj]
    return obj


def load_pipeline_config(path: str) -> SimpleNamespace:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def _get(ns, attr, default=None):
    """Safely read an optional config attribute."""
    return getattr(ns, attr, default)


# ──────────────────────────────────────────────────────────────────────────────
# Path construction
# ──────────────────────────────────────────────────────────────────────────────

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))


def safe_threshold_str(value: float) -> str:
    return str(value).replace(".", "_")


def build_clustering_suffix(cfg) -> str:
    method = cfg.PIPELINE.clustering_method
    if method == "iou":
        return f"iou{safe_threshold_str(cfg.PIPELINE.iou_threshold)}"
    elif method == "iomin":
        return f"iomin{safe_threshold_str(cfg.PIPELINE.iomin_threshold)}"
    elif method == "iomin_center_gate":
        return (
            f"iomin_cg{safe_threshold_str(cfg.PIPELINE.iomin_threshold)}"
            f"_{safe_threshold_str(cfg.PIPELINE.center_ratio_threshold)}"
        )
    return method


def build_paths(cfg) -> SimpleNamespace:
    """Derive all intermediate file paths from the config."""
    dataset_name = cfg.DATASET.name
    gmm_mode = cfg.PIPELINE.gmm_mode
    iou_rel_str = safe_threshold_str(cfg.PIPELINE.iou_reliability)
    clustering_suffix = build_clustering_suffix(cfg)

    input_csv = Path(PROJECT_ROOT) / cfg.PATHS.input_csv
    output_dir = Path(PROJECT_ROOT) / cfg.PATHS.output_dir
    reliability_dir = Path(PROJECT_ROOT) / cfg.PATHS.reliability_dir

    # Normalized CSV: same directory as input, "_normalized" suffix
    normalized_csv = input_csv.parent / f"{input_csv.stem}_normalized.csv"
    coco_json = output_dir / f"train_{dataset_name}_all.json"
    weights_csv = reliability_dir / f"annotator_reliability_scores_{dataset_name}_iou{iou_rel_str}.csv"
    priors_dir = output_dir / f"{gmm_mode}_gaussian_softlabels" / "priors"
    priors_path = priors_dir / f"class_priors_{gmm_mode}_{clustering_suffix}.json"
    output_json = (
        output_dir / f"{gmm_mode}_gaussian_softlabels"
        / f"{gmm_mode.upper()}GMM_softLabels_{clustering_suffix}_with_priors.json"
    )

    return SimpleNamespace(
        input_csv=str(input_csv),
        normalized_csv=str(normalized_csv),
        coco_json=str(coco_json),
        reliability_dir=str(reliability_dir),
        weights_csv=str(weights_csv),
        priors_path=str(priors_path),
        output_json=str(output_json),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Step command builders
# ──────────────────────────────────────────────────────────────────────────────

def build_step1_command(cfg, paths) -> list:
    return [
        sys.executable, "-m",
        "softlabel_od.softlabel_generation.core.preprocessing.manipulate_csv",
        "--input-csv", paths.input_csv,
        "--output-csv", paths.normalized_csv,
        "--adapter", cfg.DATASET.name,
    ]


def build_step2_command(cfg, paths) -> list:
    cmd = [
        sys.executable, "-m",
        "softlabel_od.softlabel_generation.core.preprocessing.convert_csv_to_json",
        "--input-csv", paths.normalized_csv,
        "--output-json", paths.coco_json,
        "--image-file-extension", _get(cfg.DATASET, "image_file_extension", ""),
    ]
    no_finding_id = _get(cfg.DATASET, "no_finding_id", None)
    if no_finding_id is not None:
        cmd.extend(["--no-finding-id", str(no_finding_id)])
    return cmd


def build_step3_command(cfg, paths) -> list:
    no_finding_id = _get(cfg.DATASET, "no_finding_id", 99)
    return [
        sys.executable, "-m",
        "softlabel_od.softlabel_generation.core.reliability.calculate_reliability",
        "--iou", str(cfg.PIPELINE.iou_reliability),
        "--no-finding-id", str(no_finding_id if no_finding_id is not None else 99),
        "--annotation-path", paths.coco_json,
        "--output-dir", paths.reliability_dir,
        "--backend", _get(cfg.PIPELINE, "reliability_backend", "kalos"),
        "--method", _get(cfg.PIPELINE, "reliability_method", "shm"),
        "--dataset-name", cfg.DATASET.name,
    ]


def build_step4_command(cfg, paths) -> list:
    gmm_mode = cfg.PIPELINE.gmm_mode
    if gmm_mode == "1d":
        module = "softlabel_od.softlabel_generation.core.label_assembly.priors.calculate_priors_1d"
    elif gmm_mode == "2d":
        module = "softlabel_od.softlabel_generation.core.label_assembly.priors.calculate_priors_2d"
    elif gmm_mode == "2d_theta":                                                # ADD
        module = "softlabel_od.softlabel_generation.theta_2d.calculate_priors_theta"            # ADD
    else:
        raise ValueError(f"Invalid gmm_mode: {gmm_mode!r}")

    no_finding_labels = _get(cfg.DATASET, "no_finding_labels", [])
    cmd = [
        sys.executable, "-m", module,
        "--train-csv", paths.normalized_csv,
        "--raters-json", paths.coco_json,
        "--weights-csv", paths.weights_csv,
        "--output", paths.priors_path,
        "--alpha", str(cfg.PIPELINE.prior_alpha),
        "--min-samples", str(cfg.PIPELINE.min_samples),
        "--clustering-method", cfg.PIPELINE.clustering_method,
        "--iou-threshold", str(cfg.PIPELINE.iou_threshold),
        "--iomin-threshold", str(cfg.PIPELINE.iomin_threshold),
        "--center-ratio-threshold", str(cfg.PIPELINE.center_ratio_threshold),
        "--iou-reliability", str(cfg.PIPELINE.iou_reliability),
        "--dataset-name", cfg.DATASET.name,
    ]
    semantic_map = _get(cfg.PIPELINE, "semantic_map", None)
    if semantic_map:
        cmd.extend(["--semantic-map", semantic_map])
    if no_finding_labels:
        cmd.extend(["--no-finding-labels"] + list(no_finding_labels))
    max_images = _get(cfg.PIPELINE, "max_images", None)
    if max_images is not None:
        cmd.extend(["--max-images", str(max_images)])
    return cmd


def build_step5_command(cfg, paths) -> list:
    gmm_mode = cfg.PIPELINE.gmm_mode
    if gmm_mode == "1d":
        module = "softlabel_od.softlabel_generation.core.label_assembly.label_constructor.gmm_1d"
    elif gmm_mode == "2d":
        module = "softlabel_od.softlabel_generation.core.label_assembly.label_constructor.gmm_2d"
    elif gmm_mode == "2d_theta":                                                # ADD
        module = "softlabel_od.softlabel_generation.theta_2d.gmm_theta"          # ADD
    else:
        raise ValueError(f"Invalid gmm_mode: {gmm_mode!r}")

    no_finding_labels = _get(cfg.DATASET, "no_finding_labels", [])
    absent_rater_vote = _get(cfg.DATASET, "absent_rater_vote", False)
    cmd = [
        sys.executable, "-m", module,
        "--train-csv", paths.normalized_csv,
        "--raters-json", paths.coco_json,
        "--weights-csv", paths.weights_csv,
        "--priors", paths.priors_path,
        "--output", paths.output_json,
        "--alpha", str(cfg.PIPELINE.prior_alpha),
        "--clustering-method", cfg.PIPELINE.clustering_method,
        "--iou-threshold", str(cfg.PIPELINE.iou_threshold),
        "--iomin-threshold", str(cfg.PIPELINE.iomin_threshold),
        "--center-ratio-threshold", str(cfg.PIPELINE.center_ratio_threshold),
        "--iou-reliability", str(cfg.PIPELINE.iou_reliability),
        "--image-id-key", _get(cfg.DATASET, "image_id_key", "file_name"),
    ]
    semantic_map = _get(cfg.PIPELINE, "semantic_map", None)
    if semantic_map:
        cmd.extend(["--semantic-map", semantic_map])
    if no_finding_labels:
        cmd.extend(["--no-finding-labels"] + list(no_finding_labels))
    if absent_rater_vote:
        cmd.append("--absent-rater-vote")
    else:
        cmd.append("--no-absent-rater-vote")
    return cmd


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

STEP_NAMES = {
    1: "Normalize CSV",
    2: "Convert CSV to COCO JSON",
    3: "Calculate Annotator Reliability (K-alpha)",
    4: "Calculate Class-Specific Priors",
    5: "Generate GMM Soft Labels",
}
ALL_STEPS = list(STEP_NAMES.keys())

STEP_BUILDERS = {
    1: build_step1_command,
    2: build_step2_command,
    3: build_step3_command,
    4: build_step4_command,
    5: build_step5_command,
}


def run_step(step_num: int, step_name: str, command: list, dry_run: bool = False) -> bool:
    print(f"\n{'='*80}")
    print(f"  STEP {step_num}/{len(STEP_NAMES)}: {step_name}")
    print(f"{'='*80}")
    print(f"  Command: {' '.join(command)}")
    if dry_run:
        print("  [DRY RUN] Skipping execution.")
        return True
    print()
    start = time.time()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = str(timedelta(seconds=int(time.time() - start)))
    if result.returncode != 0:
        print(f"\n  FAILED: Step '{step_name}' exited with code {result.returncode}")
        return False
    print(f"\n  Step '{step_name}' completed successfully. ({elapsed})")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Soft Label Generation Pipeline — dataset-agnostic orchestrator."
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to pipeline YAML config, e.g. configs/softlabel_pipeline/vindrcxr.yaml",
    )
    parser.add_argument("--steps", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    config_path = cli.config if os.path.isabs(cli.config) else os.path.join(PROJECT_ROOT, cli.config)
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    cfg = load_pipeline_config(config_path)
    paths = build_paths(cfg)

    steps_to_run = cli.steps if cli.steps else ALL_STEPS
    invalid = [s for s in steps_to_run if s not in ALL_STEPS]
    if invalid:
        print(f"Error: Invalid steps {invalid}. Valid: {ALL_STEPS}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"  Soft Label Generation Pipeline — {cfg.DATASET.name}")
    print("=" * 80)
    print(f"  Config:      {config_path}")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  GMM mode:    {cfg.PIPELINE.gmm_mode}")
    print(f"  Output:      {paths.output_json}")
    print("=" * 80)

    failed_step = None
    for step_num in sorted(steps_to_run):
        cmd = STEP_BUILDERS[step_num](cfg, paths)
        if not run_step(step_num, STEP_NAMES[step_num], cmd, dry_run=cli.dry_run):
            failed_step = step_num
            break

    print("\n" + "=" * 80)
    if failed_step:
        print(f"  Pipeline FAILED at step {failed_step}.")
    elif cli.dry_run:
        print("  Dry run complete.")
    else:
        print("  Pipeline completed successfully!")
    print("=" * 80 + "\n")
    sys.exit(1 if failed_step else 0)


if __name__ == "__main__":
    main()