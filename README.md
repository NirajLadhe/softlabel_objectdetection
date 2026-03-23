# Soft Label Object Detection

Uncertainty-aware object detection for multi-annotator medical imaging. Converts expert disagreement into probabilistic training targets and trains detectors to predict spatial uncertainty and class distributions.

## Overview

Medical imaging annotations exhibit legitimate expert disagreement due to ambiguous boundaries, overlapping disease categories, and varying clinical thresholds. This repository treats disagreement as structured uncertainty rather than noise to be removed.

**Pipeline:**
1. **Soft label generation**: Aggregate multi-rater annotations into Gaussian spatial uncertainty and class probability distributions
2. **Probabilistic detection**: Train Faster R-CNN models to predict uncertainty-aware localizations and soft class labels

**Primary dataset:** VinDr-CXR (chest X-ray abnormality detection)

## Architecture

### Package Structure

```
src/softlabel_od/
├── softlabel_generation/          # Soft label generation from multi-rater annotations
│   ├── core/
│   │   ├── reliability/           # K-alpha, I/O helpers
│   │   ├── label_assembly/        # Priors, 1D/2D assembly
│   │   └── preprocessing/         # CSV normalize, CSV→COCO JSON, enrich, extract annotators
│   ├── ports/                     # Correspondence solver
│   ├── adapters/datasets/       # Dataset adapter hooks (VinDr-CXR-oriented)
│   └── utils/                     # Clustering, anchor analysis
│
├── object_detection/              # Probabilistic object detection training
│   ├── core/                      # Model, training loops, evaluation
│   ├── ports/                     # Dataset port
│   ├── adapters/                  # Datasets, model heads, Lightning, evaluators
│   └── utils/                     # Losses, schedulers, config, augmentations
│
├── shared/                        # Shared utilities across packages
│   └── utils/                     # e.g. serialization
│
└── scripts/                       # CLI entry points (orchestration)
    ├── run_pipeline.py            # End-to-end soft label generation
    ├── gmm_1d.py, gmm_2d.py       # Soft label generation steps
    ├── calculate_priors_*.py      # Prior estimation
    ├── calculate_pairwise_reliability.py
    ├── train.py, train_lightning.py, evaluate.py
```

### Design Patterns

**Adapter pattern**: Dataset extensions (VinDr-CXR implemented; add more under `adapters/datasets/`)

**Composition pattern**: Losses and model heads

**Strategy pattern**: Label assembly methods (1D Gaussian, 2D covariance, hard labels)

**Ports and adapters**: Core logic isolated from dataset-specific and model-specific implementations

## Installation

### Requirements

- Python ≥3.10
- PyTorch ≥2.0
- CUDA (recommended for training)

### Install

```bash
git clone <repository-url>
cd softlabel_object_detection
pip install -e .
```

### Dependencies

Automatically installed via `pyproject.toml`

## Quick Start

### 1. Prepare data

Place VinDr-CXR annotations at `./data/vinbigdata/train_enriched.csv` or specify path via config.

### 2. Generate soft labels

```bash
python -m softlabel_od.scripts.run_pipeline
```

Outputs: `./outputs/1DGMM_softLabels_iou0_5_with_priors.json`

### 3. Train detector

```bash
python -m softlabel_od.scripts.train --config configs/vindr_cxr_1d_config.yaml
```

### 4. Evaluate

```bash
python -m softlabel_od.scripts.evaluate \
  --weights outputs/saved_models/model.pth \
  --gmm_type 1D
```

## Soft Label Generation

Converts multi-rater annotations into probabilistic labels via five stages:

<details>
<summary>Pipeline stages</summary>

1. **Normalize annotations**: Convert to normalized coordinates
2. **Convert to COCO JSON**: Unified format for processing
3. **Compute reliability**: Pairwise Krippendorff's Alpha per annotator
4. **Estimate priors**: Class-specific uncertainty priors from multi-annotated examples
5. **Generate soft labels**: Weighted Gaussian spatial uncertainty + class distributions
</details>

<details>
<summary>Key features</summary>

**Reliability weighting**: Annotators weighted by pairwise inter-annotator agreement (Krippendorff's Alpha)

**Unified absence/presence**: `"No Finding"` receives placeholder box `[0, 0, 1, 1]` for uniform reliability computation

**Constraint-aware clustering**: Enforces one annotation per rater per finding cluster

**Bayesian stabilization**: Class-specific priors regularize uncertainty for sparse annotations (VinDr-CXR has ≤3 annotators per image)

**Clustering strategies**: IoU, IoMin, IoMin with center-distance gating
</details>

<details>
<summary>Output format</summary>

Each soft label contains:
```json
{
  "image_id": "...",
  "spatial": {
    "mean": [x_mean, y_mean, w_mean, h_mean],
    "std": [std_x, std_y, std_w, std_h]
  },
  "semantic": {
    "class_distribution": [p_0, p_1, ..., p_C]
  },
  "cluster_size": 3,
  "annotations": [...]
}
```

Coordinates normalized to [0, 1].
</details>

<details>
<summary>Configuration</summary>

**IoMin clustering:**
```bash
python -m softlabel_od.scripts.run_pipeline \
  --clustering-method iomin \
  --iomin-threshold 0.5
```

**Center-distance gating:**
```bash
python -m softlabel_od.scripts.run_pipeline \
  --clustering-method iomin_center_gate \
  --iomin-threshold 0.5 \
  --center-ratio-threshold 0.5
```

**2D uncertainty (experimental):**
```bash
python -m softlabel_od.scripts.run_pipeline --gmm-mode 2d
```

**Partial execution:**
```bash
python -m softlabel_od.scripts.run_pipeline --steps 4 5  # priors + labels only
```

See `src/softlabel_od/softlabel_generation/README.md` for detailed documentation.
</details>

## Object Detection Training

Trains Faster R-CNN models with probabilistic ROI heads to predict spatial uncertainty and class distributions.

<details>
<summary>Model architecture</summary>

**Base detector**: Faster R-CNN with configurable backbones (ResNet50, ResNet101, MobileNet)

**Probabilistic head**: Per-class outputs (8 values per class):
- 4 mean deltas: `[dx, dy, dw, dh]`
- 4 log-variances: `[log σ²_dx, log σ²_dy, log σ²_dw, log σ²_dh]`

**Uncertainty parameterization**:
- 1D (primary): Diagonal covariance, per-coordinate independent
- 2D (experimental): Full covariance matrices at vertices
</details>

<details>
<summary>Loss design</summary>

**Decoupled mean and variance supervision** (not joint Gaussian NLL):

```
L_mean = SmoothL1(predicted_deltas, target_deltas)
L_var = SmoothL1(predicted_log_std, target_log_std)
L_cls = -Σ_c q_c log(softmax(logits)_c)
```

Decoupling prevents learned attenuation where variance absorbs mean error. Appropriate when ground truth uncertainty is available from annotation aggregation.
</details>

<details>
<summary>Training modes</summary>

**1D soft labels:**
```bash
python -m softlabel_od.scripts.train --config configs/vindr_cxr_1d_config.yaml
```

**Hard label baseline:**
```bash
python -m softlabel_od.scripts.train --config configs/vindr_cxr_hard_label_config.yaml
```

**PyTorch Lightning (optional):**
```bash
python -m softlabel_od.scripts.train_lightning --config configs/vindr_cxr_1d_config.yaml
```
</details>

<details>
<summary>Evaluation metrics</summary>

**Standard detection**: mAP, mAP@50, mAP@75, IoU

**Soft classification**: top-k accuracy, JS divergence, KL divergence, cosine similarity

**Uncertainty quality**: expected IoU, calibration (correlation between predicted uncertainty and actual error)
</details>

<details>
<summary>Configuration</summary>

Edit YAML files in `configs/`. Key parameters:

```yaml
DATA:
  ROOT: './data/vinbigdata'
  SOFT_LABEL_FILE_1D: './outputs/1DGMM_softLabels_iou0_5_with_priors.json'
  GMM_TYPE: '1D'  # '1D', '2D', or 'HARD'
  TEST_SPLIT_RATIO: 0.25

MODEL:
  NUM_CLASSES: 15
  BACKBONE: 'resnet50'
  PROBABILISTIC: True

TRAIN:
  BASE_LR: 0.0001
  BATCH_SIZE: 4
  NUM_EPOCHS: 100
  MILESTONES: [40, 70]
  GAMMA: 0.5

MISC:
  SAVE_MODEL_PATH: './outputs/saved_models'
  PRETRAINED_WEIGHTS: ''  # Path for fine-tuning
```

Override via environment variables:
```bash
export SOFTLABEL_DATA_ROOT=/path/to/data
export SOFTLABEL_OUTPUT_DIR=/path/to/outputs
```

See `src/softlabel_od/object_detection/README.md` for detailed documentation.

</details>  


## Configuration Management

All configuration via YAML files. No hard-coded paths.

**Default locations:**
- Data: `./data/`
- Outputs: `./outputs/`
- Configs: `./configs/`

**Environment variable overrides:**
- `SOFTLABEL_DATA_ROOT`: Dataset root
- `SOFTLABEL_OUTPUT_DIR`: Output directory

**Config files:**
- `vindr_cxr_1d_config.yaml`: 1D soft label training
- `vindr_cxr_2d_config.yaml`: 2D soft label training (experimental)
- `vindr_cxr_hard_label_config.yaml`: Hard label baseline

## Command-Line Interface

Entry point scripts installed as commands:

```bash
softlabel-pipeline          # Run soft label generation pipeline
softlabel-train             # Train object detection model
softlabel-train-lightning   # Train with PyTorch Lightning
softlabel-evaluate          # Evaluate trained model
```

Or use module invocation:
```bash
python -m softlabel_od.scripts.run_pipeline
python -m softlabel_od.scripts.train --config configs/vindr_cxr_1d_config.yaml
```

## Reproducibility

**Deterministic validation split**: Multilabel stratified split with `random_state=42`

**Fixed evaluation set**: Metrics computed on same validation set across runs

**Configuration tracking**: All hyperparameters specified in YAML

**Metric logging**: Training metrics saved to JSON

**Model checkpointing**: Best and final models saved

## Research Applications

- Uncertainty-aware object detection
- Multi-annotator learning
- Annotation quality analysis
- Clustering strategy ablations
- Hard vs soft label comparisons
- Calibration-aware evaluation
- Reliability-weighted supervision

## Extension Guide

### Add new dataset

1. Implement adapter in `softlabel_generation/adapters/datasets/`:
   - Parse annotations
   - Map classes
   - Normalize coordinates

2. Implement adapter in `object_detection/adapters/datasets/`:
   - Load soft labels
   - Convert to model format

3. Add config file in `configs/`

### Add new clustering strategy

Extend `softlabel_generation/ports/correspondence.py`:
- Define overlap metric
- Implement hierarchical grouping
- Enforce participant constraint

### Add new loss function

Extend `object_detection/utils/losses.py`:
- Define loss computation
- Register in training loop

### Add new evaluation metric

Extend `object_detection/core/evaluation/metrics.py`:
- Compute metric on predictions and targets
- Add to validation logging

## Key Design Decisions

### Why treat disagreement as signal?

Medical imaging exhibits legitimate expert disagreement. Forcing consensus discards information about:
- Ambiguous boundaries
- Overlapping disease categories
- Varying clinical thresholds

Probabilistic labels preserve this structure.

### Why decoupled mean/variance loss?

Joint Gaussian NLL/KL couples mean error and predicted variance. Model can reduce loss by inflating variance (learned attenuation). Decoupled supervision provides direct anchoring when ground truth uncertainty is available.

### Why Bayesian priors?

VinDr-CXR has ≤3 annotators per image. Naive variance estimation from 1-2 samples is unstable. Class-specific priors provide regularization.


## Citation

If you use this repository in your research, please cite:

```bibtex
Publication Embargo - do not publish
```

## Acknowledgments

This repository draws from:
- Faster R-CNN proposal-based detection
- Heteroscedastic uncertainty modeling
- Probabilistic object detection
- Soft-label supervision under annotation ambiguity
- Multilabel iterative stratification
- Krippendorff's Alpha for inter-annotator agreement
- NuCLS-style annotation consolidation

## Reading Guide

### For users

1. This README (overview and quick start)
2. `src/softlabel_od/softlabel_generation/README.md` (soft label generation)
3. `src/softlabel_od/object_detection/README.md` (detection training)
4. Config files in `configs/` (parameter reference)

### For developers

1. `scripts/run_pipeline.py` (orchestration)
2. `softlabel_generation/core/reliability_pairwise.py` (agreement)
3. `softlabel_generation/utils/clustering.py` (annotation grouping)
4. `scripts/gmm_1d.py` (label generation; module `softlabel_od.scripts.gmm_1d`)
5. `softlabel_generation/core/preprocessing/` (CSV normalize, CSV→JSON)
6. `object_detection/train.py` (training entry point)
7. `object_detection/core/training/train_1d.py` (training loop)
8. `object_detection/core/model/factory.py` (model construction)
9. `object_detection/utils/losses.py` (loss functions)