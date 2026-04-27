# Soft Label Object Detection

Uncertainty-aware object detection for multi-annotator medical imaging.

This repository has two linked parts:

1. `softlabel_generation`: turns multi-rater annotations into probabilistic (soft) labels
2. `object_detection`: trains detectors on those (soft) labels

The main use case is chest X-ray pathology detection on VinDr-CXR, but the code is structured so new datasets, clustering methods, and model components can be added with limited changes.

## Overview

Many medical images have valid disagreement (both, regional and semantic) between experts. This repository proposes a Deep-Learning framework to embrace these disagreements instead of collapsing it into one aggregated hard target.

The pipeline has two stages:

1. **Soft label generation**
   - groups annotations into finding clusters
   - weights annotators by agreement
   - estimates spatial uncertainty and class probabilities
   - exports canonical hard or soft label files

2. **Probabilistic object detection**
   - trains Faster R-CNN on those label files
   - predicts boxes, class distributions, and uncertainty parameters
   - supports hard labels, 1D soft labels, and 2D soft labels

<details>
<summary><strong>Core ideas</strong></summary>

- **Soft labels**: each finding can carry a class distribution instead of one class
- **Spatial uncertainty**: each finding can carry uncertainty over box parameters
- **Reliability weighting**: annotators can contribute with different weights
- **Dataset modularity**: datasets are selected from config and built through a registry
- **Image-level splitting**: train/validation splitting is deterministic and leakage-safe

</details>

<details>
<summary><strong>Main outputs</strong></summary>

**From `softlabel_generation`:**
- reliability CSVs
- Bayesian class priors
- canonical hard label JSON
- canonical soft label JSON

**From `object_detection`:**
- trained detection checkpoints
- training logs
- evaluation reports

</details>

## Installation Guide

### Requirements

- Python `>= 3.10`
- PyTorch `>= 2.0`
- CUDA is recommended for training

### Install

```bash
(setting up a virtual environment highly recommended)
git clone <repository-url>
cd softlabel_object_detection
pip install -e .
```

### Quick Start

#### 1. Prepare your data

Put your dataset files in a local path and set the paths in a YAML config.

For VinDr-CXR, the default workflow expects:
- images under `data/vinbigdata/train`
- annotation CSVs and generated outputs under `outputs/vindrcxr`

#### 2. Generate soft labels

```bash
python -m softlabel_od.scripts.run_pipeline
```

#### 3. Train a detector

```bash
python -m softlabel_od.scripts.train --config configs/<YOUR-CONFIG>.yaml
```

#### 4. Evaluate a model

```bash
python -m softlabel_od.scripts.evaluate --weights outputs/saved_models/model.pth --gmm_type 1D
```

<details>
<summary><strong>Quick start notes</strong></summary>

- Use `--gmm_type HARD_LABEL` to evaluate a deterministic model trained on Hard-Labels
- Use `python -m softlabel_od.scripts.train_lightning --config ...` for the Pytorch Lightning path
- Config files are user-specific by design. Each contributor can keep local dataset paths in their own YAML files

</details>

## Sub-Package Information

### `softlabel_generation`

Builds probabilistic labels from multi-annotator annotations.

**What it does**
- normalizes raw annotations
- converts them into a common intermediate format
- computes annotator agreement
- clusters overlapping findings
- produces hard or soft label files

<details>
<summary><strong>Structure</strong></summary>

- `core/preprocessing/`
  - CSV normalization
  - CSV to JSON conversion
  - optional data prep helpers

- `core/reliability/`
  - pairwise agreement
  - annotator weighting

- `core/label_assembly/`
  - priors
  - hard label construction
  - 1D and 2D soft label construction

- `ports/`
  - correspondence and clustering interfaces

- `adapters/datasets/`
  - dataset-specific annotation loaders

- `utils/`
  - clustering helpers
  - overlap metrics
  - analysis tools

</details>

<details>
<summary><strong>Important concepts</strong></summary>

- **Placeholder background box**: background-only findings use a normalized full-image placeholder such as `[0, 0, 1, 1]`
- **Clustering methods**: IoU, IoMin, IoMin with center-distance gating
- **Priors**: class-specific uncertainty (bayesian) priors stabilize sparse cases
- **1D vs 2D**:
  - `1D`: diagonal uncertainty over box parameters
  - `2D`: covariance over box vertices

</details>

<details>
<summary><strong>Output Label format (JSON)</strong></summary>

<details>
<summary>1D Soft label</summary>

```json
{
  "image_id": "...",
  "image_width": 1024,
  "image_height": 1024,
  "finding_id": 0,
  "original_annotations": [ ... ],
  "spatial": {
    "mean": [x_mean, y_mean, w_mean, h_mean],
    "std": [x_std, y_std, w_std, h_std]
  },
  "prior_used": "Atelectasis",
  "class_probs": { "Atelectasis": 0.7, "NoFindingInCluster": 0.3 },
  "num_annotations_in_cluster": 2
}
```
</details>

<details>
<summary>Hard label (No-Finding Instance)</summary>

```json
{
    'image_id': "...",
    'image_width': 1024,
    'image_height': 1024,
    'finding_id': 1,
    'original_annotations':[...],
    'bbox': [0.0, 0.0, 1.0, 1.0],
    'class': 'No Finding',
    'confidence': 1.0,
    'extraction_method': 'true_no_finding',
    'original_class_probs': {...},
    'renormalized_class_probs': {'No finding': 1.0},
    'num_annotations': 3
}
```
</details>

<details>
<summary>Hard label (Pathology Instance)</summary>

```json
{
    'image_id': "...",
    'image_width': 1024,
    'image_height': 1024,
    'finding_id': 0,
    'original_annotations': [...],
    'bbox': [x_mean, y_mean, w_mean, h_mean],
    'class': Aortic Enlargement,
    'confidence': 0.668,
    'extraction_method': 'majority_pathology',
    'original_class_probs': { "Aortic Enlargement": 0.668, "NoFindingInCluster": 0.332 },
    'renormalized_class_probs': {"Aortic Enlargement": 0.668},
    'num_annotations': 3
}
```
</details>
Coordinates normalized to [0, 1].
</details>  

<details>
<summary> Soft Label generation Pipeline Flow Diagram </summary>
![Flow diagram: preprocessing and aggregation](images/Pipeline_flow3_light.png)

</details>


### `object_detection`

Trains and evaluates detectors on canonical hard or soft labels.

**What it does**
- loads canonical label files
- builds deterministic train/validation splits
- trains Faster R-CNN variants
- supports probabilistic heads and standard hard-label baselines
- evaluates both standard and soft-label metrics

<details>
<summary><strong>Structure</strong></summary>

- `core/model/`
  - model factory
  - backbone selection
  - detector construction

- `core/training/`
  - training loops for hard, 1D, and 2D modes

- `core/evaluation/`
  - mAP
  - IoU
  - soft-label metrics
  - uncertainty metrics

- `core/datasets/`
  - shared canonical dataset base
  - shared dataset utilities

- `ports/`
  - dataset runtime contract

- `adapters/datasets/`
  - dataset adapters
  - dataset registry and factory

- `adapters/model_components/`
  - probabilistic ROI heads
  - custom model pieces

- `adapters/lightning/`
  - Lightning wrappers

- `utils/`
  - losses
  - schedulers
  - config loading
  - augmentations
  - samplers

</details>


<details>
<summary><strong>Important concepts</strong></summary>

- **Training modes** : `hard`, `soft_1D`, `soft_2D` (experimental)

- **Standard metrics** : mAP, avg_IoU, class_accuracy

- **Soft-label metrics**
  - top-k accuracy
  - JS divergence & KL-Divergence- measure for 'Class Probability Distribution' allignment

- **Uncertainty metrics**
  - expected IoU- measure for regression uncertainty allignment (Standard Deviation and Variances)
  - calibration-oriented error summaries

</details>

### Main callable scripts

These are the main entry points most users will call.

<details>
<summary><strong>Scripts</strong></summary>

- `softlabel_od.scripts.run_pipeline`
  - end-to-end soft label generation pipeline

- `softlabel_od.scripts.train`
  - vanilla PyTorch training

- `softlabel_od.scripts.train_lightning`
  - Lightning training path

- `softlabel_od.scripts.evaluate`
  - model evaluation

</details>

## Command Line Interface

Installed commands:

```bash
softlabel-pipeline
softlabel-train
softlabel-train-lightning
softlabel-evaluate
```

Equivalent module calls:

```bash
python -m softlabel_od.scripts.run_pipeline
python -m softlabel_od.scripts.train --config configs/vindr_cxr_1d_config.yaml
python -m softlabel_od.scripts.train_lightning --config configs/vindr_cxr_1d_config.yaml
python -m softlabel_od.scripts.evaluate --weights outputs/saved_models/model.pth --gmm_type 1D
```

<details>
<summary><strong>Common pipeline options</strong></summary>

```bash
python -m softlabel_od.scripts.run_pipeline --gmm-mode 1d
python -m softlabel_od.scripts.run_pipeline --clustering-method iomin --iomin-threshold 0.5
python -m softlabel_od.scripts.run_pipeline --clustering-method iomin_center_gate --iomin-threshold 0.5 --center-ratio-threshold 0.5
python -m softlabel_od.scripts.run_pipeline --steps 4 5
python -m softlabel_od.scripts.run_pipeline --dry-run
```

</details>

## Extension Guide

### Add a new dataset

Dataset extension is now config-driven on the object detection side.

**Object detection workflow**
[Dataset Rule Book](src/softlabel_od/object_detection/DATASET_RULEBOOK.md)

**Soft label generation workflow**
1. Add dataset-specific preprocessing / loading under `src/softlabel_od/softlabel_generation/adapters/datasets/`
2. Map raw annotations into the expected intermediate format
3. Keep coordinates and class names consistent with the downstream label schema

<details>
<summary><strong>Object detection dataset config example</strong></summary>

```yaml
DATA:
  DATASET: "vindrcxr"
  ROOT: "/path/to/images"
  LABEL_MODE: "soft_1d"
  LABEL_FILE: "/path/to/labels.json"

  OPTIONS:
    image_subdir: "train"
    image_extension: ".png"
    rater_weights_csv: "/path/to/rater_weights.csv"
```

</details>

<details>
<summary><strong>Object detection dataset rules</strong></summary>

A dataset adapter must satisfy the runtime contract in:

- `src/softlabel_od/object_detection/ports/dataset_port.py`

The detailed dataset rules are in:

- `src/softlabel_od/object_detection/DATASET_RULEBOOK.md`

Key points:
- splits happen at the image level
- `__getitem__` must return the standard target fields
- canonical label modes are `hard`, `soft_1d`, `soft_2d`
- placeholder full-image boxes are treated as background-only findings
- new adapters should not require changes to training or evaluation code

</details>

### Add a new clustering method

Extend:

- `src/softlabel_od/softlabel_generation/ports/correspondence.py`

Typical work:
- define the overlap rule
- implement the grouping logic
- keep the one-rater-per-cluster constraint if required

### Add a new loss function

Extend:

- `src/softlabel_od/object_detection/utils/losses.py`

Typical work:
- add the loss
- wire it into the training path that uses it
- document how it applies to hard or soft training modes

### Add a new evaluation metric

Extend:

- `src/softlabel_od/object_detection/core/evaluation/metrics.py`

Typical work:
- compute the metric from predictions and targets
- add it to the reporting path
- keep it consistent across train/validation use if needed

## Repository Structure

<details>
<summary><strong>Tree</strong></summary>

```text
src/softlabel_od/
├── softlabel_generation/
│   ├── core/
│   ├── ports/
│   ├── adapters/
│   └── utils/
├── object_detection/
│   ├── core/
│   ├── ports/
│   ├── adapters/
│   └── utils/
├── shared/
│   └── utils/
└── scripts/
```

</details>

## Citation

```bibtex
Publication Embargo - do not publish
```

## Acknowledgments

This repository builds on ideas from:
- probabilistic object detection
- Faster R-CNN style detection
- multi-annotator learning
- Krippendorff's Alpha
- multilabel iterative stratification