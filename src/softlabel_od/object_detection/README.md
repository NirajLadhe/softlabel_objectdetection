# Object Detection with Probabilistic Soft Labels

Train Faster R-CNN models to predict spatial uncertainty and class distributions for chest X-ray abnormality detection.

## Purpose

Extend object detection beyond deterministic box-and-class predictions to output:

- **Spatial uncertainty**: Gaussian distribution over bounding box parameters
- **Class distributions**: Probability vectors over pathology classes
- **Calibrated predictions**: Uncertainty estimates that correlate with actual error

The model learns not just *where* abnormalities appear, but *how uncertain* that localization is and *how probable* different semantic interpretations are.

## Architecture

Follows hexagonal architecture with separation between core training logic, model components, and dataset adapters.

### Core Components

**Model factory** (`core/model/`)
- Faster R-CNN with configurable backbones (ResNet50, ResNet101, MobileNet)
- Probabilistic ROI heads for 1D/2D Gaussian box uncertainty
- Class-specific regression outputs (8 values per class: 4 mean deltas + 4 log-variances)

**Training loops** (`core/training/`)
- `train_1d.py`: 1D Gaussian soft label training
- `train_2d.py`: 2D covariance soft label training (experimental)
- `train_hard.py`: Hard label baseline training
- `train_epoch.py`: Shared epoch logic with validation and metric logging

**Evaluation** (`core/evaluation/`)
- Standard detection: mAP, mAP@50, mAP@75, IoU
- Soft classification: top-k accuracy, JS divergence, KL divergence, cosine similarity
- Uncertainty quality: expected IoU, calibration metrics

### Ports

**Dataset port** (`ports/dataset_port.py`)
- Loads soft labels from JSON
- Converts normalized coordinates to pixel space
- Handles hard/soft label modes
- Implements deterministic multilabel stratified splitting (fixed `random_state=42`)

### Adapters

**Dataset adapters** (`adapters/datasets/`)
- `vindrcxr.py` — VinDr-CXR soft/hard label format handlers and methods
- Add other datasets as new modules in this package

**Model components** (`adapters/model_components/`)
- Custom probabilistic ROI heads
- Loss function variants
- Backbone configurations

**Lightning modules** (`adapters/lightning/`)
- PyTorch Lightning wrappers (optional, not primary)

### Utilities

**Losses** (`utils/losses.py`)
- Decoupled mean and variance supervision
- Soft cross-entropy for class distributions
- Class-weighted loss variants

**Schedulers** (`utils/schedulers.py`)
- Multistep learning rate schedules
- Warmup strategies

**Augmentations** (`utils/augmentations.py`)
- Class-aware augmentation probabilities
- Medical imaging appropriate transforms

**Config** (`utils/config.py`)
- YAML-based configuration via `yacs`
- Environment variable overrides

## Model Design

### Probabilistic head architecture

For 1D Gaussian uncertainty, the ROI head outputs per foreground class:
- 4 mean deltas: $[dx, dy, dw, dh]$
- 4 log-variances: $[log (σ²_{dx}), log (σ²_{dy}), log (σ²_{dw}), log (σ²_{dh})]$

Total: `num_classes × 8` regression outputs.

### Why log-variance?

Variance must be positive. Predicting log-variance ensures positivity and provides numerical stability. Standard deviation is recovered as:

```
σ = exp(0.5 × log σ²)
```

The factor 0.5 operates on the logarithm, not the variance itself.

### Bias initialization

Log-variance channels initialize with negative bias (e.g., -4) so the model starts with small uncertainty (~0.135 in normalized space) rather than exploding variances.

### Box parameterization

Follows Faster R-CNN delta encoding. For proposal `(cx_p, cy_p, w_p, h_p)` and target `(cx_t, cy_t, w_t, h_t)`:


$dx = (cx_t - cx_p) / w_p$  
$dy = (cy_t - cy_p) / h_p$  
$dw = log(w_t / w_p)$  
$dh = log(h_t / h_p)$  


### Uncertainty transformation

Ground truth uncertainty is provided in normalized pixel space `[x, y, w, h]`. The model regresses in delta space. Transformation pipeline:

1. **Normalize to pixel space:**
   
   $σ_{x-px} = σ_{x-norm} × W$  
   $σ_{y-px} = σ_{y-norm} × H$  
   

2. **Convert to center coordinates:**
   
   $σ_{cx} = sqrt(σ_x² + (0.25 × σ_w²))$  
   $σ_{cy} = sqrt(σ_y² + (0.25 × σ_h²))$
   

3. **Convert to delta space:**
   
   $σ_{dx} = σ_{cx} / w_p$  
   $σ_{dy} = σ_{cy} / h_p$  
   $σ_{dw} = σ_w / w_t$  
   $σ_{dh} = σ_h / h_t$  
   

4. **Apply floors:**
   - Pixel-space floor: 1.0 px
   - Delta-space floor: 1e-3

Floors prevent numerically singular targets and unstable gradients.

## Loss Design

### Decoupled mean and variance supervision

The model uses separate losses for mean and variance rather than joint Gaussian NLL/KL. This prevents the model from inflating variance to absorb mean error (learned attenuation effect).

**Box mean loss:**
```
L_mean = SmoothL1(predicted_deltas, target_deltas)
```

**Variance loss:**
```
predicted_log_std = 0.5 × predicted_log_vars
target_log_std = log(σ_delta)
L_var = SmoothL1(predicted_log_std, target_log_std)
```

Direct variance supervision is appropriate because ground truth uncertainty is available from multi-annotator aggregation.

### Soft classification loss

For soft class distributions `q = [q_0, q_1, ..., q_C]`:

```
L_cls = -Σ_c q_c log(softmax(logits)_c)
```

Extends standard cross-entropy to probabilistic targets. Supports class weighting for imbalance.

### Why not joint Gaussian NLL?

Joint KL/NLL losses couple mean error and predicted variance. When mean prediction is poor, the model reduces loss by inflating variance. This manifests as:
- Predicted std >> ground truth std
- Plateaued localization quality
- Uncertainty absorbing mean error instead of reflecting annotation variability

Decoupled supervision provides direct anchoring for both mean and variance.

## Class Imbalance Handling

VinDr-CXR exhibits long-tailed class distribution. Mitigation strategies:

### Stratified splitting

Train/validation split uses multilabel iterative stratification to preserve image-level label presence patterns.

### Balanced sampling

Optional balanced sampler ensures rare classes appear during training. (LVIS-style Repeat Factor Sampling)

### Class weighting

Inverse-square-root and Inverse Frquency weighting computed from per-box dominant disease counts:

$w_c = sqrt(N / (K × n_c))$  


where N = total boxes, K = num classes, n_c = count for class c.

Milder than full inverse-frequency weighting, avoids over-correction.

## Training Workflow

1. Load configuration from YAML
2. Resolve dataset type (hard, 1D soft, 2D soft)
3. Build deterministic train/val split
4. Construct model with configurable backbone
5. Load pretrained weights if provided
6. Register class weights for ROI classification
7. Freeze specified layers for fine-tuning
8. Train with SGD and multistep LR schedule
9. Validate with probabilistic and standard metrics
10. Save model and training metrics JSON

## Usage

### Training

**1D soft labels:**
```bash
python -m softlabel_od.scripts.train --config configs/vindr_cxr_1d_config.yaml
```

**Hard label baseline:**
```bash
python -m softlabel_od.scripts.train --config configs/vindr_cxr_hard_label_config.yaml
```

**Lightning (optional):**
```bash
python -m softlabel_od.scripts.train_lightning --config configs/vindr_cxr_1d_config.yaml
```

### Evaluation

**Soft label model:**
```bash
python -m softlabel_od.scripts.evaluate --weights outputs/saved_models/model.pth --gmm_type 1D
```

**Hard label model:**
```bash
python -m softlabel_od.scripts.evaluate --weights outputs/saved_models/model.pth --gmm_type HARD_LABEL
```

### Configuration

Edit YAML files in `configs/`. Key parameters:

**Data:**
- `DATA.ROOT`: Dataset root directory
- `DATA.SOFT_LABEL_FILE_1D`: Path to soft labels JSON
- `DATA.HARD_LABEL_FILE`: Path to hard labels JSON
- `DATA.GMM_TYPE`: `'1D'`, `'2D'`, or `'HARD'`
- `DATA.TEST_SPLIT_RATIO`: Validation split size (e.g., 0.25)

**Model:**
- `MODEL.NUM_CLASSES`: Number of pathology classes + background
- `MODEL.BACKBONE`: `'resnet50'`, `'resnet101'`, `'mobilenet'`
- `MODEL.PROBABILISTIC`: Enable probabilistic head

**Training:**
- `TRAIN.BASE_LR`: Base learning rate
- `TRAIN.BATCH_SIZE`: Batch size
- `TRAIN.NUM_EPOCHS`: Training epochs
- `TRAIN.MILESTONES`: LR decay milestones
- `TRAIN.GAMMA`: LR decay factor

**Paths:**
- `MISC.SAVE_MODEL_PATH`: Model checkpoint directory
- `MISC.RESULTS_CSV`: Metrics output directory
- `MISC.PRETRAINED_WEIGHTS`: Path to pretrained checkpoint for fine-tuning

Override paths via environment variables:
```bash
export SOFTLABEL_DATA_ROOT=/path/to/data
export SOFTLABEL_OUTPUT_DIR=/path/to/outputs
```

## Inference and Post-processing

### Per-class decoding

At inference, the model predicts class logits and class-specific box deltas for every proposal. After decoding:
- One candidate box per foreground class
- Each candidate has its own score (softmax probability)
- Class-specific NMS applied

### Score thresholding

In soft-label mode, a correct ambiguous object may legitimately have top probability 0.4-0.7. Score thresholds must be interpreted carefully.

### Cross-class NMS

For visualization, apply cross-class NMS to suppress overlapping detections regardless of label. Keeps highest-scoring spatial hypothesis and displays full class distribution.

## Evaluation Metrics

### Standard detection

- mAP, mAP@50, mAP@75
- IoU on matched boxes
- Precision, recall

### Soft classification

For matched detections:
- Top-1, top-3 accuracy
- Weighted accuracy using GT soft label mass
- JS divergence between predicted and target distributions
- KL divergence
- Cosine similarity

### Uncertainty quality

**Expected IoU:**
Sample boxes from predicted distribution, compute average IoU to GT. Requires translating delta-space uncertainty to pixel-space before sampling.

**Calibration:**
Compare predicted uncertainty magnitude to actual localization error. Well-calibrated models show positive correlation.

**ECE (Expected Calibration Error):**
Assess whether confidence scores align with actual correctness.

## Reproducibility

- Validation split is deterministic (`random_state=42`)
- No train/val leakage at image level
- Metrics computed on fixed validation set
- Score thresholds and NMS settings must be documented
- Probabilistic metrics should accompany precision/recall and mAP

## Fine-tuning

Load pretrained checkpoint:
```yaml
MISC:
  PRETRAINED_WEIGHTS: 'outputs/saved_models/pretrained.pth'
```

Freeze backbone layers if needed (configured in training script).

## Extension Points

**Add new datasets:**
Implement adapter in `adapters/datasets/`:
- Parse annotations
- Map classes
- Normalize coordinates

**Add new backbones:**
Extend `core/model/factory.py`:
- Register backbone
- Configure FPN if needed

**Add new loss functions:**
Extend `utils/losses.py`:
- Define loss computation
- Register in training loop

**Add new metrics:**
Extend `core/evaluation/metrics.py`:
- Compute metric on predictions and targets
- Add to validation logging


## Key Design Decisions

**Why decoupled mean/variance loss?**
Direct GT uncertainty supervision available. Joint NLL/KL causes learned attenuation where variance absorbs mean error.

**Why log-variance prediction?**
Ensures positivity, provides numerical stability, standard in heteroscedastic uncertainty modeling.

**Why deterministic validation split?**
Ensures reproducible evaluation across runs and experiments.


## Research Applications

- Uncertainty-aware object detection
- Learning from ambiguous annotations
- Reliability-weighted supervision
- Calibration-aware evaluation
- Hard vs soft label training comparisons
- Clustering strategy ablations
