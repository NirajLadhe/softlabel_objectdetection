# Object Detection with Probabilistic Soft Labels

Train Faster R-CNN models to predict spatial uncertainty and class distributions for chest X-ray abnormality detection.

## Purpose

Extend object detection beyond deterministic box-and-class predictions to output:

- **Spatial uncertainty**: Gaussian distribution over bounding box parameters
- **Class distributions**: Probability vectors over pathology classes
- **Calibrated predictions**: Uncertainty estimates that correlate with actual error

The model learns not just *where* abnormalities appear, but *how uncertain* that localization is and *how probable* different semantic interpretations are.


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

### Why not joint Gaussian NLL/ KL Loss?

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

Inverse Square Root Frequency (less aggressive): $w_c = sqrt(N / (K × n_c))$  
or
Inverse Frequency: $w_c = N / (K × n_c)$


where N = total boxes, K = num classes, n_c = count for class c.

Milder than full inverse-frequency weighting, avoids over-correction.


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


## Fine-tuning

Load pretrained checkpoint:
```yaml
MISC:
  PRETRAINED_WEIGHTS: 'outputs/saved_models/pretrained.pth'
```

Freeze backbone layers if needed (configured in training script).
```yaml
MISC:
   FROZEN_LAYER: [backbone]
```


## Key Design Decisions

**Why decoupled mean/variance loss?**
Direct GT uncertainty supervision available. Joint NLL/KL causes learned attenuation where variance absorbs mean error.

**Why log-variance prediction?**
Ensures positivity, provides numerical stability, standard in heteroscedastic uncertainty modeling.

**Why deterministic validation split?**
Ensures reproducible evaluation across runs and experiments.
