# LRP_METRIC_SPEC.md
## Localisation Recall Precision (LRP) Error

**Implementation:** `object_detection/core/evaluation/lrp.py`  
**References:**  
- Oksuz et al., "LRP: A New Performance Metric for Object Detection", ECCV 2018.  
- Oksuz et al., "One Metric to Measure Them All: LRP for Evaluating Visual Detection Tasks", TPAMI 2021.

---

## Core Calculation

LRP Error is a single scalar that jointly measures localisation quality, false positive rate, and false negative rate of an object detector. It is defined per class at a given confidence score threshold `s` and IoU threshold `τ` as:

```
LRP(X, Y_s) = ( Σ_TP E_Loc(i)  +  N_FP  +  N_FN ) / ( N_TP + N_FP + N_FN )
```

where:
- `X` is the set of ground truth boxes for the class
- `Y_s` is the set of detections surviving confidence threshold `s`
- `N_TP`, `N_FP`, `N_FN` are the counts of true positives, false positives, and false negatives at threshold `s`
- `E_Loc(i) = (1 - IoU(pred_i, gt_i)) / (1 - τ)` is the normalised localisation error for each TP, in `[0, 1]`

Each TP is penalised by its normalised localisation error. Each FP and FN contributes a penalty of exactly 1. The sum is normalised by the total number of contributing boxes.

**Optimal LRP (oLRP)** is the minimum achievable LRP over all candidate confidence thresholds:

```
oLRP = min over s of LRP(X, Y_s)
```

The threshold `s*` achieving oLRP is the class-specific optimal confidence threshold. The three components at `s*` are reported separately:

- `oLRP_IoU` — mean localisation error of TPs at `s*` → `1 - oLRP_IoU` = mean IoU
- `oLRP_FP`  — precision error at `s*` → `1 - oLRP_FP` = precision
- `oLRP_FN`  — recall error at `s*` → `1 - oLRP_FN` = recall

**Mean Optimal LRP (moLRP)** is the mean of `oLRP` over all foreground classes with at least one GT box, analogous to mAP.

Matching between predictions and GT boxes follows the greedy score-ordered protocol used by AP: predictions are sorted by score descending, each prediction is matched to the highest-IoU unmatched GT box with `IoU ≥ τ`. Each GT box can be matched at most once. Unmatched predictions are FP, unmatched GT boxes are FN.

The threshold sweep uses 101 evenly spaced candidate values `s ∈ {0.00, 0.01, ..., 1.00}`, matching the exhaustive search proposed in the ECCV 2018 paper. Matching and accumulation are performed at the dataset level — all images are pooled per class before computing LRP at each threshold.

All LRP values are in `[0, 1]`. Lower is better. LRP = 0 means perfect detection. LRP = 1 means all detections are wrong or all GT objects are missed.

---

## Input Format

The metric consumes the same `predictions` and `ground_truths` list-of-dicts format used by all other metrics in `metrics.py`.

### `predictions`
A Python list of dicts, one dict per image, in the same order as `ground_truths`.

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `boxes` | `np.ndarray` | `[N, 4]` | Predicted bounding boxes in pixel coordinates `[x1, y1, x2, y2]` |
| `scores` | `np.ndarray` | `[N]` | Confidence scores in `[0, 1]`, one per detection |
| `labels` | `np.ndarray` | `[N]` | Integer class IDs, one per detection |

`N` is the number of detections for that image. `N` can be 0 (empty array).

### `ground_truths`
A Python list of dicts, one dict per image.

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `boxes` | `np.ndarray` | `[M, 4]` | GT bounding boxes in pixel coordinates `[x1, y1, x2, y2]` |
| `labels` | `np.ndarray` | `[M]` | Integer class IDs, one per GT box |

`M` is the number of GT boxes for that image. `M` can be 0.

Probabilistic fields (`bbox_stds`, `class_probs`, `gt_class_probs`) are not used by LRP and are ignored if present.

---

## Output Format

`compute_lrp` returns a single Python dict.

### Top-level keys

| Key | Type | Description |
|-----|------|-------------|
| `moLRP` | `float` | Mean oLRP across all foreground classes with at least one GT box. Range `[0, 1]`, lower is better. `nan` if no valid classes. |
| `moLRP_IoU` | `float` | Mean of per-class `oLRP_IoU` at their respective `s*`. |
| `moLRP_FP` | `float` | Mean of per-class `oLRP_FP` at their respective `s*`. It equals mean precision error. |
| `moLRP_FN` | `float` | Mean of per-class `oLRP_FN` at their respective `s*`. It equals mean recall error. |
| `per_class` | `dict` | Per-class breakdown keyed by integer class ID. |

### `per_class[class_id]` keys

| Key | Type | Description |
|-----|------|-------------|
| `oLRP` | `float` | Optimal LRP for this class. `nan` if no GT and no predictions. |
| `oLRP_IoU` | `float` | Mean normalised localisation error of TPs at `s*`. `nan` if no TPs at `s*`. |
| `oLRP_FP` | `float` | Precision error at `s*`. It equals `N_FP / (N_TP + N_FP)`. |
| `oLRP_FN` | `float` | Recall error at `s*`. It equals `N_FN / N_GT_total`. |
| `optimal_threshold` | `float` | The confidence score threshold `s*` achieving `oLRP`. |
| `lrp_curve` | `list[float]` | LRP value at each of the 101 candidate thresholds. Length 101. Used for diagnostic plotting. Excluded from saved JSON by default. |

Classes with no GT boxes in the dataset are excluded from `moLRP` aggregation. Classes where the entire sweep produces `nan` (no GT and no predictions) return `nan` for all scalar fields.

---

## Hyperparameters

All parameters are arguments to `compute_lrp`.

| Parameter | Type | Default | Effect |
|-----------|------|---------|--------|
| `iou_threshold` | `float` | `0.5` | Minimum IoU for a detection to count as a TP (τ in the paper). Higher values require tighter bounding boxes. Standard values are 0.5 (PASCAL VOC) and 0.75 (strict COCO). Affects both matching and E_Loc normalisation. |
| `score_thresholds` | `np.ndarray` | `np.linspace(0.0, 1.0, 101)` | Candidate confidence thresholds for the oLRP sweep. The default produces 101 values from 0.00 to 1.00 in steps of 0.01, matching the exhaustive search in the paper. Coarser steps (e.g. 11 values) reduce precision of `s*`. Finer steps (e.g. 1001 values) give negligible improvement at 10× cost. |
| `background_index` | `int` or `None` | `0` | Class ID to exclude from evaluation. Set to `0` to exclude the background or NoFinding class (standard practice, matches `compute_map` convention). Set to `None` to include all classes. |

### Effect of changing `iou_threshold`

Lowering τ (e.g. to 0.3) makes matching more permissive — loose boxes pass as TP — which reduces `oLRP_FP` and `oLRP_FN` and lowers overall `oLRP`. It also changes the E_Loc normalisation denominator `(1 - τ)`, making the localisation penalty scale differently. Raising τ (e.g. to 0.75) makes matching stricter — many boxes that were TP at 0.5 become FP — which increases all error components.

### Effect of changing `score_thresholds`

Affects only the precision of `s*` and the granularity of `lrp_curve`. Does not affect the mathematical definition of oLRP — only how finely it is searched. The paper's recommended 101-step sweep is used by default.

### Effect of changing `background_index`

Including background (setting to `None`) would artificially lower `moLRP` since background is an easy negative class with few GT boxes. Excluding it (default `0`) ensures `moLRP` reflects only foreground pathology detection performance, consistent with standard mAP reporting.
