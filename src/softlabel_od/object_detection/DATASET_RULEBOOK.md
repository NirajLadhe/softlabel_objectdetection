# Dataset Rule Book

This document defines the contract contributors must follow when adding a new
dataset adapter to the object-detection package.

## Goal

A new dataset should be addable without changing:

- training loops
- evaluation logic
- samplers
- model code
- shared transforms

The only required changes should be:

- a new dataset adapter in `adapters/datasets/`
- one registration entry in `adapters/datasets/__init__.py`
- a user-specific YAML config selecting the dataset and local paths

## Runtime Contract

All dataset adapters must satisfy `DetectionDatasetPort`.

At runtime the rest of the object-detection package expects:

- standard PyTorch dataset methods: `__getitem__`, `__len__`
- a foreground class mapping via `class_to_idx`
- grouped canonical records via `image_data`
- split-specific image ids via `image_ids`
- canonical label mode via `label_mode`
- image-level presence extraction for shared split/sampler logic

## Required `__getitem__` Output

Every adapter must return:

- `boxes`: `FloatTensor [N, 4]` in pixel `[x1, y1, x2, y2]`
- `labels`: `LongTensor [N]` where background is always mapped to class `0`
- `image_id`: `Tensor [1]`

1D soft-label datasets must additionally return:

- `gt_means`: `FloatTensor [N, 4]` normalized `[x, y, w, h]`
- `gt_stds`: `FloatTensor [N, 4]`
- `gt_class_probs`: `FloatTensor [N, C]`
- `annotation_weight`: `FloatTensor [N]`

2D soft-label datasets must additionally return:

- `gt_vertices`: `FloatTensor [N, 8]`
- `gt_covariances`: list of covariance matrices
- `gt_class_probs`: `FloatTensor [N, C]`

## Canonical Label Modes

The object-detection package supports three canonical label modes:

- `hard`
- `soft_1d`
- `soft_2d`

Configs should prefer `DATA.LABEL_MODE` and `DATA.LABEL_FILE`. Legacy keys
such as `DATA.GMM_TYPE` and mode-specific file paths remain supported for
backwards compatibility.

## Canonical Record Rules

All canonical records must include:

- `image_id`
- `image_width`
- `image_height`
- `finding_id`

### Hard-label records

Hard-label records must include:

- `bbox`: normalized `[x, y, w, h]`
- `class`
- `confidence`
- `renormalized_class_probs`

Rules:

- `renormalized_class_probs` must sum to `1.0`
- `class` must agree with the argmax of `renormalized_class_probs`
- all coordinates are normalized to the source image size

### 1D soft-label records

1D soft-label records must include:

- `spatial.mean`: normalized `[x, y, w, h]`
- `spatial.std`: normalized `[sx, sy, sw, sh]`
- `class_probs`
- `original_annotations`
- `num_annotations_in_cluster`

Rules:

- `class_probs` must sum to `1.0`
- `spatial.mean` and `spatial.std` must be length `4`
- all coordinates are normalized to `[0, 1]`

### 2D soft-label records

2D soft-label records must include:

- `vertices.top_left.mean`
- `vertices.top_left.covariance`
- `vertices.top_right.mean`
- `vertices.top_right.covariance`
- `vertices.bottom_right.mean`
- `vertices.bottom_right.covariance`
- `vertices.bottom_left.mean`
- `vertices.bottom_left.covariance`
- `class_probs`

Rules:

- vertex order is fixed
- coordinates are normalized
- covariance matrices should be symmetric positive semi-definite

## Placeholder / Background Rule

The shared object-detection utilities treat full-image normalized placeholders as
background-only findings and ignore them for stratification and class-weight
computation.

By default a placeholder is identified by:

- `x` close to `0`
- `y` close to `0`
- `w` close to `1`
- `h` close to `1`

Contributors should preserve this convention whenever the dataset uses a
placeholder box for "No Finding" or background-only findings.

## Image-Level Split Rule

Train/validation splits must happen at the image level, never the record level.

The shared implementation uses multilabel iterative stratification and depends on
each adapter exposing image-level presence labels through the port.

## Registry Rule

Every new dataset adapter must be registered under a stable name in
`adapters/datasets/__init__.py`.

Contributor configs should reference only the registry name, for example:

```yaml
DATA:
  DATASET: "vindrcxr"
  ROOT: "/path/to/images"
  LABEL_MODE: "soft_1d"
  LABEL_FILE: "/path/to/labels.json"
  OPTIONS:
    rater_weights_csv: "/path/to/rater_weights.csv"
```

## Recommended Contributor Steps

1. Implement a new adapter subclassing `CanonicalDetectionDataset` when possible.
2. Define the dataset class mapping and image-path resolution.
3. Register the adapter.
4. Create a local YAML config selecting the dataset and local paths.
5. Run a small smoke test by instantiating both train and validation splits.
