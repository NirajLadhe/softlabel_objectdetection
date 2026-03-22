# Soft Label Generation

Generate reliability-weighted Gaussian uncertainty labels from multi-annotator medical imaging annotations.

## Purpose

Convert multi-rater annotations into probabilistic soft labels that preserve expert disagreement as structured uncertainty rather than forcing consensus. Each generated label contains:

- **Spatial uncertainty**: Gaussian distribution over bounding box parameters (x, y, w, h)
- **Semantic uncertainty**: Probability distribution over pathology classes
- **Reliability weighting**: Annotator contributions weighted by pairwise Krippendorff's Alpha
- **Bayesian stabilization**: Class-specific priors regularize uncertainty estimates for sparse annotations

## Architecture

### Core Components

**Reliability computation** (`core/reliability/`)
- Pairwise inter-annotator agreement via Krippendorff's Alpha
- Handles both presence and absence annotations uniformly
- Produces normalized rater weights

**Label assembly** (`core/label_assembly/`)
- Weighted mean and variance computation for bounding boxes
- Class distribution generation from weighted votes
- Prior estimation and application for uncertainty regularization

**Preprocessing** (`core/preprocessing/`)
- `manipulate_csv.py` — normalize VinDr-CXR CSV coordinates
- `convert_csv_to_json.py` — CSV → COCO-style JSON
- `enrich_csv.py`, `extract_annotators.py` — optional data prep helpers

### Ports

**Correspondence solver** (`ports/correspondence.py`)
- Finding-level clustering of annotations
- Strategies: IoU, IoMin, IoMin with center-distance gating
- Enforces participant constraint (one annotation per rater per cluster)

### Adapters

**Dataset adapters** (`adapters/datasets/`)
- VinDr-CXR-oriented pipeline (CSV + JSON); add new datasets as modules here

### Utilities

**Clustering** (`utils/clustering.py`)
- Hierarchical annotation grouping
- Overlap metrics (IoU, IoMin)
- Duplicate ejection with traceability

## Methodological Features

### Unified absence/presence handling

`"No Finding"` annotations receive a placeholder bounding box `[0, 0, 1, 1]`, enabling uniform reliability computation.

### Constraint-aware clustering

Clustering enforces that each rater contributes at most one annotation per finding cluster. Ejected duplicates become singleton clusters rather than being discarded.

### Class-specific Bayesian priors

For each pathology class, the system estimates typical coordinate-level uncertainty from multi-annotated examples. These priors regularize per-instance variance estimates, preventing degenerate uncertainty for single-annotation findings or rare classes. Falls back to global prior when class-specific support is insufficient.

### 1D vs 2D formulations

- **1D (primary)**: Diagonal covariance, per-coordinate independent uncertainty
- **2D (experimental)**: Full covariance matrices at box vertices

## Usage

### End-to-end pipeline

```bash
python -m softlabel_od.scripts.run_pipeline
```

Executes five stages:
1. Normalize CSV annotations
2. Convert to COCO JSON
3. Compute pairwise annotator reliability
4. Estimate class-specific priors
5. Generate final soft labels

### Configuration options

**Clustering method:**
```bash
python -m softlabel_od.scripts.run_pipeline --clustering-method iomin --iomin-threshold 0.5
```

**Center-distance gating:**
```bash
python -m softlabel_od.scripts.run_pipeline \
  --clustering-method iomin_center_gate \
  --iomin-threshold 0.5 \
  --center-ratio-threshold 0.5
```

**2D uncertainty:**
```bash
python -m softlabel_od.scripts.run_pipeline --gmm-mode 2d
```

**Partial execution:**
```bash
python -m softlabel_od.scripts.run_pipeline --steps 4 5  # priors + soft labels only
```

**Dry run:**
```bash
python -m softlabel_od.scripts.run_pipeline --dry-run
```

### Individual components

**Reliability scores:**
```bash
python -m softlabel_od.scripts.calculate_pairwise_reliability --iou 0.5
```

**Class priors:**
```bash
python -m softlabel_od.scripts.calculate_priors_1d \
  --clustering-method iou \
  --iou-threshold 0.5
```

**Soft label generation:**
```bash
python -m softlabel_od.scripts.gmm_1d \
  --clustering-method iou \
  --iou-threshold 0.5
```

## Input Requirements

### CSV format

Annotations must include:
- Image identifier
- Bounding box coordinates (normalized or pixel)
- Class label
- Annotator identifier
- Image dimensions (for normalization)

### Preprocessing

**Normalize coordinates:**
```bash
python -m softlabel_od.softlabel_generation.core.preprocessing.manipulate_csv \
  --input-csv data/train_enriched.csv \
  --output-csv data/train_normalized.csv
```

**Convert to COCO JSON:**
```bash
python -m softlabel_od.softlabel_generation.core.preprocessing.convert_csv_to_json \
  --input-csv data/train_normalized.csv \
  --output-json data/train_annotations.json
```

## Output Format

### Reliability CSV

Per-annotator metrics:
- Average Krippendorff's Alpha
- Standard deviation
- Participation count
- Consistency factor

### Soft label JSON

Per-finding structure:
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

Coordinates are normalized to [0, 1].

### Prior files

Class-specific uncertainty priors saved as JSON for reuse across runs.

## Configuration

Paths are specified via config files in `configs/`. Default locations:
- Input: `./data/vinbigdata/`
- Output: `./outputs/`

Override via environment variables:
```bash
export SOFTLABEL_DATA_ROOT=/path/to/data
export SOFTLABEL_OUTPUT_DIR=/path/to/outputs
```

## Research Applications

- Uncertainty-aware object detection training
- Multi-annotator learning studies
- Annotation quality analysis
- Clustering strategy ablations
- Hard vs soft label comparisons

## Dependencies

Core requirements:
- numpy, scipy
- pandas
- krippendorff
- shapely
- tqdm

Install via:
```bash
pip install -e .
```

## Extension Points

Add new datasets by implementing adapters in `adapters/datasets/`:
- CSV/JSON parsing
- Class mapping
- Coordinate normalization

Add clustering strategies in `ports/correspondence.py`:
- Define overlap metric
- Implement hierarchical grouping
- Enforce participant constraint

Add prior estimation strategies in `core/label_assembly/priors/`:
- Compute class-specific statistics
- Define regularization scheme
- Handle sparse data

## Key Design Decisions

**Why placeholder boxes for "No Finding"?**
Enables unified reliability computation without branching logic for absence vs presence of a pathological finding.

**Why participant constraint in clustering?**
Preserves interpretation that clusters represent cross-rater consensus/disagreement, not within-rater redundancy.

**Why Bayesian priors?**
VinDr-CXR has ≤3 annotators per image. Naive variance estimation from 1-2 samples is unstable. Priors provide regularization.
