# Soft Label Generation

Generate reliability-weighted Gaussian uncertainty labels from multi-annotator medical imaging annotations.

## Purpose

Convert multi-rater annotations into probabilistic soft labels that preserve expert disagreement as structured uncertainty rather than forcing consensus. Each generated label contains:

- **Spatial uncertainty**: Gaussian distribution over bounding box parameters (x, y, w, h)
- **Semantic uncertainty**: Probability distribution over pathology classes
- **Reliability weighting**: Annotator contributions weighted by pairwise Krippendorff's Alpha
- **Bayesian stabilization**: Class-specific priors regularize uncertainty estimates for sparse annotations


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


## Key Design Decisions

**Why placeholder boxes for "No Finding"?**
Enables unified reliability computation without branching logic for absence vs presence of a pathological finding.

**Why participant constraint in clustering?**
Preserves interpretation that clusters represent cross-rater consensus/disagreement, not within-rater redundancy.

**Why Bayesian priors?**
VinDr-CXR has ≤3 annotators per image. Naive variance estimation from 1-2 samples is unstable. Priors provide regularization.
