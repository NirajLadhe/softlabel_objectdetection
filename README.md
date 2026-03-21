# Soft Label Object Detection

Merged repository for soft label generation and object detection on multi-annotator medical imaging (VinDr-CXR, TexBig).

## Install

```bash
pip install -e .
```

## Usage

**Soft label pipeline** (from CSV to GMM soft labels):
```bash
python -m softlabel_od.scripts.run_pipeline
# Or: softlabel-pipeline
# Place vindr_cxr/train_enriched.csv at project root, or pass --input-csv
```

**Training (legacy PyTorch, primary):**
```bash
python -m softlabel_od.scripts.train --config configs/vindr_cxr_1d_config.yaml
# Or: softlabel-train --config configs/vindr_cxr_1d_config.yaml
```

**Training (Lightning, optional):**
```bash
python -m softlabel_od.scripts.train_lightning --config configs/vindr_cxr_1d_config.yaml
```

**Evaluation:**
```bash
python -m softlabel_od.scripts.evaluate --weights path/to/model.pth --gmm_type 1D
```

## Config

Edit `configs/vindr_cxr_1d_config.yaml` for paths. Defaults use `./data/` and `./outputs/`.
