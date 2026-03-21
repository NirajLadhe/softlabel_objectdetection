"""
Unified data loading for annotations.
Single entry point for loading COCO JSON, VinDr CSV, TexBig JSON.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path


def load_annotations(
    source: str,
    format: str = "coco",
    **kwargs
) -> Dict[str, Any]:
    """
    Load annotations from file.

    Args:
        source: Path to annotation file
        format: One of 'coco', 'vindr_csv'
        **kwargs: Format-specific options (e.g. no_finding_class_id for COCO)

    Returns:
        Unified structure: dict with 'images', 'annotations', 'categories' keys
        for COCO format; or dict with image_data for CSV-based formats.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {source}")

    if format == "coco":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif format == "vindr_csv":
        return _load_vindr_csv(source, **kwargs)
    else:
        raise ValueError(f"Unknown format: {format}")


def _load_vindr_csv(
    csv_path: str,
    class_name_col: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Load VinDr-CXR CSV to unified structure."""
    df = pd.read_csv(csv_path)
    if class_name_col is None:
        class_name_col = "class_name" if "class_name" in df.columns else "label"
    return {"_dataframe": df, "_class_col": class_name_col, "format": "vindr_csv"}


def load_all_data_for_gmm(
    train_csv_path: str,
    raters_json_path: str,
    global_normalized_weights: Dict[str, float],
) -> tuple:
    """
    Load all images and annotations for GMM soft label generation.
    Unified interface used by GMM 1D and 2D scripts.

    Returns:
        (image_ids, image_data) where image_data maps image_id -> list of annotation dicts
    """
    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(train_csv_path)
    class_name_col = "class_name" if "class_name" in df.columns else "label"
    if df.empty:
        return [], {}

    required = ["image_id", "rad_id", "x_min", "y_min", "x_max", "y_max"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV must have columns: {required}")

    df["rad_id"] = df["rad_id"].astype(str)
    all_image_ids = df["image_id"].unique()
    image_data = {}

    for img_id in tqdm(all_image_ids, desc="Loading Annotations"):
        rows = df[df["image_id"] == img_id]
        if rows.empty:
            continue

        anns = []
        for _, row in rows.iterrows():
            class_name = row[class_name_col]
            weight = global_normalized_weights.get(str(row["rad_id"]), 0.0)

            if class_name.lower() in ["no finding", "no findings"]:
                bbox = [row["x_min"], row["y_min"], row["x_max"] - row["x_min"], row["y_max"] - row["y_min"]]
            else:
                if row["x_max"] <= row["x_min"] or row["y_max"] <= row["y_min"]:
                    continue
                bbox = [row["x_min"], row["y_min"], row["x_max"] - row["x_min"], row["y_max"] - row["y_min"]]

            anns.append({
                "rater_id": str(row["rad_id"]),
                "bbox": bbox,
                "class_name": class_name,
                "weight": weight,
            })

        if anns:
            valid_raters = {a["rater_id"] for a in anns if a["weight"] > 0}
            if valid_raters:
                image_data[img_id] = anns

    return list(image_data.keys()), image_data
