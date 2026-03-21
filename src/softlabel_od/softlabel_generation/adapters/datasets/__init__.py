"""
Dataset adapters for annotation loading.

Each adapter normalises a dataset's raw annotation format into the unified
dict structure expected by the core pipeline:
    {
        'image_id': str,
        'rater_id': str,
        'class_name': str,
        'bbox': [x, y, w, h],   # normalised [0, 1]
        'weight': float,
    }

Existing adapters
-----------------
VinDrCXR — VinDr-CXR CSV + COCO JSON (see scripts/manipulate_csv.py,
            scripts/convert_csv_to_json.py for the pre-processing steps).

Adding a new dataset
--------------------
1. Create a new file, e.g. texbig.py, implementing the same output dict.
2. Register it here with an import.
"""
