# COCO-LVIS Dataset Profile

- JSON file: `/media/klee_ro/public_datasets/coco_lvis/lvis_standardized_lvis_val.json`
- Number of images: 5000
- Number of annotations: 100480
- Number of categories: 1203
- Number of raters: 2
- Missing rater_id annotations: 0
- Missing image references: 0
- Missing category references: 0
- Invalid bbox entries: 0
- Missing image files among first 1000 checked: 0

## Bounding box format

Raw LVIS bbox format is `[x, y, width, height]` in absolute pixel coordinates.
For the pipeline it must be converted to normalized `x_min, y_min, x_max, y_max`.

## Top raters

- `r2`: 51733
- `r1`: 48747

## Top categories

- `banana`: 3190
- `book`: 2446
- `carrot`: 1994
- `apple`: 1543
- `pole`: 1390
- `doughnut`: 1314
- `bird`: 1155
- `chair`: 1071
- `wheel`: 1069
- `broccoli`: 1052
- `handle`: 978
- `car_(automobile)`: 957
- `shoe`: 944
- `person`: 939
- `magnet`: 937
- `shirt`: 927
- `sheep`: 924
- `orange_(fruit)`: 905
- `kite`: 868
- `suitcase`: 865

## Raters per image

- 1 raters: 11 images
- 2 raters: 4758 images