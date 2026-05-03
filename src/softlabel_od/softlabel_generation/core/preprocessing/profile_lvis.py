import json
from pathlib import Path
from collections import Counter, defaultdict

JSON_PATH = Path("/media/klee_ro/public_datasets/coco_lvis/annotations_coco/person_keypoints_val2017.json")
IMAGE_ROOT = Path("/media/klee_ro/public_datasets/coco_lvis/val2017")

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

images = data.get("images", [])
annotations = data.get("annotations", [])
categories = data.get("categories", [])

image_by_id = {img["id"]: img for img in images}
cat_by_id = {cat["id"]: cat.get("name", str(cat["id"])) for cat in categories}

rater_counts = Counter()
category_counts = Counter()
missing_rater = 0
missing_image = 0
missing_category = 0
invalid_bbox = 0
missing_image_files = 0

anns_per_image = defaultdict(int)
raters_per_image = defaultdict(set)

for ann in annotations:
    img_id = ann.get("image_id")
    cat_id = ann.get("category_id")
    rater_id = ann.get("rater_id")

    if rater_id is None:
        missing_rater += 1
    else:
        rater_counts[str(rater_id)] += 1
        raters_per_image[img_id].add(str(rater_id))

    if img_id not in image_by_id:
        missing_image += 1
    else:
        anns_per_image[img_id] += 1

    if cat_id not in cat_by_id:
        missing_category += 1
    else:
        category_counts[cat_by_id[cat_id]] += 1

    bbox = ann.get("bbox", [])
    if len(bbox) != 4:
        invalid_bbox += 1
    else:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            invalid_bbox += 1

for img in images[:1000]:
    file_name = img.get("file_name")
    if file_name:
        p = IMAGE_ROOT / file_name
        if not p.exists():
            missing_image_files += 1

report = []
report.append("# COCO-LVIS Dataset Profile\n")
report.append(f"- JSON file: `{JSON_PATH}`")
report.append(f"- Number of images: {len(images)}")
report.append(f"- Number of annotations: {len(annotations)}")
report.append(f"- Number of categories: {len(categories)}")
report.append(f"- Number of raters: {len(rater_counts)}")
report.append(f"- Missing rater_id annotations: {missing_rater}")
report.append(f"- Missing image references: {missing_image}")
report.append(f"- Missing category references: {missing_category}")
report.append(f"- Invalid bbox entries: {invalid_bbox}")
report.append(f"- Missing image files among first 1000 checked: {missing_image_files}")
report.append("\n## Bounding box format\n")
report.append("Raw LVIS bbox format is `[x, y, width, height]` in absolute pixel coordinates.")
report.append("For the pipeline it must be converted to normalized `x_min, y_min, x_max, y_max`.")
report.append("\n## Top raters\n")
for r, c in rater_counts.most_common(20):
    report.append(f"- `{r}`: {c}")
report.append("\n## Top categories\n")
for c, n in category_counts.most_common(20):
    report.append(f"- `{c}`: {n}")
report.append("\n## Raters per image\n")
rpi = Counter(len(v) for v in raters_per_image.values())
for k, v in sorted(rpi.items()):
    report.append(f"- {k} raters: {v} images")

Path("docs").mkdir(exist_ok=True)
Path("docs/person_keypoints_val2017.json.md").write_text("\n".join(report), encoding="utf-8")
print("\n".join(report))
print("\nSaved: docs/lvis_profile.md")