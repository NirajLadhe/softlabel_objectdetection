import json
from pathlib import Path

soft_path = Path("/home/niraj/softlabel_objectdetection/outputs/lvis/1d_gaussian_softlabels/1DGMM_softLabels_iou0_5_with_priors.json")
out_path = Path("/home/niraj/Assignment2/docs/lvis_first5_softlabels.md")

out_path.parent.mkdir(parents=True, exist_ok=True)

data = json.loads(soft_path.read_text(encoding="utf-8"))

# Handle both list and dict JSON formats
if isinstance(data, dict):
    if "annotations" in data:
        items = data["annotations"]
    elif "soft_labels" in data:
        items = data["soft_labels"]
    else:
        items = next(v for v in data.values() if isinstance(v, list))
else:
    items = data

lines = ["# First 5 COCO-LVIS Soft Labels\n"]

for i, item in enumerate(items[:5], start=1):
    lines.append(f"\n## Soft label {i}\n")
    lines.append("```json")
    lines.append(json.dumps(item, indent=2)[:4000])
    lines.append("```")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved: {out_path}")