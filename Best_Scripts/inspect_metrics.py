"""
inspect_metrics_json.py
Prints all keys in a metrics JSON with sample values and length.

Usage:
    python inspect_metrics_json.py --metrics-json path/to/metrics.json
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--metrics-json", required=True)
args = ap.parse_args()

with open(args.metrics_json) as f:
    data = json.load(f)

print(f"\nTotal keys: {len(data)}\n")
print(f"{'Key':<45} {'Length':>8}   {'First value':>14}   {'Last value':>14}")
print("─" * 90)
for k, v in sorted(data.items()):
    if isinstance(v, list) and len(v) > 0:
        print(f"{k:<45} {len(v):>8}   {v[0]:>14.6f}   {v[-1]:>14.6f}")
    else:
        print(f"{k:<45} {'?':>8}   {'—':>14}   {'—':>14}")