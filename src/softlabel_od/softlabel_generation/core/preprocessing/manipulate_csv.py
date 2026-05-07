"""
Step 1: Normalize dataset CSV using the registered dataset adapter.

The adapter handles all dataset-specific column renaming and coordinate
normalization, producing a canonical CSV for the downstream pipeline steps.
"""

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Normalize a dataset CSV via its registered preprocessing adapter."
    )
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument(
        "--adapter", type=str, required=True,
        help="Dataset name whose adapter is registered in adapters/datasets/__init__.py"
        " (e.g. 'vindrcxr', 'texbig', 'null').",)
    args = parser.parse_args()

    from softlabel_od.softlabel_generation.adapters.datasets.registry import get_adapter

    print(f"Loading CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  Shape before normalization: {df.shape}")

    adapter = get_adapter(args.adapter)
    df = adapter.normalize_csv(df)
    print(f"  Shape after normalization:  {df.shape}")

    os.makedirs(os.path.dirname(args.output_csv) if os.path.dirname(args.output_csv) else ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Normalized CSV saved to: {args.output_csv}")


if __name__ == "__main__":
    main()