"""
Normalize bounding box coordinates in the VinDr-CXR training CSV.
"""

import argparse
import os

# Logic from SoftLabel_Generation/utils/manipulate_csv.py
def normalize_csv(input_csv: str, output_csv: str):
    import pandas as pd
    import numpy as np

    print(f"Loading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Dataset shape: {df.shape}")

    df['x_min'] = df['x_min'].fillna(0)
    df['y_min'] = df['y_min'].fillna(0)
    df['x_max'] = df['x_max'].fillna(1)
    df['y_max'] = df['y_max'].fillna(1)

    is_no_finding = (df['class_id'] == 14)
    is_finding = (df['class_id'] != 14)

    df.loc[is_finding, 'x_min'] = df.loc[is_finding, 'x_min'] / df.loc[is_finding, 'width']
    df.loc[is_finding, 'y_min'] = df.loc[is_finding, 'y_min'] / df.loc[is_finding, 'height']
    df.loc[is_finding, 'x_max'] = df.loc[is_finding, 'x_max'] / df.loc[is_finding, 'width']
    df.loc[is_finding, 'y_max'] = df.loc[is_finding, 'y_max'] / df.loc[is_finding, 'height']

    print("Normalization complete.")
    os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else '.', exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Modified CSV saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Normalize bounding box coordinates in VinDr-CXR training CSV.")
    parser.add_argument('--input-csv', type=str, default='vindr_cxr/train_enriched.csv')
    parser.add_argument('--output-csv', type=str, default='vindr_cxr/train_enriched_normalized.csv')
    args = parser.parse_args()
    normalize_csv(args.input_csv, args.output_csv)


if __name__ == "__main__":
    main()
