"""
transform_reliability_weights.py

Transforms K-alpha annotator reliability scores from raw [-1, 1]
to clipped [0.01, 1] range, eliminating negative weights that cause
weight sum inversion in normalize_annotator_weights().

Input:  outputs/nucls/reliability/annotator_reliability_scores_nucls_iou0_25.csv
Output: outputs/nucls/reliability/annotator_reliability_scores_nucls_iou0_25_clipped.csv

No existing pipeline code is modified. The output CSV is a drop-in
replacement — pass it via --rater_weights_csv or OPTIONS.rater_weights_csv.
"""

import pandas as pd
import numpy as np
import os

EPSILON = 0.001  # floor — same as base.py floor, so that code becomes no-op

def transform_weights(
    input_csv: str,
    output_csv: str,
    epsilon: float = EPSILON,
    verbose: bool = True,
):
    df = pd.read_csv(input_csv)

    # Expect a Avg K-alpha column — adjust column name if different in your CSV
    assert 'Avg K-alpha' in df.columns, f"Column 'Avg K-alpha' not found. Columns: {df.columns.tolist()}"

    original = df['Avg K-alpha'].copy()

    # Clip: negative → epsilon, preserves [epsilon, 1] for positive
    df['Avg K-alpha'] = df['Avg K-alpha'].clip(lower=epsilon)

    if verbose:
        n_negative  = (original < 0).sum()
        n_zero      = (original == 0).sum()
        n_positive  = (original > 0).sum()
        print(f"Original distribution:")
        print(f"  Negative K-alpha (< 0):  {n_negative} annotators → clipped to {epsilon}")
        print(f"  Zero K-alpha (= 0):       {n_zero} annotators → clipped to {epsilon}")
        print(f"  Positive K-alpha (> 0):   {n_positive} annotators → unchanged")
        print(f"  Original range: [{original.min():.4f}, {original.max():.4f}]")
        print(f"  Transformed range: [{df['Avg K-alpha'].min():.4f}, {df['Avg K-alpha'].max():.4f}]")
        print(f"  Dynamic ratio (max/min positive): "
              f"{original[original > 0].max() / original[original > 0].min():.2f}× → "
              f"{df['Avg K-alpha'].max() / df['Avg K-alpha'][original > 0].min():.2f}× (preserved)")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    return df


if __name__ == "__main__":
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )

    input_csv = os.path.join(
        project_root,
        "outputs/nucls/reliability/annotator_reliability_scores_nucls_iou0_25.csv"
    )
    output_csv = os.path.join(
        project_root,
        "outputs/nucls_transform/reliability/annotator_reliability_scores_nucls_iou0_25_clipped.csv"
    )

    transform_weights(input_csv, output_csv, epsilon=EPSILON, verbose=True)