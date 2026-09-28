"""
Blend two or more existing submission CSVs into a new one. Each input is
weighted, and final predictions are normalized so the weights sum to 1
(so a 50/50 blend is just the mean of the predictions per-cell).

Use this to combine submissions you already have:
  - mean blend of NNLS + ridge per-disease (different regularizers)
  - cross-ensemble blend (v3 NNLS + v4 NNLS)
  - hedge a per-disease submission with its uniform-mean counterpart

The blend operates on the LABEL columns and assumes:
  - all submissions have the same Id column in the same order
  - LABEL_NAMES columns are present in each

Output mirrors the submission.csv schema (Id + 9 pathologies).

Usage examples:
  # 50/50 mean of two CSVs
  python scripts/blend_submissions.py \\
      ~/Downloads/v4/submission_per_disease.csv \\
      ~/Downloads/v4/submission_per_disease_ridge.csv \\
      --output blend_v4_nnls_ridge.csv

  # weighted (0.7/0.3)
  python scripts/blend_submissions.py \\
      ~/Downloads/v4/submission_per_disease.csv \\
      ~/Downloads/v4/submission.csv \\
      --weights 0.7 0.3 \\
      --output blend_v4_nnls_uniform_7030.csv

  # 3-way blend, equal weights
  python scripts/blend_submissions.py \\
      ~/Downloads/v3/submission_per_disease.csv \\
      ~/Downloads/v4/submission_per_disease.csv \\
      ~/Downloads/v4/submission_per_disease_ridge.csv \\
      --output blend_3way.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Label columns expected in every submission CSV (from radiology_cls.data).
LABEL_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Blend existing submission CSVs by weighted per-cell mean. "
            "Output schema matches scripts/submit.py (Id + 9 pathology cols)."
        ),
    )
    p.add_argument("inputs", nargs="+", type=Path, help="Two or more submission CSVs to blend.")
    p.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Per-input weights. Length must equal number of inputs. "
            "Auto-normalized to sum to 1. Default: uniform."
        ),
    )
    p.add_argument("--output", type=Path, required=True, help="Output CSV path.")
    args = p.parse_args()

    if len(args.inputs) < 2:
        sys.exit("Need at least 2 input CSVs to blend.")
    for path in args.inputs:
        if not path.is_file():
            sys.exit(f"Input not found: {path}")

    # Resolve weights.
    if args.weights is None:
        weights = np.full(len(args.inputs), 1.0 / len(args.inputs), dtype=np.float64)
    else:
        if len(args.weights) != len(args.inputs):
            sys.exit(
                f"--weights must have {len(args.inputs)} values, got {len(args.weights)}"
            )
        w = np.asarray(args.weights, dtype=np.float64)
        if (w < 0).any():
            sys.exit("--weights must be non-negative")
        total = float(w.sum())
        if total <= 0:
            sys.exit("--weights must sum to a positive value")
        weights = w / total

    print("Inputs:")
    for path, w in zip(args.inputs, weights):
        print(f"  {w:.4f}  {path}")

    # Load all CSVs.
    dfs = [pd.read_csv(path) for path in args.inputs]

    # Validate compatibility.
    base_ids = dfs[0]["Id"].to_numpy()
    for path, df in zip(args.inputs, dfs):
        if "Id" not in df.columns:
            sys.exit(f"{path}: missing 'Id' column")
        if not (df["Id"].to_numpy() == base_ids).all():
            sys.exit(
                f"{path}: Id column doesn't match {args.inputs[0]} "
                "(rows must be in the same order)"
            )
        for col in LABEL_COLS:
            if col not in df.columns:
                sys.exit(f"{path}: missing label column '{col}'")

    # Stack predictions: (M_inputs, N_rows, C_labels)
    stack = np.stack(
        [df[LABEL_COLS].to_numpy(dtype=np.float64) for df in dfs],
        axis=0,
    )
    print(f"\nStacked predictions: {stack.shape}")

    # Weighted mean across inputs -> (N_rows, C_labels)
    blended = np.tensordot(weights, stack, axes=([0], [0]))
    print(f"Blended predictions: {blended.shape}")

    # Clip to [-1, 1] so we stay in the tanh range of the underlying members.
    blended = np.clip(blended, -1.0, 1.0)

    # Assemble output.
    out_df = pd.DataFrame(blended, columns=LABEL_COLS)
    out_df.insert(0, "Id", base_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"\nBlended submission saved: {args.output} ({len(out_df)} rows)")
    print("\nUpload to LB to compare.")


if __name__ == "__main__":
    main()
