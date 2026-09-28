"""
Post-hoc per-pathology shift on a submission CSV.

Tests the hypothesis that the train2023.csv labels are NaN-sparse in a
selection-biased way while the leaderboard's held-out labels are
denser. Under that hypothesis, our predictions on "NaN-in-train" rows
default to the labeled-subset mode and pay a large MSE on the LB when
those rows are now real labels with the opposite sign.

The shift design uses per-pathology ``train_nan_rate * sign(labeled_mean)``
as a heuristic. high-NaN pathologies whose labeled subset is heavily
positive get shifted negative (to make predictions less confidently +1
on the rows we never trained on as non-NaN), and vice versa. Pathologies
that are already balanced or have low NaN aren't touched.

Output stays clipped to ``[-1, 1]``

Usage:
    # Default shift, full strength (1.0):
    python scripts/post_hoc_shift.py \\
        runs/<user>_outputs/ensemble-kitchen-sink-v4_nnls_ridge_5050_PUBLIC.csv \\
        --output runs/<user>_outputs/ensemble-kitchen-sink-v4_nnls_ridge_5050_PUBLIC_shifted.csv

    # Half-strength variant (smaller shifts):
    python scripts/post_hoc_shift.py <input.csv> --output <out.csv> --scale 0.5

    # Print computed stats and the shift table without writing:
    python scripts/post_hoc_shift.py <input.csv> --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from radiology_cls.data import LABEL_NAMES
from radiology_cls.settings import TRAIN_CSV

# Hand-tuned per-pathology shifts based on the (NaN rate, labeled mean)
# profile reported in train2023.csv. These are the "first-experiment"
# guesses -- the magnitudes are deliberately a bit smaller than what a
# fit would prescribe so we get signal without over-committing on the
# direction. Cell-level shifts are then optionally scaled by --scale.
#
# Rationale per class:
#   - Pleural Effusion / No Finding / Pneumonia / Cardiomegaly: leave
#     alone. Either NaN rate is low (Effusion), labels are dense (No
#     Finding), or the labeled mean is already near zero (Pneumonia,
#     Cardiomegaly) so no asymmetric overconfidence to correct.
#   - Lung Opacity / Support Devices: high NaN + strongly positive mean
#     (+0.84 / +0.89). Shift negative. Support Devices is the worst
#     offender in the val/LB gap pattern (5.4x), so it gets the biggest
#     shift.
#   - Pleural Other / Fracture: very high NaN (97% / 94%) with positive
#     mean. Shift negative but more modestly than Support Devices
#     because the labeled samples are sparse and our prior on what the
#     test labels look like is weaker.
#   - Enlarged Cardiomediastinum: high NaN but negative labeled mean
#     (-0.28). If NaN-on-train rows are denser positive on test, our
#     model is under-predicting; shift positive.
DEFAULT_SHIFTS: dict[str, float] = {
    "No Finding": 0.00,
    "Enlarged Cardiomediastinum": +0.10,
    "Cardiomegaly": 0.00,
    "Lung Opacity": -0.20,
    "Pneumonia": 0.00,
    "Pleural Effusion": 0.00,
    "Pleural Other": -0.15,
    "Fracture": -0.15,
    "Support Devices": -0.25,
}


def _train_label_stats() -> pd.DataFrame:
    """Per-pathology (NaN rate, labeled mean, labeled counts) from train2023.csv."""
    df = pd.read_csv(TRAIN_CSV)
    rows = []
    for name in LABEL_NAMES:
        if name not in df.columns:
            rows.append({
                "label": name, "nan_pct": float("nan"), "labeled_mean": float("nan"),
                "n_pos": 0, "n_neg": 0, "n_unc": 0,
            })
            continue
        col = df[name]
        n_total = len(col)
        n_nan = int(col.isna().sum())
        labeled = col.dropna()
        n_pos = int((labeled == 1.0).sum())
        n_neg = int((labeled == -1.0).sum())
        n_unc = int((labeled == 0.0).sum())
        rows.append({
            "label": name,
            "nan_pct": 100.0 * n_nan / n_total,
            "labeled_mean": float(labeled.mean()) if len(labeled) else float("nan"),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_unc": n_unc,
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("input_csv", type=Path, help="Submission CSV to shift.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Required unless --dry-run.",
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help=(
            "Multiplier on DEFAULT_SHIFTS. 1.0 is full strength, 0.5 is "
            "half, 0.0 is no shift (identity output)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the shift table and pred-distribution change without writing.",
    )
    args = p.parse_args()
    if args.output is None and not args.dry_run:
        p.error("--output is required unless --dry-run")

    df = pd.read_csv(args.input_csv)
    if "Id" not in df.columns:
        sys.exit(f"{args.input_csv} has no 'Id' column")
    label_cols = list(LABEL_NAMES)
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        sys.exit(f"Input CSV is missing label columns: {missing}")

    print(f"Input:  {args.input_csv}")
    print(f"Output: {args.output if args.output else '(dry run)'}")
    print(f"Scale:  {args.scale}")
    print()

    stats = _train_label_stats().set_index("label")
    print(f"{'Pathology':<30} {'NaN%':>6} {'lab.mean':>9} {'shift':>8} "
          f"{'pred.mean_before':>17} {'pred.mean_after':>16}")
    shifts: dict[str, float] = {}
    for c in label_cols:
        shift = float(DEFAULT_SHIFTS.get(c, 0.0)) * float(args.scale)
        shifts[c] = shift
        before = float(df[c].mean())
        shifted = (df[c] + shift).clip(-1.0, 1.0)
        after = float(shifted.mean())
        if not args.dry_run:
            df[c] = shifted
        st = stats.loc[c] if c in stats.index else None
        nan_pct = float(st["nan_pct"]) if st is not None else float("nan")
        lab_mean = float(st["labeled_mean"]) if st is not None else float("nan")
        print(
            f"{c:<30} {nan_pct:>6.1f} {lab_mean:>+9.3f} {shift:>+8.3f} "
            f"{before:>+17.4f} {after:>+16.4f}"
        )

    if args.dry_run:
        print("\n(Dry run; no CSV written.)")
        return

    df.to_csv(args.output, index=False)
    print(f"\nSubmission saved: {args.output} ({len(df)} rows)")


if __name__ == "__main__":
    main()
