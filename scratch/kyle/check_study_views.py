"""
check if views of the same study consistently share pathology labels
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from radiology_cls.data import (
    LABEL_NAMES,
    load_test_df,
    load_train_df,
)


def _extract_study_id(df: pd.DataFrame) -> pd.Series:
    # Study key is `pidXXXXX/studyN`
    parts = df["Path"].str.split("/")
    return parts.str[1] + "/" + parts.str[2]


def _summarize_row_counts(df: pd.DataFrame, label: str) -> None:
    counts = df.groupby("study_id").size()
    n_studies = len(counts)
    n_multi = int((counts > 1).sum())
    pct_multi = 100.0 * n_multi / n_studies if n_studies else 0.0
    n_pooled_rows = int(counts[counts > 1].sum())
    pct_pooled_rows = 100.0 * n_pooled_rows / len(df) if len(df) else 0.0

    print(f"\n[{label}] rows per study")
    print(f"  studies: {n_studies}")
    print(f"  total rows: {len(df)}")
    print(f"  studies with >1 row: {n_multi}  ({pct_multi}%)")
    print(f"  rows that would pool: {n_pooled_rows}  ({pct_pooled_rows}% of all rows)")
    print(f"  min / median / max: {counts.min()} / {int(counts.median())} / {counts.max()}")
    print(f"  mean: {counts.mean()}")
    bins = [1, 2, 3, 4, 5, 10, 20, 50, 1000]
    binned = pd.cut(counts, bins=bins, include_lowest=True, right=True)
    for interval, n in binned.value_counts().sort_index().items():
        print(f"    rows in {str(interval)} : {n} studies")


def _label_agreement(train_df: pd.DataFrame) -> pd.DataFrame:
    multi_studies = (
        train_df[train_df.duplicated("study_id", keep=False)]
        .groupby("study_id")
    )

    summary_rows: list[dict] = []
    for label in LABEL_NAMES:
        agree = 0
        disagree = 0
        no_observation = 0
        for _, g in multi_studies:
            observed = g[label].dropna().unique()
            if len(observed) == 0:
                no_observation += 1
            elif len(observed) == 1:
                agree += 1
            else:
                disagree += 1
        total = agree + disagree
        pct = 100.0 * agree / total if total > 0 else float("nan")
        summary_rows.append({
            "label": label,
            "studies_agree": agree,
            "studies_disagree": disagree,
            "studies_no_observation": no_observation,
            "pct_agree_among_observed": pct,
        })
    return pd.DataFrame(summary_rows)


def _show_sample_studies(train_df: pd.DataFrame, n_samples: int = 5) -> None:
    multi_sids = (
        train_df.groupby("study_id").size().loc[lambda s: s > 1].index.tolist()
    )
    if not multi_sids:
        print("\nNo studies with >1 row found; skipping samples.")
        return

    rng = np.random.default_rng(42)
    sample = rng.choice(multi_sids, size=min(n_samples, len(multi_sids)), replace=False)
    cols = ["Path", "Frontal/Lateral", "AP/PA", *LABEL_NAMES]

    for sid in sample:
        print("\n" + "=" * 90)
        print(f"study: {sid}")
        print("=" * 90)
        sub = train_df.loc[train_df["study_id"] == sid, cols].copy()
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 240,
            "display.max_colwidth", 60,
        ):
            print(sub.to_string(index=False))


def main() -> None:
    train_df = load_train_df()
    train_df["study_id"] = _extract_study_id(train_df)

    _summarize_row_counts(train_df, "train")

    print("\n" + "-" * 90)
    print("LABEL AGREEMENT WITHIN A STUDY (train)")
    print("-" * 90)
    per_study = _label_agreement(train_df)
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(per_study.to_string(index=False))

    _show_sample_studies(train_df, n_samples=5)

    print("\n" + "=" * 90)
    print("TEST SET STRUCTURE")
    print("=" * 90)
    test_df = load_test_df()
    test_df = test_df.copy()
    test_df["study_id"] = _extract_study_id(test_df)
    _summarize_row_counts(test_df, "test")

    out_dir = Path("scratch/kyle")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_study.to_csv(out_dir / "study_label_agreement.csv", index=False)
    print(f"\nWrote agreement summary to {out_dir / 'study_label_agreement.csv'}")


if __name__ == "__main__":
    main()
