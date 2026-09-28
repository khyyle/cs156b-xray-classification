"""Rebuild the full v4 submission pipeline on CPU for several Stage A combiner
configurations and write a portfolio of public and private submission CSVs.

The competition pipeline has two learned stages sitting on top of the frozen
ensemble members. Stage A gives each member a weight for each pathology, fit on
the validation set with either non negative least squares or ridge regression.
Stage B replaces uniform view averaging with a per class convex combination of a
study's frontal and lateral scores. A final optional step blends two finished
submissions by averaging them cell by cell. This driver runs that exact pipeline
for a list of Stage A configurations so they can be compared on equal footing.

This script is used bc the current best model uses NNLS for Stage A, 
but study aware out of fold NMSE shows ridge with a
moderate alpha generalizes better, and the out of fold number tracks the
leaderboard closely. 

NOTE: Regenerating submissions needs no GPU because it reuses the 
cached per member validation and test predictions.

Usage
    python scratch/kyle/oof_portfolio.py
    python scratch/kyle/oof_portfolio.py --oof-only
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiology_cls.data import (
    LABEL_NAMES,
    load_test_df,
    load_train_df,
    train_val_split,
)
from radiology_cls.eval import compute_regression_metrics
from radiology_cls.utils import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import per_disease_weights as pdw  # noqa: E402
import per_view_fusion as pvf  # noqa: E402

RUN_DIR = PROJECT_ROOT / "runs" / "ensemble-kitchen-sink-v4_20260524_171917"
OUT_DIR = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
PERVIEW_ALPHA = 10.0  # prior strength for convex view fusion (champion default)
OOF_FOLDS = 5
OOF_SEED = 42

# (method, ridge_alpha, short_tag)
CONFIGS = [
    ("nnls", 1.0, "nnls"),
    ("ridge", 1.0, "ridge_a1"),
    ("ridge", 20.0, "ridge_a20"),
    ("ridge", 50.0, "ridge_a50"),
    ("ridge", 100.0, "ridge_a100"),
]

# 50/50 cell-wise blends of two configs' final fused CSVs, by tag.
BLENDS = [
    ("nnls", "ridge_a1", "blend_nnls_ridge_a1"),     # reproduces champion
    ("nnls", "ridge_a50", "blend_nnls_ridge_a50"),
    ("ridge_a20", "ridge_a100", "blend_ridge_a20_a100"),
    ("ridge_a50", "ridge_a100", "blend_ridge_a50_a100"),
]


def _resolve_members(cfg: dict) -> list[Path]:
    dirs: list[Path] = []
    for m in cfg.get("members") or []:
        p = Path(m)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        if not p.is_dir():
            sys.exit(f"Member dir missing: {p}")
        dirs.append(p)
    return dirs


def _stack(member_dirs: list[Path], base: str, suffix: str, shape: tuple) -> np.ndarray:
    arrs = []
    for d in member_dirs:
        arrs.append(
            pdw._load_member_preds(d, base, suffix, expected_shape=shape).astype(np.float64)
        )
    return np.stack(arrs, axis=0)  # (M, N, C)


def _fit_stage_a(V_val: np.ndarray, Y_val: np.ndarray, method: str, alpha: float) -> np.ndarray:
    W, _ = pdw._fit_per_disease_weights(V_val, Y_val, method=method, ridge_alpha=alpha)
    return W  # (M, C)


def _fit_view_weights(val_df: pd.DataFrame, val_comb: np.ndarray, Y_val: np.ndarray) -> np.ndarray:
    study_df, s_F, s_L, inv, _ = pvf._build_view_aggregates(val_df, val_comb)
    y_study = pvf._per_study_target(Y_val, inv)
    Wv, _ = pvf._fit_per_class_convex(
        y=y_study,
        s_F=s_F,
        s_L=s_L,
        has_F=study_df["has_frontal"].to_numpy(),
        has_L=study_df["has_lateral"].to_numpy(),
        alpha=PERVIEW_ALPHA,
    )
    return Wv  # (3, C)


def _apply_pipeline_to_test(
    W_a: np.ndarray, Wv: np.ndarray, V_test: np.ndarray, test_df: pd.DataFrame,
) -> np.ndarray:
    test_comb = np.einsum("mc,mnc->nc", W_a, V_test)
    study_df, s_F, s_L, inv, _ = pvf._build_view_aggregates(test_df, test_comb)
    fused_study = pvf._apply_fusion(
        s_F, s_L,
        study_df["has_frontal"].to_numpy(),
        study_df["has_lateral"].to_numpy(),
        Wv,
    )
    return pvf._broadcast_study_to_rows(fused_study, inv)  # (N_test, C)


def _write_submission(rows: np.ndarray, test_df: pd.DataFrame, path: Path) -> None:
    sub = pd.DataFrame(rows, columns=list(LABEL_NAMES))
    if "Id" in test_df.columns:
        sub.insert(0, "Id", test_df["Id"].to_numpy())
    else:
        sub.insert(0, "Id", range(len(sub)))
    sub.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-only", action="store_true", help="Only print the OOF table.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((RUN_DIR / "config.yaml").read_text())
    member_dirs = _resolve_members(cfg)
    suffix = pdw._tta_cache_suffix(cfg.get("tta_transforms"))
    val_frac = float(cfg.get("val_frac", 0.1))
    seed = int(cfg.get("seed", 42))

    Y_val = np.load(RUN_DIR / "val_targets.npy").astype(np.float64)
    N_val, C = Y_val.shape
    print(f"Members: {len(member_dirs)} | val rows: {N_val} | classes: {C}")

    V_val = _stack(member_dirs, "cached_val_preds", suffix, (N_val, C))
    study_ids = pdw._val_study_ids(val_frac, seed, N_val)

    ## OOF table

    # For each Stage-A config, compare the in-sample macro NMSE (optimistic)
    # against the out-of-fold macro NMSE (the number that tracks the board).
    print("\n=== Stage-A OOF NMSE (study-aware 5-fold) ===")
    print(f"{'config':<16} {'insample NMSE':>14} {'OOF NMSE':>10}")
    names = list(LABEL_NAMES)
    oof_rows = []
    for method, alpha, tag in CONFIGS:
        oof_pred = pdw._oof_predictions(
            V_val, Y_val, study_ids,
            method=method, ridge_alpha=alpha, n_folds=OOF_FOLDS, seed=OOF_SEED,
        )
        insample_pred = np.einsum("mc,mnc->nc", _fit_stage_a(V_val, Y_val, method, alpha), V_val)
        oof_macro = compute_regression_metrics(Y_val, oof_pred, names)["macro_nmse"]
        ins_macro = compute_regression_metrics(Y_val, insample_pred, names)["macro_nmse"]
        oof_rows.append((tag, ins_macro, oof_macro))
        print(f"{tag:<16} {ins_macro:>14.5f} {oof_macro:>10.5f}")
    best = min(oof_rows, key=lambda r: r[2])
    print(f"\nBest Stage-A by OOF NMSE: {best[0]} ({best[2]:.5f})")

    if args.oof_only:
        return

    # Reconstruct the exact val_df rows (for per-view fit).
    full_df = load_train_df()
    _, val_df = train_val_split(full_df, val_frac=val_frac, seed=seed)
    if len(val_df) != N_val:
        sys.exit(f"val_df rows {len(val_df)} != cached val rows {N_val}")

    # Pre-fit Stage A + Stage B weights once (val-only, phase-independent).
    fitted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method, alpha, tag in CONFIGS:
        W_a = _fit_stage_a(V_val, Y_val, method, alpha)
        val_comb = np.einsum("mc,mnc->nc", W_a, V_val)
        Wv = _fit_view_weights(val_df, val_comb, Y_val)
        fitted[tag] = (W_a, Wv)

    for phase, private in (("PUBLIC", False), ("PRIVATE", True)):
        print(f"\n=== {phase}: building submissions ===")
        test_df = load_test_df(private=private)
        test_df = pvf._ensure_frontal_lateral_column(test_df)
        N_test = len(test_df)
        V_test = _stack(member_dirs, "cached_test_preds", suffix, (N_test, C))
        print(f"  test rows: {N_test}")

        final_rows: dict[str, np.ndarray] = {}
        for method, alpha, tag in CONFIGS:
            W_a, Wv = fitted[tag]
            rows = _apply_pipeline_to_test(W_a, Wv, V_test, test_df)
            final_rows[tag] = rows
            out = OUT_DIR / f"v4_oof_perview_{tag}_{phase}.csv"
            _write_submission(rows, test_df, out)
            print(f"  wrote {out.name}")

        for tag_a, tag_b, blend_tag in BLENDS:
            rows = 0.5 * final_rows[tag_a] + 0.5 * final_rows[tag_b]
            out = OUT_DIR / f"v4_oof_{blend_tag}_{phase}.csv"
            _write_submission(rows, test_df, out)
            print(f"  wrote {out.name}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
