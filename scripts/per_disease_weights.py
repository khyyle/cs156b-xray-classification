"""
Fit per-disease ensemble weights on a trained ensemble run and optionally
generate a submission CSV.

For each of the 9 pathologies independently, this finds non-negative
member weights that minimize val MSE for that pathology. A model that's
great at "Cardiomegaly" but mediocre at "Pleural Effusion" gets high
weight on the former and low on the latter -- something uniform-mean
can't do.

Inputs (all already created by the ensemble's train() and submit() runs):
  - <run_dir>/config.yaml         (ensemble config with members + tta spec)
  - <run_dir>/val_targets.npy     (shape (N_val, 9))
  - per-member cached_val_preds[__tta-...].npy
  - per-member cached_test_preds[__tta-...].npy  (only needed for --submit)

The fitted weights are read off the same val set they minimize, so the reported
val MSE is in-sample and potentially optimistic on sparse pathologies. Passing --oof-folds
adds a study-aware K-fold out-of-fold estimate, which hopefully tracks the
public/private leaderboard better than a single fold does.

Outputs:
  - <run_dir>/per_disease_weights.npy        (shape (M, 9))
  - <run_dir>/per_disease_summary.json       (per-pathology MSE comparison)
  - <run_dir>/submission_per_disease.csv     (only if --submit)

Usage:
    # 1. Fit weights only and print diagnostics
    python scripts/per_disease_weights.py runs/ensemble-kitchen-sink-v2p5_20260523_121516

    # 2. Fit + apply to test preds + write submission (view-averaged)
    python scripts/per_disease_weights.py runs/ensemble-kitchen-sink-v2p5_20260523_121516 --submit

    # 3. Add a calibrated out-of-fold NMSE estimate for model selection
    python scripts/per_disease_weights.py runs/ensemble-kitchen-sink-v4_20260524_171917 --method ridge --ridge-alpha 50 --oof-folds 5
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiology_cls.data import (
    LABEL_NAMES,
    load_test_df,
    load_train_df,
    study_ids_from_paths,
    train_val_split,
)
from radiology_cls.eval import compute_regression_metrics
from radiology_cls.utils import PROJECT_ROOT

PROJECT_PRED_CACHE_ROOT = PROJECT_ROOT / "cache" / "ensemble_predictions"
CACHED_VAL = "cached_val_preds"
CACHED_TEST = "cached_test_preds"


def _tta_cache_suffix(tta_transforms: list[str] | None) -> str:
    """Match the suffix scheme in src/radiology_cls/models/ensemble.py."""
    if not tta_transforms:
        return ""
    canonical = "-".join(sorted(tta_transforms))
    return f"__tta-{canonical}"


def _candidate_paths(member_dir: Path, cache_filename: str) -> list[Path]:
    """Three-location lookup order: member dir, project cache, user home cache."""
    return [
        member_dir / cache_filename,
        PROJECT_PRED_CACHE_ROOT / member_dir.name / cache_filename,
        Path.home() / ".cache" / "cs156b_ensemble" / member_dir.name / cache_filename,
    ]


def _find_cache(
    member_dir: Path,
    base_name: str,
    tta_suffix: str,
    expected_shape: tuple | None = None,
) -> Path:
    """Mirror ensemble.py's cache lookup with row-count tag preference.

    For test caches, prefers the row-count-tagged filename
    (e.g. ``cached_test_preds__rows22596``) matching ``expected_shape[0]``
    so public and private caches coexist without clobbering. Falls
    back to the legacy untagged filename, validated by shape so an old
    public cache doesn't get returned when private is what we want
    (and vice versa).
    """
    untagged = f"{base_name}{tta_suffix}.npy"
    if base_name == "cached_test_preds" and expected_shape is not None:
        n_rows = expected_shape[0]
        tagged = f"{base_name}__rows{n_rows}{tta_suffix}.npy"
        candidates = _candidate_paths(member_dir, tagged) + _candidate_paths(
            member_dir, untagged,
        )
    else:
        candidates = _candidate_paths(member_dir, untagged)

    for p in candidates:
        if not p.exists():
            continue
        if expected_shape is not None:
            shape = tuple(np.load(p, mmap_mode="r").shape)
            if shape != expected_shape:
                print(
                    f"  [skip stale cache] {member_dir.name}: "
                    f"shape {shape} != expected {expected_shape} at {p}"
                )
                continue
        return p
    raise FileNotFoundError(
        f"No cached preds for {member_dir.name} ({base_name}{tta_suffix}). "
        f"Looked in: {[str(p) for p in candidates]}"
    )


def _load_member_preds(
    member_dir: Path,
    base_name: str,
    tta_suffix: str,
    expected_shape: tuple | None = None,
) -> np.ndarray:
    path = _find_cache(member_dir, base_name, tta_suffix, expected_shape)
    return np.load(path)


def _study_view_average(submission: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Replace each row's prediction with the mean across all views of the
    same study, matching the ``--view-average`` step in scripts/submit.py."""
    if len(submission) != len(test_df):
        raise ValueError(
            f"submission rows ({len(submission)}) != test_df rows ({len(test_df)})"
        )
    study_ids = study_ids_from_paths(test_df["Path"])
    if study_ids.isna().any():
        raise ValueError("Could not extract study_id from one or more test paths")
    averaged = submission.copy()
    averaged["__sid"] = study_ids.to_numpy()
    label_cols = list(LABEL_NAMES)
    averaged[label_cols] = (
        averaged.groupby("__sid", sort=False)[label_cols].transform("mean")
    )
    return averaged.drop(columns="__sid")


def _fit_per_disease_weights(
    V: np.ndarray, Y: np.ndarray, *, method: str, ridge_alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit per-pathology weights minimizing MSE(A @ w, y).

    Parameters
    ----------
    V : np.ndarray, shape (M, N, C)
        Member val predictions.
    Y : np.ndarray, shape (N, C)
        Val targets.
    method : str
        - "nnls": non-negative least squares (interpretable, non-neg only)
        - "lstsq": unconstrained least squares (best fit, can overfit)
        - "ridge": L2-regularized least squares (reduces overfit on sparse
          pathologies; alpha controls regularization strength)
    ridge_alpha : float
        Regularization strength for ridge (ignored unless method="ridge").

    Returns
    -------
    W : np.ndarray, shape (M, C)
        Per-pathology weights, normalized to sum to 1 per column.
        Negative weights are clipped to 0 before normalization for lstsq/ridge
        so the final ensemble stays in the convex hull of member predictions.
    fitted_mse_per : np.ndarray, shape (C,)
        Val MSE per pathology under the fitted weights.
    """
    M, N, C = V.shape
    W = np.zeros((M, C), dtype=np.float64)
    fitted_mse_per = np.zeros(C, dtype=np.float64)

    _nnls = None
    if method == "nnls":
        try:
            from scipy.optimize import nnls as _nnls  # type: ignore
        except ImportError:
            print(
                "WARNING: scipy not available; falling back to "
                "unconstrained lstsq + clip-and-normalize."
            )
            method = "lstsq"

    for c in range(C):
        A_full = V[:, :, c].T  # (N, M)
        y_full = Y[:, c]  # (N,)
        # CheXpert labels are NaN when a pathology wasn't annotated for that
        # patient. Drop those rows for this pathology's fit.
        mask = ~np.isnan(y_full)
        A = A_full[mask]
        y = y_full[mask]
        if len(y) == 0:
            raise ValueError(f"No non-NaN val rows for pathology index {c}")

        if method == "nnls":
            assert _nnls is not None
            w, _ = _nnls(A, y)
        elif method == "ridge":
            # Closed-form ridge: w = (A^T A + alpha * I)^-1 A^T y
            AtA = A.T @ A
            Aty = A.T @ y
            reg = ridge_alpha * np.eye(M)
            w = np.linalg.solve(AtA + reg, Aty)
            w = np.clip(w, 0.0, None)
        else:  # lstsq
            w, *_ = np.linalg.lstsq(A, y, rcond=None)
            w = np.clip(w, 0.0, None)

        s = float(w.sum())
        if s > 1e-12:
            w = w / s
        else:
            # all-zero weights would happen if every member predicts the
            # opposite sign of y on this pathology -- vanishingly unlikely
            # but fall back to uniform so we don't crash
            w = np.full(M, 1.0 / M)
        W[:, c] = w
        fitted = A @ w
        fitted_mse_per[c] = float(((fitted - y) ** 2).mean())

    return W, fitted_mse_per


def _val_study_ids(val_frac: float, seed: int, n_val: int) -> np.ndarray:
    """Reconstruct the per-row study id for the validation split.

    The train/val split is deterministic given (val_frac, seed), so this
    rebuilds the exact val rows that were cached in val_targets.npy and the
    per-member val preds, then returns their study ids. The length check
    guards against a silently changed train CSV or split, which would
    misalign the study ids against the cached predictions.
    """
    full_df = load_train_df()
    _, val_df = train_val_split(full_df, val_frac=val_frac, seed=seed)
    if len(val_df) != n_val:
        raise SystemExit(
            f"Reconstructed val_df has {len(val_df)} rows but cached val preds "
            f"have {n_val}. Has the train CSV or split logic changed?"
        )
    return study_ids_from_paths(val_df["Path"]).to_numpy()


def _assign_study_folds(study_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Assign each row to a fold, keeping every row of a study together.

    Folding by study rather than by row matters because the views of one
    study share a single study-level label and produce near-identical
    predictions. Splitting them across folds would let the combiner train on
    a study it is later scored on, leaking the label and making the
    out-of-fold estimate look better than it really generalizes.
    """
    unique_studies = np.unique(study_ids)
    rng = np.random.default_rng(seed)
    shuffled = unique_studies[rng.permutation(len(unique_studies))]
    fold_of_study = {study: i % n_folds for i, study in enumerate(shuffled)}
    return np.array([fold_of_study[s] for s in study_ids])


def _oof_predictions(
    V: np.ndarray,
    Y: np.ndarray,
    study_ids: np.ndarray,
    *,
    method: str,
    ridge_alpha: float,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Out-of-fold per-disease predictions for the validation rows.

    Each fold's weights are fit on the other folds and applied to the
    held-out fold, so no row is ever scored by weights that were fit on it.
    Feeding the result to ``compute_regression_metrics`` gives an MSE/NMSE
    that tracks the leaderboard closely, whereas the in-sample fit is potentially
    optimistic and weights can chase noise.

    Returns predictions shaped (N, C); rows whose fold produced no held-out
    members stay NaN and are ignored downstream by the metric's masking.
    """
    N, C = Y.shape
    row_fold = _assign_study_folds(study_ids, n_folds, seed)
    oof = np.full((N, C), np.nan, dtype=np.float64)
    for k in range(n_folds):
        held_out = row_fold == k
        if not held_out.any():
            continue
        W_fold, _ = _fit_per_disease_weights(
            V[:, ~held_out, :], Y[~held_out, :], method=method, ridge_alpha=ridge_alpha,
        )
        oof[held_out] = np.einsum("mc,mnc->nc", W_fold, V[:, held_out, :])
    return oof


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fit per-disease lstsq weights on a trained ensemble run.",
    )
    p.add_argument("run_dir", type=Path, help="Path to a trained ensemble run dir.")
    p.add_argument(
        "--submit",
        action="store_true",
        help="Also generate submission_per_disease.csv with view-average.",
    )
    p.add_argument(
        "--method",
        choices=("nnls", "lstsq", "ridge"),
        default="nnls",
        help=(
            "Fitting method: nnls (non-negative, default), lstsq (unconstrained "
            "then clip + normalize), or ridge (L2-regularized, then clip + "
            "normalize). Ridge usually generalizes better than lstsq on sparse "
            "pathologies."
        ),
    )
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=1.0,
        help="L2 regularization strength for method=ridge (default 1.0).",
    )
    p.add_argument(
        "--no-nnls",
        action="store_true",
        help=(
            "DEPRECATED: equivalent to --method lstsq. Kept for backward "
            "compatibility with earlier runs."
        ),
    )
    p.add_argument(
        "--oof-folds",
        type=int,
        default=0,
        help=(
            "If >1, also report study-aware K-fold out-of-fold macro MSE and NMSE."
        ),
    )
    p.add_argument(
        "--oof-seed",
        type=int,
        default=42,
        help="Seed for OOF fold assignment (only used with --oof-folds).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Submission CSV path; defaults to "
            "<run_dir>/submission_<method>.csv (or per_disease.csv for nnls)."
        ),
    )
    p.add_argument(
        "--private-set",
        action="store_true",
        dest="private_set",
        help=(
            "Load test preds and write the submission using solution_ids.csv "
            "(the private held-out set) instead of test_ids.csv. Stale "
            "public-set caches with the wrong row count are automatically "
            "skipped."
        ),
    )
    args = p.parse_args()
    if args.no_nnls:
        args.method = "lstsq"

    run_dir: Path = args.run_dir.resolve()
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")

    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        sys.exit(f"No config.yaml in {run_dir}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    members: list[str] = cfg.get("members") or []
    if not members:
        sys.exit(f"config.yaml in {run_dir} has no 'members' list")
    tta = cfg.get("tta_transforms")
    tta_suffix = _tta_cache_suffix(tta)

    print(f"Ensemble:       {run_dir.name}")
    print(f"Members:        {len(members)}")
    print(f"TTA transforms: {tta}")
    print(f"Cache suffix:   {tta_suffix or '(none)'}")

    member_dirs: list[Path] = []
    for m in members:
        path = Path(m)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_dir():
            sys.exit(f"Member dir missing: {path}")
        member_dirs.append(path)

    val_targets_path = run_dir / "val_targets.npy"
    if not val_targets_path.exists():
        sys.exit(
            f"No val_targets.npy in {run_dir}; the ensemble's train() must "
            "have completed first."
        )
    Y_val = np.load(val_targets_path).astype(np.float64)
    N_val, C = Y_val.shape
    if C != len(LABEL_NAMES):
        sys.exit(f"val_targets.npy has {C} columns, expected {len(LABEL_NAMES)}")
    print(f"Val targets:    shape {Y_val.shape}")

    print("\n=== Loading cached val preds ===")
    val_stack: list[np.ndarray] = []
    for d in member_dirs:
        preds = _load_member_preds(d, CACHED_VAL, tta_suffix, expected_shape=(N_val, C))
        if preds.shape != (N_val, C):
            sys.exit(f"{d.name} val preds shape {preds.shape} != ({N_val}, {C})")
        val_stack.append(preds.astype(np.float64))
        print(f"  loaded {d.name}: {preds.shape}")
    V = np.stack(val_stack, axis=0)  # (M, N_val, C)
    M = V.shape[0]
    print(f"Stacked val preds: {V.shape}")

    # CheXpert labels are NaN where the patient was not annotated for a given
    # pathology. Compute per-pathology counts of valid val rows.
    valid_per_path = (~np.isnan(Y_val)).sum(axis=0)
    print(f"Valid val rows per pathology: {valid_per_path.tolist()}")
    if (valid_per_path == 0).any():
        sys.exit("Some pathology has zero non-NaN val labels; cannot fit weights.")

    # Uniform mean baseline (matches MeanCombiner with weights=null).
    # Use per-pathology NaN masks so the comparison is apples-to-apples
    # vs the fitted weights below.
    uniform_pred = V.mean(axis=0)  # (N_val, C)
    uniform_mse_per = np.zeros(C, dtype=np.float64)
    for c in range(C):
        mask = ~np.isnan(Y_val[:, c])
        uniform_mse_per[c] = float(((uniform_pred[mask, c] - Y_val[mask, c]) ** 2).mean())
    uniform_macro = float(uniform_mse_per.mean())
    print(f"\nUniform-mean val macro MSE: {uniform_macro:.6f}")

    method_label = {
        "nnls": "NNLS",
        "lstsq": "unconstrained-lstsq",
        "ridge": f"Ridge(alpha={args.ridge_alpha})",
    }[args.method]
    print(f"\n=== Fitting per-disease weights ({method_label}) ===")
    W, fitted_mse_per = _fit_per_disease_weights(
        V, Y_val, method=args.method, ridge_alpha=args.ridge_alpha,
    )
    for c in range(C):
        delta = fitted_mse_per[c] - uniform_mse_per[c]
        print(
            f"  {LABEL_NAMES[c]:<30} "
            f"uniform={uniform_mse_per[c]:.4f}  "
            f"fitted={fitted_mse_per[c]:.4f}  "
            f"delta={delta:+.4f}"
        )

    fitted_macro = float(fitted_mse_per.mean())
    delta_macro = fitted_macro - uniform_macro
    print(f"\nPer-disease fitted val macro MSE: {fitted_macro:.6f}")
    print(f"Delta vs uniform: {delta_macro:+.6f}")

    oof_summary = None
    if args.oof_folds and args.oof_folds > 1:
        print(
            f"\n=== Out-of-fold evaluation ({args.oof_folds}-fold, study-aware) ==="
        )
        names = list(LABEL_NAMES)
        study_ids = _val_study_ids(
            float(cfg.get("val_frac", 0.1)), int(cfg.get("seed", 42)), N_val,
        )
        oof_pred = _oof_predictions(
            V, Y_val, study_ids,
            method=args.method, ridge_alpha=args.ridge_alpha,
            n_folds=args.oof_folds, seed=args.oof_seed,
        )
        insample_pred = np.einsum("mc,mnc->nc", W, V)
        # Reuse the shared metric so OOF, in-sample, and the uniform baseline
        # are all normalized by Var(Y)
        oof_m = compute_regression_metrics(Y_val, oof_pred, names)
        ins_m = compute_regression_metrics(Y_val, insample_pred, names)
        uniform_m = compute_regression_metrics(Y_val, uniform_pred, names)
        for name in names:
            print(
                f"  {name:<28} "
                f"insample NMSE={ins_m['per_label'][name]['nmse']:.4f}  "
                f"OOF NMSE={oof_m['per_label'][name]['nmse']:.4f}  "
                f"(uniform {uniform_m['per_label'][name]['nmse']:.4f})"
            )
        print(f"\n  Uniform mean : NMSE={uniform_m['macro_nmse']:.6f}")
        print(
            f"  In-sample fit: MSE={ins_m['macro_mse']:.6f}  "
            f"NMSE={ins_m['macro_nmse']:.6f}"
        )
        print(
            f"  Out-of-fold  : MSE={oof_m['macro_mse']:.6f}  "
            f"NMSE={oof_m['macro_nmse']:.6f}"
        )
        print(
            f"  Optimism (insample - OOF): "
            f"MSE={ins_m['macro_mse'] - oof_m['macro_mse']:+.6f}  "
            f"NMSE={ins_m['macro_nmse'] - oof_m['macro_nmse']:+.6f}"
        )
        oof_summary = {
            "n_folds": args.oof_folds,
            "oof_seed": args.oof_seed,
            "oof_macro_mse": oof_m["macro_mse"],
            "oof_macro_nmse": oof_m["macro_nmse"],
            "insample_macro_mse": ins_m["macro_mse"],
            "insample_macro_nmse": ins_m["macro_nmse"],
            "uniform_macro_nmse": uniform_m["macro_nmse"],
            "oof_per_class_nmse": {
                name: oof_m["per_label"][name]["nmse"] for name in names
            },
        }

    # Method-tagged filenames so different combiner runs don't overwrite each other.
    method_tag = args.method
    suffix = "_PRIVATE" if args.private_set else "_PUBLIC"
    try:
        probe = run_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        artifact_dir = run_dir
        weights_path = artifact_dir / f"per_disease_weights_{method_tag}.npy"
        summary_path = artifact_dir / f"per_disease_summary_{method_tag}.json"
    except PermissionError:
        artifact_dir = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        weights_path = artifact_dir / (
            f"{run_dir.name}_per_disease_weights_{method_tag}{suffix}.npy"
        )
        summary_path = artifact_dir / (
            f"{run_dir.name}_per_disease_summary_{method_tag}{suffix}.json"
        )
        print(f"Note: run dir not writable; artifacts -> {artifact_dir}")
    np.save(weights_path, W.astype(np.float32))
    print(f"\nSaved weights: {weights_path}")

    summary = {
        "ensemble_run_dir": str(run_dir),
        "members": [d.name for d in member_dirs],
        "tta_transforms": tta,
        "method": method_label,
        "ridge_alpha": args.ridge_alpha if args.method == "ridge" else None,
        "uniform_val_macro_mse": uniform_macro,
        "fitted_val_macro_mse": fitted_macro,
        "delta_macro": delta_macro,
        "oof": oof_summary,
        "per_disease": [
            {
                "label": LABEL_NAMES[c],
                "uniform_mse": float(uniform_mse_per[c]),
                "fitted_mse": float(fitted_mse_per[c]),
                "weights": [float(W[m, c]) for m in range(M)],
            }
            for c in range(C)
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    if not args.submit:
        print("\n(Skipping submission generation; pass --submit to write CSV.)")
        return

    print("\n=== Loading cached test preds ===")
    test_df = load_test_df(private=args.private_set)
    N_test = len(test_df)
    test_stack: list[np.ndarray] = []
    for d in member_dirs:
        preds = _load_member_preds(d, CACHED_TEST, tta_suffix, expected_shape=(N_test, C))
        if preds.shape != (N_test, C):
            sys.exit(
                f"{d.name} test preds shape {preds.shape} != ({N_test}, {C}). "
                "Did you run the uniform submit job first to populate the test cache?"
            )
        test_stack.append(preds.astype(np.float64))
        print(f"  loaded {d.name}: {preds.shape}")
    T = np.stack(test_stack, axis=0)  # (M, N_test, C)

    # Per-disease weighted combine: combined[n, c] = sum_m W[m, c] * T[m, n, c]
    combined = np.einsum("mc,mnc->nc", W, T)
    print(f"Combined test preds: {combined.shape}")

    submission = pd.DataFrame(combined, columns=list(LABEL_NAMES))
    if "Id" in test_df.columns:
        submission.insert(0, "Id", test_df["Id"].to_numpy())
    else:
        submission.insert(0, "Id", range(len(submission)))

    submission = _study_view_average(submission, test_df)

    if args.output is not None:
        out = args.output
    elif artifact_dir == run_dir:
        # Run dir is writable; use historical filenames so existing tooling matches.
        if args.method == "nnls":
            out = run_dir / "submission_per_disease.csv"
        else:
            out = run_dir / f"submission_per_disease_{method_tag}.csv"
    else:
        # Run dir not writable; fall back to artifact dir with a disambiguating name.
        out = artifact_dir / (
            f"{run_dir.name}_submission_per_disease_{method_tag}{suffix}.csv"
        )
    submission.to_csv(out, index=False)
    print(f"\nSubmission saved: {out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
