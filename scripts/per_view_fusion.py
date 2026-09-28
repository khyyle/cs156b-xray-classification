"""
Replace the uniform view-averaging step at submit time with per-class
learned frontal/lateral fusion fit on val.

Background
----------
Current pipeline: ``submit.py --view-average`` does a row-wise mean across
all rows of the same ``pidXXXXX/studyN``. That treats frontal and lateral
views as exchangeable for every pathology -- but Cardiomegaly is dominated
by frontal-projection geometry, while pleural lesions vary across views,
etc. Uniform averaging is the simplest fusion of heterogeneous view
signals; a per-class learned aggregator should do strictly better in
expectation.

This script fits, for each pathology ``k`` independently::

    s_study,k = b_k + w_F,k * s_F,k + w_L,k * s_L,k

where ``s_F,k`` is the mean of frontal rows for that study and ``s_L,k``
is the mean of lateral rows. Single-view studies use the available
view's mean directly. The fit is regularized toward the uniform prior
``(b, w_F, w_L) = (0, 0.5, 0.5)`` so per-class weights only deviate when
the data clearly supports it -- this protects against overfitting on
classes with few both-view val studies.

Inputs (read-only):
    <run_dir>/config.yaml         (ensemble config: members, tta, val_frac, seed)
    <run_dir>/ensemble_val_preds.npy  (N_val, 9) per-row ensembled val scores
    <run_dir>/val_targets.npy     (N_val, 9) raw labels in {-1, 0, +1, NaN}
    per-member cached_test_preds[__tta-...].npy  (needed for --submit)

Outputs:
    <run_dir or <user>_outputs>/per_view_fusion_weights.npy    (3, 9): rows are [b, w_F, w_L]
    <run_dir or <user>_outputs>/per_view_fusion_summary.json   diagnostics
    <output>                                                   submission CSV (with --submit)

Usage:
    python scripts/per_view_fusion.py runs/ensemble-kitchen-sink-v4_20260524_171917
    python scripts/per_view_fusion.py <run_dir> --submit
    python scripts/per_view_fusion.py <run_dir> --submit --private-set
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
from radiology_cls.utils import PROJECT_ROOT

PROJECT_PRED_CACHE_ROOT = PROJECT_ROOT / "cache" / "ensemble_predictions"
CACHED_VAL = "cached_val_preds"
CACHED_TEST = "cached_test_preds"


def _tta_cache_suffix(tta_transforms: list[str] | None) -> str:
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
    """Locate a cached preds file with row-count tag preference.

    For test caches, looks first for the row-count-tagged filename
    (e.g. ``cached_test_preds__rows22596``) matching
    ``expected_shape[0]`` so public and private caches don't clobber
    each other on disk. Falls back to the legacy untagged filename
    when the tagged version isn't there yet, validated by shape so an
    old cache from a different test set is correctly rejected.
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
        f"No cached preds for {member_dir.name} ({base_name}{tta_suffix})."
    )


def _combine_members(
    member_dirs: list[Path],
    base_name: str,
    tta_suffix: str,
    weights: np.ndarray,
    expected_shape: tuple,
) -> np.ndarray:
    """Stack per-member cached preds and combine with weights.

    Accepts either:
      - shape ``(M,)`` weights: scalar weight per member, broadcast over
        all pathologies (e.g. uniform 1/M, or one-vector for NNLS/Ridge
        when re-fit with a single weight per member).
      - shape ``(M, C)`` weights: per-pathology weight per member, as
        produced by ``per_disease_weights.py``. Each output cell is then
        the dot product of the per-pathology weight vector with the
        per-member preds for that cell.
    """
    stack = []
    for d in member_dirs:
        p = _find_cache(d, base_name, tta_suffix, expected_shape=expected_shape)
        arr = np.load(p)
        if arr.shape != expected_shape:
            raise ValueError(
                f"{d.name} preds shape {arr.shape} != expected {expected_shape}"
            )
        stack.append(arr.astype(np.float64))
    s = np.stack(stack, axis=0)  # (M, N, C)
    if weights.ndim == 1:
        return np.tensordot(weights, s, axes=([0], [0]))
    if weights.ndim == 2:
        # (M, C) weights -> combined[n, c] = sum_m w[m, c] * s[m, n, c]
        return np.einsum("mc,mnc->nc", weights, s)
    raise ValueError(f"weights must be 1-D or 2-D, got shape {weights.shape}")


def _ensure_frontal_lateral_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``Frontal/Lateral`` column inferred from Path filenames.

    Train CSV has this column natively; test_ids.csv / solution_ids.csv
    do not (they only have ``Id``, ``Path``). The CheXpert path
    convention encodes the view in the filename as
    ``viewN_frontal.jpg`` or ``viewN_lateral.jpg``, and on a 10k-row
    train sample the inferred value matches the explicit
    ``Frontal/Lateral`` column with zero mismatches, so this is the
    correct way to fill it on test data.
    """
    if "Frontal/Lateral" in df.columns:
        return df
    out = df.copy()
    inferred = out["Path"].str.extract(r"view\d+_(\w+)\.jpg").iloc[:, 0].str.title()
    if inferred.isna().any():
        bad = out.loc[inferred.isna(), "Path"].head(5).tolist()
        raise ValueError(
            "Could not infer Frontal/Lateral from at least one Path. "
            f"First few examples: {bad}"
        )
    out["Frontal/Lateral"] = inferred
    return out


def _build_view_aggregates(
    df: pd.DataFrame, preds: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Group per-row preds into per-study frontal and lateral means.

    Parameters:
    -----------
    df: pd.DataFrame
        Must have ``Path`` and ``Frontal/Lateral`` columns. Row ordering
        must align with ``preds``.
    preds: np.ndarray
        Shape ``(N, C)`` per-row scalar scores in ``[-1, 1]``.

    Returns:
    --------
    study_df: pd.DataFrame
        Unique-by-study frame with columns
        ``study_id``, ``has_frontal``, ``has_lateral`` (booleans).
    s_F: np.ndarray
        (S, C) frontal-mean scores per study; zero where ``has_frontal=False``.
    s_L: np.ndarray
        (S, C) lateral-mean scores per study; zero where ``has_lateral=False``.
    row_to_study: np.ndarray
        Length-N integer mapping row index -> study row index in ``study_df``.
    is_frontal: np.ndarray
        Length-N boolean True for frontal rows.
    """
    if len(df) != preds.shape[0]:
        raise ValueError(f"df rows {len(df)} != preds rows {preds.shape[0]}")
    if "Frontal/Lateral" not in df.columns:
        raise ValueError("df is missing required 'Frontal/Lateral' column")
    if "Path" not in df.columns:
        raise ValueError("df is missing required 'Path' column")

    study_ids = study_ids_from_paths(df["Path"]).to_numpy()
    is_frontal = (df["Frontal/Lateral"].to_numpy() == "Frontal")
    is_lateral = (df["Frontal/Lateral"].to_numpy() == "Lateral")

    # Stable, deterministic study ordering keyed to first occurrence.
    unique_ids, inverse = np.unique(study_ids, return_inverse=True)
    S = len(unique_ids)
    C = preds.shape[1]

    # Sum + count for each (study, view); divide for the mean. Numpy
    # add.at handles repeated indices correctly.
    frontal_sum = np.zeros((S, C), dtype=np.float64)
    frontal_count = np.zeros(S, dtype=np.int64)
    lateral_sum = np.zeros((S, C), dtype=np.float64)
    lateral_count = np.zeros(S, dtype=np.int64)

    f_idx = inverse[is_frontal]
    l_idx = inverse[is_lateral]
    np.add.at(frontal_sum, f_idx, preds[is_frontal].astype(np.float64))
    np.add.at(frontal_count, f_idx, 1)
    np.add.at(lateral_sum, l_idx, preds[is_lateral].astype(np.float64))
    np.add.at(lateral_count, l_idx, 1)

    has_frontal = frontal_count > 0
    has_lateral = lateral_count > 0
    s_F = np.where(
        has_frontal[:, None], frontal_sum / np.maximum(frontal_count[:, None], 1), 0.0,
    )
    s_L = np.where(
        has_lateral[:, None], lateral_sum / np.maximum(lateral_count[:, None], 1), 0.0,
    )

    study_df = pd.DataFrame({
        "study_id": unique_ids,
        "has_frontal": has_frontal,
        "has_lateral": has_lateral,
    })
    return study_df, s_F, s_L, inverse, is_frontal


def _per_study_target(targets: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    """Pick the first non-NaN value per study per class.

    CheXpert labels are study-level: all rows of the same study share the
    same label. We take the first non-NaN occurrence to avoid feeding
    a row with NaN into the fit when other rows of the same study have
    a real label.
    """
    S = int(inverse.max()) + 1
    C = targets.shape[1]
    out = np.full((S, C), np.nan, dtype=np.float64)
    # Iterate label-by-label to keep memory simple.
    for c in range(C):
        col = targets[:, c]
        for i in range(len(inverse)):
            v = col[i]
            sidx = inverse[i]
            if np.isnan(out[sidx, c]) and not np.isnan(v):
                out[sidx, c] = v
    return out


def _ridge_t_stats(
    X: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Compute approximate per-coefficient t-stats for a ridge fit.

    Uses the standard sandwich formula

        Var(beta_ridge) = (X'X + alpha I)^-1 X'X (X'X + alpha I)^-1 * sigma^2

    where sigma^2 = RSS / (n - p). This is the Hoerl-Kennard variance for
    ridge; it doesn't correct for the fact that ridge estimates are
    biased, but for a "is this coefficient distinguishable from zero
    given the noise level" sanity check it's the right diagnostic.
    """
    n, p = X.shape
    XtX = X.T @ X
    H = np.linalg.solve(XtX + alpha * np.eye(p), XtX)
    cov_factor = np.linalg.solve(XtX + alpha * np.eye(p), H.T).T
    residuals = y - X @ beta
    rss = float((residuals ** 2).sum())
    df = max(n - p, 1)
    sigma2 = rss / df
    var_beta = sigma2 * np.diag(cov_factor)
    se = np.sqrt(np.maximum(var_beta, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, 0.0)
    return t


def _fit_per_class_convex(
    y: np.ndarray,
    s_F: np.ndarray,
    s_L: np.ndarray,
    has_F: np.ndarray,
    has_L: np.ndarray,
    alpha: float,
    theta_prior: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Fit per-class STRICT convex view combination.

    Model:
        s_study,k = theta_k * s_F + (1 - theta_k) * s_L

    Reparametrized so the constraint ``w_F + w_L = 1`` is enforced
    automatically (no Lagrangian needed). One free parameter per class
    plus a [0, 1] box constraint enforced by clipping.

    Mathematically equivalent to solving the original constrained
    problem with the Lagrangian approach (under non-binding box
    constraints), but cheaper.

    No intercept: predictions stay strictly inside the convex hull of
    ``[s_F, s_L]``, so the [-1, 1] range guarantee from the input
    carries through to the output.

    Parameters:
    -----------
    y: np.ndarray
        (S, C) per-study targets.
    s_F, s_L: np.ndarray
        (S, C) per-study view means.
    has_F, has_L: np.ndarray
        (S,) booleans.
    alpha: float
        Ridge strength on the prior toward ``theta_prior``.
    theta_prior: float
        Target for the prior; ``0.5`` reproduces uniform view-avg as
        ``alpha -> infinity``.

    Returns:
    --------
    W: np.ndarray
        Shape (3, C), rows are [0, theta, 1 - theta] for compatibility
        with ``_apply_fusion``.
    diagnostics: dict
        Per-class (theta, n_both, fit MSE, t-stat for theta-vs-prior).
    """
    C = y.shape[1]
    W = np.zeros((3, C), dtype=np.float64)
    diag = {"per_class": []}
    both = has_F & has_L

    for c in range(C):
        mask = both & ~np.isnan(y[:, c])
        n = int(mask.sum())
        if n == 0:
            theta = theta_prior
            t_stat = 0.0
            fitted_mse = None
            uniform_mse = None
        else:
            # Reparametrized residuals: r = (y - s_L) - theta * (s_F - s_L)
            d = s_F[mask, c] - s_L[mask, c]   # (n,)
            e = y[mask, c] - s_L[mask, c]     # (n,)
            num = float((d * e).sum() + alpha * theta_prior)
            den = float((d * d).sum() + alpha)
            theta_unclipped = num / den
            theta = float(np.clip(theta_unclipped, 0.0, 1.0))
            fitted_pred = theta * s_F[mask, c] + (1.0 - theta) * s_L[mask, c]
            uniform_pred = 0.5 * s_F[mask, c] + 0.5 * s_L[mask, c]
            fitted_mse = float(np.mean((fitted_pred - y[mask, c]) ** 2))
            uniform_mse = float(np.mean((uniform_pred - y[mask, c]) ** 2))
            # Sandwich SE for theta as a single ridge coefficient on d.
            # Var(theta) ~ sigma^2 * (sum d^2) / (sum d^2 + alpha)^2
            residuals = e - theta_unclipped * d
            rss = float((residuals ** 2).sum())
            df = max(n - 1, 1)
            sigma2 = rss / df
            var_theta = sigma2 * float((d * d).sum()) / (den ** 2)
            se = float(np.sqrt(max(var_theta, 0.0)))
            t_stat = float((theta_unclipped - theta_prior) / se) if se > 0 else 0.0
        W[0, c] = 0.0
        W[1, c] = theta
        W[2, c] = 1.0 - theta
        diag["per_class"].append({
            "label": LABEL_NAMES[c],
            "n_both_view_studies": n,
            "fitted_b": 0.0,
            "fitted_w_F": float(theta),
            "fitted_w_L": float(1.0 - theta),
            "uniform_mse": uniform_mse,
            "fitted_mse": fitted_mse,
            "t_theta_vs_prior": t_stat,
        })
    return W, diag


def _fit_per_class_ridge_to_prior(
    y: np.ndarray,
    s_F: np.ndarray,
    s_L: np.ndarray,
    has_F: np.ndarray,
    has_L: np.ndarray,
    alpha: float,
    prior: tuple[float, float, float] = (0.0, 0.5, 0.5),
) -> tuple[np.ndarray, dict]:
    """Fit (b, w_F, w_L) per class via ridge toward a non-zero prior.

    Solves, per class ``c`` and using only both-view studies:
        minimize sum_{i} (y_i - b - w_F s_F,i - w_L s_L,i)^2
                 + alpha * ((b - p_b)^2 + (w_F - p_F)^2 + (w_L - p_L)^2)

    Closed-form: let A be (n, 3) features ``[1, s_F, s_L]`` and beta_p
    the prior. Then
        beta = (A^T A + alpha I)^-1 (A^T y + alpha beta_p)

    Parameters:
    -----------
    y: np.ndarray
        (S, C) per-study targets; rows with NaN in column c skipped for c.
    s_F, s_L: np.ndarray
        (S, C) per-study view means.
    has_F, has_L: np.ndarray
        (S,) booleans indicating view availability per study.
    alpha: float
        Ridge regularization toward the prior.
    prior: (float, float, float)
        (b_prior, w_F_prior, w_L_prior). ``(0, 0.5, 0.5)`` reproduces
        uniform view-averaging when alpha is large.

    Returns:
    --------
    W: np.ndarray
        Shape (3, C); rows are [b, w_F, w_L].
    diagnostics: dict
        Per-class fit info including uniform-vs-fitted val MSE.
    """
    C = y.shape[1]
    W = np.zeros((3, C), dtype=np.float64)
    diag = {"per_class": []}
    beta_prior = np.asarray(prior, dtype=np.float64)
    both = has_F & has_L

    for c in range(C):
        mask = both & ~np.isnan(y[:, c])
        n = int(mask.sum())
        if n == 0:
            # No both-view studies with this label; fall back to prior.
            W[:, c] = beta_prior
            diag["per_class"].append({
                "label": LABEL_NAMES[c],
                "n_both_view_studies": 0,
                "fitted_b": float(beta_prior[0]),
                "fitted_w_F": float(beta_prior[1]),
                "fitted_w_L": float(beta_prior[2]),
                "uniform_mse": None,
                "fitted_mse": None,
            })
            continue

        A = np.stack([
            np.ones(n, dtype=np.float64),
            s_F[mask, c],
            s_L[mask, c],
        ], axis=1)
        yv = y[mask, c]
        # Closed-form ridge with shifted prior.
        AtA = A.T @ A
        Aty = A.T @ yv
        beta = np.linalg.solve(AtA + alpha * np.eye(3), Aty + alpha * beta_prior)
        W[:, c] = beta

        # Diagnostic: uniform = (0.5, 0.5) vs fitted
        uniform_pred = 0.5 * s_F[mask, c] + 0.5 * s_L[mask, c]
        fitted_pred = beta[0] + beta[1] * s_F[mask, c] + beta[2] * s_L[mask, c]
        uniform_mse = float(np.mean((uniform_pred - yv) ** 2))
        fitted_mse = float(np.mean((fitted_pred - yv) ** 2))
        # T-stats per coefficient. We report them relative to zero (Wald-style)
        # so they show "is this coefficient nonzero given noise?", which is
        # the useful diagnostic when checking if the per-view fit signal
        # could be confused for noise on a per-class basis.
        t_stats = _ridge_t_stats(A, yv, beta, alpha=alpha)
        diag["per_class"].append({
            "label": LABEL_NAMES[c],
            "n_both_view_studies": n,
            "fitted_b": float(beta[0]),
            "fitted_w_F": float(beta[1]),
            "fitted_w_L": float(beta[2]),
            "uniform_mse": uniform_mse,
            "fitted_mse": fitted_mse,
            "t_b": float(t_stats[0]),
            "t_w_F": float(t_stats[1]),
            "t_w_L": float(t_stats[2]),
        })

    return W, diag


def _apply_fusion(
    s_F: np.ndarray, s_L: np.ndarray, has_F: np.ndarray, has_L: np.ndarray, W: np.ndarray,
) -> np.ndarray:
    """Combine per-study view means with per-class weights.

    For each study and each class:
      - Both views present: ``b + w_F * s_F + w_L * s_L``
      - Frontal only: ``s_F`` (the available view's mean, no per-class twist)
      - Lateral only: ``s_L``
      - Neither: zero (won't happen for real test data but kept defensive)

    Single-view studies bypass the learned per-class weights because the
    fit only saw both-view studies; extrapolating per-class weights to
    e.g. a frontal-only study would inject the lateral coefficient
    against an implicit ``s_L = 0`` and degrade predictions.
    """
    S = s_F.shape[0]
    C = s_F.shape[1]
    out = np.zeros((S, C), dtype=np.float64)
    both = has_F & has_L
    only_F = has_F & ~has_L
    only_L = ~has_F & has_L
    if both.any():
        bot = (
            W[0, :][None, :]
            + W[1, :][None, :] * s_F[both]
            + W[2, :][None, :] * s_L[both]
        )
        out[both] = bot
    if only_F.any():
        out[only_F] = s_F[only_F]
    if only_L.any():
        out[only_L] = s_L[only_L]
    return out


def _broadcast_study_to_rows(
    study_preds: np.ndarray, inverse: np.ndarray,
) -> np.ndarray:
    """Map (S, C) per-study preds back to the (N, C) row order."""
    return study_preds[inverse]


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fit per-class frontal/lateral ridge fusion on val and optionally "
            "apply at submit time as a replacement for uniform view-averaging."
        ),
    )
    p.add_argument("run_dir", type=Path)
    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help=(
            "Strength of regularization toward the uniform (0, 0.5, 0.5) "
            "prior. Higher = stays closer to uniform averaging. Default 10."
        ),
    )
    p.add_argument(
        "--convex",
        action="store_true",
        help=(
            "Enforce a strict convex view combination: no intercept, "
            "w_F + w_L = 1 with both >= 0. Guarantees outputs stay in "
            "[-1, 1]. Removes 2 degrees of freedom per class (likely "
            "slightly worse on val, possibly better on LB if the val "
            "intercept/amplification was overfitting selection bias)."
        ),
    )
    p.add_argument(
        "--per-disease-weights",
        type=Path,
        default=None,
        help=(
            "Optional .npy file of per-pathology member weights of shape "
            "(M, C), as written by scripts/per_disease_weights.py. When "
            "provided, members are combined with these weights (Stage A) "
            "before per-view fusion (Stage B). Without this flag, members "
            "are combined with a uniform 1/M mean."
        ),
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Also load cached test preds and write a per-view-fused submission CSV.",
    )
    p.add_argument(
        "--private-set",
        action="store_true",
        dest="private_set",
        help=(
            "Apply to the private held-out set (solution_ids.csv) instead "
            "of the public test set (test_ids.csv)."
        ),
    )
    p.add_argument("--output", type=Path, default=None, help="Submission CSV path override.")
    args = p.parse_args()

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
    val_frac = float(cfg.get("val_frac", 0.1))
    seed = int(cfg.get("seed", 42))
    tta_suffix = _tta_cache_suffix(cfg.get("tta_transforms"))

    print(f"Run dir:        {run_dir.name}")
    print(f"Members:        {len(members)}")
    print(f"TTA:            {cfg.get('tta_transforms')}")
    print(f"Ridge alpha:    {args.ridge_alpha}")
    print(f"Split:          val_frac={val_frac}, seed={seed}")

    # Resolve member dirs to absolute paths so cached preds lookups work
    # uniformly regardless of cwd.
    member_dirs: list[Path] = []
    for m in members:
        path = Path(m)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.is_dir():
            sys.exit(f"Member dir missing: {path}")
        member_dirs.append(path)

    # === Phase 1: fit on val ===
    val_targets_path = run_dir / "val_targets.npy"
    if not val_targets_path.exists():
        sys.exit(
            f"No val_targets.npy in {run_dir}; train the ensemble first to "
            "materialize it."
        )
    y_val_rows = np.load(val_targets_path).astype(np.float64)
    N_val, C = y_val_rows.shape
    if C != len(LABEL_NAMES):
        sys.exit(f"val_targets shape mismatch: {C} != {len(LABEL_NAMES)}")

    # Stage A: choose member-combination weights. Uniform unless the user
    # provided a per-disease weights file (from per_disease_weights.py).
    if args.per_disease_weights is not None:
        if not args.per_disease_weights.exists():
            sys.exit(f"--per-disease-weights file not found: {args.per_disease_weights}")
        weights = np.load(args.per_disease_weights).astype(np.float64)
        expected_w_shape = (len(member_dirs), C)
        if weights.shape != expected_w_shape:
            sys.exit(
                f"--per-disease-weights shape {weights.shape} != {expected_w_shape}"
            )
        print(f"Stage A: per-pathology weights from {args.per_disease_weights.name}")
        # When using per-pathology weights, ensemble_val_preds.npy from
        # the ensemble's uniform-mean train run is the wrong baseline;
        # always reconstruct from the per-member caches with our weights.
        val_preds = _combine_members(
            member_dirs, CACHED_VAL, tta_suffix, weights, (N_val, C),
        )
    else:
        print("Stage A: uniform-mean member combination")
        weights = np.full(len(member_dirs), 1.0 / len(member_dirs))
        val_preds_path = run_dir / "ensemble_val_preds.npy"
        if val_preds_path.exists():
            val_preds = np.load(val_preds_path).astype(np.float64)
            if val_preds.shape != (N_val, C):
                sys.exit(
                    f"ensemble_val_preds shape {val_preds.shape} != ({N_val}, {C})"
                )
            print(f"Loaded ensemble_val_preds.npy: {val_preds.shape}")
        else:
            print("Reconstructing val preds from per-member caches (no ensemble_val_preds.npy)")
            val_preds = _combine_members(
                member_dirs, CACHED_VAL, tta_suffix, weights, (N_val, C),
            )

    # Re-derive the same val_df rows the ensemble trained against. The
    # split is deterministic given (val_frac, seed); reset_index(drop=True)
    # already happened inside train_val_split, so this aligns 1:1 with
    # val_preds and val_targets.
    full_df = load_train_df()
    _, val_df = train_val_split(full_df, val_frac=val_frac, seed=seed)
    if len(val_df) != N_val:
        sys.exit(
            f"Reconstructed val_df has {len(val_df)} rows but cached val "
            f"preds have {N_val}. Has the train CSV or split logic changed?"
        )

    study_df_val, s_F_val, s_L_val, inv_val, _ = _build_view_aggregates(val_df, val_preds)
    y_study_val = _per_study_target(y_val_rows, inv_val)
    S_val = len(study_df_val)
    n_both = int((study_df_val["has_frontal"] & study_df_val["has_lateral"]).sum())
    n_only_F = int((study_df_val["has_frontal"] & ~study_df_val["has_lateral"]).sum())
    n_only_L = int((~study_df_val["has_frontal"] & study_df_val["has_lateral"]).sum())
    print(
        f"Val studies: total={S_val}, both-view={n_both}, frontal-only={n_only_F}, "
        f"lateral-only={n_only_L}"
    )

    if args.convex:
        W, fit_diag = _fit_per_class_convex(
            y=y_study_val,
            s_F=s_F_val,
            s_L=s_L_val,
            has_F=study_df_val["has_frontal"].to_numpy(),
            has_L=study_df_val["has_lateral"].to_numpy(),
            alpha=args.ridge_alpha,
        )
    else:
        W, fit_diag = _fit_per_class_ridge_to_prior(
            y=y_study_val,
            s_F=s_F_val,
            s_L=s_L_val,
            has_F=study_df_val["has_frontal"].to_numpy(),
            has_L=study_df_val["has_lateral"].to_numpy(),
            alpha=args.ridge_alpha,
        )

    # Uniform baseline at the row level so the comparison is the
    # apples-to-apples thing that --view-average would produce.
    uniform_study_preds = _apply_fusion(
        s_F_val, s_L_val,
        study_df_val["has_frontal"].to_numpy(),
        study_df_val["has_lateral"].to_numpy(),
        W=np.array([[0.0] * C, [0.5] * C, [0.5] * C]),  # b, w_F, w_L
    )
    fitted_study_preds = _apply_fusion(
        s_F_val, s_L_val,
        study_df_val["has_frontal"].to_numpy(),
        study_df_val["has_lateral"].to_numpy(),
        W=W,
    )
    # Broadcast study preds back to row order, then compute MSE against the
    # original row-level targets (with NaN masking).
    uniform_row_preds = _broadcast_study_to_rows(uniform_study_preds, inv_val)
    fitted_row_preds = _broadcast_study_to_rows(fitted_study_preds, inv_val)

    uniform_per = np.zeros(C, dtype=np.float64)
    fitted_per = np.zeros(C, dtype=np.float64)
    for c in range(C):
        m = ~np.isnan(y_val_rows[:, c])
        uniform_per[c] = float(np.mean((uniform_row_preds[m, c] - y_val_rows[m, c]) ** 2))
        fitted_per[c] = float(np.mean((fitted_row_preds[m, c] - y_val_rows[m, c]) ** 2))

    uniform_macro = float(uniform_per.mean())
    fitted_macro = float(fitted_per.mean())
    delta_macro = fitted_macro - uniform_macro

    print()
    print("Per-class fit (val MSE on row-level masked labels):")
    for c in range(C):
        pc = fit_diag["per_class"][c]
        if args.convex:
            t_info = f"  t(theta-vs-0.5)={pc.get('t_theta_vs_prior', 0.0):+.2f}"
        else:
            t_info = (
                f"  t(b)={pc.get('t_b', 0.0):+.2f} "
                f"t(w_F)={pc.get('t_w_F', 0.0):+.2f} "
                f"t(w_L)={pc.get('t_w_L', 0.0):+.2f}"
            )
        print(
            f"  {LABEL_NAMES[c]:<30} "
            f"uniform={uniform_per[c]:.4f}  "
            f"fitted={fitted_per[c]:.4f}  "
            f"delta={fitted_per[c] - uniform_per[c]:+.4f}  "
            f"(b={W[0, c]:+.3f} w_F={W[1, c]:+.3f} w_L={W[2, c]:+.3f}){t_info}"
        )
    print(f"\nUniform view-avg macro MSE: {uniform_macro:.6f}")
    print(f"Per-view-fused macro MSE:  {fitted_macro:.6f}")
    print(f"Delta vs uniform:          {delta_macro:+.6f}")

    # === Phase 2: persist weights + summary ===
    suffix = "_PRIVATE" if args.private_set else "_PUBLIC"
    try:
        probe = run_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        artifact_dir = run_dir
        weights_path = artifact_dir / "per_view_fusion_weights.npy"
        summary_path = artifact_dir / "per_view_fusion_summary.json"
    except PermissionError:
        artifact_dir = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        weights_path = artifact_dir / f"{run_dir.name}_per_view_fusion_weights{suffix}.npy"
        summary_path = artifact_dir / f"{run_dir.name}_per_view_fusion_summary{suffix}.json"
        print(f"Note: run dir not writable; artifacts -> {artifact_dir}")
    np.save(weights_path, W.astype(np.float32))
    print(f"Saved weights: {weights_path}")
    summary = {
        "run_dir": str(run_dir),
        "ridge_alpha": args.ridge_alpha,
        "uniform_val_macro_mse": uniform_macro,
        "fitted_val_macro_mse": fitted_macro,
        "delta_macro": delta_macro,
        "n_val_studies": S_val,
        "n_both_view_studies": n_both,
        "per_class": fit_diag["per_class"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary: {summary_path}")

    if not args.submit:
        print("\n(Skipping submission generation; pass --submit to write CSV.)")
        return

    # === Phase 3: apply at test time ===
    print("\n=== Loading cached test preds ===")
    test_df = load_test_df(private=args.private_set)
    test_df = _ensure_frontal_lateral_column(test_df)
    N_test = len(test_df)
    test_preds = _combine_members(
        member_dirs, CACHED_TEST, tta_suffix, weights, (N_test, C),
    )
    print(f"Test preds shape: {test_preds.shape}")

    study_df_test, s_F_test, s_L_test, inv_test, _ = _build_view_aggregates(test_df, test_preds)
    fused_study = _apply_fusion(
        s_F_test, s_L_test,
        study_df_test["has_frontal"].to_numpy(),
        study_df_test["has_lateral"].to_numpy(),
        W=W,
    )
    fused_rows = _broadcast_study_to_rows(fused_study, inv_test)

    submission = pd.DataFrame(fused_rows, columns=list(LABEL_NAMES))
    if "Id" in test_df.columns:
        submission.insert(0, "Id", test_df["Id"].to_numpy())
    else:
        submission.insert(0, "Id", range(len(submission)))

    if args.output is not None:
        out = args.output
    else:
        # Disambiguate the uniform-mean / per-disease-weighted / convex variants.
        fit_tag = "_convex" if args.convex else ""
        if args.per_disease_weights is not None:
            stem = (
                args.per_disease_weights.stem
                .replace("_PUBLIC", "")
                .replace("_PRIVATE", "")
            )
            tag = f"_per_view{fit_tag}_x_{stem}"
        else:
            tag = f"_per_view{fit_tag}"
        out = artifact_dir / f"{run_dir.name}_submission{tag}{suffix}.csv"
    submission.to_csv(out, index=False)
    print(f"\nSubmission saved: {out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
