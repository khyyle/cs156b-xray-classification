"""Greedy forward selection (with backtracking) over an ensemble member pool.

Operates entirely on cached val predictions, so it's CPU-only and fast --
no GPU or re-inference needed once each candidate has its
cached_val_preds materialized (e.g. by a throwaway ensemble train()).

Selection objective is a weighted combination of two terms computed on the
current subset:

    objective = metric_term + corr_weight * mean_residual_corr

  - metric_term: per-disease NNLS val error, either macro MSE (default) or
    macro NMSE (MSE per class divided by that class's target variance).
  - mean_residual_corr: mean pairwise correlation of member residuals,
    macro-averaged over classes. Lower = more diverse.

The correlation term is a diversity regularizer. The ensemble ambiguity
decomposition (ensemble error = mean individual error - diversity) says the
diversity term is what actually generalizes, whereas NNLS val error is fit
on the same rows it's scored on and overfits. Weighting correlation lets the
search prefer additions that decorrelate the ensemble even when their raw
val-MSE delta is within noise. corr_weight=0 recovers pure val-error
selection.

CAVEAT on the metric: every leaderboard number this project has recorded is
plain macro MSE per the competition instructions. NMSE is provided as an
exploratory option only; don't assume the LB uses it.

Usage:
    python scratch/kyle/greedy_ensemble_select.py <ensemble_run_dir> \\
        --seed-members 0 1 2 ... 14 --corr-weight 0.5 --metric mse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

from radiology_cls.data import LABEL_NAMES
from radiology_cls.utils import PROJECT_ROOT

PROJECT_PRED_CACHE_ROOT = PROJECT_ROOT / "cache" / "ensemble_predictions"
TTA_SUFFIX = "__tta-hflip-identity"
CACHED_VAL = f"cached_val_preds{TTA_SUFFIX}.npy"


def _find_val_preds(member_dir: Path, n_val: int) -> np.ndarray:
    for p in (
        member_dir / CACHED_VAL,
        PROJECT_PRED_CACHE_ROOT / member_dir.name / CACHED_VAL,
        Path.home() / ".cache" / "cs156b_ensemble" / member_dir.name / CACHED_VAL,
    ):
        if p.exists() and np.load(p, mmap_mode="r").shape[0] == n_val:
            return np.load(p).astype(np.float64)
    raise FileNotFoundError(f"No cached val preds for {member_dir.name} (N={n_val})")


def _nnls_per_class_error(
    V: np.ndarray, Y: np.ndarray, metric: str,
) -> tuple[float, np.ndarray]:
    """Per-pathology NNLS fit; return (macro error, per-class error).

    metric="mse": per-class mean squared error.
    metric="nmse": per-class MSE divided by that class's target variance,
        so every class contributes on a comparable scale regardless of
        its label spread.
    """
    try:
        from scipy.optimize import nnls
    except ImportError:
        sys.exit("scipy required for NNLS selection")
    M, N, C = V.shape
    per_class = np.zeros(C)
    for c in range(C):
        A_full = V[:, :, c].T  # (N, M)
        y_full = Y[:, c]
        mask = ~np.isnan(y_full)
        A, y = A_full[mask], y_full[mask]
        w, _ = nnls(A, y)
        s = w.sum()
        w = w / s if s > 1e-12 else np.full(M, 1.0 / M)
        mse = float(((A @ w - y) ** 2).mean())
        if metric == "nmse":
            var = float(y.var())
            per_class[c] = mse / var if var > 1e-12 else mse
        else:
            per_class[c] = mse
    return float(per_class.mean()), per_class


def _mean_residual_corr(V: np.ndarray, Y: np.ndarray) -> float:
    """Macro mean of pairwise residual correlations across members.

    Residual = member_pred - target, computed per class over non-NaN rows,
    then the off-diagonal mean of the member-by-member correlation matrix,
    macro-averaged over classes. A single-member subset has no pairs and
    returns 1.0 (maximally redundant by convention, so it never looks
    diverse).
    """
    M, N, C = V.shape
    if M < 2:
        return 1.0
    class_means = []
    for c in range(C):
        y = Y[:, c]
        mask = ~np.isnan(y)
        resid = V[:, mask, c] - y[mask]  # (M, n_valid)
        corr = np.corrcoef(resid)
        off = corr[~np.eye(M, dtype=bool)]
        class_means.append(float(np.nanmean(off)))
    return float(np.mean(class_means))


def _objective(
    all_preds: np.ndarray,
    Y: np.ndarray,
    subset: list[int],
    metric: str,
    corr_weight: float,
) -> tuple[float, float, float]:
    """Return (objective, metric_term, corr_term) for a subset."""
    V = all_preds[subset]
    err, _ = _nnls_per_class_error(V, Y, metric)
    corr = _mean_residual_corr(V, Y)
    return err + corr_weight * corr, err, corr


def _marginal_corr(
    all_preds: np.ndarray, Y: np.ndarray, current: list[int], cand: int,
) -> float:
    """Mean residual correlation of a candidate member to the current set.

    Unlike whole-set mean correlation (which grows with set size and makes
    a diversity penalty collapse the ensemble to ~2 members), this measures
    only how redundant the *new* member is relative to what's already in.
    A low value means the candidate brings genuinely new error structure.
    """
    if not current:
        return 0.0
    M, N, C = all_preds.shape
    per_class = []
    for c in range(C):
        y = Y[:, c]
        mask = ~np.isnan(y)
        cand_resid = all_preds[cand, mask, c] - y[mask]
        cors = []
        for m in current:
            other = all_preds[m, mask, c] - y[mask]
            cm = np.corrcoef(cand_resid, other)[0, 1]
            if not np.isnan(cm):
                cors.append(cm)
        if cors:
            per_class.append(float(np.mean(cors)))
    return float(np.mean(per_class)) if per_class else 1.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("run_dir", type=Path)
    p.add_argument("--metric", choices=("mse", "nmse"), default="mse")
    p.add_argument(
        "--corr-weight",
        type=float,
        default=0.0,
        help=(
            "Weight on the mean-residual-correlation diversity term. 0 = pure "
            "val-error selection. Try 0.1-1.0 to prefer decorrelating adds; "
            "val error is ~0.26 and correlation is ~0.96, so corr_weight=1.0 "
            "makes them comparable in magnitude."
        ),
    )
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--seed-members", type=int, nargs="*", default=None)
    p.add_argument(
        "--diversity-gate",
        action="store_true",
        help=(
            "Also accept an add that is metric-neutral (within --neutral-tol) "
            "when the candidate's marginal correlation to the current set is "
            "below --max-marginal-corr. Lets the ensemble grow with "
            "decorrelating members the pure-metric criterion would reject, "
            "without the size-collapse of penalizing whole-set mean corr."
        ),
    )
    p.add_argument("--neutral-tol", type=float, default=1e-4,
                   help="Metric-neutral band for the diversity gate.")
    p.add_argument("--max-marginal-corr", type=float, default=0.95,
                   help="Marginal-corr threshold for the diversity gate.")
    args = p.parse_args()

    run_dir = args.run_dir.resolve()
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    members = cfg["members"]
    Y = np.load(run_dir / "val_targets.npy").astype(np.float64)
    N = Y.shape[0]

    names = [Path(m).name for m in members]
    preds = []
    for m in members:
        d = Path(m)
        if not d.is_absolute():
            d = (PROJECT_ROOT / d).resolve()
        preds.append(_find_val_preds(d, N))
    all_preds = np.stack(preds, axis=0)
    M = all_preds.shape[0]
    print(f"Pool: {M} members, {N} val rows | metric={args.metric} corr_weight={args.corr_weight}")
    for i, nm in enumerate(names):
        print(f"  [{i:2d}] {nm}")

    full_obj, full_err, full_corr = _objective(
        all_preds, Y, list(range(M)), args.metric, args.corr_weight,
    )
    print(
        f"\nFull-pool: {args.metric}={full_err:.6f}  corr={full_corr:.4f}  "
        f"objective={full_obj:.6f}"
    )

    current = list(args.seed_members) if args.seed_members is not None else []
    if not current:
        singles = [
            (_objective(all_preds, Y, [i], args.metric, args.corr_weight)[0], i)
            for i in range(M)
        ]
        singles.sort()
        current = [singles[0][1]]
    cur_obj, cur_err, cur_corr = _objective(
        all_preds, Y, current, args.metric, args.corr_weight,
    )
    print(
        f"\nStart subset {current}\n  {args.metric}={cur_err:.6f}  "
        f"corr={cur_corr:.4f}  objective={cur_obj:.6f}"
    )

    step = 0
    while True:
        step += 1
        best_move = None
        best = (cur_obj, cur_err, cur_corr)

        moves = [("add", i) for i in range(M) if i not in current]
        if not args.forward_only and len(current) > 1:
            moves += [("remove", i) for i in current]

        for op, i in moves:
            cand = current + [i] if op == "add" else [j for j in current if j != i]
            obj, err, corr = _objective(all_preds, Y, cand, args.metric, args.corr_weight)
            if obj < best[0] - 1e-12:
                best, best_move = (obj, err, corr), (op, i)

        # Diversity gate: if no strictly-improving move exists, allow a
        # metric-neutral add whose candidate is sufficiently decorrelated.
        if best_move is None and args.diversity_gate:
            gate_best = None
            gate_best_corr = args.max_marginal_corr
            for i in range(M):
                if i in current:
                    continue
                cand = current + [i]
                _, err, _ = _objective(all_preds, Y, cand, args.metric, args.corr_weight)
                if err <= cur_err + args.neutral_tol:
                    mcorr = _marginal_corr(all_preds, Y, current, i)
                    if mcorr < gate_best_corr:
                        gate_best_corr, gate_best = mcorr, i
            if gate_best is not None:
                current.append(gate_best)
                cur_obj, cur_err, cur_corr = _objective(
                    all_preds, Y, current, args.metric, args.corr_weight,
                )
                print(
                    f"  step {step}: +*[{gate_best:2d}] {names[gate_best]:<43} "
                    f"{args.metric}={cur_err:.6f} corr={cur_corr:.4f} "
                    f"marginal_corr={gate_best_corr:.4f} (diversity gate)"
                )
                continue
            break

        if best_move is None:
            break
        op, idx = best_move
        if op == "add":
            current.append(idx)
        else:
            current.remove(idx)
        cur_obj, cur_err, cur_corr = best
        sign = "+" if op == "add" else "-"
        print(
            f"  step {step}: {sign} [{idx:2d}] {names[idx]:<45} "
            f"{args.metric}={cur_err:.6f} corr={cur_corr:.4f} obj={cur_obj:.6f}"
        )

    current_sorted = sorted(current)
    print(f"\n=== Selected subset ({len(current_sorted)} members) ===")
    for i in current_sorted:
        print(f"  [{i:2d}] {names[i]}")
    print(
        f"\nSelected: {args.metric}={cur_err:.6f}  corr={cur_corr:.4f}  "
        f"objective={cur_obj:.6f}"
    )
    excluded = [i for i in range(M) if i not in current]
    if excluded:
        print("\nExcluded:")
        for i in excluded:
            print(f"  [{i:2d}] {names[i]}")


if __name__ == "__main__":
    main()
