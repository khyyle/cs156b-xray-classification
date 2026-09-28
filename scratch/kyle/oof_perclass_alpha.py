"""Choose each pathology's Stage A regularization separately by out of fold NMSE.

A sweep on ridge alpha showed that reg helps overall, but NNLS
overfits a few sparse classes, with Pleural Other the worst. Rather than
force one global alpha on all nine pathologies, this picks each pathology's
combiner independently. For every class it tries non negative least squares and
ridge at a range of alphas, scores each one by study aware out of fold NMSE for
that class alone, and keeps whichever generalizes best. It then builds the full
v4 pipeline with those per class choices for Stage A, applies the usual convex
view fusion for Stage B, and writes public and private submissions.

Usage
    python scratch/kyle/oof_perclass_alpha.py
    python scratch/kyle/oof_perclass_alpha.py --oof-only
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
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
import oof_portfolio as op  # noqa: E402


def _per_class_nmse(Y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-class NMSE as an array in LABEL_NAMES order.

    Wraps the shared metric so the per-class selection below can index NMSE
    by class position instead of by label name.
    """
    metrics = compute_regression_metrics(Y, pred, list(LABEL_NAMES))
    return np.array([metrics["per_label"][name]["nmse"] for name in LABEL_NAMES])

GRID = [
    ("nnls", 1.0),
    ("ridge", 1.0),
    ("ridge", 5.0),
    ("ridge", 10.0),
    ("ridge", 20.0),
    ("ridge", 50.0),
    ("ridge", 100.0),
    ("ridge", 200.0),
]


def _fit_per_class_custom(
    V: np.ndarray, Y: np.ndarray, choice: list[tuple[str, float]],
) -> np.ndarray:
    """Fit Stage-A member weights, each class using its own (method, alpha)."""
    from scipy.optimize import nnls as _nnls

    M, N, C = V.shape
    W = np.zeros((M, C), dtype=np.float64)
    for c in range(C):
        method, alpha = choice[c]
        A_full = V[:, :, c].T
        y_full = Y[:, c]
        mask = ~np.isnan(y_full)
        A = A_full[mask]
        y = y_full[mask]
        if method == "nnls":
            w, _ = _nnls(A, y)
        else:
            AtA = A.T @ A
            w = np.linalg.solve(AtA + alpha * np.eye(M), A.T @ y)
            w = np.clip(w, 0.0, None)
        s = float(w.sum())
        W[:, c] = w / s if s > 1e-12 else np.full(M, 1.0 / M)
    return W


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof-only", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load((op.RUN_DIR / "config.yaml").read_text())
    member_dirs = op._resolve_members(cfg)
    suffix = pdw._tta_cache_suffix(cfg.get("tta_transforms"))
    val_frac = float(cfg.get("val_frac", 0.1))
    seed = int(cfg.get("seed", 42))

    Y_val = np.load(op.RUN_DIR / "val_targets.npy").astype(np.float64)
    N_val, C = Y_val.shape
    V_val = op._stack(member_dirs, "cached_val_preds", suffix, (N_val, C))
    study_ids = pdw._val_study_ids(val_frac, seed, N_val)

    # Per-class OOF NMSE for every grid entry.
    per_cfg_nmse: dict[tuple, np.ndarray] = {}
    for method, alpha in GRID:
        oof_pred = pdw._oof_predictions(
            V_val, Y_val, study_ids,
            method=method, ridge_alpha=alpha, n_folds=op.OOF_FOLDS, seed=op.OOF_SEED,
        )
        per_cfg_nmse[(method, alpha)] = _per_class_nmse(Y_val, oof_pred)

    # Per class, choose the grid entry with lowest OOF NMSE.
    choice: list[tuple[str, float]] = []
    print(f"\n{'class':<28} {'best cfg':<14} {'OOF NMSE':>9}  {'nnls':>7} {'r50':>7}")
    for c in range(C):
        best = min(GRID, key=lambda ma: per_cfg_nmse[ma][c])
        choice.append(best)
        tag = "nnls" if best[0] == "nnls" else f"ridge{int(best[1])}"
        print(
            f"{LABEL_NAMES[c]:<28} {tag:<14} {per_cfg_nmse[best][c]:>9.4f}"
            f"  {per_cfg_nmse[('nnls',1.0)][c]:>7.4f} {per_cfg_nmse[('ridge',50.0)][c]:>7.4f}"
        )
    macro_perclass = float(np.mean([per_cfg_nmse[choice[c]][c] for c in range(C)]))
    macro_ridge50 = float(np.nanmean(per_cfg_nmse[("ridge", 50.0)]))
    macro_nnls = float(np.nanmean(per_cfg_nmse[("nnls", 1.0)]))
    print(f"\nMacro OOF NMSE  per-class-alpha: {macro_perclass:.5f}")
    print(f"Macro OOF NMSE  ridge_a50       : {macro_ridge50:.5f}")
    print(f"Macro OOF NMSE  nnls            : {macro_nnls:.5f}")

    if args.oof_only:
        return

    # Build full pipeline with the per-class choice and emit submissions.
    full_df = load_train_df()
    _, val_df = train_val_split(full_df, val_frac=val_frac, seed=seed)
    W_a = _fit_per_class_custom(V_val, Y_val, choice)
    val_comb = np.einsum("mc,mnc->nc", W_a, V_val)
    Wv = op._fit_view_weights(val_df, val_comb, Y_val)

    for phase, private in (("PUBLIC", False), ("PRIVATE", True)):
        test_df = load_test_df(private=private)
        test_df = pvf._ensure_frontal_lateral_column(test_df)
        N_test = len(test_df)
        V_test = op._stack(member_dirs, "cached_test_preds", suffix, (N_test, C))
        rows = op._apply_pipeline_to_test(W_a, Wv, V_test, test_df)
        out = op.OUT_DIR / f"v4_oof_perview_perclass_alpha_{phase}.csv"
        op._write_submission(rows, test_df, out)
        print(f"wrote {out.name}")

    print("DONE")


if __name__ == "__main__":
    main()
