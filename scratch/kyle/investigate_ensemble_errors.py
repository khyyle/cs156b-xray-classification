"""Compute per-class residual correlations across members of an ensemble.

Ensemble averaging reduces variance proportional to (1 - mean inter-member
correlation), so this report helps determine whether a new member is adding
useful decorrelation or just noisy redundancy.

Usage:
    # Single ensemble
    python scratch/kyle/investigate_ensemble_errors.py runs/ensemble-kitchen-sink-v4_*

    # Multiple ensembles side by side
    python scratch/kyle/investigate_ensemble_errors.py \\
        runs/ensemble-kitchen-sink-v4_* \\
        runs/ensemble-kitchen-sink-v5_*

    # Highlight one specific member's correlation with the rest (e.g. for
    # the new ASL+calib or DINOv3 member)
    python scratch/kyle/investigate_ensemble_errors.py runs/ensemble-kitchen-sink-v5_* \\
        --focus-member asl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from radiology_cls.data import LABEL_NAMES
from radiology_cls.utils import PROJECT_ROOT

TTA_SUFFIX = "__tta-hflip-identity"
CACHED_VAL = f"cached_val_preds{TTA_SUFFIX}.npy"


def _candidate_paths(member_dir: Path, filename: str) -> list[Path]:
    return [
        member_dir / filename,
        PROJECT_ROOT / "cache" / "ensemble_predictions" / member_dir.name / filename,
        Path.home() / ".cache" / "cs156b_ensemble" / member_dir.name / filename,
    ]


def _find_member_val_preds(member_dir: Path, n_val: int) -> Path:
    for p in _candidate_paths(member_dir, CACHED_VAL):
        if p.exists():
            shape = tuple(np.load(p, mmap_mode="r").shape)
            if shape[0] == n_val:
                return p
    raise FileNotFoundError(f"No cached val preds for {member_dir.name} with N={n_val}")


def _load_ensemble(run_dir: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Returns (member names, val targets (N, C), stacked preds (M, N, C))."""
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    members = cfg["members"]
    val_targets = np.load(run_dir / "val_targets.npy").astype(np.float64)
    N = val_targets.shape[0]
    stack = []
    names = []
    for m in members:
        d = Path(m)
        if not d.is_absolute():
            d = (PROJECT_ROOT / d).resolve()
        preds = np.load(_find_member_val_preds(d, N)).astype(np.float64)
        stack.append(preds)
        names.append(d.name)
    return names, val_targets, np.stack(stack, axis=0)


def _per_class_corr(member_preds: np.ndarray, targets: np.ndarray, c: int) -> tuple[np.ndarray, int]:
    """Residual correlation matrix (M, M) for one class, dropping NaN rows."""
    mask = ~np.isnan(targets[:, c])
    if not mask.any():
        M = member_preds.shape[0]
        return np.full((M, M), np.nan), 0
    residuals = member_preds[:, mask, c] - targets[mask, c]  # (M, n_valid)
    return np.corrcoef(residuals), int(mask.sum())


def _off_diag_stats(corr: np.ndarray) -> tuple[float, float, float]:
    """Return (mean, min, max) of off-diagonal entries; NaN-safe."""
    M = corr.shape[0]
    off = corr.copy()
    off[np.eye(M, dtype=bool)] = np.nan
    return (
        float(np.nanmean(off)),
        float(np.nanmin(off)),
        float(np.nanmax(off)),
    )


def _focus_row(corr: np.ndarray, names: list[str], substr: str) -> tuple[int, float, float, float] | None:
    """If a member name contains substr, return (idx, mean, min, max) of its correlations with others."""
    matches = [i for i, n in enumerate(names) if substr.lower() in n.lower()]
    if not matches:
        return None
    idx = matches[0]
    row = np.delete(corr[idx], idx)
    return idx, float(np.mean(row)), float(np.min(row)), float(np.max(row))


def _print_report(run_dir: Path, focus: str | None) -> None:
    names, targets, preds = _load_ensemble(run_dir)
    M, N, C = preds.shape
    print()
    print("=" * 92)
    print(f"Ensemble: {run_dir.name}   ({M} members, {C} classes, {N} val rows)")
    print("=" * 92)
    if focus:
        match_idx = [i for i, n in enumerate(names) if focus.lower() in n.lower()]
        if match_idx:
            print(f"Focus member: [{match_idx[0]}] {names[match_idx[0]]}")
        else:
            print(f"Focus '{focus}' did not match any member name.")
    print(f"\n{'Class':<30} {'n_val':>6} {'mean_corr':>11} {'min_corr':>11} {'max_corr':>11}", end="")
    if focus:
        print(f"  {'focus_mean':>11} {'focus_min':>11} {'focus_max':>11}")
    else:
        print()

    macro_mean = 0.0
    macro_count = 0
    focus_macro_means: list[float] = []
    for c in range(C):
        corr, n_val = _per_class_corr(preds, targets, c)
        mean_off, min_off, max_off = _off_diag_stats(corr)
        line = (
            f"{LABEL_NAMES[c]:<30} {n_val:>6d} "
            f"{mean_off:>+11.3f} {min_off:>+11.3f} {max_off:>+11.3f}"
        )
        if focus:
            fr = _focus_row(corr, names, focus)
            if fr is not None:
                _, fm, fn, fx = fr
                focus_macro_means.append(fm)
                line += f"  {fm:>+11.3f} {fn:>+11.3f} {fx:>+11.3f}"
        print(line)
        if not np.isnan(mean_off):
            macro_mean += mean_off
            macro_count += 1
    macro_mean = macro_mean / max(macro_count, 1)
    print(f"\nMacro mean inter-member correlation: {macro_mean:+.3f}")
    if focus and focus_macro_means:
        print(f"Macro mean focus-vs-others correlation: {float(np.mean(focus_macro_means)):+.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("ensemble_dirs", nargs="+", type=Path)
    p.add_argument(
        "--focus-member",
        type=str,
        default=None,
        help="Substring of a member name to spotlight (e.g. 'asl', 'dinov3').",
    )
    args = p.parse_args()
    for d in args.ensemble_dirs:
        _print_report(d.resolve(), focus=args.focus_member)


if __name__ == "__main__":
    main()
