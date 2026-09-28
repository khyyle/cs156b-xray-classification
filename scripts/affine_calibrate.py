"""Fit per-class affine ``y = a*s + b`` on val for one trained run.

NNLS/ridge per-disease has no intercept and combines members within
their convex hull, so it can't fix a single member's bias or scale
distortion. Recalibrating members before they enter an ensemble gives
NNLS cleaner inputs, particularly after losses that trade calibration
for ranking (e.g. asymmetric loss).

Mirrors train.py / submit.py: sbatch from a login node, run directly
when SLURM_JOB_ID is set. Writes:

    <run_dir>/affine_calibration.npy           (2, 9) of [slope, intercept]
    <run_dir>/affine_calibration_summary.json  per-class diagnostics

Falls back to ``runs/<user>_outputs/`` when run dir isn't writable.

Notes:
- Fit on single-pass val; commutes with the ensemble's TTA mean by linearity.
- Any prior affine_calibration is disabled during inference so
  re-running this script never compounds.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from radiology_cls.data import (
    LABEL_NAMES,
    load_train_df,
    train_val_split,
)
from radiology_cls.preprocessing import encode_regression_labels
from radiology_cls.slurm import JobResources, submit_batch
from radiology_cls.utils import PROJECT_ROOT, import_class


def _load_val_split(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Reconstruct the deterministic val split from the run's config."""
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    val_frac = float(config.get("val_frac", 0.1))
    seed = int(config.get("seed", 42))
    full_df = load_train_df()
    _, val_df = train_val_split(full_df, val_frac=val_frac, seed=seed)
    y_val = encode_regression_labels(val_df, list(LABEL_NAMES))
    return val_df, y_val


def _run_val_inference(run_dir: Path, val_df: pd.DataFrame) -> np.ndarray:
    """Load model and emit single-pass val preds with any prior affine disabled."""
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)
    model_class_path = config.get("model_class")
    if not model_class_path:
        sys.exit(f"{run_dir}/config.yaml is missing 'model_class'")
    model_path = config.get("model_path")
    if model_path:
        sys.path.insert(0, str(Path(model_path).resolve()))
    model_cls = import_class(model_class_path)
    model = model_cls.from_checkpoint(run_dir)

    prior_calibration = getattr(model, "affine_calibration", None)
    try:
        if prior_calibration is not None:
            model.affine_calibration = None  # type: ignore[attr-defined]
        return model.predict(val_df)
    finally:
        if prior_calibration is not None:
            model.affine_calibration = prior_calibration  # type: ignore[attr-defined]


def _fit_per_class_affine(
    preds: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """OLS fit of ``y = a*s + b`` per class over non-NaN val rows.

    Returns ``calibration`` of shape ``(2, C)`` ([slope, intercept]) and a
    diagnostics dict with per-class ``n_val``, ``(a, b)``, raw and
    calibrated MSE.
    """
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds shape {preds.shape} != targets shape {targets.shape}"
        )
    C = preds.shape[1]
    calibration = np.zeros((2, C), dtype=np.float64)
    per_class: list[dict] = []
    for c in range(C):
        mask = ~np.isnan(targets[:, c])
        n = int(mask.sum())
        if n == 0:
            calibration[:, c] = [1.0, 0.0]
            per_class.append({
                "label": LABEL_NAMES[c],
                "n_val": 0,
                "a": 1.0,
                "b": 0.0,
                "raw_mse": None,
                "calibrated_mse": None,
            })
            continue

        s = preds[mask, c].astype(np.float64)
        y = targets[mask, c].astype(np.float64)
        A = np.stack([s, np.ones_like(s)], axis=1)  # design matrix for [a, b]
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])

        raw_mse = float(np.mean((s - y) ** 2))
        calibrated_mse = float(np.mean((a * s + b - y) ** 2))

        calibration[0, c] = a
        calibration[1, c] = b
        per_class.append({
            "label": LABEL_NAMES[c],
            "n_val": n,
            "a": a,
            "b": b,
            "raw_mse": raw_mse,
            "calibrated_mse": calibrated_mse,
        })

    return calibration, {"per_class": per_class}


def _resolve_artifact_paths(run_dir: Path) -> tuple[Path, Path]:
    """Artifact paths in the run dir when writable, else under runs/<user>_outputs/."""
    try:
        probe = run_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        return (
            run_dir / "affine_calibration.npy",
            run_dir / "affine_calibration_summary.json",
        )
    except PermissionError:
        out_dir = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Note: run dir not writable; artifacts -> {out_dir}")
        return (
            out_dir / f"{run_dir.name}_affine_calibration.npy",
            out_dir / f"{run_dir.name}_affine_calibration_summary.json",
        )


def _submit(args: argparse.Namespace) -> None:
    """Submit a SLURM job that re-runs this script on a compute node."""
    user_outputs = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
    user_outputs.mkdir(parents=True, exist_ok=True)

    try:
        probe = args.run_dir / ".write_probe"
        probe.touch()
        probe.unlink()
        log_path = args.run_dir / "affine_calibrate.out"
    except PermissionError:
        log_path = user_outputs / f"{args.run_dir.name}_affine_calibrate.out"
        print(f"Note: run dir not writable; SLURM log -> {log_path}")

    confirmation = submit_batch(
        command=["python", "-u", "scripts/affine_calibrate.py", str(args.run_dir)],
        job_name=f"calibrate-{args.run_dir.name}",
        resources=JobResources(gpu_type=args.gpu, cpus=args.cpus, mem=args.mem, time=args.time),
        log_path=log_path,
        script_path=args.run_dir / "affine_calibrate.sh",
    )
    print(f"Submitted:  {confirmation}")
    print(f"Run dir:    {args.run_dir}")
    print(f"Logs:       tail -f {log_path}")


def _run(args: argparse.Namespace) -> None:
    """Run val inference, fit calibration, write artifacts."""
    run_dir: Path = args.run_dir
    if not run_dir.is_dir():
        sys.exit(f"Not a directory: {run_dir}")
    if not (run_dir / "config.yaml").exists():
        sys.exit(f"No config.yaml in {run_dir}")

    print(f"Run dir: {run_dir}")
    val_df, y_val = _load_val_split(run_dir)
    N_val = len(val_df)
    print(f"Val set: {N_val} rows, {y_val.shape[1]} pathologies")

    print("Running val inference (single pass, no TTA)...")
    preds = _run_val_inference(run_dir, val_df)
    if preds.shape != y_val.shape:
        sys.exit(f"predict() shape {preds.shape} != val targets {y_val.shape}")

    calibration, diag = _fit_per_class_affine(preds, y_val)

    raw_mses = [e["raw_mse"] for e in diag["per_class"] if e["raw_mse"] is not None]
    cal_mses = [e["calibrated_mse"] for e in diag["per_class"] if e["calibrated_mse"] is not None]
    raw_macro = float(np.mean(raw_mses)) if raw_mses else float("nan")
    cal_macro = float(np.mean(cal_mses)) if cal_mses else float("nan")

    print()
    print(
        f"{'Class':<30} {'n_val':>7} {'a':>7} {'b':>7} "
        f"{'raw_mse':>9} {'cal_mse':>9} {'delta':>9}"
    )
    for entry in diag["per_class"]:
        if entry["raw_mse"] is None:
            print(
                f"{entry['label']:<30} {entry['n_val']:>7} {entry['a']:>+7.3f} "
                f"{entry['b']:>+7.3f}       n/a       n/a       n/a"
            )
            continue
        delta = entry["calibrated_mse"] - entry["raw_mse"]
        print(
            f"{entry['label']:<30} {entry['n_val']:>7} {entry['a']:>+7.3f} "
            f"{entry['b']:>+7.3f} {entry['raw_mse']:>9.4f} "
            f"{entry['calibrated_mse']:>9.4f} {delta:>+9.4f}"
        )
    print()
    print(f"Raw        macro MSE: {raw_macro:.6f}")
    print(f"Calibrated macro MSE: {cal_macro:.6f}")
    print(f"Delta:                {cal_macro - raw_macro:+.6f}")

    cal_path, summary_path = _resolve_artifact_paths(run_dir)
    np.save(cal_path, calibration.astype(np.float32))
    with open(summary_path, "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "n_val_rows": int(N_val),
                "raw_macro_mse": raw_macro,
                "calibrated_macro_mse": cal_macro,
                "per_class": diag["per_class"],
            },
            f,
            indent=2,
        )
    print()
    print(f"Saved calibration: {cal_path}")
    print(f"Saved summary:     {summary_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Fit per-class affine calibration y = a*s + b on val for a "
            "trained run. Mirrors train.py / submit.py: submits a SLURM "
            "job from a login node, runs directly on a compute node."
        ),
    )
    p.add_argument("run_dir", type=Path, help="Path to a completed run directory.")
    p.add_argument(
        "--gpu", type=str, default="h100",
        help="GPU type to request (h100 / v100 / p100). Default h100.",
    )
    p.add_argument("--mem", type=str, default="64G", help="Memory to request.")
    p.add_argument(
        "--time", type=str, default="00:30:00",
        help="Wall time to request (HH:MM:SS). Default 30 min covers a "
             "single-pass val inference on dinov2-base-518.",
    )
    p.add_argument(
        "--cpus", type=int, default=8,
        help="CPUs per task. Default 8 is plenty for a single inference pass.",
    )
    args = p.parse_args()
    args.run_dir = args.run_dir.resolve()

    if "SLURM_JOB_ID" in os.environ:
        _run(args)
    else:
        _submit(args)


if __name__ == "__main__":
    main()
