"""
Generate a competition submission CSV from a trained model checkpoint.

Loads the model via from_checkpoint(), runs model.predict() on the
held-out test set, and writes predictions in the format expected by the
CS156b judge.

Like train.py, this script auto-detects whether it is running on a
login node or a compute node. On a login node it submits a SLURM job
that re-invokes itself on a compute node with GPU access.

Submission format
-----------------
The judge expects a CSV with an Id column followed by one column per
pathology, with float predictions on the raw CheXpert label scale:

    Id,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Opacity,...
    18,-0.812345,0.048231,...
    ...

If test_ids.csv does not have an Id column, the row index is used.

Usage (login node -- submits a SLURM job):
    python scripts/submit.py runs/DenseNet_20260412 --gpu v100

Usage (compute node or interactive session -- runs directly):
    python scripts/submit.py runs/DenseNet_20260412

Usage (login node, foreground so output streams to terminal):
    python scripts/submit.py runs/DenseNet_20260412 --gpu v100 --foreground
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from radiology_cls.data import LABEL_NAMES, load_test_df, study_ids_from_paths
from radiology_cls.slurm import JobResources, run_foreground, submit_batch
from radiology_cls.utils import PROJECT_ROOT, import_class


def _submit(args: argparse.Namespace) -> None:
    """Submit a SLURM job from the login node to run inference.

    Parameters:
    -----------
    args: argparse.Namespace
        Parsed CLI arguments.
    """
    # Disambiguating tag so concurrent public/private jobs against the same
    # run dir don't clobber each other's logs or outputs.
    tag = "PRIVATE" if getattr(args, "private_set", False) else "PUBLIC"
    user_outputs = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
    user_outputs.mkdir(parents=True, exist_ok=True)

    # Prefer logging into the run dir when writable so it lives next to the
    # checkpoint; otherwise drop it into the user's outputs dir.
    try:
        test_probe = args.run_dir / ".write_probe"
        test_probe.touch()
        test_probe.unlink()
        log_path = args.run_dir / f"submit_inference_{tag}.out"
    except PermissionError:
        log_path = user_outputs / f"{args.run_dir.name}_submit_inference_{tag}.out"
        print(f"Note: run dir not writable; SLURM log -> {log_path}")

    resources = JobResources(gpu_type=args.gpu, cpus=args.cpus, mem=args.mem, time=args.time)
    job_name = f"submit-{args.run_dir.name}"

    submit_command = ["python", "-u", "scripts/submit.py", str(args.run_dir)]
    if args.output:
        submit_command += ["--output", str(args.output)]
    if args.view_average:
        submit_command.append("--view-average")
    if getattr(args, "private_set", False):
        submit_command.append("--private-set")

    if args.foreground:
        print(f"Run dir:  {args.run_dir}")
        print(f"Mode:     srun (foreground)\n")
        sys.exit(run_foreground(submit_command, job_name, resources))

    confirmation = submit_batch(
        submit_command,
        job_name,
        resources,
        log_path=log_path,
        script_path=args.run_dir / "submit_inference.sh",
        email=args.email,
    )
    print(f"Submitted:  {confirmation}")
    print(f"Run dir:    {args.run_dir}")
    print(f"Logs:       tail -f {log_path}")


def _run(args: argparse.Namespace) -> None:
    """
    Load the model, run inference, and write the submission CSV.

    Parameters:
    -----------
    args: argparse.Namespace
        Parsed CLI arguments.
    """
    config_path = args.run_dir / "config.yaml"
    if not config_path.exists():
        print(f"Error: no config.yaml found in {args.run_dir}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_class_path = config.get("model_class")
    if not model_class_path:
        print("Error: config.yaml must contain 'model_class'")
        sys.exit(1)

    model_path = config.get("model_path")
    if model_path:
        sys.path.insert(0, str(Path(model_path).resolve()))

    print(f"Run dir:      {args.run_dir}")
    print(f"Model class:  {model_class_path}")

    model_cls = import_class(model_class_path)
    model = model_cls.from_checkpoint(args.run_dir)  # type: ignore
    test_df = load_test_df(private=args.private_set)

    print(f"Running inference on {len(test_df)} test images...")
    preds = model.predict(test_df)

    if preds.shape[0] != len(test_df):
        raise RuntimeError(
            f"predict() returned {preds.shape[0]} rows but test set has {len(test_df)}"
        )
    if preds.shape[1] != len(LABEL_NAMES):
        raise RuntimeError(
            f"predict() returned {preds.shape[1]} columns but expected {len(LABEL_NAMES)}"
        )

    submission = pd.DataFrame(preds, columns=list(LABEL_NAMES))

    if "Id" in test_df.columns:
        submission.insert(0, "Id", test_df["Id"].to_numpy())
    else:
        submission.insert(0, "Id", range(len(submission)))

    if args.view_average:
        submission = _study_view_average(submission, test_df)
        study_ids = study_ids_from_paths(test_df["Path"])
        n_pooled = int((test_df.assign(__sid=study_ids).groupby("__sid").size() > 1).sum())
        print(
            f"Applied study-level view averaging: "
            f"{n_pooled} studies pooled across >1 view"
        )

    if args.output is not None:
        output_path = args.output
    else:
        tag = "PRIVATE" if args.private_set else "PUBLIC"
        try:
            probe = args.run_dir / ".write_probe"
            probe.touch()
            probe.unlink()
            output_path = args.run_dir / f"submission_{tag}.csv"
        except PermissionError:
            user_outputs = PROJECT_ROOT / "runs" / f"{getpass.getuser()}_outputs"
            user_outputs.mkdir(parents=True, exist_ok=True)
            output_path = user_outputs / f"{args.run_dir.name}_submission_{tag}.csv"
            print(f"Note: run dir not writable; submission -> {output_path}")
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(submission)} rows)")


def _study_view_average(
    submission: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
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
    
    # replace each row's pred with the mean across all rows of the same study (if there are multiple)
    averaged[label_cols] = (
        averaged.groupby("__sid", sort=False)[label_cols].transform("mean")
    )
    return averaged.drop(columns="__sid")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a submission CSV from a trained run directory.",
    )
    p.add_argument("run_dir", type=Path, help="Path to a completed run directory.")
    p.add_argument("--email", type=str, help="Email to send job notifications to")
    p.add_argument("--output", type=Path, default=None, help="Output CSV path. Defaults to <run_dir>/submission.csv.")
    p.add_argument("--gpu", type=str, default="v100", help="GPU type to request (e.g. h100, v100, p100).")
    p.add_argument("--mem", type=str, default="16G", help="Memory to request.")
    p.add_argument("--time", type=str, default="01:00:00", help="Wall time to request (HH:MM:SS).")
    p.add_argument(
        "--cpus", type=int, default=8,
        help=(
            "CPUs to request per task (default 8). Ensemble TTA inference "
            "creates up to 2*num_workers DataLoader workers concurrently due "
            "to persistent_workers; set this >= 2*num_workers to avoid "
            "CPU starvation causing hangs."
        ),
    )
    p.add_argument("--foreground", action="store_true", help="Run interactively via srun instead of sbatch.")
    p.add_argument(
        "--view-average",
        action="store_true",
        help=(
            "After writing submission.csv, also write submission_viewavg.csv "
            "where each row's prediction is replaced by the mean across all "
            "views of the same study. Patient-level pooling would mix "
            "different clinical encounters and is not used."
        ),
    )
    p.add_argument(
        "--private-set",
        action="store_true",
        dest="private_set",
        help=(
            "Run inference on the private held-out set (solution_ids.csv) "
            "instead of the public test set (test_ids.csv). Stale caches "
            "from the public set are automatically skipped because their "
            "row count (22596) won't match the private set (22660)."
        ),
    )
    args = p.parse_args()

    on_compute_node = "SLURM_JOB_ID" in os.environ

    if on_compute_node:
        _run(args)
    else:
        _submit(args)


if __name__ == "__main__":
    main()
