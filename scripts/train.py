"""
Training launcher that creates a run directory, saves the config, initializes
wandb, instantiates the model, and calls model.train(run_dir). The model
owns everything else, i.e., data loading, validation, evaluation, plotting.

The script auto-detects whether it is running on a login node or a compute
node by checking the SLURM_JOB_ID environment variable.

On a login node it creates a run directory under runs/, copies the config
there, and submits a SLURM job that re-invokes this same script on a compute
node. By default (sbatch) the job runs in the background and logs stream to
runs/<run>/slurm.out. Pass --foreground to use srun instead, which blocks
and streams output directly to your terminal (useful from a notebook cell).

On a compute node (SLURM_JOB_ID is set) the script skips submission and
runs training directly.

Usage:
    python scripts/train.py configs/densenet121.yaml --gpu h100
    python scripts/train.py configs/densenet121.yaml --gpu h100 --foreground
    python scripts/train.py configs/densenet121.yaml --gpu h100 --time 08:00:00

The YAML config must contain the full model import path, e.g.:
    model_class: radiology_cls.models.densenet.DenseNetModel

For models outside the installed package (e.g. in scratch/), add
model_path to the config so the module can be found. This pattern is not recommended however:
    model_class: vanilla_cnn.VanillaCNNModel
    model_path: scratch/kyle
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import wandb  # type: ignore
import yaml

from radiology_cls.slurm import JobResources, run_foreground, submit_batch
from radiology_cls.utils import PROJECT_ROOT, ensure_dir, import_class, seed_everything, write_json

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

TRAIN_CPUS = 8


def _submit(args: argparse.Namespace, run_dir: Path) -> None:
    """
    Submit a SLURM job from the login node. By default the job is
    submitted via sbatch and runs in the background with stdout captured
    to slurm.out. With --foreground it uses srun instead, which blocks
    and streams output directly to the caller's terminal.

    Parameters:
    -----------
    args: argparse.Namespace
        Parsed CLI arguments including `gpu`, `mem`, `time`, and
        `foreground`.
    run_dir: Path
        Pre-created run directory containing the copied config file.
    """
    log_path = run_dir / "slurm.out"
    resources = JobResources(gpu_type=args.gpu, cpus=TRAIN_CPUS, mem=args.mem, time=args.time)
    train_command = ["python", "-u", "scripts/train.py", str(run_dir / "config.yaml")]

    if args.foreground:
        print(f"Run dir:  {run_dir}")
        print(f"Mode:     srun (foreground)\n")
        sys.exit(run_foreground(train_command, run_dir.name, resources))

    confirmation = submit_batch(
        train_command,
        run_dir.name,
        resources,
        log_path=log_path,
        script_path=run_dir / "submit.sh",
        email=args.email,
    )
    print(f"Submitted:  {confirmation}")
    print(f"Run dir:    {run_dir}")
    print(f"Logs:       tail -f {log_path}")


def _train(config: dict, run_dir: Path) -> None:
    """
    Instantiate the model and call train(). Called if the user is already on 
    a compute node (i.e. they already ran `sbatch ...`)

    Parameters:
    -----------
    config: dict
        The full YAML config dict. `model_class` is popped out, remaining
        keys are passed as **kwargs to the model constructor.
    run_dir: Path
        Directory the model should save all artifacts into.
    """
    model_class_path = config.pop("model_class")
    model_path = config.pop("model_path", None)
    config.pop("run_name", None) # pop so doesnt enter model **kwargs
    if model_path:
        sys.path.insert(0, str(Path(model_path).resolve()))

    seed = config.get("seed", 42)

    seed_everything(seed)

    model_cls = import_class(model_class_path)

    wandb_kwargs = dict(
        project="cs156b",
        name=run_dir.name,
        config={"model_class": model_class_path, **config},
    )
    try:
        wandb.init(**wandb_kwargs)
    except Exception as e:
        print(f"wandb.init failed ({e}), falling back to offline mode")
        os.environ["WANDB_MODE"] = "offline"
        try:
            wandb.teardown()
        except Exception:
            pass
        wandb.init(**wandb_kwargs, mode="offline")

    print(f"Run dir:      {run_dir}")
    print(f"Model class:  {model_class_path}")
    print(f"Seed:         {seed}")
    print(f"Config:       {config}")
    print()

    model = model_cls(**config)
    summary = model.train(run_dir)

    write_json(summary, run_dir / "summary.json")

    wandb.finish()

    print()
    print(f"Done. Artifacts saved to {run_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train a model from a YAML config.")
    p.add_argument("config", type=Path, help="Path to YAML config file.")
    p.add_argument("--email", type=str, help="Email to send job notifications to")
    p.add_argument("--gpu", type=str, default="v100",
                   help="GPU type to request (e.g. h100, v100, p100).")
    p.add_argument("--mem", type=str, default="124G",
                   help="System RAM to request.")
    p.add_argument("--time", type=str, default="02:00:00",
                   help="Wall time to request (HH:MM:SS).")
    p.add_argument("--foreground", action="store_true",
                   help="Run interactively via srun instead of submitting with sbatch.")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_class_path = config.get("model_class")
    if not model_class_path:
        print("Error: config must contain 'model_class'")
        sys.exit(1)

    on_compute_node = "SLURM_JOB_ID" in os.environ

    if on_compute_node:
        run_dir = args.config.resolve().parent
        _train(config, run_dir)
    else:
        model_path = config.pop("model_path", None)
        if model_path:
            sys.path.insert(0, str(Path(model_path).resolve()))

        run_name = config.pop("run_name", None)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_cls = import_class(model_class_path)
        run_dir = ensure_dir(PROJECT_ROOT / "runs" / f"{run_name or model_cls.__name__}_{timestamp}")

        shutil.copy2(args.config, run_dir / "config.yaml")

        _submit(args, run_dir)


if __name__ == "__main__":
    main()
