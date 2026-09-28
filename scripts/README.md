# scripts/

These are entrypoints for data setup, training, submission, and ensemble post-processing. The GPU entrypoints (`train.py`, `submit.py`, `affine_calibrate.py`) handle SLURM job submission automatically and package results in a neat folder. They can be run from the login node or from a compute node and they take care of requesting GPU resources (or not) accordingly.

## Setup

- run `source scripts/env.sh` to load `.env` and activate the venv. Job scripts source this themselves.
- Pre-resize the dataset once per resolution so training doesn't decode full-size JPEGs each epoch. Run it for the image size your configs use:
  ```bash
  python scripts/cache_images.py --size 256
  ```

## train.py

Train.py is a dumb launcher. It only creates a run directory, initializes wandb, instantiates the model, and calls `model.train(run_dir)`. Models implement everything else like data loading, validation, evaluation plots, checkpointing.

```bash
# From login node (background, logs to runs/<run>/slurm.out):
python scripts/train.py configs/densenet121.yaml --gpu h100

# From login node (foreground, output streams to terminal/notebook cell):
python scripts/train.py configs/densenet121.yaml --gpu h100 --foreground

# From a compute node or interactive session (runs directly):
python scripts/train.py configs/densenet121.yaml
```

### How it works

The script checks for the `SLURM_JOB_ID` environment variable:

- **Not set (login node)**: creates the run directory, copies the config, and submits a SLURM job (`sbatch` or `srun`) that re-invokes this same script on a compute node.
- **Set (compute node)**: skip submission and run training directly.

This means the same command works from both contexts.

## submit.py

Generates a competition submission CSV from a trained model. Loads the model via `Model.from_checkpoint(run_dir)`, runs `model.predict()` on the test set, and writes probabilities in the format the CS156b judge expects to the corresponding run directory.

```bash
# From login node:
python scripts/submit.py runs/DenseNet_20260412 --gpu v100

# From compute node:
python scripts/submit.py runs/DenseNet_20260412
```

## Interactive sessions

These are shortcut scripts for getting an interactive shell on a compute node if you wish to request an interactive session before running train/submit.py

```bash
bash scripts/srun_cpu.sh
# or
bash scripts/srun_gpu.sh
```

Once in, load the project environment and work normally

```bash
source scripts/env.sh
python scratch/kyle/my_experiment.py
```

## Ensemble post-processing

Tools for improving ensemble submissions after training.

- `per_disease_weights.py`: fits separate member weights for each pathology on val.
  ```bash
  python scripts/per_disease_weights.py runs/<ensemble_run>
  ```
- `per_view_fusion.py`: learns how to combine frontal and lateral views per pathology instead of averaging them.
  ```bash
  python scripts/per_view_fusion.py runs/<ensemble_run>
  ```
- `affine_calibrate.py`: fits a per-pathology linear correction for one member run, which that member then applies at inference. Needs a GPU, so it submits a job like `train.py`.
  ```bash
  python scripts/affine_calibrate.py runs/<member_run>
  ```
- `blend_submissions.py`: weighted average of existing submission CSVs. Needs only pandas, so it also runs on a laptop.
  ```bash
  python scripts/blend_submissions.py a.csv b.csv --output blend.csv
  ```
- `run_ensemble_pipeline.sh`: chains the steps above into one GPU job for both the public and private sets. Usage is in the script header.

## a quick refresher on how Slurm clusters work

A Slurm cluster has two types of nodes:

- **Login nodes**: where you land after ssh. These do not have GPUs and can be used to edit code, run small CPU based scripts, submit jobs, and check results.
- **Compute nodes**: machines with GPUs/CPUs that SLURM allocates for you on request.


There are two ways to get a compute node via SLURM: 

- `sbatch` (background): submits a script that runs on a compute node. You get your terminal back immediately. Output goes to a log file.
  ```bash
  sbatch my_script.sh  # returns "Submitted batch job 12345"
  tail -f slurm.out    # watch logs
  ```
- `srun` (foreground): runs a command on a compute node and blocks your terminal until it finishes. Output streams live.
  ```bash
  srun --account=$CLUSTER_ACCOUNT --partition=$CLUSTER_GPU_PARTITION --gres=gpu:v100:1 --mem=16G --time=01:00:00 python my_script.py
  ```
- `srun --pty bash` (interactive session): gives you a shell on a compute node and is analogous to connecting to a runtime in Colab.
  ```bash
  srun --account=$CLUSTER_ACCOUNT --partition=$CLUSTER_GPU_PARTITION --gres=gpu:v100:1 --mem=16G --time=02:00:00 --pty bash
  ```

`train.py` and `submit.py` handle all of this, detecting which node you're on and requesting GPU or directly running training depending on whether you have a session or not.

### Checking on jobs

```bash
# see all your jobs (PD = pending, R = running)
squeue -u $USER

# watch your job queue, refreshes every 5s
watch -n 5 squeue -u $USER

# see full GPU partition queue
squeue -p "$CLUSTER_GPU_PARTITION"

# cancel a job
scancel <job_id>

# cancel all your jobs
scancel -u $USER

# tail logs for a running job
tail -f runs/<run_name>/slurm.out
```