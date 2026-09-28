# scratch/

Personal experimentation space for each team member. Use `.py` scripts, `.ipynb` notebooks, or both. Each person has their own subdirectory.

## What belongs here

- Investigating trained models, i.e., loading checkpoints, inspecting predictions, generating custom plots
- Data analysis and exploration, i.e., label distributions, image statistics, class imbalance
- Trying out loss functions, augmentation strategies, or preprocessing ideas
- Manual hyperparameter tuning or calibration scripts
- Anything exploratory that doesn't need the full train.py pipeline

## What doesn't belong here

Models you intend to train should go directly in `src/radiology_cls/models/`. There's no need to build here and put in the shared API later.

If you do want to develop a model here first (e.g. rapid iteration in a notebook), `train.py` supports a `model_path` config key that adds a directory to the Python path:

```yaml
model_class: my_model.MyModel
model_path: scratch/<name>
```

But the recommended workflow is to write models in `src/radiology_cls/models/` from the start.

## Running experiments

```bash
# submit a script as a batch job (no interactive session needed):
source scripts/env.sh
sbatch --gres=gpu:v100:1 --mem=16G --time=01:00:00 --wrap="python scratch/<name>/<script>.py"

# or get an interactive session (needed for notebooks):
bash scripts/srun_gpu.sh
```

Use the shared library (`radiology_cls.data`, `radiology_cls.eval`, `radiology_cls.preprocessing`) for data loading and evaluation so experiments stay comparable. 

NOTE: files here should not be imported by anything in `src/` or `scripts/`.
