# CS156b radiology classification

Authors: Kyle Berkson, Kenneth Chan, and Zarif Azher

This repository contains our codebase for Caltech's CS 156b radiology classification project on the CheXpert dataset, which placed 1st on the private leaderboard with a macro MSE of 0.653. It is configured to run training and evaluation across a Slurm GPU cluster.

*Caltech Honor Code: No member of the Caltech community shall take unfair advantage of any other member of the Caltech community. If you are taking or plan to take CS 156b, please do not view or use this code.*

## Approach

We fine-tune a mix of CNN and vision transformer backbones (DenseNet, ResNet, ConvNeXt V2, Swin V2, DINOv2) across nine chest X-ray pathologies. Models are trained using either direct regression or three-way classification (negative, uncertain, positive).

These individual models are then combined through a post-processing and ensembling pipeline. We run test-time augmentation, fit pathology-specific member weights on validation predictions using non-negative least squares and ridge regression, and fuse frontal and lateral views using regularized per-pathology weights to produce our final predictions.

## Repository structure

- `src/radiology_cls/`: shared Python library for data loading, evaluation, preprocessing, and model abstractions. Installed as an editable package.
- `src/radiology_cls/models/`: model architectures and wrappers. Each model subclasses `BaseModel` so it can be trained via `train.py`. See [models/README.md](src/radiology_cls/models/README.md) for details.
- `scripts/`: entrypoints for data caching, training, submission generation, and ensembling. Slurm batch scripts are generated automatically. See [scripts/README.md](scripts/README.md).
- `configs/`: YAML configs for training and ensemble runs.
- `scratch/`: analysis scripts and exploratory notebooks. See [scratch/README.md](scratch/README.md).

## Setup

Requires Python 3.11+. On clusters that use environment modules, `module load` a 3.11 build first.

```bash
# clone the repository
git clone <repo-url> && cd <repo>

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install dependencies, then register the notebook-output filter with git
pip install -e .
nbstripout --install

# fill in every value in .env
cp .env.example .env
```

Before running anything, be sure to activate the python virtual environment and export environment variables by running:

```bash
source scripts/env.sh
```

Log in to Weights & Biases once with an API key from the wandb dashboard:

```bash
wandb login
```
