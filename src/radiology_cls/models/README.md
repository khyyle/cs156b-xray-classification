# Adding a new model

## 1. Create your model file

Add a `.py` file in this directory. Your model must subclass `BaseModel` and implement four things: `name`, `train()`, `predict()`, and `from_checkpoint()`. During early development, `predict` and `from_checkpoint` can raise `NotImplementedError`

```python
# src/radiology_cls/models/my_resnet.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from radiology_cls.data import (
    LABEL_NAMES,
    NUM_CLASSES,
    load_train_df,
    train_val_split,
)
from radiology_cls.eval import plot_auroc
from radiology_cls.models.base import BaseModel
from radiology_cls.utils import ensure_dir


## actual model

class ResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...


## the BaseModel wrapper (what train.py sees)

class MyResNet(BaseModel):
    name = "my-resnet"

    def __init__(self, lr=1e-4, epochs=10, val_frac=0.1, seed=42, **kwargs):
        self.lr = lr
        self.epochs = epochs
        self.val_frac = val_frac
        self.seed = seed

    def train(self, run_dir: Path) -> dict:
        df = load_train_df()
        train_df, val_df = train_val_split(df, self.val_frac, self.seed)

        model = ResNet(NUM_CLASSES).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # ... build your datasets/dataloaders ...

        best_val_loss = float("inf")
        for epoch in range(self.epochs):
            # ... training loop ...
            # ... validation loop ...

            print(f"Epoch {epoch+1}/{self.epochs}  train_loss=...  val_loss=...")
            wandb.log({"train_loss": ..., "val_loss": ..., "epoch": epoch})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), run_dir / "best.pt")

        # generate eval plots (recommended)
        plots_dir = ensure_dir(run_dir / "plots")
        fig = plot_auroc(y_true, y_pred, list(LABEL_NAMES), plots_dir)
        fig.savefig(plots_dir / "auroc.png")

        return {"best_val_loss": best_val_loss}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError  # implement when ready to submit

    @classmethod
    def from_checkpoint(cls, run_dir: Path, **kwargs) -> MyResNet:
        raise NotImplementedError  # implement when ready to submit
```

## 2. Create a YAML config

Add a config file in `configs/`. The only required key is `model_class`
-- everything else is passed as `**kwargs` to your model's `__init__`.

```yaml
# configs/my_resnet.yaml
model_class: radiology_cls.models.my_resnet.MyResNet
lr: 1.0e-4
epochs: 10
seed: 42
val_frac: 0.1
batch_size: 32
```

## 3. Submit a training run

```bash
# from the login node:
python scripts/train.py configs/my_resnet.yaml --gpu v100

# check logs:
tail -f runs/MyResNet_<timestamp>/slurm.out
```

This creates a run directory with your config, checkpoints, plots, and a summary.json. Everything is tracked in wandb.

## 4. Iterate

edit the `.py` file -> resubmit -> check logs. You do not need a GPU or interactive session to edit the model and short runs (few epochs, `--time 00:15:00`) should be utilized to verify things work before committing to a full training run.

## What a model is responsible for

- Loading data (`load_train_df`, `load_test_df`)
- Train/val splitting (`train_val_split`)
- The training loop (optimizer, loss, forward/backward)
- Saving checkpoints to `run_dir/`
- Logging metrics to stdout (captured by SLURM) and wandb
- Generating eval plots to `run_dir/plots/` (using `radiology_cls.eval`)

## What train.py handles

- Run directory creation (`runs/<ModelName>_<timestamp>/`)
- Copying the config into the run directory
- wandb init/finish
- SLURM job submission (sbatch or srun)
- Seeding (stdlib random + numpy)

## Composing trained models: EnsembleModel

`EnsembleModel` is a `BaseModel` that wraps a list of trained run dirs and combines their tanh outputs via a `Combiner` strategy (currently `MeanCombiner`; add new ones by subclassing `Combiner` and registering in `_COMBINERS`).

Because it is just another `BaseModel`, the same `train.py` / `submit.py` SLURM flow applies. "Training" computes diagnostics on the shared val split and persists a `members.json` manifest; `predict` loads each member, runs cached inference, and aggregates. Per-member predictions are cached inside each member's own run dir, so two ensembles that share members reuse work, and re-mixing weights costs only the aggregation. See `scripts/README.md` for the workflow.

Ensemble members can themselves be ensembles; the abstraction composes.

### Test-time augmentation

`EnsembleModel` and `ImageClassifierModel` both support an optional `tta_transforms` field (list of TTA op names from `radiology_cls.preprocessing.TTA_OPS`). When set, each member runs one inference pass per TTA op and averages -- variance reduction with no retraining. The ensemble pushes its TTA spec down into each member at inference time, so existing trained checkpoints gain TTA without modification. Cached preds are TTA-aware: non-TTA and TTA preds for the same member coexist as separate files. See `configs/ensemble_example.yaml`.

