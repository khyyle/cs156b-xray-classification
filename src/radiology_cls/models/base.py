from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Every model must implement this abstract base class so that the 
    train.py wrapper can train it.

    train.py passes the YAML config dict as **kwargs to the constructor,
    so any key in the config becomes a constructor argument. Each model
    defines its own __init__ signature with whatever it needs (lr, epochs,
    pretrained backbone, etc.).

    Models are responsible for:
      - Loading data (using helpers from radiology_cls.data)
      - Train/val splitting (using train_val_split from radiology_cls.data)
      - Training and saving checkpoints to run_dir
      - Logging metrics to stdout (captured by SLURM) and wandb
      - Generating eval plots to run_dir/plots/
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in run directory names and wandb tags."""
        ...

    @abstractmethod
    def train(self, run_dir: Path) -> dict:
        """
        This method should load data, train the model, and save checkpoints to run_dir.

        Models own data loading, train/val splitting, evaluation, and
        plot generation. Use the eval library or custom logic
        to produce diagnostic plots in run_dir/plots/.

        Print progress to stdout so SLURM logs capture live output. Log
        metrics to wandb for the team dashboard.

        Return a summary dict with whatever the model wants to report
        (e.g. best_val_loss, best_epoch). This is for informational
        printing only--train.py does not act on the contents.
        """
        ...

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Run inference on a dataframe with an `image_path` column.

        Used by submit.py to generate predictions on the held-out test
        set. Returns an (N, num_classes) float array of predicted
        probabilities.
        """
        ...

    @classmethod
    @abstractmethod
    def from_checkpoint(cls, run_dir: Path, **kwargs) -> BaseModel:
        """
        Reconstruct a trained model from a saved run directory.

        Used by submit.py and any post-hoc analysis scripts to load a
        model without retraining. The model should load its weights from
        the checkpoint files it saved during train().
        """
        ...
