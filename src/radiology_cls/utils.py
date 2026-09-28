"""
Small generic helpers that don't belong in any specific module like filesystem
shortcuts, seeding, JSON I/O.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory (and parents) if it doesn't already exist.

    Parameters:
    -----------
    path: str | Path
        Directory to create.

    Returns:
    --------
    target: Path
        The same path, as a `Path` object.
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def seed_everything(seed: int = 42) -> None:
    """
    Seed stdlib random and numpy for reproducibility. Models that use
    torch should also call `torch.manual_seed` in their own __init__.

    Parameters:
    -----------
    seed: int
        Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


def import_class(dotted_path: str) -> type:
    """
    Dynamically import a class from a fully-qualified dotted path.

    Parameters:
    -----------
    dotted_path: str
        Full Python import path, e.g.
        `radiology_cls.models.densenet.DenseNetModel`.

    Returns:
    --------
    cls: type
        The imported class object.
    """
    import importlib

    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(
            f"dotted_path must be like 'pkg.mod.Class', got '{dotted_path}'"
        )
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def write_json(payload: dict, path: str | Path) -> None:
    """
    Write a dict as pretty-printed JSON, creating parent directories
    if they don't exist.

    Parameters:
    -----------
    payload: dict
        Data to serialize.
    path: str | Path
        Destination file path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
