"""
Anything that answers "where is the data and how do I read it" should be
written here. Preprocessing and transforms belong in preprocessing.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image  # type: ignore
from torch.utils.data import Dataset

from radiology_cls.settings import DATA_ROOT, SOLUTION_CSV, TEST_CSV, TRAIN_CSV
from radiology_cls.utils import PROJECT_ROOT

DEFAULT_CACHE_ROOT = PROJECT_ROOT / "cache"

LABEL_NAMES = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Pneumonia",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)
_NON_LABEL_COLUMNS = frozenset({
    "Path",
    "Sex",
    "Age",
    "Frontal/Lateral",
    "AP/PA",
    "image_path",
    "patient_id",
})
NUM_CLASSES: int = len(LABEL_NAMES)


def _drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    return df.drop(columns=unnamed)

## DataFrame loaders

def load_train_df() -> pd.DataFrame:
    """
    Read `TRAIN_CSV`, drop index artifact columns, add an absolute
    `image_path` column, and derive `patient_id` from the path.

    Returns:
    --------
    df: pd.DataFrame
        DataFrame with one row per image path. Includes the original columns
        from the CSV plus resolved `image_path` and `patient_id` columns.
    """
    df = _drop_unnamed(pd.read_csv(TRAIN_CSV))

    # train2023.csv has stray rows whose Path uses the upstream
    # 'CheXpert-v1.0/train/patientXXXXX/...' format instead of the
    # expected 'train/pidXXXXX/...' layout. Those files don't exist.
    bad_mask = df["Path"].str.contains("CheXpert-v1.0", na=False)
    if bad_mask.any():
        print(f"Dropping {bad_mask.sum()} row(s) with non-standard Path format")
        df = df[~bad_mask].reset_index(drop=True)

    df["image_path"] = df["Path"].apply(lambda p: str(DATA_ROOT / p))
    df["patient_id"] = df["Path"].str.split("/").str[1]
    if df["patient_id"].isna().any():
        raise ValueError("Failed to extract patient_id from one or more Path values")
    print(f"Loaded {len(df)} training rows from {TRAIN_CSV}")
    return df


def load_test_df(private: bool = False) -> pd.DataFrame:
    """
    Read a held-out test CSV used for competition submissions.
    Contains image paths but no labels.

    Parameters:
    -----------
    private: bool, Default=False
        Read the private held-out set (`SOLUTION_CSV`) instead of the
        public test set (`TEST_CSV`).

    Returns:
    --------
    df: pd.DataFrame
        DataFrame with one row per test image path, including a resolved
        `image_path` column.
    """
    path = SOLUTION_CSV if private else TEST_CSV
    df = pd.read_csv(path)
    df["image_path"] = df["Path"].apply(lambda p: str(DATA_ROOT / p))
    print(f"Loaded {len(df)} test rows from {path}")
    return df


def get_label_columns(df: pd.DataFrame) -> list[str]:
    """Return the pathology label column names present in a dataframe,
    filtering out metadata columns like Path, Sex, Age, etc.

    Parameters:
    -----------
    df: pd.DataFrame
        A training or test dataframe loaded by `load_train_df` or
        `load_test_df`.

    Returns:
    --------
    columns: list[str]
        Column names that correspond to pathology labels.
    """
    return [c for c in df.columns if c not in _NON_LABEL_COLUMNS]


def study_ids_from_paths(paths: pd.Series) -> pd.Series:
    """
    Extract the per-row study identifier from CheXpert image paths.

    Every image lives under a path shaped like
    `train/pidXXXXX/studyN/viewM_frontal.jpg`, where one patient can have
    several studies and each study several views. CheXpert assigns labels at
    the study level, so anything that needs to treat the views of one study
    as a unit (view averaging at submit time, study-aware cross validation)
    groups rows by the `pidXXXXX/studyN` key returned here.

    Parameters:
    -----------
    paths: pd.Series
        The `Path` column of a train or test dataframe.

    Returns:
    --------
    study_ids: pd.Series
        One `pidXXXXX/studyN` string per input row, aligned to the input
        index.
    """
    parts = paths.str.split("/")
    return parts.str[1] + "/" + parts.str[2]


def train_val_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deterministic patient-level train/val split. Models should call this
    inside their train() method, and any post-hoc scripts (calibration,
    analysis) can reconstruct the same split by using the same seed and
    val_frac from the run's saved config.

    Parameters:
    -----------
    df: pd.DataFrame
        The full training dataframe with a `patient_id` column.
    val_frac: float
        Fraction of patients to hold out for validation.
    seed: int
        Random seed for reproducibility.

    Returns:
    --------
    train_df: pd.DataFrame
        Training portion.
    val_df: pd.DataFrame
        Validation portion.
    """
    if "patient_id" not in df.columns:
        raise ValueError("train_val_split requires a patient_id column")

    patients = df["patient_id"].drop_duplicates()
    val_patients = patients.sample(frac=val_frac, random_state=seed)

    val_df = df[df["patient_id"].isin(val_patients)]
    train_df = df.drop(val_df.index)

    if not set(train_df["patient_id"]).isdisjoint(set(val_df["patient_id"])):
        raise ValueError("Patient-level split leaked at least one patient into both train and val")

    print(
        f"Patient split: "
        f"{len(train_df)} train rows from {train_df['patient_id'].nunique()} patients, "
        f"{len(val_df)} val rows from {val_df['patient_id'].nunique()} patients"
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def filter_frontal(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)


def filter_lateral(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Frontal/Lateral"] == "Lateral"].reset_index(drop=True)


def load_image(path: str | Path, *, rgb: bool = False) -> Image.Image:
    """
    Load a single chest X-ray as a PIL Image.

    Parameters:
    -----------
    path: str | Path
        Absolute or relative path to the image file.
    rgb: bool
        If True, convert to 3-channel RGB. If False, load as grayscale.

    Returns:
    --------
    img: Image.Image
        PIL Image in mode `L` (grayscale) or `RGB`.
    """
    mode = "RGB" if rgb else "L"
    with Image.open(path) as img:
        return img.convert(mode)


## Dataset

def _find_cache_dir(image_size: int | None) -> Path | None:
    """Return the cache directory for a given image size, or None if it
    doesn't exist or has no files."""
    if image_size is None:
        return None
    cache_dir = DEFAULT_CACHE_ROOT / str(image_size)
    if cache_dir.is_dir():
        return cache_dir
    return None


class ChestXrayDataset(Dataset):
    """
    Reusable dataset that loads images from disk and applies a transform.
    Works for both training (with labels) and inference (without).

    When a pre-resized cache exists (created by ``scripts/cache_images.py``),
    images are loaded from cache instead of the full-resolution originals.

    Parameters:
    -----------
    df: pd.DataFrame
        Must have an `image_path` column with absolute paths and a
        `Path` column with relative paths (for cache lookups).
    labels: np.ndarray | None
        Shape (N, num_labels) float32 array from encode_labels. Pass
        None for test-time inference (no labels available).
    transform: Callable
        A torchvision v2 transform pipeline.
    rgb: bool
        If True, load images as 3-channel RGB. If False, grayscale.
    image_size: int | None
        If set, attempt to load from the pre-resized cache at this size.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        labels: np.ndarray | None,
        transform: Callable,
        rgb: bool = False,
        image_size: int | None = None,
    ) -> None:
        self.paths = df["image_path"].tolist()
        self.rel_paths = df["Path"].tolist() if "Path" in df.columns else None
        self.labels = labels
        self.transform = transform
        self.rgb = rgb
        self.cache_dir = _find_cache_dir(image_size)

        if self.cache_dir is not None:
            print(f"Using image cache: {self.cache_dir}")

        if labels is not None and len(self.paths) != len(labels):
            raise ValueError(
                f"paths ({len(self.paths)}) and labels ({len(labels)}) length mismatch"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def _load_image(self, idx: int) -> Image.Image:
        if self.cache_dir is not None and self.rel_paths is not None:
            cached = self.cache_dir / self.rel_paths[idx]
            if cached.exists():
                return load_image(str(cached), rgb=self.rgb)
        return load_image(self.paths[idx], rgb=self.rgb)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        img = self._load_image(idx)
        x = self.transform(img)

        if self.labels is None:
            return x

        y = torch.from_numpy(self.labels[idx])
        return x, y
