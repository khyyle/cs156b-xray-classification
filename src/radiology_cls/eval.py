"""
Evaluation utilities for classification/regression metrics and diagnostic
plots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib  # type: ignore
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)  # type: ignore
import pandas as pd
from matplotlib.figure import Figure  # type: ignore

from .utils import write_json



def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Compute per-label and macro-averaged AUROC and PR-AUC.

    NaN entries in y_true are masked per column so partially-labeled
    samples are handled gracefully. Predictions may be probabilities or
    any score where larger values mean more likely positive.

    Parameters:
    -----------
    y_true: np.ndarray
        Ground-truth labels with shape `(N, num_labels)`. May contain
        NaN for missing annotations.
    y_pred: np.ndarray
        Predicted probabilities or ranking scores with shape
        `(N, num_labels)`.
    label_names: list[str]
        Human-readable name for each label column, in the same order as
        the columns of y_true and y_pred.

    Returns:
    --------
    results: dict
        Keys: `per_label` (dict mapping label name to `auroc` and
        `pr_auc`), `macro_auroc`, `macro_pr_auc`. Values are
        `None` when a label has fewer than two unique ground-truth
        values.
    """
    results: dict = {"per_label": {}}

    aurocs, pr_aucs = [], []
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        yt = y_true[mask, i]
        yp = y_pred[mask, i]

        if len(np.unique(yt)) < 2:
            results["per_label"][name] = {"auroc": None, "pr_auc": None}
            continue

        auroc = float(roc_auc_score(yt, yp))
        pr_auc = float(average_precision_score(yt, yp))
        results["per_label"][name] = {"auroc": auroc, "pr_auc": pr_auc}
        aurocs.append(auroc)
        pr_aucs.append(pr_auc)

    results["macro_auroc"] = float(np.mean(aurocs)) if aurocs else None
    results["macro_pr_auc"] = float(np.mean(pr_aucs)) if pr_aucs else None
    return results


def binarize_raw_labels(
    y_true: np.ndarray,
) -> np.ndarray:
    """
    Convert raw CheXpert labels to binary labels for ranking metrics.

    Positive labels (`1`) become `1`, negative labels (`-1`) become `0`,
    and uncertain (`0`) / blank (`NaN`) labels become `NaN` so they are
    ignored by `compute_classification_metrics`.

    Parameters:
    -----------
    y_true: np.ndarray
        Raw labels with shape `(N, num_labels)` on the CheXpert label
        scale {-1, 0, 1, NaN}.

    Returns:
    --------
    y_binary: np.ndarray
        Binary labels with shape `(N, num_labels)`, containing 0, 1, and
        NaN values.
    """
    y_binary = np.full_like(y_true, np.nan, dtype=np.float32)
    y_binary[y_true == 1.0] = 1.0
    y_binary[y_true == -1.0] = 0.0
    return y_binary


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
) -> dict:
    """
    Compute masked MSE and MAE for outputs in [-1, 1]

    Parameters:
    -----------
    y_true: np.ndarray
        Ground-truth labels with shape `(N, num_labels)`. May contain
        NaN for blank / unmentioned labels.
    y_pred: np.ndarray
        Predicted raw-scale values with shape `(N, num_labels)`.
    label_names: list[str]
        Human-readable name for each label column, in the same order as
        the columns of y_true and y_pred.

    Returns:
    --------
    results: dict
        Keys: `per_label` (dict mapping label name to `mse`, `mae`, `nmse`),
        `macro_mse`, `macro_mae`, and `macro_nmse`. Per-label values are
        `None` when that label has no non-NaN validation entries.
    """
    results: dict = {"per_label": {}}

    mses, maes, nmses = [], [], []
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        if not mask.any():
            results["per_label"][name] = {"mse": None, "mae": None, "nmse": None}
            continue

        errors = y_pred[mask, i] - y_true[mask, i]
        mse = float(np.mean(errors ** 2))
        mae = float(np.mean(np.abs(errors)))
        var = float(np.var(y_true[mask, i]))
        nmse = mse / var if var > 1e-12 else float("nan")
        results["per_label"][name] = {"mse": mse, "mae": mae, "nmse": nmse}
        mses.append(mse)
        maes.append(mae)
        if not np.isnan(nmse):
            nmses.append(nmse)

    results["macro_mse"] = float(np.mean(mses)) if mses else None
    results["macro_mae"] = float(np.mean(maes)) if maes else None
    results["macro_nmse"] = float(np.mean(nmses)) if nmses else None
    return results


def plot_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    save_dir: str | Path | None = None,
) -> Figure:
    """
    Plot per-label ROC curves on a single figure. Each curve is
    labeled with its AUROC value in the legend. If save_dir is provided
    the figure is saved as `auroc.png` there.

    Parameters:
    -----------
    y_true: np.ndarray
        Ground-truth labels, shape `(N, num_labels)`.
    y_pred: np.ndarray
        Predicted probabilities, shape `(N, num_labels)`.
    label_names: list[str]
        Human-readable name for each label column.
    save_dir: str | Path | None
        Directory to write the plot to. Pass `None` to skip saving

    Returns:
    --------
    fig: Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        yt = y_true[mask, i]
        yp = y_pred[mask, i]
        if len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc_val = roc_auc_score(yt, yp)
        ax.plot(fpr, tpr, label=f"{name} ({auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (per pathology)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "auroc.png", dpi=150)

    return fig


def plot_pr_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    save_dir: str | Path | None = None,
) -> Figure:
    """
    Plot per-label precision-recall curves on a single figure. Each
    curve is labeled with its average precision in the legend. If
    save_dir is provided the figure is saved as `pr_curve.png` there.

    Parameters:
    -----------
    y_true: np.ndarray
        Ground-truth labels, shape `(N, num_labels)`.
    y_pred: np.ndarray
        Predicted probabilities, shape `(N, num_labels)`.
    label_names: list[str]
        Human-readable name for each label column.
    save_dir: str | Path | None
        Directory to write the plot to. Pass `None` to skip saving.

    Returns:
    --------
    fig: Figure
        The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(label_names):
        mask = ~np.isnan(y_true[:, i])
        yt = y_true[mask, i]
        yp = y_pred[mask, i]
        if len(np.unique(yt)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(yt, yp)
        ap = average_precision_score(yt, yp)
        ax.plot(recall, precision, label=f"{name} ({ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (per pathology)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "pr_curve.png", dpi=150)

    return fig


def plot_label_distribution(
    df: pd.DataFrame,
    label_names: list[str],
    save_dir: str | Path | None = None,
) -> Figure:
    """
    Bar chart showing positive, negative, uncertain, and blank counts
    per label.

    Parameters:
    -----------
    df: pd.DataFrame
        Training dataframe with pathology label columns.
    label_names: list[str]
        Which columns to include in the chart.
    save_dir: str | Path | None
        Directory to write the plot to. Pass `None` to skip saving.

    Returns:
    --------
    fig: Figure
        The matplotlib figure.
    """
    counts: dict[str, list[int]] = {"positive": [], "negative": [], "uncertain": [], "blank": []}
    for name in label_names:
        col = df[name]
        counts["positive"].append(int((col == 1.0).sum()))
        counts["negative"].append(int((col == -1.0).sum()))
        counts["uncertain"].append(int((col == 0.0).sum()))
        counts["blank"].append(int(col.isna().sum()))

    x = np.arange(len(label_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5 * width, counts["positive"], width, label="Positive (1)")
    ax.bar(x - 0.5 * width, counts["negative"], width, label="Negative (-1)")
    ax.bar(x + 0.5 * width, counts["uncertain"], width, label="Uncertain (0)")
    ax.bar(x + 1.5 * width, counts["blank"], width, label="Blank (NaN)")

    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("Label Distribution")
    ax.legend()
    fig.tight_layout()

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / "label_distribution.png", dpi=150)

    return fig


def save_metrics(metrics: dict, save_dir: str | Path) -> Path:
    """
    Serialize a metrics dict to metrics.json in the given directory.

    Parameters:
    -----------
    metrics: dict
        Metrics returned by `compute_classification_metrics` or
        `compute_regression_metrics`.
    save_dir: str | Path
        Directory to write `metrics.json` into.

    Returns:
    --------
    path: Path
        Path to the written JSON file.
    """
    path = Path(save_dir) / "metrics.json"
    write_json(metrics, path)
    return path
