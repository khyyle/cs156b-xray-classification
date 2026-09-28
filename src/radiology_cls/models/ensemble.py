"""
Test-time ensemble of multiple trained checkpoints, implemented as a
BaseModel so it slots into the existing ``train.py`` and ``submit.py``
SLURM machinery exactly like any other model.

Conceptual layout
-----------------
- ``Combiner`` is an ABC for fusing per-member predictions. Today there
  is one implementation (``MeanCombiner``); future combiners (median,
  rank-average, logistic stacking) only need to subclass and register.
- ``EnsembleModel`` wraps a list of member run directories plus a
  combiner. Each member is loaded via that member's own
  ``from_checkpoint``, so the ensemble doesn't care whether members are
  ResNets, ConvNeXts, DenseNets, ... or themselves ensembles.
- Member predictions are cached per member as ``cached_test_preds.npy`` /
  ``cached_val_preds.npy``. The primary location is the member's own run
  dir (so teammates sharing a checkout see the cache). When that dir
  isn't writable -- common on a multi-user shared HPC checkout where
  different teammates created different run dirs -- caching falls back
  to a project-local writable path under ``cache/ensemble_predictions/<member_name>/``.
  Reads check the member dir first, then the fallback. Either way two
  ensembles that share members reuse the cache.

Workflow (mirrors every other model in the project)
---------------------------------------------------
``train.py`` "trains" the ensemble: it reconstructs the patient-level
val split that the members trained on, runs cached val inference for
each, computes per-member and ensemble val MSE, and writes diagnostics
to the run dir. Then ``submit.py`` runs cached test inference per
member, averages, and writes the submission CSV.

Members must agree on ``val_frac`` and ``seed`` so their val splits are
identical -- otherwise an "ensemble val MSE" would be comparing
predictions on different rows. The script aborts loudly if they don't.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import wandb  # type: ignore
import yaml

from radiology_cls.data import (
    LABEL_NAMES,
    load_train_df,
    train_val_split,
)
from radiology_cls.eval import (
    binarize_raw_labels,
    compute_classification_metrics,
    compute_regression_metrics,
    plot_auroc,
    plot_pr_curve,
    save_metrics,
)
from radiology_cls.models.base import BaseModel
from radiology_cls.preprocessing import encode_regression_labels
from radiology_cls.utils import PROJECT_ROOT, ensure_dir, import_class, write_json

logger = logging.getLogger(__name__)

CACHED_TEST_PREDS = "cached_test_preds.npy"
CACHED_VAL_PREDS = "cached_val_preds.npy"
PROJECT_PRED_CACHE_ROOT = PROJECT_ROOT / "cache" / "ensemble_predictions"


def _tag_test_cache_with_rows(cache_name: str, n_rows: int) -> str:
    """Add a row-count tag to a test-pred cache filename.

    Converts e.g. ``cached_test_preds.npy`` to
    ``cached_test_preds__rows22596.npy`` so the public (22596) and
    private (22660) test sets get distinct cache files and don't
    clobber each other on disk.

    Idempotent: an already-tagged name is returned unchanged.

    Val caches don't get tagged because there's only one val set per
    (val_frac, seed) pair, and that's encoded by the ensemble config
    rather than the cache filename.
    """
    if "cached_test_preds" not in cache_name:
        return cache_name
    if "__rows" in cache_name:
        return cache_name
    stem, _, ext = cache_name.rpartition(".")
    return f"{stem}__rows{n_rows}.{ext}"


## Combiner strategy

class Combiner(ABC):
    """
    Strategy for fusing a stack of per-member tanh outputs into a single
    ensemble prediction.

    Implementations should be stateless and operate on numpy arrays.
    Adding a new combiner = subclass + register in ``_COMBINERS``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def combine(self, stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Parameters:
        -----------
        stack: np.ndarray
            Per-member predictions, shape ``(M, N, C)`` with values in
            ``[-1, 1]``.
        weights: np.ndarray
            Length-``M`` non-negative weights summing to 1.

        Returns:
        --------
        combined: np.ndarray
            Ensemble predictions, shape ``(N, C)``.
        """
        ...


class MeanCombiner(Combiner):
    """Weighted arithmetic mean of tanh outputs."""

    @property
    def name(self) -> str:
        return "mean"

    def combine(self, stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return np.tensordot(weights, stack, axes=([0], [0]))


class MedianCombiner(Combiner):
    """
    Per-cell median across members. Robust to a single member making a
    wildly wrong prediction on a given (row, label) cell -- the outlier
    gets ignored instead of pulling the average.

    Median is unweighted by construction. If non-uniform weights are
    supplied we log a warning so the user knows they're being ignored;
    use ``MeanCombiner`` if weights matter for your experiment.
    """

    @property
    def name(self) -> str:
        return "median"

    def combine(self, stack: np.ndarray, weights: np.ndarray) -> np.ndarray:
        if not np.allclose(weights, weights[0]):
            logger.warning(
                "MedianCombiner ignores non-uniform weights %s; "
                "use MeanCombiner if weighting matters",
                weights.tolist(),
            )
        return np.median(stack, axis=0)


_COMBINERS: dict[str, Combiner] = {
    "mean": MeanCombiner(),
    "median": MedianCombiner(),
}


## Helpers

def _resolve_member_dir(member: str | Path) -> Path:
    """
    Resolve a member spec to an absolute run dir. Relative paths are
    resolved against ``PROJECT_ROOT`` so configs can use short paths
    like ``runs/foo_20260506``.
    """
    p = Path(member)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    if not p.is_dir():
        raise FileNotFoundError(f"Ensemble member is not a directory: {p}")
    if not (p / "best.pt").exists():
        raise FileNotFoundError(f"No best.pt in {p}")
    if not (p / "config.yaml").exists():
        raise FileNotFoundError(f"No config.yaml in {p}")
    return p


def _normalize_weights(
    weights: Sequence[float] | None, n: int,
) -> np.ndarray:
    """Validate weights and rescale to sum to 1; default uniform."""
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n,):
        raise ValueError(f"weights must have length {n}, got shape {arr.shape}")
    if (arr < 0).any():
        raise ValueError(f"weights must be non-negative, got {arr.tolist()}")
    total = float(arr.sum())
    if total <= 0:
        raise ValueError("weights must sum to a positive value")
    return arr / total


def _load_member(run_dir: Path) -> tuple[BaseModel, str]:
    """
    Reconstruct a trained ``BaseModel`` from a run dir using its own
    ``config.yaml``. Honors the optional ``model_path`` config key so
    members defined in ``scratch/`` are importable too.
    """
    with open(run_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    model_class_path = cfg.get("model_class")
    if not model_class_path:
        raise ValueError(f"{run_dir}/config.yaml is missing 'model_class'")
    extra_path = cfg.get("model_path")
    if extra_path:
        sys.path.insert(0, str(Path(extra_path).resolve()))
    model_cls = import_class(model_class_path)
    return model_cls.from_checkpoint(run_dir), model_class_path  # type: ignore[attr-defined]


def _free_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _warn_if_out_of_range(run_dir_name: str, preds: np.ndarray) -> None:
    """
    Flag a member whose predictions don't look like tanh outputs. The
    earliest runs in this project trained with BCE (outputs in
    ``[0, 1]``); silently averaging them with current ``tanh`` outputs
    in ``[-1, 1]`` would corrupt the ensemble without raising.
    """
    pmin, pmax = float(preds.min()), float(preds.max())
    if pmin < -1.05 or pmax > 1.05:
        logger.warning(
            "[%s] preds outside expected [-1, 1]: min=%.3f max=%.3f. "
            "If this member was trained with BCE (outputs [0, 1]) it is "
            "incompatible with the rest of the ensemble.",
            run_dir_name, pmin, pmax,
        )


## EnsembleModel

class EnsembleModel(BaseModel):
    """
    Test-time ensemble of trained checkpoints.

    Each member must be a completed run dir containing ``best.pt`` and
    ``config.yaml`` (i.e. anything previously trained via ``train.py``
    whose ``BaseModel`` subclass implements ``from_checkpoint``).

    Parameters:
    -----------
    members: list[str | Path]
        Paths to member run dirs. Relative paths are resolved against
        ``PROJECT_ROOT``.
    weights: list[float] | None
        Optional non-negative per-member weights; auto-normalized to
        sum to 1. Defaults to uniform.
    combiner: str
        Aggregation strategy key from ``_COMBINERS``. Defaults to "mean".
    refresh: bool
        If True, ignore any cached member predictions and recompute.
    tta_transforms: list[str] | None
        Optional test-time augmentation spec. When set, each member runs
        one inference pass per TTA op and averages -- variance reduction
        for free, no retraining. Names come from
        ``radiology_cls.preprocessing.TTA_OPS`` (e.g. ``["identity", "hflip"]``).
        Cached preds get a TTA-specific filename so non-TTA caches stay
        intact.
    """

    def __init__(
        self,
        members: list[str | Path],
        weights: list[float] | None = None,
        combiner: str = "mean",
        refresh: bool = False,
        tta_transforms: list[str] | None = None,
        **kwargs,
    ) -> None:
        if len(members) < 1:
            raise ValueError("EnsembleModel requires at least one member")
        if combiner not in _COMBINERS:
            raise ValueError(
                f"Unknown combiner '{combiner}'. Supported: {sorted(_COMBINERS)}"
            )

        self.member_dirs: list[Path] = [_resolve_member_dir(m) for m in members]
        self.weights: np.ndarray = _normalize_weights(weights, len(self.member_dirs))
        self.combiner: Combiner = _COMBINERS[combiner]
        self.refresh: bool = refresh
        self.tta_transforms: list[str] | None = (
            list(tta_transforms) if tta_transforms else None
        )

    @property
    def name(self) -> str:
        suffix = ""
        if self.tta_transforms:
            suffix = f"-tta{len(self.tta_transforms)}"
        return f"ensemble-{self.combiner.name}-{len(self.member_dirs)}way{suffix}"

    def _tta_cache_suffix(self) -> str:
        """Return a deterministic filename suffix encoding the TTA spec."""
        if not self.tta_transforms:
            return ""
        canonical = "-".join(sorted(self.tta_transforms))
        return f"__tta-{canonical}"

    def _members_share_split(self) -> tuple[float, int]:
        """
        Return the shared ``(val_frac, seed)`` across members or raise.

        Identical val splits are required for an honest ensemble val MSE
        because per-member predictions must align row-by-row.
        """
        val_frac: float | None = None
        seed: int | None = None
        for d in self.member_dirs:
            with open(d / "config.yaml") as f:
                cfg = yaml.safe_load(f)
            vf = float(cfg.get("val_frac", 0.1))
            sd = int(cfg.get("seed", 42))
            if val_frac is None:
                val_frac, seed = vf, sd
                continue
            if vf != val_frac or sd != seed:
                raise ValueError(
                    "Members disagree on val split: "
                    f"{self.member_dirs[0].name} uses val_frac={val_frac}, seed={seed}; "
                    f"{d.name} uses val_frac={vf}, seed={sd}. "
                    "Cannot compute an honest ensemble val MSE."
                )
        assert val_frac is not None and seed is not None
        return val_frac, seed

    def _candidate_cache_paths(self, run_dir: Path, cache_name: str) -> list[Path]:
        """
        Cache locations in preference order.

        Member dir is best (cross-user visible on a shared checkout when
        writable). Project cache is next (team-shared if ``cache/`` is
        team-writable). User home is the last-resort fallback that
        always works.

        When TTA is enabled, the cache filename is suffixed so TTA preds
        live alongside non-TTA preds without colliding.
        """
        suffix = self._tta_cache_suffix()
        if suffix:
            stem, _, ext = cache_name.rpartition(".")
            cache_name = f"{stem}{suffix}.{ext}"
        return [
            run_dir / cache_name,
            PROJECT_PRED_CACHE_ROOT / run_dir.name / cache_name,
            Path.home() / ".cache" / "cs156b_ensemble" / run_dir.name / cache_name,
        ]

    def _member_predict_cached(
        self,
        run_dir: Path,
        df: pd.DataFrame,
        cache_name: str,
    ) -> np.ndarray:
        """
        Return cached preds for a member when shape matches, else load
        the member, run ``predict``, persist to the first writable
        cache location, and free GPU memory.

        Test caches are tagged with the row count of ``df`` so the
        public and private test sets can't clobber each other's caches
        on disk. Legacy untagged caches written by an older version of
        this code are still discoverable through the read fallback and
        validated by shape, so historical caches don't have to be
        regenerated.
        """
        expected = (len(df), len(LABEL_NAMES))
        tagged_name = _tag_test_cache_with_rows(cache_name, expected[0])
        candidates = self._candidate_cache_paths(run_dir, tagged_name)
        # Backward-compat: also look for the legacy untagged filename
        # written by earlier runs. Shape validation below filters out
        # the wrong test set if its untagged cache happens to be present.
        if tagged_name != cache_name:
            candidates.extend(self._candidate_cache_paths(run_dir, cache_name))

        if not self.refresh:
            for cache_path in candidates:
                if not cache_path.exists():
                    continue
                preds = np.load(cache_path)
                if preds.shape == expected:
                    logger.info(
                        "[%s] using cached %s at %s",
                        run_dir.name, tagged_name, cache_path,
                    )
                    _warn_if_out_of_range(run_dir.name, preds)
                    return preds
                logger.info(
                    "[%s] cache shape %s != expected %s at %s; ignoring",
                    run_dir.name, preds.shape, expected, cache_path,
                )

        logger.info("[%s] loading checkpoint", run_dir.name)
        member, _ = _load_member(run_dir)
        # Push our TTA spec down so the member's predict() averages
        # across augmented passes. Members trained without TTA simply
        # gain it at inference time -- no retraining needed. Members
        # whose class doesn't recognize tta_transforms (e.g. ones that
        # don't subclass ImageClassifierModel) silently ignore the
        # attribute, which is the right fallback.
        if self.tta_transforms is not None:
            try:
                member.tta_transforms = list(self.tta_transforms)
            except AttributeError:
                logger.warning(
                    "[%s] member type %s does not accept tta_transforms; "
                    "running inference without TTA",
                    run_dir.name, type(member).__name__,
                )
        try:
            if self.tta_transforms:
                logger.info(
                    "[%s] inference on %d rows with TTA %s",
                    run_dir.name, len(df), self.tta_transforms,
                )
            else:
                logger.info("[%s] inference on %d rows", run_dir.name, len(df))
            preds = member.predict(df)
        finally:
            del member
            _free_gpu_memory()

        if preds.shape != expected:
            raise RuntimeError(
                f"{run_dir.name}.predict returned shape {preds.shape}, "
                f"expected {expected}"
            )
        preds = preds.astype(np.float32)
        _warn_if_out_of_range(run_dir.name, preds)

        # Writes only target the tagged cache locations so the public
        # and private test sets each get distinct files on disk. The
        # legacy-untagged paths checked above are read-only fallbacks
        # for historical caches; we never write back to them.
        write_candidates = self._candidate_cache_paths(run_dir, tagged_name)
        last_err: Exception | None = None
        for i, cache_path in enumerate(write_candidates):
            try:
                ensure_dir(cache_path.parent)
                np.save(cache_path, preds)
                if i == 0:
                    logger.info("[%s] cached preds -> %s", run_dir.name, cache_path)
                else:
                    logger.warning(
                        "[%s] preferred cache %s not writable; "
                        "cached preds -> %s instead",
                        run_dir.name, write_candidates[0], cache_path,
                    )
                return preds
            except (PermissionError, OSError) as e:
                last_err = e
                continue

        raise RuntimeError(
            f"[{run_dir.name}] could not write member preds to any cache "
            f"location: {[str(p) for p in write_candidates]}"
        ) from last_err

    def _stack_member_preds(
        self, df: pd.DataFrame, cache_name: str,
    ) -> np.ndarray:
        return np.stack(
            [self._member_predict_cached(d, df, cache_name) for d in self.member_dirs],
            axis=0,
        )

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        stack = self._stack_member_preds(df, CACHED_TEST_PREDS)
        return self.combiner.combine(stack, self.weights).astype(np.float32)

    def train(self, run_dir: Path) -> dict:
        """
        "Training" an ensemble = computing diagnostics on the shared val
        split and persisting the ensemble manifest. No gradients flow.
        """
        val_frac, seed = self._members_share_split()
        full_df = load_train_df()
        _, val_df = train_val_split(full_df, val_frac=val_frac, seed=seed)
        y_val = encode_regression_labels(val_df, list(LABEL_NAMES))

        member_val_preds = self._stack_member_preds(val_df, CACHED_VAL_PREDS)
        ensemble_val_preds = self.combiner.combine(member_val_preds, self.weights)

        per_member_macro: list[float] = []
        for i, d in enumerate(self.member_dirs):
            mi = compute_regression_metrics(
                y_val, member_val_preds[i], list(LABEL_NAMES),
            )
            per_member_macro.append(float(mi["macro_mse"] or float("nan")))

        ensemble_metrics = compute_regression_metrics(
            y_val, ensemble_val_preds, list(LABEL_NAMES),
        )
        classification_metrics = compute_classification_metrics(
            binarize_raw_labels(y_val), ensemble_val_preds, list(LABEL_NAMES),
        )
        ensemble_metrics["classification_diagnostics"] = classification_metrics
        save_metrics(ensemble_metrics, run_dir)

        plots_dir = ensure_dir(run_dir / "plots")
        auroc_fig = plot_auroc(
            binarize_raw_labels(y_val), ensemble_val_preds, list(LABEL_NAMES), plots_dir,
        )
        pr_fig = plot_pr_curve(
            binarize_raw_labels(y_val), ensemble_val_preds, list(LABEL_NAMES), plots_dir,
        )
        import matplotlib.pyplot as plt  # type: ignore
        plt.close(auroc_fig)
        plt.close(pr_fig)

        np.save(run_dir / "ensemble_val_preds.npy", ensemble_val_preds.astype(np.float32))
        np.save(run_dir / "val_targets.npy", y_val.astype(np.float32))

        members_summary = [
            {
                "run_dir": str(d),
                "weight": float(self.weights[i]),
                "val_macro_mse": per_member_macro[i],
            }
            for i, d in enumerate(self.member_dirs)
        ]
        write_json(
            {
                "combiner": self.combiner.name,
                "members": members_summary,
                "val_split": {"val_frac": val_frac, "seed": seed},
            },
            run_dir / "members.json",
        )

        best_member = min(members_summary, key=lambda m: m["val_macro_mse"])
        ensemble_macro = ensemble_metrics["macro_mse"] or float("inf")
        gain = best_member["val_macro_mse"] - ensemble_macro

        try:
            wandb.log({
                "ensemble_val_macro_mse": ensemble_metrics["macro_mse"],
                "ensemble_val_macro_auroc": classification_metrics.get("macro_auroc"),
                "ensemble_val_macro_pr_auc": classification_metrics.get("macro_pr_auc"),
                "ensemble_gain_vs_best_member": gain,
                **{
                    f"member_{i}_val_macro_mse": v
                    for i, v in enumerate(per_member_macro)
                },
            })
        except Exception:
            pass

        rows = "\n".join(
            f"  {Path(m['run_dir']).name:<60s} w={m['weight']:.3f}  "
            f"val_macro_mse={m['val_macro_mse']:.4f}"
            for m in members_summary
        )
        logger.info("Per-member val MSE:\n%s", rows)
        logger.info(
            "Ensemble (%s, %d members) val_macro_mse=%.4f  gain_vs_best=%+.4f",
            self.combiner.name, len(self.member_dirs), ensemble_macro, gain,
        )

        return {
            "name": self.name,
            "combiner": self.combiner.name,
            "ensemble_val_macro_mse": ensemble_metrics["macro_mse"],
            "ensemble_val_macro_auroc": classification_metrics.get("macro_auroc"),
            "ensemble_val_macro_pr_auc": classification_metrics.get("macro_pr_auc"),
            "per_member_val_macro_mse": dict(zip(
                [d.name for d in self.member_dirs], per_member_macro,
            )),
            "gain_vs_best_member": gain,
            "weights": self.weights.tolist(),
            "members": [str(d) for d in self.member_dirs],
        }

    @classmethod
    def from_checkpoint(cls, run_dir: Path, **kwargs) -> "EnsembleModel":
        """
        Reconstruct from the ensemble run dir's ``config.yaml``. Members
        themselves are loaded lazily inside ``predict`` via their own
        ``from_checkpoint``.
        """
        with open(run_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg.pop("model_class", None)
        cfg.pop("model_path", None)
        cfg.pop("run_name", None)
        cfg.update(kwargs)
        return cls(**cfg)
