from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb  # type: ignore
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from radiology_cls import (
    BaseModel,
    ChestXrayDataset,
    LABEL_NAMES,
    load_train_df,
    train_val_split,
)
from radiology_cls import ensure_dir
from radiology_cls.eval import (
    binarize_raw_labels,
    compute_classification_metrics,
    compute_regression_metrics,
    plot_auroc,
    plot_pr_curve,
    save_metrics,
)
from radiology_cls.preprocessing import (
    encode_regression_labels,
    Masked3ClassLoss,
    MaskedRegressionLoss,
    noisy_or_positive_prob,
    threeclass_logits_to_score,
    train_transform,
    val_transform,
    val_transform_with_tta,
)

_SUPPORTED_TARGET_TYPES = ("regression", "3class")

logger = logging.getLogger(__name__)


class ImageClassifierModel(BaseModel):
    """
    Shared training wrapper for neural net ablations.

    Subclasses only define the architecture via `build_model` and whether
    they expect RGB inputs. The training/eval/checkpoint behavior stays
    consistent across architecture sweeps.
    """

    def __init__(
        self,
        image_size: int = 256,
        batch_size: int = 32,
        lr: float = 1e-4,
        epochs: int = 50,
        val_frac: float = 0.1,
        seed: int = 42,
        num_workers: int = 4,
        loss_type: str = "smooth_l1",
        weight_decay: float = 1e-4,
        scheduler: str = "none",
        min_lr: float = 1e-6,
        warmup_epochs: int = 0,
        rgb: bool = False,
        tta_transforms: list[str] | None = None,
        target_type: str = "regression",
        llrd_decay: float | None = None,
        strong_aug: bool = False,
        
        # Auxiliary hard-class asymmetric loss configuration.
        asl_alpha: float = 0.0,
        asl_hard_classes: list[str] | None = None,   # i.e. rare-positive classes like pleural other, pneumonia
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        # Per-class ASL override. When provided, the keys (class names)
        # define which classes get ASL and each value is a dict with
        # gamma_pos / gamma_neg / clip, letting positive-flooded and
        # negative-flooded classes get opposite focal directions in one run.
        asl_per_class: dict[str, dict] | None = None,
        
        # Noisy-OR auxiliary architectural configuration
        # Designed for heterogenous classes whose visual signature is a
        # union of distinct subtypes (e.g. Pleural Other).
        noisy_or_classes: list[str] | None = None,
        noisy_or_K: int = 4,
        noisy_or_beta: float = 0.0,
        noisy_or_gamma_pos: float = 0.0,
        noisy_or_gamma_neg: float = 4.0,
        noisy_or_clip: float = 0.05,
        **kwargs,
    ) -> None:
        if target_type not in _SUPPORTED_TARGET_TYPES:
            raise ValueError(
                f"Unsupported target_type {target_type!r}. "
                f"Expected one of: {_SUPPORTED_TARGET_TYPES}"
            )
        if llrd_decay is not None and not (0.0 < llrd_decay <= 1.0):
            raise ValueError(
                f"llrd_decay must be in (0, 1], got {llrd_decay!r}"
            )

        self.image_size = image_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.val_frac = val_frac
        self.seed = seed
        self.num_workers = num_workers
        self.loss_type = loss_type
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.rgb = rgb
        self.target_type = target_type
        self.llrd_decay = llrd_decay
        self.strong_aug = strong_aug

        # 1 output per label for regression (tanh scalar), 3 outputs per label for 3-class CE
        self.num_classes_per_label = 3 if target_type == "3class" else 1
        # tta_transforms can also be set after construction by an outer
        # caller (e.g. EnsembleModel pushes its own TTA spec down into
        # each member at inference time). When None, predict behaves
        # exactly as before.
        self.tta_transforms: list[str] | None = list(tta_transforms) if tta_transforms else None

        # Resolve hard-class / noisy-OR class-name lists to label indices.
        # Both auxiliary mechanisms require target_type=3class because
        # they read positive softmax probabilities; reject misconfigs
        # early rather than fail with a confusing tensor-shape error
        # mid-training.
        self.asl_alpha = float(asl_alpha)
        self.asl_hard_classes: list[str] = list(asl_hard_classes) if asl_hard_classes else []
        self.asl_hard_indices: list[int] = self._resolve_label_indices(
            self.asl_hard_classes, knob_name="asl_hard_classes",
        )
        self.asl_gamma_pos = float(asl_gamma_pos)
        self.asl_gamma_neg = float(asl_gamma_neg)
        self.asl_clip = float(asl_clip)

        # Reject double-config to avoid silently dropping one path.
        if asl_per_class and asl_hard_classes:
            raise ValueError(
                "Specify either asl_hard_classes (uniform gammas) or "
                "asl_per_class (per-class gammas), not both."
            )
        per_class_indices = self._resolve_label_indices(
            list(asl_per_class.keys()) if asl_per_class else [],
            knob_name="asl_per_class",
        )
        self.asl_per_class: dict[int, dict] | None = (
            {idx: asl_per_class[name] for idx, name in zip(per_class_indices, asl_per_class)}
            if asl_per_class else None
        )

        self.noisy_or_classes: list[str] = list(noisy_or_classes) if noisy_or_classes else []
        self.noisy_or_indices: list[int] = self._resolve_label_indices(
            self.noisy_or_classes, knob_name="noisy_or_classes",
        )
        self.noisy_or_K = int(noisy_or_K)
        self.noisy_or_beta = float(noisy_or_beta)
        self.noisy_or_gamma_pos = float(noisy_or_gamma_pos)
        self.noisy_or_gamma_neg = float(noisy_or_gamma_neg)
        self.noisy_or_clip = float(noisy_or_clip)

        asl_active = self.asl_alpha > 0.0 and (self.asl_hard_indices or self.asl_per_class)
        if asl_active and target_type != "3class":
            raise ValueError(
                "ASL auxiliary loss requires target_type='3class' "
                "(it reads positive softmax probabilities)."
            )
        if (self.noisy_or_beta > 0.0 and self.noisy_or_indices) and target_type != "3class":
            raise ValueError(
                "Noisy-OR auxiliary branch requires target_type='3class'."
            )

        # Per-class blending lambdas fit on val after training; populated
        # by ``_fit_aux_blend`` at the end of train() and persisted to
        # disk as ``noisy_or_blend_lambdas.npy`` for reload.
        # Shape: (len(noisy_or_indices),). lambda=1 means "ignore aux".
        self.noisy_or_lambdas: np.ndarray | None = None

        # Per-class affine y = a*s + b fit by scripts/affine_calibrate.py; None = identity.
        self.affine_calibration: np.ndarray | None = None  # (2, NUM_CLASSES): slope, intercept

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.build_model().to(self.device)

    @staticmethod
    def _resolve_label_indices(names: list[str], knob_name: str) -> list[int]:
        """Map human-readable label names to their position in ``LABEL_NAMES``."""
        if not names:
            return []
        label_lookup = {name: i for i, name in enumerate(LABEL_NAMES)}
        indices: list[int] = []
        unknown: list[str] = []
        for n in names:
            if n not in label_lookup:
                unknown.append(n)
            else:
                indices.append(label_lookup[n])
        if unknown:
            raise ValueError(
                f"{knob_name} includes unknown label(s) {unknown}. "
                f"Valid labels: {list(LABEL_NAMES)}"
            )
        return indices

    def _logits_to_score(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Convert raw model logits to a scalar prediction in `[-1, 1]`.
        """
        if self.target_type == "3class":
            return threeclass_logits_to_score(logits)
        return torch.tanh(logits)

    def _build_criterion(self) -> nn.Module:
        if self.target_type == "3class":
            return Masked3ClassLoss(
                asl_alpha=self.asl_alpha,
                asl_hard_indices=self.asl_hard_indices,
                asl_gamma_pos=self.asl_gamma_pos,
                asl_gamma_neg=self.asl_gamma_neg,
                asl_clip=self.asl_clip,
                asl_per_class=self.asl_per_class,
                noisy_or_beta=self.noisy_or_beta,
                noisy_or_indices=self.noisy_or_indices,
                noisy_or_gamma_pos=self.noisy_or_gamma_pos,
                noisy_or_gamma_neg=self.noisy_or_gamma_neg,
                noisy_or_clip=self.noisy_or_clip,
            )
        return MaskedRegressionLoss(loss_type=self.loss_type)

    @staticmethod
    def _unpack_forward(out: torch.Tensor | tuple) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return ``(main_logits, aux_subtype_logits)`` whether the model
        emits a bare tensor (no aux branches) or a tuple."""
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None

    def _compute_loss(
        self,
        criterion: nn.Module,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_subtype_logits: torch.Tensor | None,
    ) -> torch.Tensor:
        if isinstance(criterion, Masked3ClassLoss):
            return criterion(logits, targets, aux_subtype_logits=aux_subtype_logits)
        return criterion(logits, targets)

    @abstractmethod
    def build_model(self) -> nn.Module:
        ...

    def _optimizer_param_groups(self) -> list[dict]:
        """
        Build AdamW parameter groups.

        With ``llrd_decay`` set, uses timm's layer-wise learning rate decay
        (LLRD): each layer gets a per-group lr scaled by
        ``llrd_decay ** (max_depth - depth)``, so the patch embed / early
        transformer blocks see a much smaller LR than the head. This is the
        standard recipe for fine-tuning large pretrained ViTs (DINOv2, BEiT,
        MAE) -- it protects the pretrained features at the bottom while
        letting the new classification head train at full speed. timm's
        helper uses each model's own ``group_matcher()`` to assign depths
        correctly across architectures (ViT, Swin, ConvNeXt).

        Without ``llrd_decay``, falls back to the original 2-group split
        that decays only matrix weights, leaving biases and norm scales
        unregularized (Wang et al. 2025 ConvNeXt recipe).

        Returns:
        --------
        param_groups: list[dict]
            Parameter groups for torch optimizers.
        """
        if self.llrd_decay is not None:
            from timm.optim.optim_factory import param_groups_layer_decay  # type: ignore[attr-defined]
            param_groups = param_groups_layer_decay(
                self.model,
                weight_decay=self.weight_decay,
                layer_decay=self.llrd_decay,
            )
            # timm returns lr_scale metadata for each group; when using
            # torch.optim.AdamW directly (instead of timm.create_optimizer_v2),
            # materialize it as an actual per-group learning rate.
            for group in param_groups:
                lr_scale = float(group.pop("lr_scale", 1.0))
                group["lr"] = self.lr * lr_scale
            return param_groups

        decay = []
        no_decay = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.endswith(".bias"):
                no_decay.append(param)
            else:
                decay.append(param)

        return [
            {"params": decay, "weight_decay": self.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def train(self, run_dir: Path) -> dict:
        use_amp = self.device.type == "cuda"
        print(f"Device: {self.device}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if use_amp:
            print(f"GPU: {torch.cuda.get_device_name(self.device)}")
            print("Mixed precision: float16 (AMP enabled)")
            torch.backends.cudnn.benchmark = True

        df = load_train_df()
        train_df, val_df = train_val_split(df, self.val_frac, self.seed)

        y_train = encode_regression_labels(train_df, list(LABEL_NAMES))
        y_val = encode_regression_labels(val_df, list(LABEL_NAMES))

        train_ds = ChestXrayDataset(
            train_df,
            y_train,
            train_transform(self.image_size, rgb=self.rgb, strong_aug=self.strong_aug),
            rgb=self.rgb,
            image_size=self.image_size,
        )
        val_ds = ChestXrayDataset(
            val_df,
            y_val,
            val_transform(self.image_size, rgb=self.rgb),
            rgb=self.rgb,
            image_size=self.image_size,
        )

        if self.num_workers > 0:
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=use_amp,
                persistent_workers=True,
                prefetch_factor=4,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=use_amp,
                persistent_workers=True,
                prefetch_factor=4,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=use_amp,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=use_amp,
            )

        optimizer = torch.optim.AdamW(
            self._optimizer_param_groups(),
            lr=self.lr,
        )
        if self.scheduler == "none":
            lr_scheduler = None
        elif self.scheduler == "cosine":
            cosine_epochs = max(self.epochs - self.warmup_epochs, 1)
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cosine_epochs, eta_min=self.min_lr,
            )
            if self.warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    total_iters=self.warmup_epochs,
                )
                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs],
                )
            else:
                lr_scheduler = cosine_scheduler # type: ignore
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler}")

        # .to(device) so any buffers the criterion registers (e.g. per-class
        # ASL gamma tensors) match the input device at forward time.
        criterion = self._build_criterion().to(self.device)
        scaler = torch.amp.GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        best_epoch = 0
        final_val_loss = float("nan")
        best_val_logits_all: list[torch.Tensor] | None = None
        best_val_labels_all: list[torch.Tensor] | None = None
        best_val_aux_all: list[torch.Tensor] | None = None

        with tqdm(range(self.epochs), desc="epochs", unit="epoch") as epoch_bar:
            for epoch in epoch_bar:
                epoch_num = epoch + 1
                self.model.train()
                train_loss_sum = 0.0
                train_batches = 0

                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                        out = self.model(x)
                        logits, aux = self._unpack_forward(out)
                        loss = self._compute_loss(criterion, logits, y, aux)

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    train_loss_sum += loss.item()
                    train_batches += 1

                train_loss = train_loss_sum / train_batches

                self.model.eval()
                val_loss_sum = 0.0
                val_batches = 0
                val_logits_all: list[torch.Tensor] = []
                val_aux_all: list[torch.Tensor] = []
                val_labels_all: list[torch.Tensor] = []

                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(self.device), y.to(self.device)
                        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                            out = self.model(x)
                            logits, aux = self._unpack_forward(out)
                        aux_float = aux.float() if aux is not None else None
                        loss = self._compute_loss(criterion, logits.float(), y, aux_float)

                        val_loss_sum += loss.item()
                        val_batches += 1
                        val_logits_all.append(logits.float().cpu())
                        if aux_float is not None:
                            val_aux_all.append(aux_float.cpu())
                        val_labels_all.append(y.cpu())

                val_loss = val_loss_sum / val_batches
                final_val_loss = val_loss
                current_lr = optimizer.param_groups[0]["lr"]

                epoch_bar.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_loss=f"{val_loss:.4f}",
                    lr=f"{current_lr:.2e}",
                )

                if val_loss < best_val_loss:
                    previous_best = best_val_loss
                    best_val_loss = val_loss
                    best_epoch = epoch_num
                    best_val_logits_all = [batch.clone() for batch in val_logits_all]
                    best_val_labels_all = [batch.clone() for batch in val_labels_all]
                    best_val_aux_all = (
                        [batch.clone() for batch in val_aux_all] if val_aux_all else None
                    )
                    torch.save(self.model.state_dict(), run_dir / "best.pt")
                    previous_msg = "n/a" if previous_best == float("inf") else f"{previous_best:.4f}"
                    logger.info(
                        "Saved new best checkpoint at epoch %d: val_loss=%.4f (previous best=%s)",
                        best_epoch,
                        best_val_loss,
                        previous_msg,
                    )

                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "lr": current_lr,
                    "epoch": epoch_num,
                })

                if lr_scheduler is not None:
                    lr_scheduler.step()

        torch.save(self.model.state_dict(), run_dir / "last.pt")
        if best_val_logits_all is None or best_val_labels_all is None:
            raise RuntimeError("No best checkpoint was saved; epochs must be at least 1")

        state_dict = torch.load(run_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(state_dict)

        y_true = torch.cat(best_val_labels_all).numpy()
        val_logits = torch.cat(best_val_logits_all)
        y_pred = self._logits_to_score(val_logits).numpy()

        # If the model has noisy-OR aux branches, fit per-class blending
        # lambdas on best-epoch val outputs. Persist as ``noisy_or_blend_lambdas.npy``
        # so the same blending is applied at inference time when this run is
        # reloaded via ``from_checkpoint``. Skipping when no aux branches exist
        # is a no-op for any pre-noisy-OR config.
        if (
            self.noisy_or_indices
            and best_val_aux_all is not None
            and len(best_val_aux_all) > 0
        ):
            val_aux_subtype = torch.cat(best_val_aux_all)  # (N, num_aux, K)
            val_p_no = noisy_or_positive_prob(val_aux_subtype).numpy()  # (N, num_aux)
            lambdas, blended_mse, main_mse = self._fit_noisy_or_blend(
                main_scores=y_pred,
                aux_p_pos=val_p_no,
                targets=y_true,
                aux_indices=self.noisy_or_indices,
            )
            self.noisy_or_lambdas = lambdas
            np.save(run_dir / "noisy_or_blend_lambdas.npy", lambdas.astype(np.float32))
            for i, lbl_idx in enumerate(self.noisy_or_indices):
                logger.info(
                    "Noisy-OR blend [%s]: lambda=%.2f  main_mse=%.4f  blended_mse=%.4f  delta=%+.4f",
                    LABEL_NAMES[lbl_idx],
                    float(lambdas[i]),
                    float(main_mse[i]),
                    float(blended_mse[i]),
                    float(blended_mse[i] - main_mse[i]),
                )
            # Re-derive y_pred for downstream metrics with the learned blend.
            y_pred = self._apply_noisy_or_blend(y_pred, val_p_no)

        plots_dir = ensure_dir(run_dir / "plots")
        metrics = compute_regression_metrics(y_true, y_pred, list(LABEL_NAMES))
        classification_metrics = compute_classification_metrics(
            binarize_raw_labels(y_true),
            y_pred,
            list(LABEL_NAMES),
        )
        metrics["classification_diagnostics"] = classification_metrics
        save_metrics(metrics, run_dir)

        auroc_fig = plot_auroc(binarize_raw_labels(y_true), y_pred, list(LABEL_NAMES), plots_dir)
        pr_fig = plot_pr_curve(binarize_raw_labels(y_true), y_pred, list(LABEL_NAMES), plots_dir)
        import matplotlib.pyplot as plt  # type: ignore
        plt.close(auroc_fig)
        plt.close(pr_fig)

        logger.info(f"Macro MSE: {metrics.get('macro_mse', 'n/a')}")
        logger.info(f"Macro MAE: {metrics.get('macro_mae', 'n/a')}")
        logger.info(
            "Diagnostic macro AUROC: %s",
            classification_metrics.get("macro_auroc", "n/a"),
        )
        logger.info(
            "Best checkpoint: epoch %d val_loss=%.4f; final epoch val_loss=%.4f",
            best_epoch,
            best_val_loss,
            final_val_loss,
        )

        return {
            "metric_checkpoint": "best.pt",
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "final_epoch": self.epochs,
            "final_val_loss": final_val_loss,
            "macro_mse": metrics.get("macro_mse"),
            "macro_mae": metrics.get("macro_mae"),
            "diagnostic_macro_auroc": classification_metrics.get("macro_auroc"),
            "diagnostic_macro_pr_auc": classification_metrics.get("macro_pr_auc"),
        }

    @staticmethod
    def _fit_noisy_or_blend(
        main_scores: np.ndarray,
        aux_p_pos: np.ndarray,
        targets: np.ndarray,
        aux_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Grid-search per-aux-class blending weights against val labels.

        For each aux class ``c`` at label index ``aux_indices[c]``, search
        ``lambda in {0, 0.02, ..., 1.0}`` for the value that minimizes
        MSE between ``y[lbl_idx]`` and
        ``lambda * s_main + (1 - lambda) * (2 * p_pos - 1)`` over valid
        (non-NaN) val rows.

        Parameters:
        -----------
        main_scores: np.ndarray
            (N, num_labels) scalar scores from the main 3-class softmax,
            already in ``[-1, 1]``.
        aux_p_pos: np.ndarray
            (N, num_aux) noisy-OR positive probabilities in ``[0, 1]``.
        targets: np.ndarray
            (N, num_labels) raw labels in ``{-1, 0, +1, NaN}``.
        aux_indices: list[int]
            Length ``num_aux`` mapping aux column to label column.

        Returns:
        --------
        lambdas: np.ndarray
            Optimal blending weight per aux class in ``[0, 1]``.
        blended_mse: np.ndarray
            Val per-aux MSE after blending.
        main_mse: np.ndarray
            Val per-aux MSE without blending (main only).
        """
        num_aux = len(aux_indices)
        lambdas = np.ones(num_aux, dtype=np.float64)
        blended_mse = np.zeros(num_aux, dtype=np.float64)
        main_mse = np.zeros(num_aux, dtype=np.float64)
        # 51 lambda candidates: 0.00, 0.02, ..., 1.00. Fine-grained enough
        # for a 1-D search and trivial to compute (N << 100k).
        grid = np.linspace(0.0, 1.0, 51)
        for c, lbl_idx in enumerate(aux_indices):
            y = targets[:, lbl_idx]
            valid = ~np.isnan(y)
            if not valid.any():
                continue
            s_main = main_scores[valid, lbl_idx]
            s_aux = 2.0 * aux_p_pos[valid, c] - 1.0  # map [0,1] -> [-1,1]
            y_v = y[valid]
            mse_main = float(np.mean((s_main - y_v) ** 2))
            main_mse[c] = mse_main
            best_lam, best_mse = 1.0, mse_main
            for lam in grid:
                s = lam * s_main + (1.0 - lam) * s_aux
                mse = float(np.mean((s - y_v) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_lam = float(lam)
            lambdas[c] = best_lam
            blended_mse[c] = best_mse
        return lambdas, blended_mse, main_mse

    def _apply_affine_calibration(self, scores: np.ndarray) -> np.ndarray:
        """Apply per-class ``s' = a*s + b`` when calibration is loaded; no-op otherwise."""
        if self.affine_calibration is None:
            return scores
        if self.affine_calibration.shape != (2, scores.shape[1]):
            raise RuntimeError(
                f"affine_calibration shape {self.affine_calibration.shape} != "
                f"(2, {scores.shape[1]})"
            )
        a = self.affine_calibration[0].astype(np.float32)
        b = self.affine_calibration[1].astype(np.float32)
        return (scores.astype(np.float32) * a + b).astype(np.float32)

    def _apply_noisy_or_blend(
        self,
        main_scores: np.ndarray,
        aux_p_pos: np.ndarray,
    ) -> np.ndarray:
        """Overlay aux-blended scores onto ``main_scores`` for noisy-OR classes.

        Parameters:
        -----------
        main_scores: np.ndarray
            Shape (N, num_labels). All non-aux columns are returned unchanged.
        aux_p_pos: np.ndarray
            Shape (N, num_aux); column order matches ``self.noisy_or_indices``.

        Returns:
        --------
        blended: np.ndarray
            Copy of ``main_scores`` with aux columns replaced by the
            learned-lambda blend.
        """
        if self.noisy_or_lambdas is None or not self.noisy_or_indices:
            return main_scores
        blended = main_scores.astype(np.float32, copy=True)
        for c, lbl_idx in enumerate(self.noisy_or_indices):
            lam = float(self.noisy_or_lambdas[c])
            s_main = main_scores[:, lbl_idx]
            s_aux = 2.0 * aux_p_pos[:, c] - 1.0
            blended[:, lbl_idx] = (lam * s_main + (1.0 - lam) * s_aux).astype(np.float32)
        return blended

    def _predict_with_transform(
        self,
        df: pd.DataFrame,
        transform,
    ) -> np.ndarray:
        """
        Single inference pass over ``df`` using the provided transform.

        Used both by the default ``predict`` (with ``val_transform``) and
        by the TTA path (with each ``val_transform_with_tta(...)``).
        """
        use_amp = self.device.type == "cuda"
        ds = ChestXrayDataset(
            df,
            labels=None,
            transform=transform,
            rgb=self.rgb,
            image_size=self.image_size,
        )
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=use_amp,
            persistent_workers=self.num_workers > 0,
        )

        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_aux_p_pos: list[np.ndarray] = []  # used only when noisy-OR is active
        with torch.no_grad():
            for x in loader:
                x = x.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    out = self.model(x)
                    logits, aux = self._unpack_forward(out)
                all_preds.append(self._logits_to_score(logits.float()).cpu().numpy())
                if aux is not None and self.noisy_or_indices:
                    p_no = noisy_or_positive_prob(aux.float()).cpu().numpy()
                    all_aux_p_pos.append(p_no)

        main = np.concatenate(all_preds, axis=0)
        if (
            self.noisy_or_lambdas is not None
            and self.noisy_or_indices
            and all_aux_p_pos
        ):
            aux_p = np.concatenate(all_aux_p_pos, axis=0)
            main = self._apply_noisy_or_blend(main, aux_p)
        return self._apply_affine_calibration(main)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict on ``df``. When ``self.tta_transforms`` is set, runs one
        pass per TTA op and averages -- this is test-time augmentation
        for variance reduction. Otherwise behaves identically to the
        original single-pass inference.
        """
        if not self.tta_transforms:
            return self._predict_with_transform(
                df, val_transform(self.image_size, rgb=self.rgb),
            )

        per_op_preds: list[np.ndarray] = []
        for op_name in self.tta_transforms:
            transform = val_transform_with_tta(self.image_size, self.rgb, op_name)
            per_op_preds.append(self._predict_with_transform(df, transform))

        return np.mean(np.stack(per_op_preds, axis=0), axis=0).astype(np.float32)

    @classmethod
    def from_checkpoint(cls, run_dir: Path, **kwargs) -> "ImageClassifierModel":
        with open(run_dir / "config.yaml") as f:
            config = yaml.safe_load(f)

        config.pop("model_class", None)
        config.pop("model_path", None)
        config.pop("run_name", None)
        wrapper = cls(**config)

        state_dict = torch.load(run_dir / "best.pt", map_location=wrapper.device)
        wrapper.model.load_state_dict(state_dict)

        # Restore noisy-OR blending lambdas fit during training so
        # `predict` reproduces the post-calibration scores. Silent when
        # the run never had noisy-OR enabled.
        lambdas_path = run_dir / "noisy_or_blend_lambdas.npy"
        if lambdas_path.exists() and wrapper.noisy_or_indices:
            lambdas = np.load(lambdas_path)
            if lambdas.shape != (len(wrapper.noisy_or_indices),):
                raise RuntimeError(
                    f"noisy_or_blend_lambdas.npy shape {lambdas.shape} != "
                    f"expected ({len(wrapper.noisy_or_indices)},)"
                )
            wrapper.noisy_or_lambdas = lambdas.astype(np.float64)

        # Per-class affine calibration from scripts/affine_calibrate.py; silent when absent.
        affine_path = run_dir / "affine_calibration.npy"
        if affine_path.exists():
            affine = np.load(affine_path)
            expected = (2, len(LABEL_NAMES))
            if affine.shape != expected:
                raise RuntimeError(
                    f"affine_calibration.npy shape {affine.shape} != "
                    f"expected {expected}"
                )
            wrapper.affine_calibration = affine.astype(np.float64)

        return wrapper
