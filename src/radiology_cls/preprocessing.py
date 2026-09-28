"""
Transforms, normalization, augmentation, and feature extraction. Anything that
changes raw data into model-ready inputs should go here. Dataset loading and
path resolution go in data.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2  # type: ignore

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Placeholder until compute_dataset_stats.py is run on the HPC.
GRAYSCALE_MEAN = (0.503320,)
GRAYSCALE_STD = (0.291820,)


## Image transforms

def train_transform(
    image_size: int,
    rgb: bool = False,
    strong_aug: bool = False,
) -> Callable:
    """
    Generates a transform with augmentation for regularization during
    training.

    Parameters:
    -----------
    image_size: int
        Target height and width in pixels.
    rgb: bool
        If True, normalize with ImageNet stats (3 channels). If False,
        normalize with grayscale stats (1 channel).
    strong_aug: bool
        If True, layer in chest-X-ray-appropriate stronger augmentations
        on top of the default crop/flip/rotate: brightness/contrast jitter
        (X-ray exposure varies between scans) and ``RandomErasing`` (p=0.25,
        on the normalized tensor). Saturation/hue jitter is intentionally
        omitted because chest X-rays are effectively grayscale -- introducing
        chromatic noise has no medical analog and tends to hurt. Off by
        default to preserve recipe parity with already-trained checkpoints.

    Returns:
    --------
    transform: Callable
        A torchvision v2 Compose pipeline.
    """
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if rgb else (GRAYSCALE_MEAN, GRAYSCALE_STD)

    ops: list = [
        v2.ToImage(),
        v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(degrees=10),
    ]
    if strong_aug:
        ops.append(v2.ColorJitter(brightness=0.2, contrast=0.2))
    ops.extend([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    if strong_aug:
        # RandomErasing operates on tensors after Normalize. Default magnitudes;
        # value="random" fills with random noise rather than a constant, which
        # tends to be slightly stronger regularization on small datasets.
        ops.append(v2.RandomErasing(p=0.25, value="random"))

    return v2.Compose(ops)


def val_transform(image_size: int, rgb: bool = False) -> Callable:
    """
    Generates a transform with deterministic resize and no augmentation.

    Parameters:
    -----------
    image_size: int
        Target height and width in pixels.
    rgb: bool
        If True, normalize with ImageNet stats (3 channels). If False,
        normalize with grayscale stats (1 channel).

    Returns:
    --------
    transform: Callable
        A torchvision v2 Compose pipeline.
    """
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if rgb else (GRAYSCALE_MEAN, GRAYSCALE_STD)
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(image_size, image_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


## Test-time augmentation (TTA)
#
# At inference, run the model on multiple deterministic augmented views
# of each image and average the predictions. Each TTA op is just a
# torchvision v2 transform inserted between Resize and Normalize in the
# usual val pipeline; "identity" means no augmentation (= val_transform).
#
# Adding a new TTA op = add an entry to TTA_OPS. Configs reference ops
# by name, so YAML stays clean.

TTA_OPS: dict[str, Callable | None] = {
    "identity": None,
    "hflip": v2.RandomHorizontalFlip(p=1.0),
    # Small rotations: chest X-rays have real positioning variance of a few
    # degrees due to patient stance; +-3 through +-10 stay within that
    # envelope so the augmented views are still anatomically plausible.
    # Bigger angles (e.g. +-30) push beyond clinical realism and start
    # hurting. We sweep 3/5/7/10 so the 10-pass TTA ensemble can test
    # whether rotation density helps beyond 5+10 alone.
    "rot+3": v2.RandomRotation(degrees=(3, 3)),
    "rot-3": v2.RandomRotation(degrees=(-3, -3)),
    "rot+5": v2.RandomRotation(degrees=(5, 5)),
    "rot-5": v2.RandomRotation(degrees=(-5, -5)),
    "rot+7": v2.RandomRotation(degrees=(7, 7)),
    "rot-7": v2.RandomRotation(degrees=(-7, -7)),
    "rot+10": v2.RandomRotation(degrees=(10, 10)),
    "rot-10": v2.RandomRotation(degrees=(-10, -10)),
}


def val_transform_with_tta(
    image_size: int,
    rgb: bool = False,
    tta_op_name: str | None = None,
) -> Callable:
    """
    Variant of ``val_transform`` that inserts a deterministic TTA
    augmentation between resize and normalize.

    Parameters:
    -----------
    image_size: int
        Target height and width in pixels.
    rgb: bool
        If True, normalize with ImageNet stats; else grayscale stats.
    tta_op_name: str | None
        Key into ``TTA_OPS``. ``None`` or ``"identity"`` returns a
        pipeline equivalent to ``val_transform``.

    Returns:
    --------
    transform: Callable
        A torchvision v2 Compose pipeline.
    """
    if tta_op_name is None or tta_op_name == "identity":
        return val_transform(image_size, rgb=rgb)
    if tta_op_name not in TTA_OPS:
        raise ValueError(
            f"Unknown TTA op '{tta_op_name}'. Available: {sorted(TTA_OPS)}"
        )
    op = TTA_OPS[tta_op_name]
    if op is None:
        return val_transform(image_size, rgb=rgb)

    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if rgb else (GRAYSCALE_MEAN, GRAYSCALE_STD)
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(image_size, image_size), antialias=True),
        op,
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])


## Label encoding

def encode_labels(
    df: pd.DataFrame,
    label_names: list[str],
    uncertain_as: float = 1.0,
) -> np.ndarray:
    """
    Convert CheXpert's 4-state CSV labels into a float32 array for BCE.

    The raw CSV encodes:
        1.0   positive finding
        0.0   UNCERTAIN
       -1.0   negative finding
        NaN   blank / unmentioned

    Positives become 1, negatives and blanks become 0. The uncertain_as
    parameter controls what happens to the uncertain entries (0.0 in the
    CSV). Common choices:
        1.0   treat uncertain as positive
        0.0   treat uncertain as negative
        NaN   mask out of loss/metrics

    Parameters:
    -----------
    df: pd.DataFrame
        Dataframe containing the label columns.
    label_names: list[str]
        Which columns to extract, in order.
    uncertain_as: float, default = 1.0
        Value to assign to uncertain entries.

    Returns:
    --------
    labels: np.ndarray
        Shape (N, len(label_names)), dtype float32.
    """
    raw = df[label_names].to_numpy(dtype=np.float32)

    out = np.zeros_like(raw, dtype=np.float32)
    out[raw == 1.0] = 1.0
    out[raw == 0.0] = uncertain_as

    return out


def encode_regression_labels(
    df: pd.DataFrame,
    label_names: list[str],
) -> np.ndarray:
    """
    Return CheXpert labels on original scale:
        1.0   positive finding
        0.0   uncertain
       -1.0   negative finding
        NaN   blank / unmentioned, masked out of loss and metrics

    Parameters:
    -----------
    df: pd.Dataframe
        The dataframe containing labels to cast to numpy
    label_names: list[str]
        The labels to cast to numpy
    
    Returns:
    --------
    df: pd.Dataframe
        Dataframe with casted labels
    """
    return df[label_names].to_numpy(dtype=np.float32)


## Loss helpers

def masked_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Binary cross-entropy for labels in [0, 1].
    For training with uncertain labels as 1s, regular 
    F.binary_cross_entropy_with_logits works fine.

    Parameters:
    -----------
    logits: torch.Tensor
        Raw model output, shape (B, num_labels).
    targets: torch.Tensor
        Labels with possible NaN entries, shape (B, num_labels).

    Returns:
    --------
    loss: torch.Tensor
        Scalar mean loss over non-NaN entries.
    """
    mask = ~torch.isnan(targets)
    safe_targets = targets.clone()
    safe_targets[~mask] = 0.0
    per_element = F.binary_cross_entropy_with_logits(logits, safe_targets, reduction="none")
    return (per_element * mask).sum() / mask.sum().clamp(min=1)


class MaskedRegressionLoss(nn.Module):
    """
    Regression loss for raw CheXpert labels in [-1, 1], masking NaNs.

    Parameters:
    -----------
    loss_type: str
        One of "smooth_l1" or "mse".
    """

    def __init__(self, loss_type: str = "smooth_l1") -> None:
        super().__init__()
        if loss_type == "smooth_l1":
            self.loss = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.loss = nn.MSELoss(reduction="none")
        else:
            raise ValueError(f"Unsupported regression loss_type: {loss_type}")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute masked regression loss on tanh-transformed logits.

        Parameters:
        -----------
        logits: torch.Tensor
            Raw model output, shape (B, num_labels).
        targets: torch.Tensor
            Raw CheXpert labels with possible NaN entries, shape
            (B, num_labels).

        Returns:
        --------
        loss: torch.Tensor
            Scalar mean loss over non-NaN entries.
        """
        mask = ~torch.isnan(targets)
        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0
        preds = torch.tanh(logits)
        per_element = self.loss(preds, safe_targets)
        return (per_element * mask).sum() / mask.sum().clamp(min=1)


def asymmetric_binary_loss(
    p: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma_pos: float | torch.Tensor = 0.0,
    gamma_neg: float | torch.Tensor = 4.0,
    clip: float | torch.Tensor = 0.05,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Asymmetric binary focal loss (Ridnik et al. 2021, "Asymmetric Loss For
    Multi-Label Classification") with stronger focusing on easy negatives
    than easy positives. Designed for long-tailed multi-label problems
    where abundant easy-negative gradients drown out rare positives.

    For a target ``t in {0, 1}`` and predicted probability ``p``,

        loss_pos = -t       * (1 - p)^gamma_pos        * log(p)
        loss_neg = -(1 - t) * (clamp(p - clip, 0))^gamma_neg * log(1 - p_neg)

    With ``gamma_neg > gamma_pos`` and a small negative-margin ``clip``, the
    contribution of confident negatives collapses while positives keep
    their gradient.

    Parameters:
    -----------
    p: torch.Tensor
        Positive-class probability in ``[0, 1]``, any shape.
    target: torch.Tensor
        Binary target in ``{0, 1}``, same shape as ``p``. Uncertain or
        invalid labels must be filtered via ``mask`` rather than encoded.
    mask: torch.Tensor
        Boolean tensor, same shape as ``p``; ``True`` where the entry is
        valid and should contribute to the mean.
    gamma_pos, gamma_neg, clip: float or torch.Tensor
        ASL focal exponents and negative margin. Pass scalars for uniform
        ASL across all entries, or per-class tensors broadcastable to ``p``
        (typically shape ``(C,)`` for a ``(B, C)`` input) to assign
        different ASL directions to different classes in a single loss call.
    eps: float
        Numerical floor used to keep ``log(p)`` and ``log(1 - p)`` finite.

    Returns:
    --------
    loss: torch.Tensor
        Scalar mean loss over ``mask=True`` entries. When ``mask`` is
        entirely ``False``, returns ``p.sum() * 0`` so autograd stays
        connected without contributing a gradient.
    """
    p = p.clamp(eps, 1.0 - eps)
    p_neg = (p - clip).clamp(min=eps, max=1.0 - eps)
    pos_term = -target * ((1.0 - p) ** gamma_pos) * torch.log(p)
    neg_term = -(1.0 - target) * (p_neg ** gamma_neg) * torch.log(1.0 - p_neg)
    loss = pos_term + neg_term
    if not mask.any():
        return p.sum() * 0.0
    return loss[mask].mean()


def noisy_or_positive_prob(subtype_logits: torch.Tensor) -> torch.Tensor:
    """
    Combine ``K`` subtype detector logits via noisy-OR.

        p_pos = 1 - prod_j (1 - sigmoid(a_j))

    each detector ``a_j`` independently votes on whether
    its latent subtype is present; the class is positive if at least one
    subtype fires. This helps for heterogenous labels where the visual signature
    is the union of several distinct subtypes, which avoids forcing a awkward or noisy
    decision boundary.

    Parameters:
    -----------
    subtype_logits: torch.Tensor
        Shape `(..., K)` raw logits from the `K` subtype detectors.

    Returns:
    --------
    p_pos: torch.Tensor
        Shape `(...)` positive probability in `[0, 1]`.
    """
    p_subtypes = torch.sigmoid(subtype_logits)
    # 1 - prod is numerically stable enough for K <= 16 in float32.
    return 1.0 - (1.0 - p_subtypes).prod(dim=-1)


class Masked3ClassLoss(nn.Module):
    """
    Per-pathology 3-way cross-entropy over the CheXpert label states with
    optional auxiliary losses for long-tail / heterogeneous classes.

    Primary loss:
        Raw labels in `{-1, 0, +1}` are mapped to classes `{0, 1, 2}`
        (negative, uncertain, positive), masking NaNs. This treats
        ``uncertain`` as its own class instead of forcing it to a
        regression midpoint.

    Optional auxiliary terms (both default to disabled):
        1. **Asymmetric loss on hard classes.** For each class in
           `asl_hard_indices`, an auxiliary binary ASL is computed
           against the softmax positive probability `p^+` with target
           `1[y=+1]`. Labels with `y=0` are masked out of the binary
           target so they keep contributing to CE3 only.

        2. **Noisy-OR auxiliary branch on heterogeneous classes.** When
           the caller passes `aux_subtype_logits` of shape
           `(B, |noisy_or_indices|, K)`, the K subtype detector logits
           per class are combined via noisy-OR into `p^+_{NO}` and an
           ASL is computed against the same binary target.

    Parameters:
    -----------
    asl_alpha: float
        Weight on the hard-class ASL term. `0.0`` disables.
    asl_hard_indices: list[int] | None
        Label indices (against ``LABEL_NAMES``) to apply ASL to. Empty
        list or None disables the ASL term regardless of ``asl_alpha``.
    asl_gamma_pos, asl_gamma_neg, asl_clip: float
        ASL hyperparameters (see ``asymmetric_binary_loss``).
    noisy_or_beta: float
        Weight on the noisy-OR ASL term. ``0.0`` disables.
    noisy_or_indices: list[int] | None
        Label indices that have a noisy-OR aux branch on the model.
        Order must match the second axis of ``aux_subtype_logits``.
    noisy_or_gamma_pos, noisy_or_gamma_neg, noisy_or_clip: float
        ASL hyperparameters for the noisy-OR positive term.
    """

    def __init__(
        self,
        asl_alpha: float = 0.0,
        asl_hard_indices: list[int] | None = None,
        asl_gamma_pos: float = 0.0,
        asl_gamma_neg: float = 4.0,
        asl_clip: float = 0.05,
        # Per-class ASL: dict mapping class index -> {"gamma_pos", "gamma_neg",
        # "clip"}. When provided, overrides the uniform asl_hard_indices +
        # scalar gammas above. Lets the same loss apply opposite ASL
        # directions to different classes in one pass.
        asl_per_class: dict[int, dict] | None = None,
        noisy_or_beta: float = 0.0,
        noisy_or_indices: list[int] | None = None,
        noisy_or_gamma_pos: float = 0.0,
        noisy_or_gamma_neg: float = 4.0,
        noisy_or_clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.asl_alpha = float(asl_alpha)

        if asl_per_class:
            # Sort keys for deterministic ordering of the (|H|,) gamma tensors.
            indices = sorted(asl_per_class.keys())
            gp = [float(asl_per_class[i].get("gamma_pos", 0.0)) for i in indices]
            gn = [float(asl_per_class[i].get("gamma_neg", 4.0)) for i in indices]
            cl = [float(asl_per_class[i].get("clip", 0.05)) for i in indices]
            self.asl_hard_indices = indices
            gamma_pos_t = torch.tensor(gp, dtype=torch.float32)
            gamma_neg_t = torch.tensor(gn, dtype=torch.float32)
            clip_t = torch.tensor(cl, dtype=torch.float32)
        else:
            self.asl_hard_indices = list(asl_hard_indices) if asl_hard_indices else []
            n = len(self.asl_hard_indices)
            gamma_pos_t = torch.full((n,), float(asl_gamma_pos), dtype=torch.float32)
            gamma_neg_t = torch.full((n,), float(asl_gamma_neg), dtype=torch.float32)
            clip_t = torch.full((n,), float(asl_clip), dtype=torch.float32)

        # Buffers so they migrate with .to(device) but aren't trainable.
        self.register_buffer("_asl_gamma_pos", gamma_pos_t)
        self.register_buffer("_asl_gamma_neg", gamma_neg_t)
        self.register_buffer("_asl_clip", clip_t)

        self.noisy_or_beta = float(noisy_or_beta)
        self.noisy_or_indices = list(noisy_or_indices) if noisy_or_indices else []
        self.noisy_or_gamma_pos = float(noisy_or_gamma_pos)
        self.noisy_or_gamma_neg = float(noisy_or_gamma_neg)
        self.noisy_or_clip = float(noisy_or_clip)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        aux_subtype_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute masked 3-way cross-entropy plus optional auxiliary terms.

        Parameters:
        -----------
        logits: torch.Tensor
            Raw model output, shape ``(B, num_labels, 3)`` or
            ``(B, num_labels * 3)``.
        targets: torch.Tensor
            Raw CheXpert labels in ``{-1, 0, +1, NaN}``, shape
            ``(B, num_labels)``.
        aux_subtype_logits: torch.Tensor | None
            Optional auxiliary subtype detector logits with shape
            ``(B, |noisy_or_indices|, K)``. Ignored when
            ``noisy_or_beta == 0`` or ``noisy_or_indices`` is empty.

        Returns:
        --------
        loss: torch.Tensor
            Scalar weighted sum of CE3 plus any active auxiliary terms.
            When all label entries in the batch are NaN, returns
            ``0 * logits.sum()`` so the autograd graph stays connected
            without contributing.
        """
        if logits.ndim == 2:
            batch_size, _ = logits.shape
            num_labels = targets.shape[1]
            logits = logits.view(batch_size, num_labels, 3)
        elif logits.ndim != 3 or logits.shape[-1] != 3:
            raise ValueError(
                f"Expected logits shape (B, L, 3) or (B, L*3); got {tuple(logits.shape)}"
            )

        mask = ~torch.isnan(targets)
        if not mask.any():
            # No labels in this batch contribute; keep graph connected.
            return logits.sum() * 0.0

        safe_targets = targets.clone()
        safe_targets[~mask] = 0.0
        classes = (safe_targets + 1.0).long()  # -1 -> 0, 0 -> 1, +1 -> 2

        flat_logits = logits.reshape(-1, 3)
        flat_classes = classes.reshape(-1)
        flat_mask = mask.reshape(-1)

        ce_loss = F.cross_entropy(
            flat_logits[flat_mask],
            flat_classes[flat_mask],
            reduction="mean",
        )
        total = ce_loss

        if self.asl_alpha > 0.0 and self.asl_hard_indices:
            # Positive softmax probability from main head.
            probs = F.softmax(logits, dim=-1)
            p_pos = probs[..., 2]  # (B, L)
            hard_idx = torch.as_tensor(
                self.asl_hard_indices, dtype=torch.long, device=logits.device,
            )
            p_hard = p_pos.index_select(1, hard_idx)        # (B, |H|)
            t_hard = targets.index_select(1, hard_idx)      # (B, |H|), raw {-1,0,+1,nan}
            asl_mask = ~torch.isnan(t_hard) & (t_hard != 0.0)
            t_binary = (t_hard == 1.0).float()
            asl = asymmetric_binary_loss(
                p_hard, t_binary, asl_mask,
                gamma_pos=self._asl_gamma_pos,
                gamma_neg=self._asl_gamma_neg,
                clip=self._asl_clip,
            )
            total = total + self.asl_alpha * asl

        if (
            self.noisy_or_beta > 0.0
            and self.noisy_or_indices
            and aux_subtype_logits is not None
        ):
            if aux_subtype_logits.shape[1] != len(self.noisy_or_indices):
                raise ValueError(
                    f"aux_subtype_logits has {aux_subtype_logits.shape[1]} aux "
                    f"classes but noisy_or_indices has {len(self.noisy_or_indices)}"
                )
            p_no = noisy_or_positive_prob(aux_subtype_logits)  # (B, num_aux)
            no_idx = torch.as_tensor(
                self.noisy_or_indices, dtype=torch.long, device=logits.device,
            )
            t_no = targets.index_select(1, no_idx)             # (B, num_aux)
            no_mask = ~torch.isnan(t_no) & (t_no != 0.0)
            t_binary = (t_no == 1.0).float()
            no_aux = asymmetric_binary_loss(
                p_no, t_binary, no_mask,
                gamma_pos=self.noisy_or_gamma_pos,
                gamma_neg=self.noisy_or_gamma_neg,
                clip=self.noisy_or_clip,
            )
            total = total + self.noisy_or_beta * no_aux

        return total


def threeclass_logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert 3-class logits to the scalar `p_pos - p_neg` score in
    `[-1, 1]` used for the leaderboard MSE evaluation.

    Parameters:
    -----------
    logits: torch.Tensor
        Either `(B, num_labels, 3)` or `(B, num_labels * 3)`. The
        last dim is interpreted as `[neg, uncertain, pos]`.

    Returns:
    --------
    score: torch.Tensor
        Shape `(B, num_labels)`, values in `[-1, 1]`. This is the
        expected value of the categorical posterior under the
        `{-1, 0, +1}` encoding and is MSE-optimal under the held-out
        posterior.
    """
    if logits.ndim == 2:
        batch_size, flat = logits.shape
        if flat % 3 != 0:
            raise ValueError(f"Cannot reshape {flat} into (num_labels, 3)")
        logits = logits.view(batch_size, flat // 3, 3)
    elif logits.ndim != 3 or logits.shape[-1] != 3:
        raise ValueError(
            f"Expected logits shape (B, L, 3) or (B, L*3); got {tuple(logits.shape)}"
        )
    probs = F.softmax(logits, dim=-1)
    return probs[..., 2] - probs[..., 0]
