from __future__ import annotations

import timm  # type: ignore
import torch
import torch.nn as nn

from radiology_cls import NUM_CLASSES
from radiology_cls.models.base_image_classifier import ImageClassifierModel


class TimmBackboneWithAux(nn.Module):
    """
    Wrap a timm classifier model to optionally emit auxiliary subtype
    detector logits alongside the main classification head's logits.

    Forward returns ``(main_logits, aux_subtype_logits)`` where
    ``aux_subtype_logits`` has shape ``(B, num_aux_classes, K)`` when
    ``aux_indices`` is non-empty and ``None`` otherwise.

    The wrapper routes through timm's standard
    ``forward_features`` + ``forward_head(pre_logits=True)`` API to
    expose pooled features, then applies both the original classifier and
    ``K`` parallel ``nn.Linear`` detectors per auxiliary class. This works
    uniformly across ViT, Swin, and ConvNeXt families.

    Parameters:
    -----------
    backbone: nn.Module
        A timm model created with the desired ``num_classes`` for the
        main head. Its ``get_classifier()`` is reused as-is.
    aux_indices: list[int]
        Label indices (against ``LABEL_NAMES``) that should receive a
        noisy-OR auxiliary branch. Order is preserved in the second axis
        of ``aux_subtype_logits``. Pass ``[]`` to disable the aux path
        (forward then returns ``(main_logits, None)``).
    aux_K: int
        Number of latent subtype detectors per auxiliary class. Each
        detector is a single linear projection ``in_features -> 1``.
    """

    def __init__(self, backbone: nn.Module, aux_indices: list[int], aux_K: int = 4):
        super().__init__()
        self.backbone = backbone
        self.aux_indices: list[int] = list(aux_indices)
        self.aux_K = int(aux_K)
        if self.aux_indices:
            classifier = backbone.get_classifier()
            in_features = getattr(classifier, "in_features", None)
            if in_features is None:
                raise RuntimeError(
                    "Could not determine pooled feature dim from "
                    f"backbone.get_classifier()={type(classifier).__name__}"
                )
            self.aux_detectors = nn.ModuleList(
                [nn.Linear(in_features, self.aux_K) for _ in self.aux_indices]
            )
        else:
            self.aux_detectors = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.backbone.forward_features(x)
        # pre_logits=True returns the pooled feature vector (B, D) before
        # the timm head; this is the same tensor the head's Linear would
        # receive in the unwrapped forward.
        pooled = self.backbone.forward_head(features, pre_logits=True)
        main_logits = self.backbone.get_classifier()(pooled)

        if self.aux_detectors is None:
            return main_logits, None

        # Each detector emits (B, K); stack into (B, num_aux, K).
        aux_each = [det(pooled) for det in self.aux_detectors]
        aux_subtype_logits = torch.stack(aux_each, dim=1)
        return main_logits, aux_subtype_logits

# Architectures available via timm. Mirrors the torchvision_cnn.py pattern but
# delegates head replacement and weight loading to timm.create_model. All
# entries currently assume ImageNet-style normalization (handled via rgb=True
# in the parent class).
_SUPPORTED_ARCHITECTURES = {
    "convnextv2_base",
    "convnextv2_large",
    # Swin-V2 (windowed-attention transformer, IN-22k -> IN-1k pretraining).
    # The 192to256 tag means the checkpoint was trained at 192 then fine-tuned
    # to support 256, so it loads cleanly at our pipeline's image_size=256.
    "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k",
    "swinv2_large_window12to16_192to256.ms_in22k_ft_in1k",
    # DINOv2 with registers (self-supervised ViT, LVD-142M natural images,
    # no medical data). The reg4 variants add 4 learned register tokens that
    # absorb attention-map artifacts; strictly better than vanilla DINOv2 on
    # downstream tasks at no inference cost (Darcet et al. 2024,
    # "Vision Transformers Need Registers"). Native input size is 224
    # (16x16 patches at patch14), but we pass img_size to timm so they
    # interpolate position embeddings cleanly to whatever our pipeline uses
    # (typically 256 to reuse the existing image cache).
    "vit_base_patch14_reg4_dinov2.lvd142m",
    "vit_large_patch14_reg4_dinov2.lvd142m",
    "vit_giant_patch14_reg4_dinov2.lvd142m",
    # DINOv3 successor (LVD-1689M, 10x larger pretraining set, patch16). New
    # architectural family for ensemble diversity. patch16 means image_size
    # should be a multiple of 16 (we use 512 -> 1024 patches, matching the
    # existing 512 image cache).
    "vit_base_patch16_dinov3.lvd1689m",
    # SigLIP 2 base @ 512 (vision-language pretraining on WebLI). Different
    # objective (contrastive sigmoid loss against text) so error structure
    # should be meaningfully decorrelated from the DINO/Swin/ConvNeXt
    # supervised + self-supervised families. patch16 + 512 = 32x32 patches.
    "vit_base_patch16_siglip_512.v2_webli",
    "vit_base_patch16_siglip_384.v2_webli",
    # EVA-02 Large @ 448 (M38M-pretrained, IN22k+IN1k fine-tuned). Largest
    # capacity tier we'd add. patch14 -- can run at 518 via position-embed
    # interpolation to reuse the existing 518 image cache.
    "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    # OmniRad-base: ViT-B/14 pretrained on RadImageNet (~1.2M radiology
    # images) via a modified DINOv2. Loaded straight from HF Hub through
    # timm's hf_hub: prefix. Medical-domain pretraining is the diversity
    # axis ImageNet/LVD models can't cover.
    "hf_hub:Snarcy/OmniRad-base",
    # Swin-V2 large fine-tuned for 384 (`192to384` tag). Higher input
    # resolution -> better small-structure pathology localization, at the
    # cost of needing a 384 image cache and ~2x training time vs 256.
    "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
    # Swin-V2 base counterpart at 384 -- same recipe, half the params.
    # Useful for ensemble diversity (different capacity tier at the same
    # resolution) without giving up native window alignment.
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
}


class TimmCNNModel(ImageClassifierModel):
    def __init__(
        self,
        architecture: str,
        pretrained: bool = True,
        dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs,
    ) -> None:
        if architecture not in _SUPPORTED_ARCHITECTURES:
            supported = ", ".join(sorted(_SUPPORTED_ARCHITECTURES))
            raise ValueError(f"Unsupported architecture '{architecture}'. Expected one of: {supported}")

        self.architecture = architecture
        self.pretrained = pretrained
        self.dropout_prob = dropout_prob
        self.drop_path_rate = drop_path_rate
        super().__init__(rgb=True, **kwargs)

    @property
    def name(self) -> str:
        suffix = "pretrained" if self.pretrained else "scratch"
        return f"{self.architecture}-{suffix}"

    def build_model(self) -> nn.Module:
        # ViT/Swin position embeddings are size-locked at create time.
        # Pass img_size so timm can interpolate them to match our pipeline's
        # image_size when it differs from the model's native resolution
        # (e.g. DINOv2 native 224 used at 256 to reuse the 256 cache, or
        # Swin-V2-384 used at 384). Pure CNNs (ConvNeXt) ignore img_size,
        # so we only forward it when the architecture name signals ViT/Swin.
        # ViT/Swin/EVA position embeddings are size-locked at create time.
        # Pass img_size so timm interpolates them when our pipeline's image_size
        # differs from the model's native resolution. Pure CNNs (ConvNeXt)
        # ignore img_size. hf_hub: prefix is treated as a ViT because our
        # only such entry (OmniRad) is one; revisit if we add a CNN via hf_hub.
        extra: dict = {}
        if (
            self.architecture.startswith("vit_")
            or "swin" in self.architecture
            or self.architecture.startswith("eva")
            or self.architecture.startswith("hf_hub:")
        ):
            extra["img_size"] = self.image_size

        backbone = timm.create_model(
            self.architecture,
            pretrained=self.pretrained,
            num_classes=NUM_CLASSES * self.num_classes_per_label,
            drop_rate=self.dropout_prob,
            drop_path_rate=self.drop_path_rate,
            **extra,
        )

        # When noisy-OR is configured (base class resolves class names to
        # indices on self.noisy_or_indices), wrap the backbone so forward
        # also emits subtype detector logits per aux class. The wrapper
        # routes through forward_features/forward_head(pre_logits=True),
        # which is uniform across ViT/Swin/ConvNeXt.
        aux_indices = list(getattr(self, "noisy_or_indices", []) or [])
        if aux_indices:
            return TimmBackboneWithAux(
                backbone, aux_indices=aux_indices, aux_K=int(self.noisy_or_K),
            )
        return backbone
