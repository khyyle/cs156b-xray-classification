from __future__ import annotations

import torch.nn as nn
from torchvision import models  # type: ignore

from radiology_cls import NUM_CLASSES
from radiology_cls.models.base_image_classifier import ImageClassifierModel

# TODO: explore attention resnets via CBAM as extension to convnext-base/convnext-large

_SUPPORTED_ARCHITECTURES = {
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "convnext_base",
    "convnext_large",
    "densenet121",
}


class TorchvisionCNNModel(ImageClassifierModel):
    def __init__(
        self,
        architecture: str,
        pretrained: bool = True,
        dropout_prob: float = 0.0,
        **kwargs,
    ) -> None:
        if architecture not in _SUPPORTED_ARCHITECTURES:
            supported = ", ".join(sorted(_SUPPORTED_ARCHITECTURES))
            raise ValueError(f"Unsupported architecture '{architecture}'. Expected one of: {supported}")

        self.architecture = architecture
        self.pretrained = pretrained
        self.dropout_prob = dropout_prob
        super().__init__(rgb=True, **kwargs)

    @property
    def name(self) -> str:
        suffix = "pretrained" if self.pretrained else "scratch"
        return f"{self.architecture}-{suffix}"

    def build_model(self) -> nn.Module:
        if self.architecture == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if self.pretrained else None
            model = models.resnet18(weights=weights)
            model.fc = self._linear_head(model.fc.in_features)
            return model

        if self.architecture == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if self.pretrained else None
            model = models.resnet34(weights=weights)
            model.fc = self._linear_head(model.fc.in_features)
            return model

        if self.architecture == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if self.pretrained else None
            model = models.resnet50(weights=weights)
            model.fc = self._linear_head(model.fc.in_features)
            return model
        
        if self.architecture == "resnet101":
            weights = models.ResNet101_Weights.DEFAULT if self.pretrained else None
            model = models.resnet101(weights=weights)
            model.fc = self._linear_head(model.fc.in_features)
            return model
        
        if self.architecture == "resnet152":
            weights = models.ResNet152_Weights.DEFAULT if self.pretrained else None
            model = models.resnet152(weights=weights)
            model.fc = self._linear_head(model.fc.in_features)
            return model
        
        if self.architecture == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.DEFAULT if self.pretrained else None
            model = models.convnext_base(weights=weights)
            # ConvNeXt head is model.classifier[2]
            in_features = model.classifier[2].in_features
            model.classifier[2] = self._linear_head(in_features)
            return model

        if self.architecture == "convnext_large":
            weights = models.ConvNeXt_Large_Weights.DEFAULT if self.pretrained else None
            model = models.convnext_large(weights=weights)
            in_features = model.classifier[2].in_features
            model.classifier[2] = self._linear_head(in_features)
            return model

        if self.architecture == "densenet121":
            weights = models.DenseNet121_Weights.DEFAULT if self.pretrained else None
            model = models.densenet121(weights=weights)
            model.classifier = self._linear_head(model.classifier.in_features)
            return model

        raise AssertionError(f"Unhandled architecture: {self.architecture}")

    def _linear_head(self, in_features: int) -> nn.Module:
        out_features = NUM_CLASSES * self.num_classes_per_label
        if self.dropout_prob <= 0:
            return nn.Linear(in_features, out_features)
        return nn.Sequential(
            nn.Dropout(self.dropout_prob),
            nn.Linear(in_features, out_features),
        )
