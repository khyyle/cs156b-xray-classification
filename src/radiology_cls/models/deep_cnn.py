from __future__ import annotations

import torch
import torch.nn as nn

from radiology_cls import NUM_CLASSES
from radiology_cls.models.base_image_classifier import ImageClassifierModel


def _conv_block(
    in_channels: int,
    out_channels: int,
    num_convs: int,
    dropout_prob: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_channels = in_channels
    for _ in range(num_convs):
        layers.extend([
            nn.Conv2d(current_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ])
        current_channels = out_channels

    layers.extend([
        nn.MaxPool2d(2),
        nn.Dropout2d(dropout_prob),
    ])
    return nn.Sequential(*layers)


class DeepCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = NUM_CLASSES,
        dropout_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32, num_convs=2, dropout_prob=dropout_prob),
            _conv_block(32, 64, num_convs=2, dropout_prob=dropout_prob),
            _conv_block(64, 128, num_convs=3, dropout_prob=dropout_prob),
            _conv_block(128, 256, num_convs=3, dropout_prob=dropout_prob),
            _conv_block(256, 512, num_convs=3, dropout_prob=dropout_prob),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


class DeepCNNModel(ImageClassifierModel):
    @property
    def name(self) -> str:
        return "deep-cnn"

    def __init__(
        self,
        dropout_prob: float = 0.2,
        **kwargs,
    ) -> None:
        self.dropout_prob = dropout_prob
        super().__init__(rgb=False, **kwargs)

    def build_model(self) -> nn.Module:
        return DeepCNN(in_channels=1, dropout_prob=self.dropout_prob)
