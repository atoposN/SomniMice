from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn

from .baseline import BaselineModelConfig, SequenceEncoder


class FeatureEncoder(nn.Module):
    def __init__(self, hidden_dims: Sequence[int] = (256, 128), dropout: float = 0.2):
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers.append(nn.LazyLinear(dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class SpectrogramEncoder(nn.Module):
    def __init__(self, base_channels: int = 16, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LazyConv2d(base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.LazyConv2d(base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.LazyLinear(base_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
        x = self.net(spec)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)


class FusionClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int], num_classes: int):
        super().__init__()
        layers = []
        in_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            in_dim = dim
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class MultimodalConfig(BaselineModelConfig):
    feature_dims: Sequence[int] = (256, 128)
    spec_base_channels: int = 16
    fusion_hidden_dims: Sequence[int] = (256, 128)


class RawFeatureFusionModel(nn.Module):
    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(
            input_channels=cfg.input_channels,
            base_filters=cfg.base_filters,
            kernel_size=cfg.kernel_size,
            residual_blocks=cfg.residual_blocks,
            reduction_ratio=cfg.reduction_ratio,
            lstm_hidden_size=cfg.lstm_hidden_size,
            lstm_layers=cfg.lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            bidirectional=cfg.bidirectional,
        )
        self.feature_encoder = FeatureEncoder(cfg.feature_dims)
        fused_dim = self.sequence_encoder.output_dim + cfg.feature_dims[-1]
        self.classifier = FusionClassifier(fused_dim, cfg.fusion_hidden_dims, cfg.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_repr = self.sequence_encoder(batch["sequence"])
        feat_repr = self.feature_encoder(batch["handcrafted"])
        fused = torch.cat([seq_repr, feat_repr], dim=-1)
        return self.classifier(fused)


class RawSpectrogramFusionModel(nn.Module):
    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(
            input_channels=cfg.input_channels,
            base_filters=cfg.base_filters,
            kernel_size=cfg.kernel_size,
            residual_blocks=cfg.residual_blocks,
            reduction_ratio=cfg.reduction_ratio,
            lstm_hidden_size=cfg.lstm_hidden_size,
            lstm_layers=cfg.lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            bidirectional=cfg.bidirectional,
        )
        self.spectrogram_encoder = SpectrogramEncoder(cfg.spec_base_channels)
        fused_dim = self.sequence_encoder.output_dim + cfg.spec_base_channels * 4
        self.classifier = FusionClassifier(fused_dim, cfg.fusion_hidden_dims, cfg.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_repr = self.sequence_encoder(batch["sequence"])
        spec_repr = self.spectrogram_encoder(batch["stft"])
        fused = torch.cat([seq_repr, spec_repr], dim=-1)
        return self.classifier(fused)


class FullMultimodalModel(nn.Module):
    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.sequence_encoder = SequenceEncoder(
            input_channels=cfg.input_channels,
            base_filters=cfg.base_filters,
            kernel_size=cfg.kernel_size,
            residual_blocks=cfg.residual_blocks,
            reduction_ratio=cfg.reduction_ratio,
            lstm_hidden_size=cfg.lstm_hidden_size,
            lstm_layers=cfg.lstm_layers,
            lstm_dropout=cfg.lstm_dropout,
            bidirectional=cfg.bidirectional,
        )
        self.feature_encoder = FeatureEncoder(cfg.feature_dims)
        self.spectrogram_encoder = SpectrogramEncoder(cfg.spec_base_channels)
        fused_dim = (
            self.sequence_encoder.output_dim
            + cfg.feature_dims[-1]
            + cfg.spec_base_channels * 4
        )
        self.classifier = FusionClassifier(fused_dim, cfg.fusion_hidden_dims, cfg.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        seq_repr = self.sequence_encoder(batch["sequence"])
        feat_repr = self.feature_encoder(batch["handcrafted"])
        spec_repr = self.spectrogram_encoder(batch["stft"])
        fused = torch.cat([seq_repr, feat_repr, spec_repr], dim=-1)
        return self.classifier(fused)


class FeatureOnlyModel(nn.Module):
    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.feature_encoder = FeatureEncoder(cfg.feature_dims)
        self.classifier = FusionClassifier(cfg.feature_dims[-1], cfg.fusion_hidden_dims, cfg.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        feat_repr = self.feature_encoder(batch["handcrafted"])
        return self.classifier(feat_repr)


class SpectrogramOnlyModel(nn.Module):
    def __init__(self, cfg: MultimodalConfig):
        super().__init__()
        self.spectrogram_encoder = SpectrogramEncoder(cfg.spec_base_channels)
        self.classifier = FusionClassifier(cfg.spec_base_channels * 4, cfg.fusion_hidden_dims, cfg.num_classes)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if "stft" not in batch:
            raise KeyError("SpectrogramOnlyModel expects 'stft' modality in the batch.")
        spec_repr = self.spectrogram_encoder(batch["stft"])
        return self.classifier(spec_repr)
