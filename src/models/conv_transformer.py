from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFeatureExtractor(nn.Module):
    def __init__(self, in_channels: int, base_filters: int, kernel_size: int, conv_blocks: int, embed_dim: int):
        super().__init__()
        padding = kernel_size // 2
        layers = []
        current_channels = in_channels
        filters = base_filters
        for _ in range(conv_blocks):
            layers.append(nn.Conv1d(current_channels, filters, kernel_size, padding=padding))
            layers.append(nn.BatchNorm1d(filters))
            layers.append(nn.ReLU(inplace=True))
            current_channels = filters
            filters *= 2
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(current_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        pooled = self.pool(features).squeeze(-1)
        return F.relu(self.proj(pooled))


@dataclass
class ConvTransformerConfig:
    input_channels: int = 2
    base_filters: int = 16
    kernel_size: int = 7
    conv_blocks: int = 2
    embed_dim: int = 128
    num_heads: int = 4
    ff_multiplier: float = 4.0
    num_layers: int = 2
    dropout: float = 0.1
    sequence_length: int = 11
    classifier_hidden_dims: tuple[int, ...] = (128,)
    num_classes: int = 3


class ConvTransformerSequenceModel(nn.Module):
    """1D convolutional feature extractor followed by a Transformer encoder."""

    def __init__(self, cfg: ConvTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_extractor = ConvFeatureExtractor(
            in_channels=cfg.input_channels,
            base_filters=cfg.base_filters,
            kernel_size=cfg.kernel_size,
            conv_blocks=cfg.conv_blocks,
            embed_dim=cfg.embed_dim,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=int(cfg.embed_dim * cfg.ff_multiplier),
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.positional_embedding = nn.Parameter(torch.zeros(1, cfg.sequence_length, cfg.embed_dim))

        classifier_layers = []
        in_dim = cfg.embed_dim
        for hidden_dim in cfg.classifier_hidden_dims:
            classifier_layers.append(nn.Linear(in_dim, hidden_dim))
            classifier_layers.append(nn.ReLU(inplace=True))
            classifier_layers.append(nn.Dropout(0.2))
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = batch["sequence"]
        batch_size, seq_len, channels, samples = sequence.shape
        x = sequence.view(batch_size * seq_len, channels, samples)
        embeddings = self.feature_extractor(x)
        embeddings = embeddings.view(batch_size, seq_len, -1)

        pos = self.positional_embedding[:, :seq_len, :]
        transformer_input = embeddings + pos
        encoded = self.transformer(transformer_input)
        pooled = encoded.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
