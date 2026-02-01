from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int | None = None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced, bias=False)
        self.fc2 = nn.Linear(reduced, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_pool = torch.mean(x, dim=-1)
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(avg_pool))))
        return x * weights.unsqueeze(-1)


class SequenceEncoder(nn.Module):
    """Encode sequence context into a fixed-length representation."""

    def __init__(
        self,
        input_channels: int = 2,
        base_filters: int = 32,
        kernel_size: int = 7,
        residual_blocks: int = 2,
        reduction_ratio: int = 8,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        bidirectional: bool = True,
        use_residual: bool = True,
        use_attention: bool = True,
        use_lstm: bool = True,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_residual = use_residual and residual_blocks > 0
        self.use_attention = use_attention
        self.use_lstm = use_lstm
        filters = base_filters
        self.encoder = nn.Sequential(
            ConvBNReLU(input_channels, filters, kernel_size=kernel_size),
            ConvBNReLU(filters, filters * 2, kernel_size=kernel_size),
        )

        if self.use_residual:
            self.residual_layers = nn.Sequential(
                *[ResidualBlock(filters * 2, kernel_size=kernel_size) for _ in range(residual_blocks)]
            )
        else:
            self.residual_layers = nn.Identity()
        self.attention = (
            ChannelAttention(filters * 2, reduction=reduction_ratio) if self.use_attention else nn.Identity()
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        feature_dim = filters * 2

        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=feature_dim,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
            )
            self.output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        else:
            self.lstm = None
            self.output_dim = feature_dim

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels, samples = sequence.shape
        x = sequence.view(batch_size * seq_len, channels, samples)
        x = self.encoder(x)
        x = self.residual_layers(x)
        x = self.attention(x)
        x = self.global_pool(x).squeeze(-1)
        x = x.view(batch_size, seq_len, -1)
        if self.use_lstm and self.lstm is not None:
            lstm_out, _ = self.lstm(x)
            return lstm_out[:, -1, :]
        return x.mean(dim=1)


@dataclass
class BaselineModelConfig:
    input_channels: int = 2
    base_filters: int = 32
    kernel_size: int = 7
    residual_blocks: int = 2
    reduction_ratio: int = 8
    lstm_hidden_size: int = 128
    lstm_layers: int = 2
    lstm_dropout: float = 0.2
    bidirectional: bool = True
    use_residual: bool = True
    use_attention: bool = True
    use_lstm: bool = True
    classifier_hidden_dims: tuple[int, ...] = (128,)
    num_classes: int = 3
    sequence_length: int = 11


class BaselineSequenceModel(nn.Module):
    """CNN encoder + channel attention + BiLSTM classifier."""

    def __init__(self, cfg: BaselineModelConfig):
        super().__init__()
        self.cfg = cfg
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
            use_residual=cfg.use_residual,
            use_attention=cfg.use_attention,
            use_lstm=cfg.use_lstm,
        )

        lstm_out_dim = self.sequence_encoder.output_dim
        classifier_layers = []
        in_dim = lstm_out_dim
        for hidden_dim in cfg.classifier_hidden_dims:
            classifier_layers.append(nn.Linear(in_dim, hidden_dim))
            classifier_layers.append(nn.ReLU())
            classifier_layers.append(nn.Dropout(0.3))
            in_dim = hidden_dim
        classifier_layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        sequence = batch["sequence"]
        representation = self.sequence_encoder(sequence)
        logits = self.classifier(representation)
        return logits
