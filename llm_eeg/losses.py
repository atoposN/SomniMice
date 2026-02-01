from __future__ import annotations

import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """
    Center loss: encourages embeddings of the same class to cluster tightly.
    """

    def __init__(self, num_classes: int, feat_dim: int) -> None:
        super().__init__()
        self.centers = nn.Parameter(torch.zeros(num_classes, feat_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long()
        batch_centers = self.centers[labels]
        return 0.5 * ((features - batch_centers).pow(2).sum(dim=1)).mean()
