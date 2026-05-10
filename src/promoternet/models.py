"""Neural model definitions for promoter strength regression."""

from __future__ import annotations

import torch
from torch import nn


class PromoterCNN(nn.Module):
    """DeepBind/Basset-flavored 1D CNN for fixed-length promoter sequences.

    Input: (batch, 4, L) one-hot tensor.
    Output: (batch,) scalar predicted log-expression.
    """

    def __init__(
        self,
        seq_len: int = 150,
        conv1_filters: int = 128,
        conv1_kernel: int = 12,
        conv2_filters: int = 64,
        conv2_kernel: int = 6,
        pool_size: int = 4,
        hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(4, conv1_filters, conv1_kernel, padding=conv1_kernel // 2)
        self.bn1 = nn.BatchNorm1d(conv1_filters)
        self.pool1 = nn.MaxPool1d(pool_size)

        self.conv2 = nn.Conv1d(
            conv1_filters, conv2_filters, conv2_kernel, padding=conv2_kernel // 2
        )
        self.bn2 = nn.BatchNorm1d(conv2_filters)
        self.pool2 = nn.MaxPool1d(pool_size)

        self.relu = nn.ReLU()

        with torch.no_grad():
            dummy = torch.zeros(1, 4, seq_len)
            flat = self._features(dummy).numel()

        self.fc1 = nn.Linear(flat, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, 1)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._features(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)
