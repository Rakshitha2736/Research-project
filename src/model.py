"""Baseline LSTM for rainfall prediction (base paper Sec 3.3.1, scalar target)."""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """2-layer LSTM (64 units) -> FC -> 1 rainfall output."""

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # (batch, hidden)
        return self.fc(last).squeeze(-1)  # (batch,)
