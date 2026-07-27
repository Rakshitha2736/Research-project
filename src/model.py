"""Models for rainfall prediction (LSTM baseline + GNN-LSTM)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GNNLSTM(nn.Module):
    """Per-day shared-weight 2-layer GCN encoder -> per-station LSTM -> FC.

    GCN uses a *per-date* masked adjacency (invalid stations isolated with
    self-loop only), renormalized as D^{-0.5} A_masked D^{-0.5}.
    """

    def __init__(
        self,
        adjacency: torch.Tensor,
        in_features: int = 8,
        gcn_hidden: int = 16,
        gcn_out: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"adjacency must be square, got {tuple(adjacency.shape)}")
        # A already includes self-loops; stored as buffer (not trained)
        self.register_buffer("adjacency", adjacency.to(dtype=torch.float32))

        self.w1 = nn.Parameter(torch.empty(in_features, gcn_hidden))
        self.w2 = nn.Parameter(torch.empty(gcn_hidden, gcn_out))
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        self.lstm = nn.LSTM(
            input_size=gcn_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def _masked_a_norm(self, mask: torch.Tensor) -> torch.Tensor:
        """Per-date symmetric-normalized adjacency.

        mask: (B, N) bool — True = valid station that day.
        Returns A_norm: (B, N, N).
        """
        a = self.adjacency  # (N, N), includes self-loops
        n = a.size(0)
        both = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, N, N)
        am = a * both.to(dtype=a.dtype)
        # Isolate masked nodes with self-loop only; keep self-loops on valid nodes
        idx = torch.arange(n, device=a.device)
        am = am.clone()
        am[:, idx, idx] = 1.0

        deg = am.sum(dim=-1).clamp_min(1e-12)  # (B, N)
        d_inv_sqrt = deg.pow(-0.5)
        # A_norm[b,i,j] = d_i^{-1/2} * A[b,i,j] * d_j^{-1/2}
        return d_inv_sqrt.unsqueeze(2) * am * d_inv_sqrt.unsqueeze(1)

    def _gcn_encode(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        """Shared-weight GCN over all timesteps.

        x: (B, N, T, F_in)  a_norm: (B, N, N)  -> (B, N, T, F_out)
        """
        b, n, t, f = x.shape
        # (B*T, N, F)
        h = x.permute(0, 2, 1, 3).reshape(b * t, n, f)
        a_rep = a_norm.unsqueeze(1).expand(b, t, n, n).reshape(b * t, n, n)

        h = torch.bmm(a_rep, h @ self.w1)
        h = F.relu(h)
        h = torch.bmm(a_rep, h @ self.w2)
        h = F.relu(h)

        return h.reshape(b, t, n, -1).permute(0, 2, 1, 3)  # (B, N, T, F_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, N, T, F)
        mask: (B, N) bool
        returns pred: (B, N) scaled rainfall
        """
        a_norm = self._masked_a_norm(mask)
        h = self._gcn_encode(x, a_norm)  # (B, N, T, 32)

        b, n, t, f = h.shape
        h_flat = h.reshape(b * n, t, f)
        out, _ = self.lstm(h_flat)
        last = out[:, -1, :]
        pred = self.fc(last).squeeze(-1)  # (B*N,)
        return pred.reshape(b, n)
