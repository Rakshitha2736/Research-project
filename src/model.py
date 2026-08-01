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


class CNNLSTMTemporalBaseline(nn.Module):
    """
    Temporal CNN-LSTM ablation.
    NOTE: This is NOT the base paper's spatial CNN-LSTM (paper Sec 3.3.3),
    which convolves over a 2D lat/lon grid. This variant convolves over
    the TIME axis instead, since no valid spatial grid exists for
    irregular station data. Included as a supplementary ablation only.
    """

    def __init__(
        self,
        n_features: int = 8,
        conv_channels: tuple[int, int] = (16, 32),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
        use_pooling: bool = False,
    ):
        super().__init__()
        c1, c2 = conv_channels

        self.conv1 = nn.Conv1d(n_features, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.use_pooling = use_pooling
        if use_pooling:
            # kernel=2, stride=1, padding=0 keeps length close to original
            # (29 instead of 30) - opt-in only, not the default.
            self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 30, 8)
        x = x.transpose(1, 2)          # (B, 8, 30)
        x = self.relu(self.conv1(x))   # (B, 16, 30)
        x = self.relu(self.conv2(x))   # (B, 32, 30)
        if self.use_pooling:
            x = self.pool(x)           # (B, 32, ~29)
        x = x.transpose(1, 2)          # (B, T, 32)

        lstm_out, _ = self.lstm(x)     # (B, T, 64)
        last_hidden = lstm_out[:, -1, :]  # (B, 64)

        out = self.fc(last_hidden)     # (B, 1)
        return out.squeeze(-1)         # (B,)


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor):
        # lstm_outputs: (B, T, H)
        scores = self.v(torch.tanh(self.W(lstm_outputs)))   # (B, T, 1)
        weights = torch.softmax(scores, dim=1)               # (B, T, 1)
        context = torch.sum(weights * lstm_outputs, dim=1)   # (B, H)
        return context, weights.squeeze(-1)                  # weights: (B, T)


class CNNLSTMAttention(nn.Module):
    """
    Temporal CNN-LSTM with additive attention over LSTM hidden states.
    Attention placed AFTER the LSTM (not before), following the pattern
    in cited literature (LSTM-SelfAttention, CNN-Attention-BiLSTM for
    precipitation forecasting): LSTM builds contextualized per-day
    representations, attention then weights their relevance.
    """

    def __init__(
        self,
        n_features: int = 8,
        conv_channels: tuple[int, int] = (16, 32),
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
    ):
        super().__init__()
        c1, c2 = conv_channels

        self.conv1 = nn.Conv1d(n_features, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=c2,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
        )
        self.attention = AdditiveAttention(lstm_hidden)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        # x: (B, 30, 8)
        x = x.transpose(1, 2)          # (B, 8, 30)
        x = self.relu(self.conv1(x))   # (B, 16, 30)
        x = self.relu(self.conv2(x))   # (B, 32, 30)
        x = x.transpose(1, 2)          # (B, 30, 32)

        lstm_out, _ = self.lstm(x)     # (B, 30, 64) - ALL hidden states
        context, attn_weights = self.attention(lstm_out)  # (B,64), (B,30)

        out = self.fc(context).squeeze(-1)  # (B,)

        if return_attention:
            return out, attn_weights
        return out


class TransformerEncoderBaseline(nn.Module):
    """Pre-norm Transformer encoder -> last timestep -> FC (scalar rainfall).

    Adapted from base-paper Sec 3.3.4 for per-station (non-spatial) output.
    Input: (batch, seq_len, input_size) with input_size=8 (v2 feature set).
    """

    def __init__(
        self,
        input_size: int = 8,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        h = self.input_proj(x) + self.pos_embed[:, : x.size(1), :]
        h = self.encoder(h)
        last = h[:, -1, :]
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
