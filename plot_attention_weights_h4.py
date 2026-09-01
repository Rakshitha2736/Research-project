"""
Mean attention profile for CNNLSTMAttention h=4 (seed 42).

Regenerates (N, 30) attention weights on X_test_h4, averages over samples,
plots days-before-target (1=most recent ... 30=oldest).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.model import CNNLSTMAttention

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
FIGURES = BASE / "reports" / "figures"

SEQ_LEN = 30
H = 4
SEED = 42


@torch.no_grad()
def collect_attention(model: CNNLSTMAttention, loader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            _, attn = model(xb, return_attention=True)
        chunks.append(attn.float())
    return torch.cat(chunks, dim=0).cpu().numpy()


def main() -> None:
    device = require_cuda()
    X_test = np.load(DATA / f"X_test_h{H}.npy")
    y_dummy = np.zeros(len(X_test), dtype=np.float32)
    loader = make_loader(X_test, y_dummy, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    model = CNNLSTMAttention(n_features=8).to(device)
    ckpt = torch.load(
        MODELS / f"cnn_lstm_attention_h{H}_seed{SEED}.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    attn = collect_attention(model, loader, device)  # (N, 30), idx 0=oldest, 29=most recent
    assert attn.ndim == 2 and attn.shape[1] == SEQ_LEN
    np.save(DATA / f"attention_weights_h{H}_seed{SEED}.npy", attn)

    mean_w = attn.mean(axis=0)  # (30,), chronological oldest→newest
    assert np.isclose(mean_w.sum(), 1.0, atol=1e-3)

    # Plot axis: 1=most recent ... 30=oldest  → reverse chronological mean
    days_before = np.arange(1, SEQ_LEN + 1)
    mean_for_plot = mean_w[::-1]

    peak_idx_chrono = int(np.argmax(mean_w))  # 0=oldest
    peak_day = SEQ_LEN - peak_idx_chrono  # 1=most recent

    recent7 = float(mean_w[-7:].sum())
    oldest7 = float(mean_w[:7].sum())

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(days_before, mean_for_plot, width=0.8, color="#2c6e8a", edgecolor="none")
    ax.set_xlabel("Days before target (1 = most recent, 30 = oldest)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(
        f"Temporal CNN-LSTM+Attention — mean α over test set (h={H}, seed {SEED})"
    )
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0.5, 30.5)
    ax.axvline(peak_day, color="gray", linestyle="--", linewidth=1, label=f"peak day={peak_day}")
    ax.legend(frameon=False)
    plt.tight_layout()
    out = FIGURES / "attention_weights_h4_mean.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"peak_day_position: {peak_day}")
    print(f"recent_7_days_share: {recent7:.4f}")
    print(f"oldest_7_days_share: {oldest7:.4f}")


if __name__ == "__main__":
    main()
