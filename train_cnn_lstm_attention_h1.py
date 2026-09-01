"""
Train temporal CNN-LSTM + additive attention h=1 (8 features, v2 data) — seed 42.
Same training protocol as LSTM v2 / temporal CNN-LSTM (batch 256). Requires CUDA.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.cuda_setup import (
    DEFAULT_BATCH_SIZE,
    make_loader,
    require_cuda,
    set_seed,
    to_device,
)
from src.model import CNNLSTMAttention

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
FIGURES = BASE / "reports" / "figures"

SEED = 42
BATCH_SIZE = DEFAULT_BATCH_SIZE  # 256 — match LSTM v2 protocol
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
SEQ_LEN = 30

FEATURE_COLS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "rainfall",
    "doy_sin",
    "doy_cos",
]


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = torch.zeros((), device=device)
    n = 0
    for xb, yb in loader:
        xb, yb = to_device(xb, yb, device)
        with autocast("cuda"):
            pred = model(xb)
            loss = criterion(pred, yb)
        total_loss += loss.detach() * xb.size(0)
        n += xb.size(0)
    return float(total_loss.item() / max(n, 1))


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device) -> np.ndarray:
    """Run inference on GPU; move to CPU once at the end."""
    model.eval()
    chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            pred = model(xb)
        chunks.append(pred.float())
    if not chunks:
        return np.array([], dtype=np.float32)
    return torch.cat(chunks, dim=0).cpu().numpy()


@torch.no_grad()
def predict_with_attention(
    model: CNNLSTMAttention, loader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return predictions and attention weights (N, T) for the full loader."""
    model.eval()
    pred_chunks: list[torch.Tensor] = []
    attn_chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            pred, attn = model(xb, return_attention=True)
        pred_chunks.append(pred.float())
        attn_chunks.append(attn.float())
    if not pred_chunks:
        return (
            np.array([], dtype=np.float32),
            np.zeros((0, SEQ_LEN), dtype=np.float32),
        )
    y_pred = torch.cat(pred_chunks, dim=0).cpu().numpy()
    attn = torch.cat(attn_chunks, dim=0).cpu().numpy()
    return y_pred, attn


def metrics_mm(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def main() -> None:
    set_seed(SEED)
    device = require_cuda()

    X_train = np.load(DATA / "X_train_v2.npy")
    X_val = np.load(DATA / "X_val_v2.npy")
    X_test = np.load(DATA / "X_test_v2.npy")
    y_train = np.load(DATA / "y_train_v2.npy")
    y_val = np.load(DATA / "y_val_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")

    assert X_train.shape[-1] == 8, f"Expected 8 features, got {X_train.shape[-1]}"

    train_loader = make_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    model = CNNLSTMAttention(n_features=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler("cuda")

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = torch.zeros((), device=device)
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = to_device(xb, yb, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                pred = model(xb)
                loss = criterion(pred, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = float(running.item() / max(n_seen, 1))
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val - MIN_DELTA:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                stopped_epoch = epoch
                break
    else:
        stopped_epoch = MAX_EPOCHS

    if best_state is not None:
        model.load_state_dict(best_state)

    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODELS / "cnn_lstm_attention_h1_seed42.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": best_val,
            "history": history,
            "seed": SEED,
            "n_features": 8,
            "feature_cols": FEATURE_COLS,
            "device": str(device),
            "batch_size": BATCH_SIZE,
        },
        ckpt_path,
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train")
    ax.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="val")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (scaled y)")
    ax.set_title(
        "Temporal CNN-LSTM+Attention h=1 Training Curve (seed 42, batch 256)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "cnn_lstm_attention_h1_training_curve.png", dpi=150)
    plt.close()

    y_pred_scaled, attn_weights = predict_with_attention(model, test_loader, device)
    np.save(DATA / "attention_weights_h1_seed42.npy", attn_weights)

    y_pred_mm = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_mm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    m = metrics_mm(y_test_mm, y_pred_mm)

    print(f"stopped_epoch: {stopped_epoch}")
    print(f"best_epoch: {best_epoch}")
    print(f"MSE:  {m['MSE']:.4f}")
    print(f"RMSE: {m['RMSE']:.4f}")
    print(f"MAE:  {m['MAE']:.4f}")
    print(f"R2:   {m['R2']:.4f}")


if __name__ == "__main__":
    main()
