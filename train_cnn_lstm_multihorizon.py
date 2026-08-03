"""
Train CNN-LSTM Temporal + CNN-LSTM Attention for horizons h=2,3,4 (seed 42).

Uses existing X/y_*_h{h}.npy and minmax_scaler_y_h{h}.joblib — does not regenerate.
Protocol matches h=1: Adam lr=1e-3, batch_size=256, patience=15, min_delta=1e-5,
grad clip=1.0. Requires CUDA.
"""

from __future__ import annotations

from pathlib import Path

import joblib
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
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEED = 42
BATCH_SIZE = DEFAULT_BATCH_SIZE  # 256 — match h=1 protocol
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
HORIZONS = (2, 3, 4)

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
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for xb, yb in loader:
        xb, yb = to_device(xb, yb, device)
        with autocast("cuda"):
            loss = criterion(model(xb), yb)
        total += loss.detach() * xb.size(0)
        n += xb.size(0)
    return float(total.item() / max(n, 1))


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(model(xb).float())
    return torch.cat(chunks, dim=0).cpu().numpy()


def metrics_mm(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def build_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "attention":
        return CNNLSTMAttention(n_features=8).to(device)
    if model_name == "temporal":
        return CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
    raise ValueError(model_name)


def ckpt_name(model_name: str, h: int) -> Path:
    if model_name == "attention":
        return MODELS / f"cnn_lstm_attention_h{h}_seed42.pt"
    if model_name == "temporal":
        return MODELS / f"cnn_lstm_temporal_h{h}_seed42.pt"
    raise ValueError(model_name)


def train_one(model_name: str, h: int, device: torch.device) -> dict:
    set_seed(SEED)

    X_train = np.load(DATA / f"X_train_h{h}.npy")
    X_val = np.load(DATA / f"X_val_h{h}.npy")
    X_test = np.load(DATA / f"X_test_h{h}.npy")
    y_train = np.load(DATA / f"y_train_h{h}.npy")
    y_val = np.load(DATA / f"y_val_h{h}.npy")
    y_test = np.load(DATA / f"y_test_h{h}.npy")
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")

    assert X_train.shape[-1] == 8, f"Expected 8 features, got {X_train.shape[-1]}"

    train_loader = make_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(model_name, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    amp_scaler = GradScaler("cuda")

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = to_device(xb, yb, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = criterion(model(xb), yb)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()

        val_loss = evaluate(model, val_loader, criterion, device)
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
    payload = {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val,
        "seed": SEED,
        "horizon": h,
        "n_features": 8,
        "feature_cols": FEATURE_COLS,
        "batch_size": BATCH_SIZE,
        "device": str(device),
    }
    if model_name == "temporal":
        payload["use_pooling"] = False
    torch.save(payload, ckpt_name(model_name, h))

    y_pred_s = predict(model, test_loader, device)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    m = metrics_mm(y_true, y_pred)

    label = "CNN-LSTM+Attn" if model_name == "attention" else "CNN-LSTM-Temp"
    return {
        "horizon": h,
        "model": label,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        **m,
    }


def main() -> None:
    device = require_cuda()
    rows: list[dict] = []
    for h in HORIZONS:
        for model_name in ("temporal", "attention"):
            rows.append(train_one(model_name, h, device))

    print(
        f"{'horizon':>7} | {'model':<15} | {'stopped':>7} | {'best':>5} | "
        f"{'RMSE':>8} | {'MAE':>8} | {'R2':>8}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['horizon']:>7} | {r['model']:<15} | {r['stopped_epoch']:>7} | "
            f"{r['best_epoch']:>5} | {r['RMSE']:8.4f} | {r['MAE']:8.4f} | {r['R2']:8.4f}"
        )


if __name__ == "__main__":
    main()
