"""
Train baseline LSTM for horizons h=2,3,4 (seed 42).

Architecture/hparams match lstm_baseline_v2 (except batch_size=64 as specified):
  2-layer LSTM 64 hidden -> FC->1, Adam lr=1e-3, max_epochs=100,
  early stop patience=15 / min_delta=1e-5, grad clip=1.0.

Prints a single table: horizon | stopped_epoch | best_epoch | RMSE | MAE | R2.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.cuda_setup import make_loader, require_cuda, set_seed, to_device
from src.model import LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEED = 42
BATCH_SIZE = 64
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 30  # was 15; allow longer plateau before stopping
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
HORIZONS = (2, 3, 4)


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


def train_horizon(h: int, device: torch.device) -> dict:
    set_seed(SEED)

    X_train = np.load(DATA / f"X_train_h{h}.npy")
    X_val = np.load(DATA / f"X_val_h{h}.npy")
    X_test = np.load(DATA / f"X_test_h{h}.npy")
    y_train = np.load(DATA / f"y_train_h{h}.npy")
    y_val = np.load(DATA / f"y_val_h{h}.npy")
    y_test = np.load(DATA / f"y_test_h{h}.npy")
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")

    assert X_train.shape[-1] == 8

    train_loader = make_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
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
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": best_val,
            "seed": SEED,
            "horizon": h,
            "input_size": 8,
            "batch_size": BATCH_SIZE,
            "device": str(device),
        },
        MODELS / f"lstm_h{h}_seed42.pt",
    )

    y_pred_s = predict(model, test_loader, device)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    m = metrics_mm(y_true, y_pred)
    return {
        "horizon": h,
        "stopped_epoch": stopped_epoch,
        "best_epoch": best_epoch,
        "pred_std": float(np.std(y_pred)),
        "target_std": float(np.std(y_true)),
        **m,
    }


def main() -> None:
    device = require_cuda()
    rows = [train_horizon(h, device) for h in HORIZONS]

    print(
        f"{'horizon':>7}  {'stopped':>7}  {'best':>5}  "
        f"{'RMSE':>8}  {'MAE':>8}  {'R2':>8}  "
        f"{'pred_std':>8}  {'tgt_std':>8}"
    )
    print("-" * 78)
    for r in rows:
        print(
            f"{r['horizon']:>7}  {r['stopped_epoch']:>7}  {r['best_epoch']:>5}  "
            f"{r['RMSE']:8.4f}  {r['MAE']:8.4f}  {r['R2']:8.4f}  "
            f"{r['pred_std']:8.4f}  {r['target_std']:8.4f}"
        )


if __name__ == "__main__":
    main()
