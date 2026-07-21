"""
Multi-seed LSTM v2 training (seeds 13, 42, 123).
Requires CUDA (project .venv with torch+cu126).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.cuda_setup import (
    DEFAULT_BATCH_SIZE,
    make_loader,
    print_gpu_diagnostics,
    print_gpu_memory,
    require_cuda,
    set_seed,
    to_device,
)
from src.model import LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEEDS = [13, 42, 123]
BATCH_SIZE = DEFAULT_BATCH_SIZE
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0


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


def train_one(seed: int, loaders, scaler_y, device, *, print_diag: bool = False) -> dict[str, float]:
    set_seed(seed)
    train_loader, val_loader, test_loader = loaders
    input_size = 8

    model = LSTMBaseline(input_size=input_size, hidden_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    amp_scaler = GradScaler("cuda")

    if print_diag:
        xb0, _ = next(iter(train_loader))
        print_gpu_diagnostics(model=model, sample_batch=xb0.to(device, non_blocking=True))

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
        if epoch == 1 and print_diag:
            print_gpu_memory(f"seed {seed} after epoch 1")

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

    ckpt = MODELS / f"lstm_baseline_v2_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": best_val,
            "seed": seed,
            "input_size": input_size,
            "device": str(device),
            "batch_size": BATCH_SIZE,
        },
        ckpt,
    )

    y_test = np.load(DATA / "y_test_v2.npy")
    y_pred_s = predict(model, test_loader, device)
    y_pred = scaler_y.inverse_transform(y_pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    m = metrics_mm(y_true, y_pred)

    with open(MODELS / f"lstm_baseline_v2_seed{seed}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stopped_epoch": stopped_epoch,
                "best_epoch": best_epoch,
                "test_metrics_mm": m,
                "device": str(device),
            },
            f,
            indent=2,
        )
    return m


def main() -> None:
    device = require_cuda()
    X_train = np.load(DATA / "X_train_v2.npy")
    X_val = np.load(DATA / "X_val_v2.npy")
    X_test = np.load(DATA / "X_test_v2.npy")
    y_train = np.load(DATA / "y_train_v2.npy")
    y_val = np.load(DATA / "y_val_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")

    loaders = (
        make_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
        make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False),
        make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False),
    )

    all_metrics: dict[int, dict[str, float]] = {}

    # Always retrain all seeds under CUDA path when this script is run
    first = True
    for seed in SEEDS:
        all_metrics[seed] = train_one(
            seed, loaders, scaler_y, device, print_diag=first
        )
        first = False

    keys = ["MSE", "RMSE", "MAE", "R2"]
    rows = []
    for k in keys:
        vals = np.array([all_metrics[s][k] for s in SEEDS], dtype=np.float64)
        rows.append((k, vals.mean(), vals.std(ddof=1)))

    summary = {k: {"mean": float(m), "std": float(s)} for k, m, s in rows}
    with open(MODELS / "lstm_baseline_v2_multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump({"seeds": SEEDS, "per_seed": all_metrics, "summary": summary}, f, indent=2)

    print("Metric   mean ± std (seeds 13, 42, 123)")
    print("-" * 42)
    for k, mean, std in rows:
        print(f"{k:<6}  {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
