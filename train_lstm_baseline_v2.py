"""
Train baseline LSTM v2 (8 features, past rainfall in X) — seed 42.
Requires CUDA (project .venv with torch+cu126).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
from src.persistence_baseline import eval_persistence_horizon

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
FIGURES = BASE / "reports" / "figures"

SEED = 42
BATCH_SIZE = DEFAULT_BATCH_SIZE  # 256 — fits RTX 2050 4GB for this LSTM
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
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-02-10")


def split_name(target_date: pd.Timestamp) -> str | None:
    if target_date <= TRAIN_END:
        return "train"
    if VAL_START <= target_date <= VAL_END:
        return "val"
    if TEST_START <= target_date <= TEST_END:
        return "test"
    return None


def rebuild_test_meta(df: pd.DataFrame) -> list[dict]:
    meta: list[dict] = []
    need = SEQ_LEN + 1
    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        n = len(g)
        if n < need:
            continue
        day_ints = dates.astype("datetime64[D]").astype(np.int64)
        breaks = np.where(np.diff(day_ints) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g["rainfall"].to_numpy(dtype=np.float64)
        for seg_start, seg_end in zip(starts, ends):
            if seg_end - seg_start < need:
                continue
            for i in range(seg_start, seg_end - SEQ_LEN):
                target_idx = i + SEQ_LEN
                target_date = pd.Timestamp(dates[target_idx])
                if split_name(target_date) != "test":
                    continue
                x_seq = feats[i : i + SEQ_LEN]
                y_val = targets[target_idx]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
                    continue
                meta.append(
                    {
                        "station_id": station_id,
                        "target_date": str(target_date.date()),
                    }
                )
    return meta


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

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    scaler = GradScaler("cuda")

    # Diagnostics (once)
    xb0, _ = next(iter(train_loader))
    xb0 = xb0.to(device, non_blocking=True)
    print_gpu_diagnostics(model=model, sample_batch=xb0)
    del xb0

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
            # Accumulate on GPU — single .item() sync per epoch (not per batch)
            running += loss.detach() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = float(running.item() / max(n_seen, 1))
        val_loss = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1:
            print_gpu_memory("after epoch 1")

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
    ckpt_path = MODELS / "lstm_baseline_v2_seed42.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": best_val,
            "history": history,
            "seed": SEED,
            "input_size": 8,
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
    ax.set_title("LSTM Baseline v2 Training Curve (seed 42, 8 features, CUDA)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "lstm_baseline_v2_training_curve.png", dpi=150)
    plt.close()

    y_pred_scaled = predict(model, test_loader, device)
    y_pred_mm = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_test_mm = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    m = metrics_mm(y_test_mm, y_pred_mm)
    persistence_m = eval_persistence_horizon(1, BASE)

    df = pd.read_csv(DATA / "feature_engineered_v2.csv", parse_dates=["date_of_record"])
    test_meta = rebuild_test_meta(df)
    assert len(test_meta) == len(y_test_mm)
    meta_df = pd.DataFrame(test_meta)
    meta_df["y_true"] = y_test_mm
    meta_df["y_pred"] = y_pred_mm
    sample_station = meta_df["station_id"].value_counts().index[0]
    sample = meta_df[meta_df["station_id"] == sample_station].copy()
    sample["target_date"] = pd.to_datetime(sample["target_date"])
    sample = sample.sort_values("target_date")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample["target_date"], sample["y_true"], label="Actual", linewidth=1.2)
    ax.plot(sample["target_date"], sample["y_pred"], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rainfall (mm/day)")
    ax.set_title(f"Predicted vs Actual — {sample_station} (test, v2)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "lstm_baseline_v2_pred_vs_actual_sample.png", dpi=150)
    plt.close()

    with open(MODELS / "lstm_baseline_v2_seed42_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stopped_epoch": stopped_epoch,
                "best_epoch": best_epoch,
                "best_val_loss_scaled": best_val,
                "test_metrics_mm": m,
                "sample_station": sample_station,
                "persistence_baseline_mm": persistence_m,
                "device": str(device),
                "batch_size": BATCH_SIZE,
            },
            f,
            indent=2,
        )

    print(f"stopped_epoch: {stopped_epoch}")
    print(f"best_epoch: {best_epoch}")
    print(f"MSE:  {m['MSE']:.4f}")
    print(f"RMSE: {m['RMSE']:.4f}")
    print(f"MAE:  {m['MAE']:.4f}")
    print(f"R2:   {m['R2']:.4f}")
    print(f"Persistence RMSE: {persistence_m['RMSE']:.4f}")
    print(f"Persistence MAE:  {persistence_m['MAE']:.4f}")
    print(f"Persistence R2:   {persistence_m['R2']:.4f}")


if __name__ == "__main__":
    main()
