"""
Multi-seed CNN-LSTM Temporal + CNN-LSTM Attention (h=1).

Trains seeds 13 and 123 (seed 42 checkpoints expected to already exist).
Reports mean±std RMSE/MAE/R2, then DM / paired-t / bootstrap CI (seed 42):
  1. Attention vs LSTM
  2. Attention vs Temporal (no-attention)
Requires CUDA (project .venv).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.amp import GradScaler, autocast

from arima_and_significance import diebold_mariano
from src.cuda_setup import (
    DEFAULT_BATCH_SIZE,
    make_loader,
    require_cuda,
    set_seed,
    to_device,
)
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEEDS = [13, 42, 123]
TRAIN_SEEDS = [13, 123]  # seed 42 already trained
BATCH_SIZE = DEFAULT_BATCH_SIZE  # 256
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
N_BOOT = 1000
BOOT_SEED = 42

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


def train_one(
    model_name: str,
    seed: int,
    loaders,
    device: torch.device,
) -> None:
    set_seed(seed)
    train_loader, val_loader, _ = loaders

    if model_name == "attention":
        model = CNNLSTMAttention(n_features=8).to(device)
        ckpt_path = MODELS / f"cnn_lstm_attention_h1_seed{seed}.pt"
    elif model_name == "temporal":
        model = CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
        ckpt_path = MODELS / f"cnn_lstm_temporal_h1_seed{seed}.pt"
    else:
        raise ValueError(model_name)

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

    payload = {
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "best_val_loss": best_val,
        "seed": seed,
        "n_features": 8,
        "feature_cols": FEATURE_COLS,
        "device": str(device),
        "batch_size": BATCH_SIZE,
    }
    if model_name == "temporal":
        payload["use_pooling"] = False
    torch.save(payload, ckpt_path)


def load_and_predict_mm(
    model_name: str,
    seed: int,
    test_loader,
    scaler_y,
    y_test: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    if model_name == "attention":
        model = CNNLSTMAttention(n_features=8).to(device)
        ckpt_path = MODELS / f"cnn_lstm_attention_h1_seed{seed}.pt"
    elif model_name == "temporal":
        model = CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
        ckpt_path = MODELS / f"cnn_lstm_temporal_h1_seed{seed}.pt"
    elif model_name == "lstm":
        model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
        ckpt_path = MODELS / f"lstm_baseline_v2_seed{seed}.pt"
    else:
        raise ValueError(model_name)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    pred_s = predict(model, test_loader, device)
    y_pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    return y_pred, metrics_mm(y_true, y_pred)


def bootstrap_rmse_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_resamples: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """95% CI for (RMSE_A - RMSE_B) via paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        rmse_a = float(np.sqrt(np.mean((yt - pa) ** 2)))
        rmse_b = float(np.sqrt(np.mean((yt - pb) ** 2)))
        diffs[b] = rmse_a - rmse_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def agg_mean_std(per_seed: dict[int, dict[str, float]], key: str) -> tuple[float, float]:
    vals = np.array([per_seed[s][key] for s in SEEDS], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    device = require_cuda()
    MODELS.mkdir(parents=True, exist_ok=True)

    X_train = np.load(DATA / "X_train_v2.npy")
    X_val = np.load(DATA / "X_val_v2.npy")
    X_test = np.load(DATA / "X_test_v2.npy")
    y_train = np.load(DATA / "y_train_v2.npy")
    y_val = np.load(DATA / "y_val_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")

    assert X_train.shape[-1] == 8

    loaders = (
        make_loader(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
        make_loader(X_val, y_val, batch_size=BATCH_SIZE, shuffle=False),
        make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False),
    )
    test_loader = loaders[2]

    # --- Train missing seeds ---
    for model_name in ("temporal", "attention"):
        for seed in TRAIN_SEEDS:
            train_one(model_name, seed, loaders, device)

    # --- Evaluate all 3 seeds ---
    attn_metrics: dict[int, dict[str, float]] = {}
    temp_metrics: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        _, attn_metrics[seed] = load_and_predict_mm(
            "attention", seed, test_loader, scaler_y, y_test, device
        )
        _, temp_metrics[seed] = load_and_predict_mm(
            "temporal", seed, test_loader, scaler_y, y_test, device
        )

    # --- Seed-42 paired significance ---
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    attn42, _ = load_and_predict_mm(
        "attention", 42, test_loader, scaler_y, y_test, device
    )
    temp42, _ = load_and_predict_mm(
        "temporal", 42, test_loader, scaler_y, y_test, device
    )
    lstm42, _ = load_and_predict_mm(
        "lstm", 42, test_loader, scaler_y, y_test, device
    )
    assert len(attn42) == len(temp42) == len(lstm42) == len(y_true)

    err_attn = (y_true - attn42) ** 2
    err_temp = (y_true - temp42) ** 2
    err_lstm = (y_true - lstm42) ** 2

    dm_vs_lstm = diebold_mariano(err_attn, err_lstm, h=1)
    tt_vs_lstm = float(stats.ttest_rel(err_attn, err_lstm).pvalue)
    ci_vs_lstm = bootstrap_rmse_diff_ci(y_true, attn42, lstm42)

    dm_vs_temp = diebold_mariano(err_attn, err_temp, h=1)
    tt_vs_temp = float(stats.ttest_rel(err_attn, err_temp).pvalue)
    ci_vs_temp = bootstrap_rmse_diff_ci(y_true, attn42, temp42)

    a_rmse = agg_mean_std(attn_metrics, "RMSE")
    a_mae = agg_mean_std(attn_metrics, "MAE")
    a_r2 = agg_mean_std(attn_metrics, "R2")
    t_rmse = agg_mean_std(temp_metrics, "RMSE")
    t_mae = agg_mean_std(temp_metrics, "MAE")
    t_r2 = agg_mean_std(temp_metrics, "R2")

    # ONE table
    print("model | RMSE(mean±std) | MAE(mean±std) | R2(mean±std)")
    print("-" * 72)
    print(
        f"CNN-LSTM+Attention | {fmt_ms(*a_rmse)} | {fmt_ms(*a_mae)} | {fmt_ms(*a_r2)}"
    )
    print(
        f"CNN-LSTM-Temporal  | {fmt_ms(*t_rmse)} | {fmt_ms(*t_mae)} | {fmt_ms(*t_r2)}"
    )
    print("-" * 72)
    print(
        f"Attn vs LSTM (seed42): DM p={dm_vs_lstm:.6e}  "
        f"paired-t p={tt_vs_lstm:.6e}  "
        f"bootstrap 95% CI RMSE_attn-RMSE_lstm=({ci_vs_lstm[0]:.4f}, {ci_vs_lstm[1]:.4f})"
    )
    print(
        f"Attn vs Temporal (seed42): DM p={dm_vs_temp:.6e}  "
        f"paired-t p={tt_vs_temp:.6e}  "
        f"bootstrap 95% CI RMSE_attn-RMSE_temp=({ci_vs_temp[0]:.4f}, {ci_vs_temp[1]:.4f})"
    )


if __name__ == "__main__":
    main()
