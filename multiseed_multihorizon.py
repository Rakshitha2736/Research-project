"""
Multi-seed LSTM + GNN-LSTM for horizons h=2,3,4 (seeds 13, 42, 123).

Part 1 — Train seeds 13 and 123 (reuse any existing checkpoint, incl. seed 42):
  LSTM: batch=64, patience=30, Adam lr=1e-3, grad clip=1.0
  GNN:  batch_size=1 date/step, patience=30, per-date masked A_norm

Part 2 — Bootstrap 95% CI + DM / paired-t (seed=42) for h=1,2,3,4.

Prints ONE final table per horizon:
  horizon | LSTM RMSE(mean±std) | GNN RMSE(mean±std) |
  DM p-value | paired-t p-value | bootstrap 95% CI (lo, hi)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from torch.amp import GradScaler, autocast

from arima_and_significance import diebold_mariano
from build_graph_batches import replay_meta
from src.cuda_setup import make_loader, require_cuda, set_seed, to_device
from src.model import GNNLSTM, LSTMBaseline
from train_gnn_lstm import (
    build_and_save_adjacency,
    filter_train,
    make_date_loader,
    metrics_mm,
    predict_all,
    train_loop,
)
import train_gnn_lstm as _tgl

_tgl.PATIENCE = 30

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

TRAIN_HORIZONS = (2, 3, 4)
ALL_HORIZONS = (1, 2, 3, 4)
SEEDS = (13, 42, 123)
LSTM_BS = 64
GNN_BS = 1
LR = 1e-3
MAX_EPOCHS = 100
LSTM_PATIENCE = 30
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
N_BOOT = 1000
BOOT_SEED = 42


@torch.no_grad()
def lstm_evaluate(model, loader, criterion, device) -> float:
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
def lstm_predict(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(model(xb).float())
    return torch.cat(chunks, dim=0).cpu().numpy()


def train_lstm_seed(h: int, seed: int, device: torch.device) -> dict[str, float]:
    ckpt_path = MODELS / f"lstm_h{h}_seed{seed}.pt"
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")
    X_test = np.load(DATA / f"X_test_h{h}.npy")
    y_test = np.load(DATA / f"y_test_h{h}.npy")
    test_loader = make_loader(X_test, y_test, batch_size=LSTM_BS, shuffle=False)

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[monitor] reuse LSTM h={h} seed={seed}", file=sys.stderr, flush=True)
    else:
        set_seed(seed)
        X_train = np.load(DATA / f"X_train_h{h}.npy")
        X_val = np.load(DATA / f"X_val_h{h}.npy")
        y_train = np.load(DATA / f"y_train_h{h}.npy")
        y_val = np.load(DATA / f"y_val_h{h}.npy")
        train_loader = make_loader(X_train, y_train, batch_size=LSTM_BS, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size=LSTM_BS, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.MSELoss()
        amp_scaler = GradScaler("cuda")
        best_val = float("inf")
        best_state = None
        best_epoch = 0
        wait = 0
        stopped = MAX_EPOCHS

        print(f"[monitor] train LSTM h={h} seed={seed}", file=sys.stderr, flush=True)
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

            val_loss = lstm_evaluate(model, val_loader, criterion, device)
            if val_loss < best_val - MIN_DELTA:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= LSTM_PATIENCE:
                    stopped = epoch
                    break
        else:
            stopped = MAX_EPOCHS

        if best_state is not None:
            model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": best_epoch,
                "stopped_epoch": stopped,
                "best_val_loss": best_val,
                "seed": seed,
                "horizon": h,
                "batch_size": LSTM_BS,
                "patience": LSTM_PATIENCE,
            },
            ckpt_path,
        )
        print(
            f"[monitor] LSTM h={h} seed={seed} stopped={stopped} best={best_epoch}",
            file=sys.stderr,
            flush=True,
        )

    pred_s = lstm_predict(model, test_loader, device)
    yp = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    yt = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    return metrics_mm(yt, yp)


def train_gnn_seed(
    h: int, seed: int, adjacency: torch.Tensor, device: torch.device
) -> dict[str, float]:
    ckpt_path = MODELS / f"gnn_lstm_h{h}_seed{seed}.pt"
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")

    X_te = np.load(DATA / f"X_test_graph_h{h}.npy")
    y_te = np.load(DATA / f"y_test_graph_h{h}.npy")
    m_te = np.load(DATA / f"mask_test_graph_h{h}.npy")
    test_loader = make_date_loader(X_te, y_te, m_te, shuffle=False, batch_size=8)

    model = GNNLSTM(adjacency=adjacency).to(device)

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[monitor] reuse GNN h={h} seed={seed}", file=sys.stderr, flush=True)
    else:
        set_seed(seed)
        X_tr = np.load(DATA / f"X_train_graph_h{h}.npy")
        y_tr = np.load(DATA / f"y_train_graph_h{h}.npy")
        m_tr = np.load(DATA / f"mask_train_graph_h{h}.npy")
        X_va = np.load(DATA / f"X_val_graph_h{h}.npy")
        y_va = np.load(DATA / f"y_val_graph_h{h}.npy")
        m_va = np.load(DATA / f"mask_val_graph_h{h}.npy")
        X_tr, y_tr, m_tr, n_dates = filter_train(X_tr, y_tr, m_tr)
        train_loader = make_date_loader(X_tr, y_tr, m_tr, shuffle=True, batch_size=GNN_BS)
        val_loader = make_date_loader(X_va, y_va, m_va, shuffle=False, batch_size=8)

        print(
            f"[monitor] train GNN h={h} seed={seed} n_train_dates={n_dates}",
            file=sys.stderr,
            flush=True,
        )
        history, stopped, best, best_val, best_state = train_loop(
            model, train_loader, val_loader, device, log_epochs=False
        )
        if best_state:
            model.load_state_dict(best_state)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_epoch": best,
                "stopped_epoch": stopped,
                "best_val_loss": best_val,
                "history": history,
                "seed": seed,
                "horizon": h,
                "n_train_dates_after_filter": n_dates,
                "batch_size": GNN_BS,
                "patience": 30,
            },
            ckpt_path,
        )
        print(
            f"[monitor] GNN h={h} seed={seed} stopped={stopped} best={best}",
            file=sys.stderr,
            flush=True,
        )

    pred_s, true_s, mask = predict_all(model, test_loader, device)
    valid = mask.astype(bool)
    yp = scaler_y.inverse_transform(pred_s[valid].reshape(-1, 1)).ravel()
    yt = scaler_y.inverse_transform(true_s[valid].reshape(-1, 1)).ravel()
    return metrics_mm(yt, yp)


def eval_lstm_h1_seed(seed: int, device: torch.device) -> dict[str, float]:
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")
    X_test = np.load(DATA / "X_test_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    test_loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    ckpt = torch.load(
        MODELS / f"lstm_baseline_v2_seed{seed}.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])

    pred_s = lstm_predict(model, test_loader, device)
    yp = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    yt = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    return metrics_mm(yt, yp)


def eval_gnn_h1_seed(seed: int, adjacency: torch.Tensor, device: torch.device) -> dict[str, float]:
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")
    X_te = np.load(DATA / "X_test_graph.npy")
    y_te = np.load(DATA / "y_test_graph.npy")
    m_te = np.load(DATA / "mask_test_graph.npy")
    test_loader = make_date_loader(X_te, y_te, m_te, shuffle=False, batch_size=8)

    model = GNNLSTM(adjacency=adjacency).to(device)
    ckpt = torch.load(
        MODELS / f"gnn_lstm_seed{seed}.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state_dict"])

    pred_s, true_s, mask = predict_all(model, test_loader, device)
    valid = mask.astype(bool)
    yp = scaler_y.inverse_transform(pred_s[valid].reshape(-1, 1)).ravel()
    yt = scaler_y.inverse_transform(true_s[valid].reshape(-1, 1)).ravel()
    return metrics_mm(yt, yp)


@torch.no_grad()
def paired_preds_h1(device: torch.device, adjacency: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned masked test preds: y_true, gnn_pred, lstm_pred (seed 42)."""
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")

    X_te = np.load(DATA / "X_test_graph.npy")
    y_te = np.load(DATA / "y_test_graph.npy")
    m_te = np.load(DATA / "mask_test_graph.npy")
    test_loader = make_date_loader(X_te, y_te, m_te, shuffle=False, batch_size=8)

    gnn = GNNLSTM(adjacency=adjacency).to(device)
    gckpt = torch.load(MODELS / "gnn_lstm_seed42.pt", map_location=device, weights_only=False)
    gnn.load_state_dict(gckpt["model_state_dict"])
    gnn.eval()
    g_pred_s, g_true_s, mask = predict_all(gnn, test_loader, device)
    valid = mask.astype(bool)
    g_pred = scaler_y.inverse_transform(g_pred_s[valid].reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(g_true_s[valid].reshape(-1, 1)).ravel()

    df = pd.read_csv(DATA / "feature_engineered_v2.csv", parse_dates=["date_of_record"])
    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)
    meta = replay_meta(df)["test"]

    with open(DATA / "station_id_to_index.json", encoding="utf-8") as f:
        id_to_index = json.load(f)
    idx_to_station = {v: k for k, v in id_to_index.items()}
    with open(DATA / "graph_date_index_test.json", encoding="utf-8") as f:
        date_list = json.load(f)

    rows_date, rows_node = np.where(valid)

    lstm = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    lckpt = torch.load(
        MODELS / "lstm_baseline_v2_seed42.pt", map_location=device, weights_only=False
    )
    lstm.load_state_dict(lckpt["model_state_dict"])
    lstm.eval()

    X_test = np.load(DATA / "X_test_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(lstm(xb).float().cpu())
    l_pred_s = torch.cat(chunks, dim=0).numpy()
    l_pred_all = scaler_y.inverse_transform(l_pred_s.reshape(-1, 1)).ravel()

    l_pred_map: dict[tuple, float] = {}
    for r, lp in zip(meta, l_pred_all):
        key = (r["station_id"], r["target_date"])
        l_pred_map[key] = float(lp)

    # Build aligned arrays in GNN masked order
    l_pred = np.array(
        [l_pred_map[(idx_to_station[j], date_list[i])] for i, j in zip(rows_date, rows_node)],
        dtype=np.float64,
    )
    assert len(l_pred) == len(y_true) == len(g_pred)
    return y_true, g_pred, l_pred


@torch.no_grad()
def paired_preds_multi(
    h: int, device: torch.device, adjacency: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aligned masked test preds for horizon h>=2 (seed 42)."""
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")
    Xg = np.load(DATA / f"X_test_graph_h{h}.npy")
    yg = np.load(DATA / f"y_test_graph_h{h}.npy")
    mg = np.load(DATA / f"mask_test_graph_h{h}.npy")

    gnn = GNNLSTM(adjacency=adjacency).to(device)
    gckpt = torch.load(
        MODELS / f"gnn_lstm_h{h}_seed42.pt", map_location=device, weights_only=False
    )
    gnn.load_state_dict(gckpt["model_state_dict"])
    gnn.eval()
    loader = make_date_loader(Xg, yg, mg, shuffle=False, batch_size=8)
    g_pred_s, g_true_s, mask = predict_all(gnn, loader, device)
    valid = mask.astype(bool)
    g_pred = scaler_y.inverse_transform(g_pred_s[valid].reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(g_true_s[valid].reshape(-1, 1)).ravel()

    lstm = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    lckpt = torch.load(
        MODELS / f"lstm_h{h}_seed42.pt", map_location=device, weights_only=False
    )
    lstm.load_state_dict(lckpt["model_state_dict"])
    lstm.eval()

    Xs = []
    for di in range(len(mg)):
        v = mg[di]
        if v.any():
            Xs.append(Xg[di][v])
    X_flat = np.concatenate(Xs, axis=0)
    assert len(X_flat) == len(y_true)

    lloader = make_loader(
        X_flat, np.zeros(len(X_flat), dtype=np.float32), batch_size=256, shuffle=False
    )
    chunks = []
    for xb, _ in lloader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(lstm(xb).float().cpu())
    l_pred_s = torch.cat(chunks, dim=0).numpy()
    l_pred = scaler_y.inverse_transform(l_pred_s.reshape(-1, 1)).ravel()
    return y_true, g_pred, l_pred


def bootstrap_rmse_diff_ci(
    y_true: np.ndarray,
    g_pred: np.ndarray,
    l_pred: np.ndarray,
    n_resamples: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """95% CI for (RMSE_GNN - RMSE_LSTM) via paired bootstrap resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        gp = g_pred[idx]
        lp = l_pred[idx]
        rmse_g = float(np.sqrt(np.mean((yt - gp) ** 2)))
        rmse_l = float(np.sqrt(np.mean((yt - lp) ** 2)))
        diffs[b] = rmse_g - rmse_l
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def significance_and_bootstrap(
    h: int, device: torch.device, adjacency: torch.Tensor
) -> tuple[float, float, float, float]:
    if h == 1:
        y_true, g_pred, l_pred = paired_preds_h1(device, adjacency)
    else:
        y_true, g_pred, l_pred = paired_preds_multi(h, device, adjacency)

    err_g = (y_true - g_pred) ** 2
    err_l = (y_true - l_pred) ** 2
    # DM HAC lag = forecast_horizon - 1 (h=1 => lag 0, no serial correction)
    dm_p = diebold_mariano(err_g, err_l, h=h)
    tt_p = float(stats.ttest_rel(err_g, err_l).pvalue)
    ci_lo, ci_hi = bootstrap_rmse_diff_ci(y_true, g_pred, l_pred)
    print(
        f"[h={h}] DM p={dm_p:.6e}  paired-t p={tt_p:.6e}  "
        f"bootstrap CI=({ci_lo:.4f}, {ci_hi:.4f})  n={len(y_true)}",
        file=sys.stderr,
        flush=True,
    )
    return dm_p, tt_p, ci_lo, ci_hi


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def agg_rmse(store: dict[int, dict[str, float]]) -> tuple[float, float]:
    vals = np.array([store[s]["RMSE"] for s in SEEDS], dtype=np.float64)
    return float(vals.mean()), float(vals.std(ddof=1))


def main() -> None:
    device = require_cuda()
    adjacency = build_and_save_adjacency()

    lstm_metrics: dict[int, dict[int, dict[str, float]]] = {h: {} for h in ALL_HORIZONS}
    gnn_metrics: dict[int, dict[int, dict[str, float]]] = {h: {} for h in ALL_HORIZONS}

    # --- Part 1: train multi-seed for h=2,3,4 ---
    for h in TRAIN_HORIZONS:
        print(f"\n========== PART 1 — TRAIN h={h} ==========", file=sys.stderr, flush=True)
        for seed in SEEDS:
            lstm_metrics[h][seed] = train_lstm_seed(h, seed, device)
        for seed in SEEDS:
            gnn_metrics[h][seed] = train_gnn_seed(h, seed, adjacency, device)

    # --- h=1: evaluate existing v2 / gnn checkpoints (no training) ---
    print("\n========== h=1 — evaluate existing checkpoints ==========", file=sys.stderr, flush=True)
    for seed in SEEDS:
        lstm_metrics[1][seed] = eval_lstm_h1_seed(seed, device)
        gnn_metrics[1][seed] = eval_gnn_h1_seed(seed, adjacency, device)

    # --- Part 2: significance + bootstrap for all horizons ---
    results: dict[int, dict] = {}
    for h in ALL_HORIZONS:
        dm_p, tt_p, ci_lo, ci_hi = significance_and_bootstrap(h, device, adjacency)
        lr_m, lr_s = agg_rmse(lstm_metrics[h])
        gr_m, gr_s = agg_rmse(gnn_metrics[h])
        results[h] = {
            "lstm_rmse": (lr_m, lr_s),
            "gnn_rmse": (gr_m, gr_s),
            "dm_p": dm_p,
            "tt_p": tt_p,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
        }

    # --- Final tables (one per horizon) ---
    print()
    for h in ALL_HORIZONS:
        r = results[h]
        lr_m, lr_s = r["lstm_rmse"]
        gr_m, gr_s = r["gnn_rmse"]
        print(
            f"horizon={h} | "
            f"LSTM RMSE={fmt_ms(lr_m, lr_s)} | "
            f"GNN RMSE={fmt_ms(gr_m, gr_s)} | "
            f"DM p-value={r['dm_p']:.6e} | "
            f"paired-t p-value={r['tt_p']:.6e} | "
            f"bootstrap 95% CI=({r['ci_lo']:.4f}, {r['ci_hi']:.4f})"
        )


if __name__ == "__main__":
    main()
