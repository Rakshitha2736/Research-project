"""
Multi-seed LSTM + GNN-LSTM for horizons h=2,3,4 (seeds 13, 42, 123).

- Reuses existing *_h{h}_seed42.pt checkpoints.
- Trains seeds 13 and 123 with identical configs:
    LSTM: batch=64, patience=30, Adam lr=1e-3, grad clip=1.0
    GNN:  batch_size=1 date/step, patience=30, per-date masked A_norm,
          train coverage>=20% filter on existing *_graph_h{h}.npy

Prints per-horizon mean±std and a final summary table with DM p-values
(GNN seed42 vs LSTM seed42 on the same masked test targets).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
from torch.amp import GradScaler, autocast

from arima_and_significance import diebold_mariano
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

HORIZONS = (2, 3, 4)
SEEDS = (13, 42, 123)
LSTM_BS = 64
GNN_BS = 1
LR = 1e-3
MAX_EPOCHS = 100
LSTM_PATIENCE = 30
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0


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

    if ckpt_path.exists() and seed == 42:
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

    if ckpt_path.exists() and seed == 42:
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


@torch.no_grad()
def significance_seed42(h: int, adjacency: torch.Tensor, device: torch.device) -> float:
    """DM p-value: GNN(seed42) vs LSTM(seed42) on same masked graph test samples."""
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")
    Xg = np.load(DATA / f"X_test_graph_h{h}.npy")
    yg = np.load(DATA / f"y_test_graph_h{h}.npy")
    mg = np.load(DATA / f"mask_test_graph_h{h}.npy")

    # --- GNN preds on graph batches ---
    gnn = GNNLSTM(adjacency=adjacency).to(device)
    gckpt = torch.load(MODELS / f"gnn_lstm_h{h}_seed42.pt", map_location=device, weights_only=False)
    gnn.load_state_dict(gckpt["model_state_dict"])
    gnn.eval()
    loader = make_date_loader(Xg, yg, mg, shuffle=False, batch_size=8)
    g_pred_s, g_true_s, mask = predict_all(gnn, loader, device)
    valid = mask.astype(bool)
    g_pred = scaler_y.inverse_transform(g_pred_s[valid].reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(g_true_s[valid].reshape(-1, 1)).ravel()

    # --- LSTM on the same valid windows, same date order ---
    lstm = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    lckpt = torch.load(MODELS / f"lstm_h{h}_seed42.pt", map_location=device, weights_only=False)
    lstm.load_state_dict(lckpt["model_state_dict"])
    lstm.eval()

    Xs = []
    for di in range(len(mg)):
        v = mg[di]
        if v.any():
            Xs.append(Xg[di][v])
    X_flat = np.concatenate(Xs, axis=0)
    assert len(X_flat) == len(y_true)

    lloader = make_loader(X_flat, np.zeros(len(X_flat), dtype=np.float32), batch_size=256, shuffle=False)
    chunks = []
    for xb, _ in lloader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(lstm(xb).float().cpu())
    l_pred_s = torch.cat(chunks, dim=0).numpy()
    l_pred = scaler_y.inverse_transform(l_pred_s.reshape(-1, 1)).ravel()

    err_g = (y_true - g_pred) ** 2
    err_l = (y_true - l_pred) ** 2
    dm_p = diebold_mariano(err_g, err_l, h=1)
    tt_p = float(stats.ttest_rel(err_g, err_l).pvalue)
    print(
        f"[h={h}] DM p={dm_p:.6e}  paired-t p={tt_p:.6e}  n={len(y_true)}",
        file=sys.stderr,
        flush=True,
    )
    return dm_p


def fmt_ms(mean: float, std: float) -> str:
    return f"{mean:.4f}±{std:.4f}"


def main() -> None:
    device = require_cuda()
    adjacency = build_and_save_adjacency()

    lstm_metrics: dict[int, dict[int, dict[str, float]]] = {h: {} for h in HORIZONS}
    gnn_metrics: dict[int, dict[int, dict[str, float]]] = {h: {} for h in HORIZONS}
    dm_pvals: dict[int, float] = {}

    for h in HORIZONS:
        print(f"\n========== HORIZON h={h} ==========", file=sys.stderr, flush=True)
        for seed in SEEDS:
            lstm_metrics[h][seed] = train_lstm_seed(h, seed, device)
        for seed in SEEDS:
            gnn_metrics[h][seed] = train_gnn_seed(h, seed, adjacency, device)

        def agg(store: dict[int, dict[str, float]], key: str) -> tuple[float, float]:
            vals = np.array([store[s][key] for s in SEEDS], dtype=np.float64)
            return float(vals.mean()), float(vals.std(ddof=1))

        print(f"\n--- h={h} mean±std (seeds {list(SEEDS)}) ---")
        for name, store in (("LSTM", lstm_metrics[h]), ("GNN-LSTM", gnn_metrics[h])):
            for key in ("RMSE", "MAE", "R2"):
                m, s = agg(store, key)
                print(f"  {name:8} {key}: {fmt_ms(m, s)}")

        dm_pvals[h] = significance_seed42(h, adjacency, device)

    # Final summary table
    print()
    print(
        f"{'h':>3}  {'LSTM RMSE':>14}  {'LSTM R2':>14}  "
        f"{'GNN RMSE':>14}  {'GNN R2':>14}  {'DM p':>12}"
    )
    print("-" * 82)
    for h in HORIZONS:
        lr = np.array([lstm_metrics[h][s]["RMSE"] for s in SEEDS])
        l2 = np.array([lstm_metrics[h][s]["R2"] for s in SEEDS])
        gr = np.array([gnn_metrics[h][s]["RMSE"] for s in SEEDS])
        g2 = np.array([gnn_metrics[h][s]["R2"] for s in SEEDS])
        print(
            f"{h:>3}  {fmt_ms(lr.mean(), lr.std(ddof=1)):>14}  "
            f"{fmt_ms(l2.mean(), l2.std(ddof=1)):>14}  "
            f"{fmt_ms(gr.mean(), gr.std(ddof=1)):>14}  "
            f"{fmt_ms(g2.mean(), g2.std(ddof=1)):>14}  "
            f"{dm_pvals[h]:.4e}"
        )


if __name__ == "__main__":
    main()
