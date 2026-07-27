"""
Train GNN-LSTM for horizons h=2,3,4 (seed 42).

Config matches densest h=1 run: per-date masked A_norm, train coverage>=20%,
batch_size=1 date/step, patience=30, Adam lr=1e-3, grad clip=1.0.

Requires *_graph_h{h}.npy from build_graph_batches_multihorizon.py.

Prints: horizon | n_train_dates_after_filter | stopped | best | RMSE | MAE | R2
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from src.cuda_setup import require_cuda, set_seed
from src.model import GNNLSTM
from train_gnn_lstm import (
    build_and_save_adjacency,
    evaluate,
    filter_train,
    make_date_loader,
    masked_mse,
    metrics_mm,
    predict_all,
    train_loop,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEED = 42
TRAIN_BS = 1
HORIZONS = (2, 3, 4)
# Override module patience used inside train_loop
import train_gnn_lstm as _tgl

_tgl.PATIENCE = 30


def train_one(h: int, adjacency: torch.Tensor, device: torch.device) -> dict:
    set_seed(SEED)
    X_tr = np.load(DATA / f"X_train_graph_h{h}.npy")
    y_tr = np.load(DATA / f"y_train_graph_h{h}.npy")
    m_tr = np.load(DATA / f"mask_train_graph_h{h}.npy")
    X_va = np.load(DATA / f"X_val_graph_h{h}.npy")
    y_va = np.load(DATA / f"y_val_graph_h{h}.npy")
    m_va = np.load(DATA / f"mask_val_graph_h{h}.npy")
    X_te = np.load(DATA / f"X_test_graph_h{h}.npy")
    y_te = np.load(DATA / f"y_test_graph_h{h}.npy")
    m_te = np.load(DATA / f"mask_test_graph_h{h}.npy")

    X_tr, y_tr, m_tr, n_train_dates = filter_train(X_tr, y_tr, m_tr)
    train_loader = make_date_loader(X_tr, y_tr, m_tr, shuffle=True, batch_size=TRAIN_BS)
    val_loader = make_date_loader(X_va, y_va, m_va, shuffle=False, batch_size=8)
    test_loader = make_date_loader(X_te, y_te, m_te, shuffle=False, batch_size=8)

    model = GNNLSTM(adjacency=adjacency).to(device)
    print(
        f"[monitor] h={h} training bs={TRAIN_BS} n_train_dates={n_train_dates} patience=30",
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
            "seed": SEED,
            "horizon": h,
            "n_train_dates_after_filter": n_train_dates,
            "batch_size": TRAIN_BS,
            "patience": 30,
            "device": str(device),
        },
        MODELS / f"gnn_lstm_h{h}_seed42.pt",
    )

    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")
    pred_s, true_s, mask = predict_all(model, test_loader, device)
    valid = mask.astype(bool)
    yp = scaler_y.inverse_transform(pred_s[valid].reshape(-1, 1)).ravel()
    yt = scaler_y.inverse_transform(true_s[valid].reshape(-1, 1)).ravel()
    m = metrics_mm(yt, yp)
    print(
        f"[monitor] h={h} done stopped={stopped} best={best} RMSE={m['RMSE']:.4f}",
        file=sys.stderr,
        flush=True,
    )
    return {
        "horizon": h,
        "n_train_dates_after_filter": n_train_dates,
        "stopped_epoch": stopped,
        "best_epoch": best,
        **m,
    }


def main() -> None:
    device = require_cuda()
    adjacency = build_and_save_adjacency()
    rows = [train_one(h, adjacency, device) for h in HORIZONS]

    print(
        f"{'horizon':>7}  {'n_train':>7}  {'stopped':>7}  {'best':>5}  "
        f"{'RMSE':>8}  {'MAE':>8}  {'R2':>8}"
    )
    print("-" * 66)
    for r in rows:
        print(
            f"{r['horizon']:>7}  {r['n_train_dates_after_filter']:>7}  "
            f"{r['stopped_epoch']:>7}  {r['best_epoch']:>5}  "
            f"{r['RMSE']:8.4f}  {r['MAE']:8.4f}  {r['R2']:8.4f}"
        )


if __name__ == "__main__":
    main()
