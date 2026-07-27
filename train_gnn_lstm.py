"""
Phase 7c — GNN-LSTM (seed 42), denser date-steps.

- Global A (+ self-loops) -> models/adjacency_norm.pt (artifact)
- Train: filter dates with coverage >= 20% (>= 83 valid stations)
- Val/test: all dates
- Per-date masked A_norm inside the model (invalid stations isolated)
- Masked MSE loss / metrics on mask=True only
- Default batch_size=1 date/step (fallback to 2 if unstable)

Prints: steps/epoch, stopped_epoch, best_epoch, full-test metrics,
        dense/sparse 2x2 (GNN vs LSTM).
"""

from __future__ import annotations

import json
import math
import sys
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
from torch.utils.data import DataLoader, TensorDataset

from src.cuda_setup import make_loader, require_cuda, set_seed
from src.model import GNNLSTM, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
FIGURES = BASE / "reports" / "figures"

SEED = 42
BATCH_SIZE = 1  # dates per step (fallback to 2 if unstable)
LR = 1e-3
MAX_EPOCHS = 100
PATIENCE = 15
MIN_DELTA = 1e-5
GRAD_CLIP = 1.0
N_NODES = 414
COV_THRESH = 0.20
MIN_VALID = int(np.ceil(COV_THRESH * N_NODES))  # 83
DENSE_MIN = 207  # 50% of 414 for stratified diagnostic


def build_and_save_adjacency() -> torch.Tensor:
    """Build A (+I), save global A_norm, return A (with self-loops) for the model."""
    with open(DATA / "station_id_to_index.json", encoding="utf-8") as f:
        id_to_index = json.load(f)
    edges = pd.read_csv(DATA / "station_graph_edges.csv")

    a = np.zeros((N_NODES, N_NODES), dtype=np.float64)
    for _, row in edges.iterrows():
        i = id_to_index[row["source"]]
        j = id_to_index[row["target"]]
        a[i, j] = 1.0
    a = a + np.eye(N_NODES, dtype=np.float64)

    deg = a.sum(axis=1)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.clip(deg, 1e-12, None)))
    a_norm = d_inv_sqrt @ a @ d_inv_sqrt

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save(torch.from_numpy(a_norm.astype(np.float32)), MODELS / "adjacency_norm.pt")
    return torch.from_numpy(a.astype(np.float32))


def filter_train(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    counts = mask.sum(axis=1)
    keep = counts >= MIN_VALID
    return X[keep], y[keep], mask[keep], int(keep.sum())


def make_date_loader(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    shuffle: bool,
    batch_size: int | None = None,
) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(X),  # (D, N, T, F)
        torch.from_numpy(y),  # (D, N)
        torch.from_numpy(mask),  # (D, N) bool
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE if batch_size is None else batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
    )


def masked_mse(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(dtype=pred.dtype)
    denom = m.sum().clamp_min(1.0)
    return ((pred - y).pow(2) * m).sum() / denom


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n = torch.zeros((), device=device)
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        with autocast("cuda"):
            pred = model(xb, mb)
            m = mb.to(dtype=pred.dtype)
            total += ((pred - yb).pow(2) * m).sum()
            n += m.sum()
    return float((total / n.clamp_min(1.0)).item())


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    preds, ys, masks = [], [], []
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        mb_dev = mb.to(device, non_blocking=True)
        with autocast("cuda"):
            pred = model(xb, mb_dev).float()
        preds.append(pred.cpu().numpy())
        ys.append(yb.numpy())
        masks.append(mb.numpy())
    return np.concatenate(preds), np.concatenate(ys), np.concatenate(masks)


def metrics_mm(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def _unstable(history: dict[str, list[float]], n_check: int = 3) -> bool:
    """True if early train/val losses are NaN/Inf or explode vs epoch 1."""
    tr = history["train_loss"][:n_check]
    va = history["val_loss"][:n_check]
    if not tr:
        return False
    for v in tr + va:
        if not np.isfinite(v):
            return True
    if len(tr) >= 2 and tr[-1] > max(tr[0] * 50.0, 1.0):
        return True
    return False


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    *,
    log_epochs: bool = True,
) -> tuple[dict, int, int, float, dict]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    amp_scaler = GradScaler("cuda")
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    stopped_epoch = MAX_EPOCHS

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running = torch.zeros((), device=device)
        n_seen = torch.zeros((), device=device)
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                pred = model(xb, mb)
                loss = masked_mse(pred, yb, mb)
            if not torch.isfinite(loss):
                history["train_loss"].append(float("nan"))
                history["val_loss"].append(float("nan"))
                if log_epochs:
                    print(f"[monitor] epoch {epoch}: non-finite loss — aborting", file=sys.stderr)
                return history, epoch, best_epoch, best_val, best_state or {}
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            msum = mb.to(dtype=pred.dtype).sum()
            running += loss.detach() * msum
            n_seen += msum

        train_loss = float((running / n_seen.clamp_min(1.0)).item())
        val_loss = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if log_epochs and epoch <= 5:
            print(
                f"[monitor] epoch {epoch}: train={train_loss:.6f} val={val_loss:.6f}",
                file=sys.stderr,
                flush=True,
            )

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

        # After 3 epochs, bail early if unstable so we can fall back
        if epoch == 3 and _unstable(history):
            if log_epochs:
                print("[monitor] early epochs unstable — aborting for fallback", file=sys.stderr)
            break
    else:
        stopped_epoch = MAX_EPOCHS

    return history, stopped_epoch, best_epoch, best_val, best_state or {}


@torch.no_grad()
def stratified_metrics(
    gnn: nn.Module,
    Xg: np.ndarray,
    yg: np.ndarray,
    mg: np.ndarray,
    scaler_y,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Dense/sparse RMSE/MAE/R2 for GNN and Phase-6 LSTM on the same samples."""
    counts = mg.sum(axis=1)
    subsets = {
        "Dense": np.where(counts >= DENSE_MIN)[0],
        "Sparse": np.where(counts < DENSE_MIN)[0],
    }

    lstm = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    lckpt = torch.load(MODELS / "lstm_baseline_v2_seed42.pt", map_location=device, weights_only=False)
    lstm.load_state_dict(lckpt["model_state_dict"])
    lstm.eval()
    gnn.eval()

    out: dict[str, dict[str, float]] = {}

    for name, idxs in subsets.items():
        # GNN
        g_preds, g_trues = [], []
        for start in range(0, len(idxs), 8):
            sl = idxs[start : start + 8]
            xb = torch.from_numpy(Xg[sl]).to(device)
            mb = torch.from_numpy(mg[sl]).to(device)
            with autocast("cuda"):
                pred = gnn(xb, mb).float().cpu().numpy()
            valid = mg[sl].astype(bool)
            g_preds.append(pred[valid])
            g_trues.append(yg[sl][valid])
        gp = np.concatenate(g_preds)
        gt = np.concatenate(g_trues)
        gp_mm = scaler_y.inverse_transform(gp.reshape(-1, 1)).ravel()
        gt_mm = scaler_y.inverse_transform(gt.reshape(-1, 1)).ravel()
        out[f"{name}_GNN"] = metrics_mm(gt_mm, gp_mm)

        # LSTM on the same valid windows
        Xs = np.concatenate([Xg[di][mg[di]] for di in idxs], axis=0)
        ys = np.concatenate([yg[di][mg[di]] for di in idxs], axis=0)
        loader = make_loader(Xs, ys, batch_size=256, shuffle=False)
        chunks = []
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            with autocast("cuda"):
                chunks.append(lstm(xb).float().cpu())
        lp = torch.cat(chunks, dim=0).numpy()
        lp_mm = scaler_y.inverse_transform(lp.reshape(-1, 1)).ravel()
        lt_mm = scaler_y.inverse_transform(ys.reshape(-1, 1)).ravel()
        out[f"{name}_LSTM"] = metrics_mm(lt_mm, lp_mm)

    return out


def main() -> None:
    device = require_cuda()
    adjacency = build_and_save_adjacency()

    X_tr = np.load(DATA / "X_train_graph.npy")
    y_tr = np.load(DATA / "y_train_graph.npy")
    m_tr = np.load(DATA / "mask_train_graph.npy")
    X_va = np.load(DATA / "X_val_graph.npy")
    y_va = np.load(DATA / "y_val_graph.npy")
    m_va = np.load(DATA / "mask_val_graph.npy")
    X_te = np.load(DATA / "X_test_graph.npy")
    y_te = np.load(DATA / "y_test_graph.npy")
    m_te = np.load(DATA / "mask_test_graph.npy")

    X_tr, y_tr, m_tr, n_train_dates = filter_train(X_tr, y_tr, m_tr)
    val_loader = make_date_loader(X_va, y_va, m_va, shuffle=False, batch_size=8)
    test_loader = make_date_loader(X_te, y_te, m_te, shuffle=False, batch_size=8)

    used_bs = 1
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    stopped_epoch = 0
    best_epoch = 0
    best_val = float("inf")
    best_state: dict = {}
    model = GNNLSTM(adjacency=adjacency).to(device)

    for attempt_bs in (1, 2):
        used_bs = attempt_bs
        set_seed(SEED)
        model = GNNLSTM(adjacency=adjacency).to(device)
        train_loader = make_date_loader(
            X_tr, y_tr, m_tr, shuffle=True, batch_size=used_bs
        )
        print(f"[monitor] training with batch_size={used_bs}", file=sys.stderr, flush=True)
        history, stopped_epoch, best_epoch, best_val, best_state = train_loop(
            model, train_loader, val_loader, device
        )
        aborted_unstable = (
            len(history["train_loss"]) <= 3 and _unstable(history)
        ) or any(not np.isfinite(v) for v in history["train_loss"] + history["val_loss"])
        if not aborted_unstable and best_state:
            if used_bs != 1:
                print(f"[monitor] fell back to batch_size={used_bs}", file=sys.stderr)
            break
        if attempt_bs == 2:
            break
        print("[monitor] retrying with batch_size=2", file=sys.stderr, flush=True)

    if best_state:
        model.load_state_dict(best_state)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="train")
    ax.plot(range(1, len(history["val_loss"]) + 1), history["val_loss"], label="val")
    ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked MSE (scaled y)")
    ax.set_title(f"GNN-LSTM Training Curve (seed 42, batch_size={used_bs})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "gnn_lstm_training_curve.png", dpi=150)
    plt.close()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": best_val,
            "history": history,
            "seed": SEED,
            "n_train_dates_after_filter": n_train_dates,
            "min_valid_stations": MIN_VALID,
            "device": str(device),
            "batch_size": used_bs,
        },
        MODELS / "gnn_lstm_seed42.pt",
    )

    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")
    y_pred_s, y_true_s, mask = predict_all(model, test_loader, device)
    valid = mask.astype(bool)
    y_pred_mm = scaler_y.inverse_transform(y_pred_s[valid].reshape(-1, 1)).ravel()
    y_true_mm = scaler_y.inverse_transform(y_true_s[valid].reshape(-1, 1)).ravel()
    m = metrics_mm(y_true_mm, y_pred_mm)
    strat = stratified_metrics(model, X_te, y_te, m_te, scaler_y, device)

    steps_per_epoch = math.ceil(n_train_dates / used_bs)
    print(f"steps/epoch: {steps_per_epoch}  (batch_size={used_bs}, n_train_dates={n_train_dates})")
    print(f"stopped_epoch: {stopped_epoch}")
    print(f"best_epoch: {best_epoch}")
    print(f"MSE:  {m['MSE']:.4f}")
    print(f"RMSE: {m['RMSE']:.4f}")
    print(f"MAE:  {m['MAE']:.4f}")
    print(f"R2:   {m['R2']:.4f}")
    print()
    print(f"{'':10} {'GNN-LSTM':^28} {'LSTM':^28}")
    print(f"{'':10} {'RMSE':>8} {'MAE':>8} {'R2':>8}  {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    for subset in ("Dense", "Sparse"):
        g = strat[f"{subset}_GNN"]
        l = strat[f"{subset}_LSTM"]
        print(
            f"{subset:10} {g['RMSE']:8.4f} {g['MAE']:8.4f} {g['R2']:8.4f}  "
            f"{l['RMSE']:8.4f} {l['MAE']:8.4f} {l['R2']:8.4f}"
        )


if __name__ == "__main__":
    main()
