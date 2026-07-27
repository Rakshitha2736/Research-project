"""
Multi-seed GNN-LSTM at h=1 (seeds 13, 42, 123) + significance vs LSTM v2.

- Reuses gnn_lstm_seed42.pt (already trained, batch_size=1 config).
- Trains seeds 13 and 123 with identical config via train_gnn_lstm helpers.
- Full-test masked, inverse-transformed metrics per seed -> mean +/- std.
- Diebold-Mariano + paired t-test: GNN-LSTM(seed42) vs LSTM v2(seed42)
  per-sample squared errors on the SAME masked (station, date) targets.

Prints only: mean +/- std RMSE/MAE/R2, and the two p-values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats

from arima_and_significance import diebold_mariano
from build_graph_batches import replay_meta
from src.cuda_setup import make_loader, require_cuda, set_seed
from src.model import GNNLSTM, LSTMBaseline
from train_gnn_lstm import (
    build_and_save_adjacency,
    filter_train,
    make_date_loader,
    metrics_mm,
    predict_all,
    train_loop,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEEDS = [13, 42, 123]
TRAIN_BS = 1


def gnn_test_frame(model, test_loader, device, scaler_y) -> pd.DataFrame:
    """Per masked (station_id, target_date) -> y_true_mm, y_pred_mm for a GNN model."""
    with open(DATA / "station_id_to_index.json", encoding="utf-8") as f:
        id_to_index = json.load(f)
    idx_to_station = {v: k for k, v in id_to_index.items()}
    with open(DATA / "graph_date_index_test.json", encoding="utf-8") as f:
        date_list = json.load(f)

    pred_s, true_s, mask = predict_all(model, test_loader, device)
    valid = mask.astype(bool)
    rows_date, rows_node = np.where(valid)

    y_pred_mm = scaler_y.inverse_transform(pred_s[valid].reshape(-1, 1)).ravel()
    y_true_mm = scaler_y.inverse_transform(true_s[valid].reshape(-1, 1)).ravel()

    return pd.DataFrame(
        {
            "station_id": [idx_to_station[j] for j in rows_node],
            "target_date": [date_list[i] for i in rows_date],
            "y_true": y_true_mm,
            "gnn_pred": y_pred_mm,
        }
    )


def lstm_test_frame(device, scaler_y) -> pd.DataFrame:
    """Per (station_id, target_date) -> y_true_mm, y_pred_mm for LSTM v2 seed 42."""
    df = pd.read_csv(DATA / "feature_engineered_v2.csv", parse_dates=["date_of_record"])
    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)
    meta = replay_meta(df)["test"]  # aligned with X_test_v2 order

    X_test = np.load(DATA / "X_test_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    assert len(meta) == len(y_test) == len(X_test)

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    ckpt = torch.load(MODELS / "lstm_baseline_v2_seed42.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    loader = make_loader(X_test, y_test, batch_size=256, shuffle=False)
    chunks = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            chunks.append(model(xb).float().cpu())
    pred_s = torch.cat(chunks, dim=0).numpy()
    y_pred_mm = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()

    return pd.DataFrame(
        {
            "station_id": [r["station_id"] for r in meta],
            "target_date": [r["target_date"] for r in meta],
            "lstm_pred": y_pred_mm,
        }
    )


def main() -> None:
    device = require_cuda()
    adjacency = build_and_save_adjacency()
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")

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

    per_seed: dict[int, dict[str, float]] = {}
    gnn_seed42_model = None

    for seed in SEEDS:
        ckpt_path = MODELS / f"gnn_lstm_seed{seed}.pt"
        model = GNNLSTM(adjacency=adjacency).to(device)

        if seed == 42 and ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"[monitor] reusing seed 42 checkpoint", file=sys.stderr, flush=True)
        else:
            set_seed(seed)
            train_loader = make_date_loader(X_tr, y_tr, m_tr, shuffle=True, batch_size=TRAIN_BS)
            print(f"[monitor] training seed {seed} (bs={TRAIN_BS})", file=sys.stderr, flush=True)
            _, stopped, best, best_val, best_state = train_loop(
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
                    "seed": seed,
                    "n_train_dates_after_filter": n_train_dates,
                    "batch_size": TRAIN_BS,
                    "device": str(device),
                },
                ckpt_path,
            )
            print(f"[monitor] seed {seed} done: stopped={stopped} best={best}",
                  file=sys.stderr, flush=True)

        pred_s, true_s, mask = predict_all(model, test_loader, device)
        valid = mask.astype(bool)
        yp = scaler_y.inverse_transform(pred_s[valid].reshape(-1, 1)).ravel()
        yt = scaler_y.inverse_transform(true_s[valid].reshape(-1, 1)).ravel()
        per_seed[seed] = metrics_mm(yt, yp)

        if seed == 42:
            gnn_seed42_model = model

    # --- mean +/- std across seeds ---
    def agg(key: str) -> tuple[float, float]:
        vals = np.array([per_seed[s][key] for s in SEEDS], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=1))

    rmse_m, rmse_s = agg("RMSE")
    mae_m, mae_s = agg("MAE")
    r2_m, r2_s = agg("R2")

    # --- significance: GNN(seed42) vs LSTM(seed42) on same masked targets ---
    gnn_df = gnn_test_frame(gnn_seed42_model, test_loader, device, scaler_y)
    lstm_df = lstm_test_frame(device, scaler_y)
    merged = gnn_df.merge(
        lstm_df, on=["station_id", "target_date"], how="inner", validate="one_to_one"
    )
    assert len(merged) == len(gnn_df) == len(lstm_df), (
        f"alignment mismatch: gnn={len(gnn_df)} lstm={len(lstm_df)} merged={len(merged)}"
    )

    err_gnn_sq = (merged["y_true"] - merged["gnn_pred"]).to_numpy() ** 2
    err_lstm_sq = (merged["y_true"] - merged["lstm_pred"]).to_numpy() ** 2
    dm_p = diebold_mariano(err_gnn_sq, err_lstm_sq, h=1)
    tt_p = float(stats.ttest_rel(err_gnn_sq, err_lstm_sq).pvalue)

    print("Metric   mean +/- std (seeds 13, 42, 123)")
    print(f"RMSE   {rmse_m:.4f} +/- {rmse_s:.4f}")
    print(f"MAE    {mae_m:.4f} +/- {mae_s:.4f}")
    print(f"R2     {r2_m:.4f} +/- {r2_s:.4f}")
    print(f"n_matched_samples: {len(merged)}")
    print(f"Diebold-Mariano p-value (GNN vs LSTM, seed42): {dm_p:.6e}")
    print(f"Paired t-test p-value   (GNN vs LSTM, seed42): {tt_p:.6e}")


if __name__ == "__main__":
    main()
