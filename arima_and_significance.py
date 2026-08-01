"""
Task 1 — ARIMA baseline (rolling 1-step-ahead) on a random 30-station sample.
Task 2 — Statistical significance of seed-42 LSTM vs persistence on the full test set.

Prints only:
  - ARIMA RMSE / MAE / R2 + n_stations_sampled
  - Diebold-Mariano p-value + paired t-test p-value

Run with the project CUDA venv:
  & "D:\\project\\Research Project\\.venv\\Scripts\\python.exe" arima_and_significance.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA

from src.cuda_setup import make_loader, require_cuda
from src.model import LSTMBaseline

warnings.filterwarnings("ignore")  # silence ARIMA convergence/stationarity chatter

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

SEQ_LEN = 30
RAIN_IDX = 5  # rainfall position in FEATURE_COLS (v2)
N_STATIONS = 30
SAMPLE_SEED = 42
ARIMA_ORDER = (2, 0, 2)

FEATURE_COLS = [
    "avg_temp", "min_temp", "max_temp", "wind_speed",
    "air_pressure", "rainfall", "doy_sin", "doy_cos",
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


def build_test_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild test-set rows in the SAME order as X_test_v2 / y_test_v2.

    Mirrors generate_sequences_v2.build_sequences exactly so row j here aligns
    with row j of the saved arrays. Also captures the persistence prediction
    (rainfall on the last day of the input window = day before target).
    """
    rows: list[dict] = []
    need = SEQ_LEN + 1
    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g["rainfall"].to_numpy(dtype=np.float64)
        n = len(g)
        if n < need:
            continue
        day_ints = dates.astype("datetime64[D]").astype(np.int64)
        breaks = np.where(np.diff(day_ints) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))
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
                rows.append(
                    {
                        "station_id": station_id,
                        "target_date": target_date,
                        "y_true": float(y_val),
                        "persistence": float(targets[target_idx - 1]),  # last window day
                    }
                )
    return pd.DataFrame(rows)


def lstm_predict_mm(y_test_scaled: np.ndarray, scaler_y) -> np.ndarray:
    """Inference with the seed-42 v2 checkpoint -> predictions in mm/day."""
    device = require_cuda()
    X_test = np.load(DATA / "X_test_v2.npy")
    loader = make_loader(X_test, y_test_scaled, batch_size=256, shuffle=False)

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    ckpt = torch.load(MODELS / "lstm_baseline_v2_seed42.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            chunks.append(model(xb).float().cpu())
    pred_scaled = torch.cat(chunks, dim=0).numpy()
    return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()


def diebold_mariano(e1_sq: np.ndarray, e2_sq: np.ndarray, h: int = 1) -> float:
    """Two-sided DM p-value with HAC/Newey-West (Bartlett) LRV and HLN correction.

    Loss differential d = e1_sq - e2_sq.
    Long-run variance uses Bartlett kernel with max lag L = h-1
    (weight w_k = 1 - k/h). For h=1 only lag-0 (gamma0) is used.
    """
    d = e1_sq - e2_sq
    n = d.size
    dbar = d.mean()
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        w = 1.0 - k / h  # Bartlett / Newey-West: 1 - k/(L+1), L=h-1
        lrv += 2.0 * w * gk
    lrv = max(float(lrv), float(gamma0) * 1e-12, 1e-18)
    dm = dbar / np.sqrt(lrv / n)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_star = dm * hln
    return float(2.0 * stats.t.cdf(-abs(dm_star), df=n - 1))


def rolling_arima_metrics(df: pd.DataFrame, test_rows: pd.DataFrame) -> dict:
    """Fit ARIMA per sampled station on TRAIN rainfall; rolling 1-step-ahead on test."""
    stations_with_test = test_rows["station_id"].unique()
    rng = np.random.default_rng(SAMPLE_SEED)
    sampled = rng.choice(stations_with_test, size=N_STATIONS, replace=False)

    # Per-station train rainfall series (chronological, train period only).
    train_df = df[df["date_of_record"] <= TRAIN_END]
    train_series = {
        sid: g.sort_values("date_of_record")["rainfall"].to_numpy(dtype=np.float64)
        for sid, g in train_df[train_df["station_id"].isin(sampled)].groupby("station_id")
    }

    all_true: list[float] = []
    all_pred: list[float] = []
    n_used = 0

    for sid in sampled:
        y_train = train_series.get(sid)
        st = test_rows[test_rows["station_id"] == sid].sort_values("target_date")
        if y_train is None or len(y_train) < 10 or st.empty:
            continue
        y_test_vals = st["y_true"].to_numpy(dtype=np.float64)
        try:
            res = ARIMA(
                y_train, order=ARIMA_ORDER,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit()
        except Exception:
            continue

        preds: list[float] = []
        ok = True
        for y_obs in y_test_vals:
            try:
                fc = np.asarray(res.forecast(steps=1)).ravel()[0]
                preds.append(float(fc))
                res = res.append([y_obs], refit=False)  # roll history forward
            except Exception:
                ok = False
                break
        if not ok or len(preds) != len(y_test_vals):
            continue

        all_true.extend(y_test_vals.tolist())
        all_pred.extend(preds)
        n_used += 1

    yt = np.asarray(all_true)
    yp = np.asarray(all_pred)
    mse = float(np.mean((yt - yp) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yt - yp)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "n_stations": n_used}


def main() -> None:
    df = pd.read_csv(DATA / "feature_engineered_v2.csv", parse_dates=["date_of_record"])
    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)

    test_rows = build_test_rows(df)

    # --- Task 2: significance (LSTM seed-42 vs persistence, full test set) ---
    y_test_scaled = np.load(DATA / "y_test_v2.npy")
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")
    y_true_mm = scaler_y.inverse_transform(y_test_scaled.reshape(-1, 1)).ravel()

    assert len(test_rows) == len(y_true_mm), (
        f"meta rows {len(test_rows)} != y_test {len(y_true_mm)}"
    )
    assert np.allclose(test_rows["y_true"].to_numpy(), y_true_mm, atol=1e-3), (
        "rebuilt test targets do not align with saved y_test_v2"
    )

    y_pred_lstm = lstm_predict_mm(y_test_scaled, scaler_y)
    y_pred_persist = test_rows["persistence"].to_numpy(dtype=np.float64)

    err_lstm_sq = (y_true_mm - y_pred_lstm) ** 2
    err_persist_sq = (y_true_mm - y_pred_persist) ** 2

    dm_p = diebold_mariano(err_lstm_sq, err_persist_sq, h=1)
    tt_p = float(stats.ttest_rel(err_lstm_sq, err_persist_sq).pvalue)

    # --- Task 1: ARIMA baseline ---
    arima = rolling_arima_metrics(df, test_rows)

    # --- Required output only ---
    print(f"ARIMA RMSE: {arima['rmse']:.4f}")
    print(f"ARIMA MAE:  {arima['mae']:.4f}")
    print(f"ARIMA R2:   {arima['r2']:.4f}")
    print(f"n_stations_sampled: {arima['n_stations']}")
    print(f"Diebold-Mariano p-value: {dm_p:.6e}")
    print(f"Paired t-test p-value:   {tt_p:.6e}")


if __name__ == "__main__":
    main()
