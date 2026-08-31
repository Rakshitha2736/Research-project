"""
Refresh lstm_baseline_v2_seed42_metrics.json without retraining.

Recomputes test_metrics_mm and persistence_baseline_mm from the existing
checkpoint using the canonical eval path (CUDA+autocast + src.persistence_baseline).

Usage (from RainfallPrediction/, project CUDA venv):
    python refresh_lstm_baseline_v2_metrics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.model import LSTMBaseline
from src.persistence_baseline import eval_persistence_horizon, metrics_mm

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
METRICS_PATH = MODELS / "lstm_baseline_v2_seed42_metrics.json"
CKPT_PATH = MODELS / "lstm_baseline_v2_seed42.pt"


@torch.no_grad()
def lstm_predict_mm(device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    X_test = np.load(DATA / "X_test_v2.npy")
    y_test = np.load(DATA / "y_test_v2.npy")
    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")
    loader = make_loader(X_test, y_test, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)

    model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(model(xb).float())
    pred_scaled = torch.cat(chunks, dim=0).cpu().numpy()
    y_pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    return y_true, y_pred


def _diff_keys(before: dict, after: dict) -> None:
    all_keys = sorted(set(before) | set(after))
    print("\n=== JSON before/after diff ===")
    for key in all_keys:
        b = before.get(key, "<missing>")
        a = after.get(key, "<missing>")
        if b == a:
            continue
        print(f"  {key}:")
        print(f"    before: {b}")
        print(f"    after:  {a}")


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(METRICS_PATH)
    if not CKPT_PATH.exists():
        raise FileNotFoundError(CKPT_PATH)

    before = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print("=== Before (LSTM test_metrics_mm) ===")
    print(json.dumps(before.get("test_metrics_mm"), indent=2))
    print(f"persistence_baseline_rmse: {before.get('persistence_baseline_rmse', '<missing>')}")
    print(f"persistence_baseline_mm: {before.get('persistence_baseline_mm', '<missing>')}")

    device = require_cuda()
    y_true, y_pred = lstm_predict_mm(device)
    lstm_m = metrics_mm(y_true, y_pred)
    persistence_m = eval_persistence_horizon(1, BASE)

    ref_rmse = before["test_metrics_mm"]["RMSE"]
    if abs(lstm_m["RMSE"] - ref_rmse) > 0.001:
        raise SystemExit(
            f"STOP: LSTM RMSE drift {lstm_m['RMSE']:.6f} vs JSON {ref_rmse:.6f} "
            "(tolerance 0.001 mm/day)"
        )

    after = dict(before)
    after["test_metrics_mm"] = lstm_m
    after["persistence_baseline_mm"] = persistence_m
    after.pop("persistence_baseline_rmse", None)

    METRICS_PATH.write_text(json.dumps(after, indent=2), encoding="utf-8")

    print("\n=== After (LSTM test_metrics_mm) ===")
    print(json.dumps(after["test_metrics_mm"], indent=2))
    print(f"persistence_baseline_mm: {json.dumps(after['persistence_baseline_mm'], indent=2)}")
    _diff_keys(before, after)
    print("\nLSTM metrics unchanged within tolerance; persistence now genuinely computed.")
    print(f"Wrote {METRICS_PATH}")


if __name__ == "__main__":
    main()
