"""Persistence baseline: last-window-day rainfall as multi-horizon forecast."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

RAINFALL_FEATURE_IDX = 5
HORIZONS_DEFAULT = (1, 2, 3, 4)


def data_paths(horizon: int, base: Path | None = None) -> dict[str, Path]:
    """Test arrays and scalers for one forecast horizon."""
    root = base or Path(__file__).resolve().parent.parent
    data = root / "data" / "processed"
    models = root / "models"
    if horizon == 1:
        return {
            "X_test": data / "X_test_v2.npy",
            "y_test": data / "y_test_v2.npy",
            "scaler_y": models / "minmax_scaler_y_v2.joblib",
            "scaler_x": models / "minmax_scaler_v2.joblib",
        }
    return {
        "X_test": data / f"X_test_h{horizon}.npy",
        "y_test": data / f"y_test_h{horizon}.npy",
        "scaler_y": models / f"minmax_scaler_y_h{horizon}.joblib",
        "scaler_x": models / "minmax_scaler_v2.joblib",
    }


def persistence_mm(
    X: np.ndarray,
    y_scaled: np.ndarray,
    scaler_x,
    scaler_y,
) -> tuple[np.ndarray, np.ndarray]:
    """Persistence: last window day's rainfall (feature index 5) in mm."""
    last_day = X[:, -1, :]  # (N, 8) scaled
    last_mm = scaler_x.inverse_transform(last_day)[:, RAINFALL_FEATURE_IDX]
    y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    return y_true, last_mm.astype(np.float64)


def metrics_mm(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


def eval_persistence_horizon(horizon: int, base: Path | None = None) -> dict[str, float]:
    """Load test split and return persistence regression metrics in mm/day."""
    paths = data_paths(horizon, base)
    X_test = np.load(paths["X_test"])
    y_test = np.load(paths["y_test"])
    scaler_x = joblib.load(paths["scaler_x"])
    scaler_y = joblib.load(paths["scaler_y"])
    y_true, y_pred = persistence_mm(X_test, y_test, scaler_x, scaler_y)
    return metrics_mm(y_true, y_pred)


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    print("=== Persistence baseline (src.persistence_baseline) ===")
    for h in HORIZONS_DEFAULT:
        m = eval_persistence_horizon(h, base)
        print(
            f"h={h}: RMSE={m['RMSE']:.6f} MAE={m['MAE']:.6f} "
            f"MSE={m['MSE']:.6f} R2={m['R2']:.6f}"
        )
