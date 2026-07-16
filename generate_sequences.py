"""
Phase 5 — Sequence generation for rainfall LSTM baseline
========================================================
- Group by station_id
- Build sequences only from naturally contiguous calendar days (no fill)
- SEQ_LEN=30 -> predict rainfall on day 31
- Chronological split by target date:
    Train: 2015-01-01 .. 2022-12-31
    Val:   2023-01-01 .. 2023-12-31
    Test:  2024-01-01 .. 2025-02-10
- MinMaxScaler fit on train X only
- X features exclude lat/lon/elevation (metadata only)

Usage (from RainfallPrediction/):
    python generate_sequences.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"

INPUT_CSV = DATA / "feature_engineered_v2.csv"
SEQ_LEN = 30
FEATURE_COLS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "doy_sin",
    "doy_cos",
]
TARGET_COL = "rainfall"
METADATA_COLS = ["station_id", "date_of_record", "latitude", "longitude", "elevation"]

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


def build_sequences(df: pd.DataFrame) -> tuple[dict[str, list], set[str]]:
    """Return dict of lists per split + set of station_ids that contributed any sequence."""
    buckets: dict[str, list] = {
        "train": {"X": [], "y": [], "meta": []},
        "val": {"X": [], "y": [], "meta": []},
        "test": {"X": [], "y": [], "meta": []},
    }
    used_stations: set[str] = set()
    need = SEQ_LEN + 1

    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g[TARGET_COL].to_numpy(dtype=np.float64)
        lats = g["latitude"].to_numpy(dtype=np.float64)
        lons = g["longitude"].to_numpy(dtype=np.float64)
        elevs = g["elevation"].to_numpy(dtype=np.float64)

        n = len(g)
        if n < need:
            continue

        day_ints = dates.astype("datetime64[D]").astype(np.int64)
        # Contiguous runs: break wherever calendar gap != 1 day
        breaks = np.where(np.diff(day_ints) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))

        for seg_start, seg_end in zip(starts, ends):
            seg_len = seg_end - seg_start
            if seg_len < need:
                continue
            # Every starting index in [seg_start, seg_end - need] is a valid window
            for i in range(seg_start, seg_end - SEQ_LEN):
                target_idx = i + SEQ_LEN
                target_date = pd.Timestamp(dates[target_idx])
                split = split_name(target_date)
                if split is None:
                    continue

                x_seq = feats[i : i + SEQ_LEN]
                y_val = targets[target_idx]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
                    continue

                buckets[split]["X"].append(x_seq)
                buckets[split]["y"].append(y_val)
                buckets[split]["meta"].append(
                    {
                        "station_id": station_id,
                        "target_date": str(target_date.date()),
                        "latitude": float(lats[target_idx]),
                        "longitude": float(lons[target_idx]),
                        "elevation": float(elevs[target_idx]),
                    }
                )
                used_stations.add(station_id)

    return buckets, used_stations


def stack_split(bucket: dict) -> tuple[np.ndarray, np.ndarray, list]:
    if not bucket["X"]:
        n_feat = len(FEATURE_COLS)
        return (
            np.empty((0, SEQ_LEN, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            [],
        )
    X = np.stack(bucket["X"]).astype(np.float32)
    y = np.asarray(bucket["y"], dtype=np.float32)
    return X, y, bucket["meta"]


def scale_features(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    n_feat = X_train.shape[-1]
    scaler = MinMaxScaler()
    if len(X_train) == 0:
        raise RuntimeError("No training sequences — cannot fit scaler.")

    # Fit on all timesteps of train, flattened
    scaler.fit(X_train.reshape(-1, n_feat))

    def transform(X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return X
        shape = X.shape
        return scaler.transform(X.reshape(-1, n_feat)).reshape(shape).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def main() -> None:
    print("Loading", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, parse_dates=["date_of_record"])
    missing = [c for c in FEATURE_COLS + [TARGET_COL, "station_id"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)

    print("Building contiguous sequences...")
    buckets, used_stations = build_sequences(df)

    X_train, y_train, meta_train = stack_split(buckets["train"])
    X_val, y_val, meta_val = stack_split(buckets["val"])
    X_test, y_test, meta_test = stack_split(buckets["test"])

    print("Fitting MinMaxScaler on train only...")
    X_train, X_val, X_test, scaler = scale_features(X_train, X_val, X_test)

    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    np.save(DATA / "X_train.npy", X_train)
    np.save(DATA / "X_val.npy", X_val)
    np.save(DATA / "X_test.npy", X_test)
    np.save(DATA / "y_train.npy", y_train)
    np.save(DATA / "y_val.npy", y_val)
    np.save(DATA / "y_test.npy", y_test)
    joblib.dump(scaler, MODELS / "minmax_scaler.joblib")

    # Optional metadata for GNN / debugging (not model inputs)
    meta_path = DATA / "sequence_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
                "seq_len": SEQ_LEN,
                "splits": {
                    "train": "2015-01-01 .. 2022-12-31",
                    "val": "2023-01-01 .. 2023-12-31",
                    "test": "2024-01-01 .. 2025-02-10",
                },
                "n_stations_used": len(used_stations),
                "counts": {
                    "train": int(len(y_train)),
                    "val": int(len(y_val)),
                    "test": int(len(y_test)),
                },
                "shapes": {
                    "X_train": list(X_train.shape),
                    "X_val": list(X_val.shape),
                    "X_test": list(X_test.shape),
                    "y_train": list(y_train.shape),
                    "y_val": list(y_val.shape),
                    "y_test": list(y_test.shape),
                },
            },
            f,
            indent=2,
        )

    def nan_report(name: str, arr: np.ndarray) -> str:
        return f"{name} NaNs: {int(np.isnan(arr).sum())}"

    print("\n=== VERIFICATION ===")
    print(f"n_stations_used: {len(used_stations)}")
    print(f"sequence counts: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_val shape:   {y_val.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(nan_report("X_train", X_train))
    print(nan_report("X_val", X_val))
    print(nan_report("X_test", X_test))
    print(nan_report("y_train", y_train))
    print(nan_report("y_val", y_val))
    print(nan_report("y_test", y_test))
    print(f"feature_cols: {FEATURE_COLS}")
    print(f"scaler saved: {MODELS / 'minmax_scaler.joblib'}")
    print(f"meta saved:   {meta_path}")


if __name__ == "__main__":
    main()
