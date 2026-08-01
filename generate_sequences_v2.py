"""
Phase 5 v2 — Sequences with past rainfall in X (8 features).

X uses rainfall on days 1..30 only; y is rainfall on day 31.
Explicitly asserts target-day date is never in the input window.

Saves *_v2.npy and minmax_scaler_v2.joblib / minmax_scaler_y_v2.joblib
without overwriting v1 artifacts.

Usage:
    python generate_sequences_v2.py
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
    "rainfall",
    "doy_sin",
    "doy_cos",
]
TARGET_COL = "rainfall"

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


def build_sequences(df: pd.DataFrame) -> tuple[dict[str, list], set[str], int]:
    buckets: dict[str, list] = {
        "train": {"X": [], "y": [], "meta": []},
        "val": {"X": [], "y": [], "meta": []},
        "test": {"X": [], "y": [], "meta": []},
    }
    used_stations: set[str] = set()
    leakage_checks = 0
    need = SEQ_LEN + 1

    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g[TARGET_COL].to_numpy(dtype=np.float64)

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
                window_dates = dates[i : i + SEQ_LEN]
                target_date = pd.Timestamp(dates[target_idx])

                # CRITICAL: target day must not appear in X window
                assert target_date not in {pd.Timestamp(d) for d in window_dates}, (
                    f"LEAKAGE: target date {target_date.date()} in window "
                    f"for station {station_id}"
                )
                # Calendar: window ends day before target
                assert pd.Timestamp(window_dates[-1]) + pd.Timedelta(days=1) == target_date
                leakage_checks += 1

                split = split_name(target_date)
                if split is None:
                    continue

                x_seq = feats[i : i + SEQ_LEN]
                y_val = targets[target_idx]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
                    continue

                # X rows are indices [i, i+SEQ_LEN); target is index i+SEQ_LEN
                assert x_seq.shape == (SEQ_LEN, len(FEATURE_COLS))

                buckets[split]["X"].append(x_seq)
                buckets[split]["y"].append(y_val)
                buckets[split]["meta"].append(
                    {
                        "station_id": station_id,
                        "target_date": str(target_date.date()),
                        "window_end_date": str(pd.Timestamp(window_dates[-1]).date()),
                    }
                )
                used_stations.add(station_id)

    return buckets, used_stations, leakage_checks


def stack_split(bucket: dict) -> tuple[np.ndarray, np.ndarray]:
    if not bucket["X"]:
        n_feat = len(FEATURE_COLS)
        return (
            np.empty((0, SEQ_LEN, n_feat), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    X = np.stack(bucket["X"]).astype(np.float32)
    y = np.asarray(bucket["y"], dtype=np.float32)
    return X, y


def scale_X(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    n_feat = X_train.shape[-1]
    scaler = MinMaxScaler()
    if len(X_train) == 0:
        raise RuntimeError("No training sequences — cannot fit X scaler.")
    scaler.fit(X_train.reshape(-1, n_feat))

    def transform(X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return X
        return scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype(np.float32)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def scale_y(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    y_train_s = scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)
    return y_train_s, y_val_s, y_test_s, scaler


def main() -> None:
    print("Loading", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, parse_dates=["date_of_record"])
    missing = [c for c in FEATURE_COLS + [TARGET_COL, "station_id"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)

    print("Building contiguous sequences (v2, rainfall in X)...")
    buckets, used_stations, leakage_checks = build_sequences(df)

    X_train, y_train = stack_split(buckets["train"])
    X_val, y_val = stack_split(buckets["val"])
    X_test, y_test = stack_split(buckets["test"])

    print(f"Leakage assertions passed on {leakage_checks:,} candidate windows")

    print("Fitting MinMaxScaler on train X only (8 features)...")
    X_train, X_val, X_test, scaler_x = scale_X(X_train, X_val, X_test)

    print("Fitting MinMaxScaler on train y only...")
    y_train, y_val, y_test, scaler_y = scale_y(y_train, y_val, y_test)

    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    np.save(DATA / "X_train_v2.npy", X_train)
    np.save(DATA / "X_val_v2.npy", X_val)
    np.save(DATA / "X_test_v2.npy", X_test)
    np.save(DATA / "y_train_v2.npy", y_train)
    np.save(DATA / "y_val_v2.npy", y_val)
    np.save(DATA / "y_test_v2.npy", y_test)
    joblib.dump(scaler_x, MODELS / "minmax_scaler_v2.joblib")
    joblib.dump(scaler_y, MODELS / "minmax_scaler_y_v2.joblib")

    meta_path = DATA / "sequence_metadata_v2.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "version": 2,
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
                "seq_len": SEQ_LEN,
                "rainfall_in_X": "days 1-30 only; target day excluded",
                "leakage_checks_passed": leakage_checks,
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
                    "X_train_v2": list(X_train.shape),
                    "X_val_v2": list(X_val.shape),
                    "X_test_v2": list(X_test.shape),
                    "y_train_v2": list(y_train.shape),
                    "y_val_v2": list(y_val.shape),
                    "y_test_v2": list(y_test.shape),
                },
            },
            f,
            indent=2,
        )

    print("\n=== VERIFICATION ===")
    print(f"n_stations_used: {len(used_stations)}")
    print(f"sequence counts: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")
    print(f"X_train_v2 shape: {X_train.shape}")
    print(f"X_val_v2 shape:   {X_val.shape}")
    print(f"X_test_v2 shape:  {X_test.shape}")
    print(f"y_train_v2 shape: {y_train.shape}")
    print(f"target-day leakage checks passed: {leakage_checks:,} (0 failures)")
    print(f"X NaNs: train={int(np.isnan(X_train).sum())} val={int(np.isnan(X_val).sum())} test={int(np.isnan(X_test).sum())}")
    print(f"y NaNs: train={int(np.isnan(y_train).sum())} val={int(np.isnan(y_val).sum())} test={int(np.isnan(y_test).sum())}")
    print(f"feature_cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"saved: minmax_scaler_v2.joblib, minmax_scaler_y_v2.joblib")
    print(f"meta:  {meta_path}")


if __name__ == "__main__":
    main()
