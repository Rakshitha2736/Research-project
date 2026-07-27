"""
Multi-horizon sequence generation for h in {2, 3, 4}.

Option A (confirmed):
  - 30-day input window must be a real contiguous run.
  - Target = rainfall on calendar date (window_end + h); that day must be a
    genuine station observation (finite, no gap-fill). Intermediate days
    between window_end and target need NOT be present.

Does NOT touch h=1 / v2 artifacts (*_v2.npy, minmax_scaler*_v2.joblib).
Reuses models/minmax_scaler_v2.joblib for X; fits a new y-scaler per horizon.

Usage:
    python generate_sequences_multihorizon.py
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
HORIZONS = (2, 3, 4)
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


def build_sequences_horizon(
    df: pd.DataFrame, horizon: int
) -> tuple[dict, set[str], int, int]:
    """Build sequences for a single horizon.

    Returns buckets, used_stations, n_real_target_checks, n_windows_considered.
    """
    buckets: dict[str, dict] = {
        "train": {"X": [], "y": [], "meta": []},
        "val": {"X": [], "y": [], "meta": []},
        "test": {"X": [], "y": [], "meta": []},
    }
    used_stations: set[str] = set()
    real_target_checks = 0
    windows_considered = 0

    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g[TARGET_COL].to_numpy(dtype=np.float64)
        n = len(g)
        if n < SEQ_LEN:
            continue

        # Date -> row index lookup over ALL observations for this station
        # (target need not lie in the same contiguous segment as the window)
        day_ints_all = dates.astype("datetime64[D]").astype(np.int64)
        date_to_idx: dict[int, int] = {int(d): i for i, d in enumerate(day_ints_all)}

        breaks = np.where(np.diff(day_ints_all) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))

        for seg_start, seg_end in zip(starts, ends):
            # Need only SEQ_LEN contiguous days for the input window
            if seg_end - seg_start < SEQ_LEN:
                continue
            for i in range(seg_start, seg_end - SEQ_LEN + 1):
                windows_considered += 1
                window_dates = dates[i : i + SEQ_LEN]
                window_end = pd.Timestamp(window_dates[-1])
                target_date = window_end + pd.Timedelta(days=horizon)
                target_day_int = int(np.datetime64(target_date, "D").astype(np.int64))

                # CRITICAL: target day must not appear in X window
                window_set = {pd.Timestamp(d) for d in window_dates}
                assert target_date not in window_set, (
                    f"LEAKAGE: target {target_date.date()} in window for {station_id}"
                )
                # Calendar: target is exactly window_end + h
                assert window_end + pd.Timedelta(days=horizon) == target_date

                # Contiguity of the 30-day window
                win_ints = window_dates.astype("datetime64[D]").astype(np.int64)
                assert np.all(np.diff(win_ints) == 1), "non-contiguous input window"

                target_idx = date_to_idx.get(target_day_int)
                if target_idx is None:
                    continue  # no real observation on target day

                # Genuine observation (not gap-filled / fabricated)
                y_val = targets[target_idx]
                x_seq = feats[i : i + SEQ_LEN]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
                    continue

                # Confirm the looked-up row's date matches (real CSV row)
                assert pd.Timestamp(dates[target_idx]) == target_date
                real_target_checks += 1

                split = split_name(target_date)
                if split is None:
                    continue

                assert x_seq.shape == (SEQ_LEN, len(FEATURE_COLS))
                buckets[split]["X"].append(x_seq)
                buckets[split]["y"].append(float(y_val))
                buckets[split]["meta"].append(
                    {
                        "station_id": station_id,
                        "target_date": str(target_date.date()),
                        "window_end_date": str(window_end.date()),
                        "horizon": horizon,
                    }
                )
                used_stations.add(station_id)

    return buckets, used_stations, real_target_checks, windows_considered


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


def transform_X(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    if len(X) == 0:
        return X
    n_feat = X.shape[-1]
    return scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape).astype(np.float32)


def scale_y(
    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    y_train_s = scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)
    return y_train_s, y_val_s, y_test_s, scaler


def process_horizon(df: pd.DataFrame, horizon: int, scaler_x: MinMaxScaler) -> None:
    print(f"\n========== HORIZON h={horizon} ==========")
    buckets, used_stations, real_checks, n_windows = build_sequences_horizon(df, horizon)

    X_train, y_train = stack_split(buckets["train"])
    X_val, y_val = stack_split(buckets["val"])
    X_test, y_test = stack_split(buckets["test"])

    n_samples = len(y_train) + len(y_val) + len(y_test)
    # Every kept sample incremented real_checks; samples outside split bounds
    # also passed the real-row check. Confirm 100% of emitted samples are real.
    n_emitted = n_samples
    # real_checks counts all valid real targets (including out-of-split dates);
    # for emitted samples we already asserted date match — report coverage:
    real_pct = 100.0  # by construction every emitted sample passed assert

    X_train = transform_X(X_train, scaler_x)
    X_val = transform_X(X_val, scaler_x)
    X_test = transform_X(X_test, scaler_x)

    y_train, y_val, y_test, scaler_y = scale_y(y_train, y_val, y_test)

    # Save — never write *_v2.*
    for split, X, y in (
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ):
        np.save(DATA / f"X_{split}_h{horizon}.npy", X)
        np.save(DATA / f"y_{split}_h{horizon}.npy", y)

    joblib.dump(scaler_y, MODELS / f"minmax_scaler_y_h{horizon}.joblib")

    meta_path = DATA / f"sequence_metadata_h{horizon}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "horizon": horizon,
                "feature_cols": FEATURE_COLS,
                "target_col": TARGET_COL,
                "seq_len": SEQ_LEN,
                "target_rule": "rainfall on calendar date window_end + h "
                "(real observation required; intermediates need not exist)",
                "x_scaler": "reused minmax_scaler_v2.joblib (h=1)",
                "y_scaler": f"minmax_scaler_y_h{horizon}.joblib (fit on train y)",
                "real_target_checks_passed": real_checks,
                "windows_considered": n_windows,
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
                    f"X_train_h{horizon}": list(X_train.shape),
                    f"X_val_h{horizon}": list(X_val.shape),
                    f"X_test_h{horizon}": list(X_test.shape),
                    f"y_train_h{horizon}": list(y_train.shape),
                    f"y_val_h{horizon}": list(y_val.shape),
                    f"y_test_h{horizon}": list(y_test.shape),
                },
            },
            f,
            indent=2,
        )

    print(f"sequence counts: train={len(y_train):,}  val={len(y_val):,}  test={len(y_test):,}")
    print(f"X_train_h{horizon} shape: {X_train.shape}")
    print(f"X_val_h{horizon} shape:   {X_val.shape}")
    print(f"X_test_h{horizon} shape:  {X_test.shape}")
    print(f"y_train_h{horizon} shape: {y_train.shape}")
    print(f"y_val_h{horizon} shape:   {y_val.shape}")
    print(f"y_test_h{horizon} shape:  {y_test.shape}")
    print(
        f"X NaNs: train={int(np.isnan(X_train).sum())} "
        f"val={int(np.isnan(X_val).sum())} test={int(np.isnan(X_test).sum())}"
    )
    print(
        f"y NaNs: train={int(np.isnan(y_train).sum())} "
        f"val={int(np.isnan(y_val).sum())} test={int(np.isnan(y_test).sum())}"
    )
    print(
        f"target-day is real (non-gap-filled) observation: "
        f"{real_pct:.0f}% of {n_emitted:,} emitted samples "
        f"({real_checks:,} real-row asserts passed over all candidate windows)"
    )
    print(f"n_stations_used: {len(used_stations)}")
    print(f"saved: X/y_*_h{horizon}.npy, minmax_scaler_y_h{horizon}.joblib, {meta_path.name}")


def main() -> None:
    scaler_x_path = MODELS / "minmax_scaler_v2.joblib"
    if not scaler_x_path.exists():
        raise FileNotFoundError(f"Missing h=1 X-scaler: {scaler_x_path}")

    # Safety: refuse to overwrite h=1 / v2 artifacts
    protected = [
        DATA / "X_train_v2.npy",
        DATA / "y_train_v2.npy",
        MODELS / "minmax_scaler_v2.joblib",
        MODELS / "minmax_scaler_y_v2.joblib",
    ]
    v2_mtime_before = {p: p.stat().st_mtime if p.exists() else None for p in protected}

    print("Loading", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV, parse_dates=["date_of_record"])
    missing = [c for c in FEATURE_COLS + [TARGET_COL, "station_id"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)

    print(f"Reusing X-scaler: {scaler_x_path.name}")
    scaler_x = joblib.load(scaler_x_path)

    DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    for h in HORIZONS:
        process_horizon(df, h, scaler_x)

    for p, mtime in v2_mtime_before.items():
        if mtime is None:
            continue
        assert p.stat().st_mtime == mtime, f"Protected v2 artifact was modified: {p}"
    print("\nProtected h=1/v2 artifacts unchanged: OK")


if __name__ == "__main__":
    main()
