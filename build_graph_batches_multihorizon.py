"""
Build per-date graph-batch tensors for horizons h=2,3,4 (Phase 7b logic).

Saves:
  data/processed/{X,y,mask}_{train,val,test}_graph_h{h}.npy
  data/processed/graph_date_index_{split}_h{h}.json

Does not modify h=1 *_graph.npy artifacts.
Prints alignment checks + train coverage>=20% survival counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
INPUT_CSV = DATA / "feature_engineered_v2.csv"
INDEX_JSON = DATA / "station_id_to_index.json"

SEQ_LEN = 30
N_NODES = 414
N_FEAT = 8
HORIZONS = (2, 3, 4)
MIN_VALID = int(np.ceil(0.20 * N_NODES))  # 83

FEATURE_COLS = [
    "avg_temp", "min_temp", "max_temp", "wind_speed",
    "air_pressure", "rainfall", "doy_sin", "doy_cos",
]
TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-02-10")
SPLITS = ("train", "val", "test")


def split_name(target_date: pd.Timestamp) -> str | None:
    if target_date <= TRAIN_END:
        return "train"
    if VAL_START <= target_date <= VAL_END:
        return "val"
    if TEST_START <= target_date <= TEST_END:
        return "test"
    return None


def replay_meta_horizon(df: pd.DataFrame, horizon: int) -> dict[str, list[dict]]:
    """Same order as generate_sequences_multihorizon for this horizon."""
    meta: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g["rainfall"].to_numpy(dtype=np.float64)
        n = len(g)
        if n < SEQ_LEN:
            continue

        day_ints_all = dates.astype("datetime64[D]").astype(np.int64)
        date_to_idx = {int(d): i for i, d in enumerate(day_ints_all)}
        breaks = np.where(np.diff(day_ints_all) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))

        for seg_start, seg_end in zip(starts, ends):
            if seg_end - seg_start < SEQ_LEN:
                continue
            for i in range(seg_start, seg_end - SEQ_LEN + 1):
                window_end = pd.Timestamp(dates[i + SEQ_LEN - 1])
                target_date = window_end + pd.Timedelta(days=horizon)
                target_day_int = int(np.datetime64(target_date, "D").astype(np.int64))
                target_idx = date_to_idx.get(target_day_int)
                if target_idx is None:
                    continue
                x_seq = feats[i : i + SEQ_LEN]
                y_val = targets[target_idx]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
                    continue
                split = split_name(target_date)
                if split is None:
                    continue
                meta[split].append(
                    {
                        "station_id": station_id,
                        "target_date": str(target_date.date()),
                        "raw_target": float(y_val),
                    }
                )
    return meta


def build_split(
    rows: list[dict],
    X_h: np.ndarray,
    y_h: np.ndarray,
    id_to_index: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    unique_dates = sorted({r["target_date"] for r in rows})
    date_to_idx = {d: i for i, d in enumerate(unique_dates)}
    n_dates = len(unique_dates)

    X_graph = np.zeros((n_dates, N_NODES, SEQ_LEN, N_FEAT), dtype=np.float32)
    y_graph = np.zeros((n_dates, N_NODES), dtype=np.float32)
    mask_graph = np.zeros((n_dates, N_NODES), dtype=bool)

    for j, r in enumerate(rows):
        node = id_to_index[r["station_id"]]
        di = date_to_idx[r["target_date"]]
        X_graph[di, node] = X_h[j]
        y_graph[di, node] = y_h[j]
        mask_graph[di, node] = True

    return X_graph, y_graph, mask_graph, unique_dates


def process_horizon(df: pd.DataFrame, h: int, id_to_index: dict[str, int]) -> None:
    print(f"\n========== GRAPH BATCHES h={h} ==========")
    scaler_y = joblib.load(MODELS / f"minmax_scaler_y_h{h}.joblib")
    meta = replay_meta_horizon(df, h)

    print("=== ALIGNMENT SANITY CHECKS ===")
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        X_h = np.load(DATA / f"X_{split}_h{h}.npy")
        y_h = np.load(DATA / f"y_{split}_h{h}.npy")
        rows = meta[split]
        assert len(rows) == len(y_h) == len(X_h), (
            f"[h={h} {split}] replay {len(rows)} != X {len(X_h)} / y {len(y_h)}"
        )
        raw = np.array([r["raw_target"] for r in rows], dtype=np.float64)
        y_inv = scaler_y.inverse_transform(y_h.reshape(-1, 1)).ravel()
        max_dev = float(np.max(np.abs(raw - y_inv)))
        assert max_dev < 1e-2, f"[h={h} {split}] y misalignment max_dev={max_dev}"
        print(
            f"[h={h} {split}] length match OK ({len(rows)}); "
            f"y-value match OK (max |raw-inv|={max_dev:.2e} mm)"
        )
        arrays[split] = (X_h, y_h)

    for split in SPLITS:
        X_h, y_h = arrays[split]
        Xg, yg, mg, dates = build_split(meta[split], X_h, y_h, id_to_index)
        np.save(DATA / f"X_{split}_graph_h{h}.npy", Xg)
        np.save(DATA / f"y_{split}_graph_h{h}.npy", yg)
        np.save(DATA / f"mask_{split}_graph_h{h}.npy", mg)
        with open(DATA / f"graph_date_index_{split}_h{h}.json", "w", encoding="utf-8") as f:
            json.dump(dates, f, indent=2)
        print(f"[h={h} {split}] X{Xg.shape} y{yg.shape} mask{mg.shape} n_dates={len(dates)}")

    m_tr = np.load(DATA / f"mask_train_graph_h{h}.npy")
    n_surv = int((m_tr.sum(axis=1) >= MIN_VALID).sum())
    print(
        f"[h={h}] train dates surviving coverage>={100*MIN_VALID/N_NODES:.0f}% "
        f"(>={MIN_VALID} stations): {n_surv} / {m_tr.shape[0]}"
    )


def main() -> None:
    with open(INDEX_JSON, encoding="utf-8") as f:
        id_to_index = json.load(f)
    assert len(id_to_index) == N_NODES

    df = pd.read_csv(INPUT_CSV, parse_dates=["date_of_record"])
    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)

    for h in HORIZONS:
        process_horizon(df, h, id_to_index)


if __name__ == "__main__":
    main()
