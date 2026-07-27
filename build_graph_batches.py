"""
Build per-date, all-station graph-batch tensors for the GNN phase (Phase 7).

For each split, produces dense tensors over the 414-station node set:
  X_graph:    (n_dates, 414, 30, 8)  float32
  y_graph:    (n_dates, 414)         float32  (scaled, same space as LSTM)
  mask_graph: (n_dates, 414)         bool

Saves per split:
  data/processed/{X,y,mask}_{train,val,test}_graph.npy
  data/processed/graph_date_index_{split}.json   (date_idx -> ISO date string)

Node ordering reference: data/processed/station_id_to_index.json (unchanged).

Prints diagnostics only.
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


def replay_meta(df: pd.DataFrame) -> dict[str, list[dict]]:
    """Recover (station_id, target_date, raw_target) per sample in the SAME order
    as generate_sequences_v2 wrote X_*_v2.npy / y_*_v2.npy."""
    meta: dict[str, list[dict]] = {s: [] for s in SPLITS}
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
                split = split_name(target_date)
                if split is None:
                    continue
                x_seq = feats[i : i + SEQ_LEN]
                y_val = targets[target_idx]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
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
    split: str,
    rows: list[dict],
    X_v2: np.ndarray,
    y_v2: np.ndarray,
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
        X_graph[di, node] = X_v2[j]
        y_graph[di, node] = y_v2[j]
        mask_graph[di, node] = True

    return X_graph, y_graph, mask_graph, unique_dates


def main() -> None:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date_of_record"])
    df = df.sort_values(["station_id", "date_of_record"]).reset_index(drop=True)

    with open(INDEX_JSON, encoding="utf-8") as f:
        id_to_index = json.load(f)
    assert len(id_to_index) == N_NODES, f"index file has {len(id_to_index)} nodes"

    scaler_y = joblib.load(MODELS / "minmax_scaler_y_v2.joblib")
    meta = replay_meta(df)

    print("=== ALIGNMENT SANITY CHECKS (replay vs saved arrays) ===")
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in SPLITS:
        X_v2 = np.load(DATA / f"X_{split}_v2.npy")
        y_v2 = np.load(DATA / f"y_{split}_v2.npy")
        rows = meta[split]
        assert len(rows) == len(y_v2) == len(X_v2), (
            f"[{split}] replay {len(rows)} != X {len(X_v2)} / y {len(y_v2)}"
        )
        raw_target = np.array([r["raw_target"] for r in rows], dtype=np.float64)
        y_inv = scaler_y.inverse_transform(y_v2.reshape(-1, 1)).ravel()
        max_dev = float(np.max(np.abs(raw_target - y_inv)))
        assert max_dev < 1e-2, f"[{split}] y-value misalignment, max_dev={max_dev}"
        print(
            f"[{split}] length match OK ({len(rows)} samples); "
            f"y-value match OK (max |raw - inverse(scaled)| = {max_dev:.2e} mm)"
        )
        arrays[split] = (X_v2, y_v2)

    print("\n=== PER-SPLIT DIAGNOSTICS ===")
    for split in SPLITS:
        X_v2, y_v2 = arrays[split]
        X_graph, y_graph, mask_graph, unique_dates = build_split(
            split, meta[split], X_v2, y_v2, id_to_index
        )

        xf = DATA / f"X_{split}_graph.npy"
        yf = DATA / f"y_{split}_graph.npy"
        mf = DATA / f"mask_{split}_graph.npy"
        np.save(xf, X_graph)
        np.save(yf, y_graph)
        np.save(mf, mask_graph)
        with open(DATA / f"graph_date_index_{split}.json", "w", encoding="utf-8") as f:
            json.dump(unique_dates, f, indent=2)

        total_mb = sum(p.stat().st_size for p in (xf, yf, mf)) / (1024 ** 2)

        coverage = mask_graph.mean(axis=1) * 100.0  # % of 414 valid, per date
        # mask=False rows must be all-zero in X and y
        inv = ~mask_graph
        x_zero_ok = not X_graph[inv].any()
        y_zero_ok = not y_graph[inv].any()
        nan_ok = not (np.isnan(X_graph).any() or np.isnan(y_graph).any())

        print(f"\n[{split}]")
        print(f"  n_dates: {len(unique_dates)}")
        print(f"  X_graph shape:    {X_graph.shape}  ({X_graph.dtype})")
        print(f"  y_graph shape:    {y_graph.shape}  ({y_graph.dtype})")
        print(f"  mask_graph shape: {mask_graph.shape}  ({mask_graph.dtype})")
        print(f"  total file size on disk: {total_mb:.1f} MB")
        print(f"  mean mask coverage: {coverage.mean():.2f}% of {N_NODES} stations")
        print(f"  min/max coverage on any date: {coverage.min():.2f}% / {coverage.max():.2f}%")
        print(f"  mask=False rows all-zero in X_graph: {x_zero_ok}")
        print(f"  mask=False rows all-zero in y_graph: {y_zero_ok}")
        print(f"  no NaNs anywhere (X & y): {nan_ok}")

    print(f"\nNode ordering reference: {INDEX_JSON.name} (unchanged, {N_NODES} nodes)")


if __name__ == "__main__":
    main()
