"""
PHASE 0 — Offline forecast_cache for the Historical Rainfall Forecasting Dashboard.

One-time CUDA+autocast seed=42 inference for LSTM, CNN-LSTM-Temporal, and
CNN-LSTM+Attention at h=1..4. Reuses Features 3–7 provenance:
  - require_cuda + autocast
  - rebuild_test_meta() for station_id / target_date
  - existing MinMax scalers for inverse-transform
  - attention_weights_h{1..4}_seed42.npy reused (no attention re-inference)

Writes ONLY:
  reports/dashboard_data/forecast_cache.parquet
  reports/dashboard_data/station_metadata.parquet

Does NOT retrain. Does NOT modify any verified result table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.eval_attention import (
    TEST_END,
    TEST_START,
    paths_for_horizon,
    rebuild_test_meta,
)
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
TABLES = BASE / "reports" / "tables"
OUT_DIR = BASE / "reports" / "dashboard_data"
FEAT_CSV = DATA / "feature_engineered_v2.csv"
ABLATION_CSV = TABLES / "ablation_study.csv"
LSTM_H1_METRICS = MODELS / "lstm_baseline_v2_seed42_metrics.json"

FORECAST_PQ = OUT_DIR / "forecast_cache.parquet"
STATION_PQ = OUT_DIR / "station_metadata.parquet"

SEED = 42
HORIZONS = (1, 2, 3, 4)
BATCH_SIZE = DEFAULT_BATCH_SIZE
SPOT_N = 5
SPOT_SEED = 42
RMSE_ATOL = 1e-3
PRED_ATOL = 1e-4

MODEL_SPECS = (
    ("LSTM", "lstm"),
    ("CNN-LSTM-Temporal", "temporal"),
    ("CNN-LSTM+Attention", "attention"),
)

VERIFIED = [
    TABLES / "master_results.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
    TABLES / "station_wise_error.csv",
    TABLES / "seasonal_performance.csv",
    TABLES / "rain_classification_metrics.csv",
    TABLES / "extreme_rainfall_evaluation.csv",
    TABLES / "attention_extreme_vs_normal.csv",
]

STATION_COLS = [
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "state",
    "district",
    "elevation",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_paths(horizon: int) -> dict[str, Path]:
    if horizon == 1:
        return {
            "X_test": DATA / "X_test_v2.npy",
            "y_test": DATA / "y_test_v2.npy",
            "scaler_y": MODELS / "minmax_scaler_y_v2.joblib",
        }
    return {
        "X_test": DATA / f"X_test_h{horizon}.npy",
        "y_test": DATA / f"y_test_h{horizon}.npy",
        "scaler_y": MODELS / f"minmax_scaler_y_h{horizon}.joblib",
    }


def ckpt_path(model_key: str, horizon: int) -> Path:
    if model_key == "attention":
        return MODELS / f"cnn_lstm_attention_h{horizon}_seed{SEED}.pt"
    if model_key == "temporal":
        return MODELS / f"cnn_lstm_temporal_h{horizon}_seed{SEED}.pt"
    if model_key == "lstm":
        if horizon == 1:
            return MODELS / f"lstm_baseline_v2_seed{SEED}.pt"
        return MODELS / f"lstm_h{horizon}_seed{SEED}.pt"
    raise ValueError(model_key)


def attn_cache_path(horizon: int) -> Path:
    return Path(str(paths_for_horizon(BASE, horizon)["attn_cache"]).format(seed=SEED))


def build_model(model_key: str, device: torch.device) -> torch.nn.Module:
    if model_key == "attention":
        return CNNLSTMAttention(n_features=8).to(device)
    if model_key == "temporal":
        return CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
    if model_key == "lstm":
        return LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    raise ValueError(model_key)


@torch.no_grad()
def predict_mm(
    model: torch.nn.Module,
    X: np.ndarray,
    y_scaled: np.ndarray,
    scaler_y,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(X, y_scaled, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(model(xb).float())
    pred_s = torch.cat(chunks, dim=0).cpu().numpy()
    y_pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    return y_true.astype(np.float64), y_pred.astype(np.float64)


def rmse_mm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def station_lookup(feat: pd.DataFrame) -> pd.DataFrame:
    return (
        feat[STATION_COLS]
        .drop_duplicates(subset=["station_id"])
        .reset_index(drop=True)
    )


def print_step1(n_per_h: dict[int, int]) -> None:
    total_samples = sum(n_per_h.values())
    expected_rows = total_samples * len(MODEL_SPECS)
    print("=" * 72)
    print("STEP 1 — Scope confirmation (before any computation)")
    print("=" * 72)
    print("IN SCOPE (primary forecast_cache, seed=42 checkpoints only):")
    print("  Models:   LSTM, CNN-LSTM-Temporal, CNN-LSTM+Attention")
    print("  Horizons: h=1, 2, 3, 4")
    print(
        f"  Test dates: {TEST_START.date()} to {TEST_END.date()} "
        "(locked chronological test split)"
    )
    print("OUT OF SCOPE for this cache:")
    print("  GNN-LSTM, Transformer (may appear later as secondary comparison only)")
    print("Demo/deployment layer — NOT a re-run of multi-seed statistical analysis.")
    print("n_test_samples per horizon:")
    for h in HORIZONS:
        print(f"  h={h}: {n_per_h[h]:,}")
    print(
        f"Expected forecast_cache rows: "
        f"({total_samples:,}) x 3 models = {expected_rows:,}"
    )
    print()


def load_ablation_attn_delta_h1() -> float:
    abl = pd.read_csv(ABLATION_CSV)
    row = abl[
        (abl["Horizon"].astype(int) == 1)
        & (abl["Model"] == "CNN-LSTM+Attention")
    ].iloc[0]
    return float(row["Delta_RMSE_seed42_vs_LSTM"])


def load_lstm_seed42_rmse_h1() -> float:
    with open(LSTM_H1_METRICS, encoding="utf-8") as f:
        meta = json.load(f)
    return float(meta["test_metrics_mm"]["RMSE"])


def build_cache(
    device: torch.device, feat: pd.DataFrame, stations: pd.DataFrame
) -> tuple[pd.DataFrame, dict[int, np.ndarray], dict[tuple[str, int], float]]:
    """Return forecast DataFrame, X_test by horizon (for spot-check), live RMSEs."""
    frames: list[pd.DataFrame] = []
    x_by_h: dict[int, np.ndarray] = {}
    live_rmse: dict[tuple[str, int], float] = {}
    inference_calls = 0

    for h in HORIZONS:
        paths = data_paths(h)
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"])
        scaler_y = joblib.load(paths["scaler_y"])
        x_by_h[h] = X_test
        n = len(X_test)

        print(f"\n--- h={h}: rebuild_test_meta() ---", flush=True)
        meta = rebuild_test_meta(feat, h)
        if len(meta) != n:
            raise RuntimeError(
                f"Meta/test length mismatch h={h}: meta={len(meta)} vs X={n}"
            )

        meta_df = pd.DataFrame(meta)
        meta_df = meta_df.merge(
            stations, on="station_id", how="left", validate="many_to_one"
        )
        if meta_df["station_name"].isna().any():
            missing = int(meta_df["station_name"].isna().sum())
            raise RuntimeError(f"h={h}: {missing} rows missing station metadata join")

        attn_path = attn_cache_path(h)
        if not attn_path.exists():
            raise FileNotFoundError(
                f"Required attention cache missing: {attn_path} "
                "(do not re-infer; Feature 5 caches must already exist)"
            )
        attn_arr = np.load(attn_path)
        if attn_arr.shape != (n, 30):
            raise RuntimeError(
                f"Attention cache shape mismatch: {attn_path} "
                f"got {attn_arr.shape}, expected ({n}, 30)"
            )
        print(f"  Reused attention cache: {attn_path.name} {attn_arr.shape}", flush=True)
        attn_lists = attn_arr.astype(np.float32).tolist()  # list[list[float]] len n

        for label, key in MODEL_SPECS:
            ckpt = ckpt_path(key, h)
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            print(f"--- h={h}: {label} CUDA+autocast ({ckpt.name}) ---", flush=True)
            model = build_model(key, device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            y_true, y_pred = predict_mm(model, X_test, y_test, scaler_y, device)
            inference_calls += 1
            live_rmse[(label, h)] = rmse_mm(y_true, y_pred)
            print(
                f"  live seed-42 RMSE={live_rmse[(label, h)]:.6f}  n={n:,}",
                flush=True,
            )
            del model
            torch.cuda.empty_cache()

            part = pd.DataFrame(
                {
                    "station_id": meta_df["station_id"].to_numpy(),
                    "station_name": meta_df["station_name"].to_numpy(),
                    "latitude": meta_df["latitude"].to_numpy(dtype=np.float64),
                    "longitude": meta_df["longitude"].to_numpy(dtype=np.float64),
                    "state": meta_df["state"].to_numpy(),
                    "district": meta_df["district"].to_numpy(),
                    "target_date": meta_df["target_date"].to_numpy(),
                    "horizon": np.full(n, h, dtype=np.int32),
                    "model_name": np.full(n, label, dtype=object),
                    "y_true_mm": y_true,
                    "y_pred_mm": y_pred,
                    "abs_error_mm": np.abs(y_true - y_pred),
                    "sample_idx": np.arange(n, dtype=np.int32),
                    "attention_weights": (
                        attn_lists if key == "attention" else [None] * n
                    ),
                }
            )
            frames.append(part)

    print(f"\nTotal CUDA+autocast prediction passes this run: {inference_calls}")
    return pd.concat(frames, ignore_index=True), x_by_h, live_rmse


def write_forecast_parquet(df: pd.DataFrame) -> None:
    """Write with pyarrow list<float32> attention_weights (null for non-Attention)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Variable-size list (not FixedSizeList): FixedSizeList encodes Python None as
    # empty list (len 0) which fails round-trip validation.
    attn_type = pa.list_(pa.float32())
    arrays = {
        "station_id": pa.array(df["station_id"].astype(str).tolist(), type=pa.string()),
        "station_name": pa.array(df["station_name"].astype(str).tolist(), type=pa.string()),
        "latitude": pa.array(df["latitude"].to_numpy(dtype=np.float64)),
        "longitude": pa.array(df["longitude"].to_numpy(dtype=np.float64)),
        "state": pa.array(df["state"].astype(str).tolist(), type=pa.string()),
        "district": pa.array(df["district"].astype(str).tolist(), type=pa.string()),
        "target_date": pa.array(df["target_date"].astype(str).tolist(), type=pa.string()),
        "horizon": pa.array(df["horizon"].to_numpy(dtype=np.int32)),
        "model_name": pa.array(df["model_name"].astype(str).tolist(), type=pa.string()),
        "y_true_mm": pa.array(df["y_true_mm"].to_numpy(dtype=np.float64)),
        "y_pred_mm": pa.array(df["y_pred_mm"].to_numpy(dtype=np.float64)),
        "abs_error_mm": pa.array(df["abs_error_mm"].to_numpy(dtype=np.float64)),
        "attention_weights": pa.array(df["attention_weights"].tolist(), type=attn_type),
    }
    table = pa.table(arrays)
    pq.write_table(table, FORECAST_PQ, compression="zstd")


def build_station_metadata(forecast: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    # Per-station test-sample count across h=1..4 (one model — avoids x3 inflation)
    one_model = forecast[forecast["model_name"] == "LSTM"]
    n_test = (
        one_model.groupby("station_id")
        .size()
        .rename("n_test_samples_available")
        .astype(int)
    )
    out = stations.merge(n_test, on="station_id", how="inner")
    out = out.sort_values("station_id").reset_index(drop=True)
    return out[
        [
            "station_id",
            "station_name",
            "latitude",
            "longitude",
            "state",
            "district",
            "elevation",
            "n_test_samples_available",
        ]
    ]


def validate(
    forecast: pd.DataFrame,
    stations_meta: pd.DataFrame,
    x_by_h: dict[int, np.ndarray],
    live_rmse: dict[tuple[str, int], float],
    device: torch.device,
    hashes_before: dict[str, str],
) -> None:
    print("\n" + "=" * 72)
    print("STEP 4 — Validation checks")
    print("=" * 72)

    # (a) row count
    n_per_h = {h: int((forecast["horizon"] == h).sum() // 3) for h in HORIZONS}
    # more reliable: count from one model
    n_per_h = {
        h: int(
            ((forecast["horizon"] == h) & (forecast["model_name"] == "LSTM")).sum()
        )
        for h in HORIZONS
    }
    expected = sum(n_per_h.values()) * 3
    actual = len(forecast)
    print(f"(a) Row count: expected={expected:,}  actual={actual:,}  "
          f"{'PASS' if expected == actual else 'FAIL'}")
    for h in HORIZONS:
        print(f"    h={h}: n_test={n_per_h[h]:,}")

    # (b) Attention h=1 RMSE self-check + ablation cross-check + spot-check
    attn_h1 = forecast[
        (forecast["model_name"] == "CNN-LSTM+Attention") & (forecast["horizon"] == 1)
    ]
    cache_rmse = rmse_mm(
        attn_h1["y_true_mm"].to_numpy(), attn_h1["y_pred_mm"].to_numpy()
    )
    live = live_rmse[("CNN-LSTM+Attention", 1)]
    self_ok = abs(cache_rmse - live) <= RMSE_ATOL
    print("\n(b) Attention h=1 RMSE validation (seed-42 single-run, NOT 3-seed mean)")
    print(f"    Cache self-computed RMSE:     {cache_rmse:.6f}")
    print(f"    Live inference RMSE (same run): {live:.6f}")
    print(
        f"    |cache - live| = {abs(cache_rmse - live):.2e}  "
        f"(atol={RMSE_ATOL})  {'PASS' if self_ok else 'FAIL'}"
    )
    print(
        "    NOTE: ablation_study.csv RMSE_mean=9.4440 is the 3-seed mean — "
        "NOT the validation target."
    )

    lstm_seed42 = load_lstm_seed42_rmse_h1()
    delta42 = load_ablation_attn_delta_h1()
    expected_from_ablation = lstm_seed42 + delta42
    cross_ok = abs(cache_rmse - expected_from_ablation) <= 5e-2  # 0.05 mm tolerance
    print("\n    Independent cross-check vs ablation_study.csv seed-42 deltas:")
    print(f"    LSTM_seed42_RMSE (metrics JSON):     {lstm_seed42:.6f}")
    print(f"    Delta_RMSE_seed42_vs_LSTM (ablation): {delta42:.6f}")
    print(f"    LSTM_seed42 + Delta:                  {expected_from_ablation:.6f}")
    print(f"    Cache Attention h=1 RMSE:             {cache_rmse:.6f}")
    print(
        f"    |cache - (LSTM+Delta)| = {abs(cache_rmse - expected_from_ablation):.6f}  "
        f"(atol=0.05)  {'PASS' if cross_ok else 'FAIL'}"
    )

    # Point-level spot-check: re-infer FULL Attention h=1 test set (same batch=256
    # CUDA+autocast path as the build). Mini-batches of 5 are NOT used — cudnn /
    # autocast can be batch-composition dependent and create false mismatches.
    rng = np.random.default_rng(SPOT_SEED)
    pick_idx = rng.choice(len(attn_h1), size=SPOT_N, replace=False)
    picked = attn_h1.iloc[pick_idx].copy()
    sample_idxs = picked["sample_idx"].to_numpy(dtype=np.int64)

    print(
        f"\n    Point-level spot-check ({SPOT_N} random Attention h=1 rows; "
        f"FULL-test re-inference at batch={BATCH_SIZE}, seed={SPOT_SEED}):",
        flush=True,
    )
    paths1 = data_paths(1)
    X_test = x_by_h[1]
    y_test = np.load(paths1["y_test"])
    scaler_y = joblib.load(paths1["scaler_y"])
    ckpt = ckpt_path("attention", 1)
    model = build_model("attention", device)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    _, live_full = predict_mm(model, X_test, y_test, scaler_y, device)
    del model
    torch.cuda.empty_cache()
    live_preds = live_full[sample_idxs]

    spot_ok = True
    for j, (_, r) in enumerate(picked.iterrows()):
        cached = float(r["y_pred_mm"])
        live_p = float(live_preds[j])
        diff = abs(cached - live_p)
        ok = diff <= PRED_ATOL
        spot_ok = spot_ok and ok
        print(
            f"      [{j+1}] station={r['station_id']}  date={r['target_date']}  "
            f"idx={int(r['sample_idx'])}  cache={cached:.6f}  live={live_p:.6f}  "
            f"|d|={diff:.2e}  {'PASS' if ok else 'FAIL'}"
        )
    print(f"    Spot-check overall: {'PASS' if spot_ok else 'FAIL'}")

    # (c) NaNs / duplicates
    check_cols = [
        "station_id",
        "target_date",
        "horizon",
        "model_name",
        "y_true_mm",
        "y_pred_mm",
        "abs_error_mm",
    ]
    nan_count = int(forecast[check_cols].isna().sum().sum())
    dup_mask = forecast.duplicated(
        subset=["model_name", "station_id", "target_date", "horizon"], keep=False
    )
    n_dup = int(dup_mask.sum())
    print(f"\n(c) NaNs in core columns: {nan_count}  "
          f"{'PASS' if nan_count == 0 else 'FAIL'}")
    print(
        f"    Duplicate (model, station_id, target_date, horizon) rows: {n_dup}  "
        f"{'PASS' if n_dup == 0 else 'FAIL'}"
    )

    # (d) attention weights presence
    is_attn = forecast["model_name"] == "CNN-LSTM+Attention"
    attn_present = forecast["attention_weights"].notna()
    # list column: None vs list — after parquet round-trip may differ; check in-memory
    n_attn_with = int((is_attn & attn_present).sum())
    n_attn_without = int((is_attn & ~attn_present).sum())
    n_other_with = int((~is_attn & attn_present).sum())
    n_other_without = int((~is_attn & ~attn_present).sum())
    # verify list length 30 for attention rows
    lens_ok = True
    for w in forecast.loc[is_attn, "attention_weights"]:
        if w is None or len(w) != 30:
            lens_ok = False
            break
    d_ok = (
        n_attn_without == 0
        and n_other_with == 0
        and n_attn_with == int(is_attn.sum())
        and lens_ok
    )
    print("\n(d) Attention weight vectors:")
    print(f"    Attention rows with weights:     {n_attn_with:,}")
    print(f"    Attention rows missing weights:  {n_attn_without:,}")
    print(f"    Non-Attention rows with weights: {n_other_with:,}")
    print(f"    Non-Attention rows null weights: {n_other_without:,}")
    print(f"    All Attention vectors len==30:   {lens_ok}")
    print(f"    Result: {'PASS' if d_ok else 'FAIL'}")

    # (e) file sizes
    print("\n(e) Output artifacts:")
    for path in (FORECAST_PQ, STATION_PQ):
        size_mb = path.stat().st_size / (1024 * 1024)
        if path == FORECAST_PQ:
            pf = pq.ParquetFile(path)
            print(
                f"    {path.name}: {size_mb:.2f} MB  "
                f"rows={pf.metadata.num_rows:,}  cols={pf.metadata.num_columns}"
            )
            # Round-trip attention nulls from on-disk parquet
            t_attn = pq.read_table(path, columns=["model_name", "attention_weights"])
            md = t_attn.column("model_name").to_pylist()
            aw = t_attn.column("attention_weights")
            n_null = 0
            n_ok = 0
            n_bad = 0
            for i, name in enumerate(md):
                val = aw[i].as_py()
                if name == "CNN-LSTM+Attention":
                    if val is not None and len(val) == 30:
                        n_ok += 1
                    else:
                        n_bad += 1
                else:
                    if val is None:
                        n_null += 1
                    else:
                        n_bad += 1
            print(
                f"    On-disk attention round-trip: "
                f"Attention ok={n_ok:,}  non-Attention null={n_null:,}  bad={n_bad:,}  "
                f"{'PASS' if n_bad == 0 else 'FAIL'}"
            )
        else:
            sm = pd.read_parquet(path)
            print(
                f"    {path.name}: {size_mb:.2f} MB  "
                f"rows={len(sm):,}  cols={len(sm.columns)}"
            )

    # hash integrity
    print("\nVerified-file hash check (must be unchanged):")
    all_hash_ok = True
    for path in VERIFIED:
        after = sha256_file(path)
        before = hashes_before[path.name]
        ok = before == after
        all_hash_ok = all_hash_ok and ok
        print(f"  {path.name}: {'UNCHANGED' if ok else 'MODIFIED!'}  {after}")
    print(f"  Overall hash integrity: {'PASS' if all_hash_ok else 'FAIL'}")

    # STEP 5 print schema + samples
    print("\n" + "=" * 72)
    print("STEP 5 — Schema, samples, inference scope")
    print("=" * 72)
    fc = pq.read_table(FORECAST_PQ)
    print("\nforecast_cache.parquet schema:")
    for field in fc.schema:
        print(f"  {field.name}: {field.type}")
    sm = pd.read_parquet(STATION_PQ)
    print("\nstation_metadata.parquet schema:")
    for col, dtype in sm.dtypes.items():
        print(f"  {col}: {dtype}")

    print("\n5 sample rows from forecast_cache (mixed models):")
    sample = forecast.sample(n=5, random_state=0)[
        [
            "station_id",
            "station_name",
            "target_date",
            "horizon",
            "model_name",
            "y_true_mm",
            "y_pred_mm",
            "abs_error_mm",
        ]
    ]
    print(sample.to_string(index=False))

    print("\nInference scope confirmation:")
    print(
        "  This script performed one-time CUDA+autocast prediction passes to build "
        "the cache (plus a 5-row Attention h=1 spot-check re-inference)."
    )
    print(
        "  Attention weight vectors were LOADED from existing "
        "attention_weights_h{1..4}_seed42.npy — not recomputed."
    )
    print(
        "  The dashboard must READ these parquet files only — "
        "do NOT re-run this script on every dashboard open."
    )
    print(f"\nstation_metadata: {len(stations_meta):,} stations")


def main() -> None:
    hashes_before = {p.name: sha256_file(p) for p in VERIFIED}
    print("SHA-256 BEFORE build:")
    for name, h in hashes_before.items():
        print(f"  {name}: {h}")

    n_per_h = {}
    for h in HORIZONS:
        n_per_h[h] = int(np.load(data_paths(h)["X_test"], mmap_mode="r").shape[0])
    print_step1(n_per_h)

    device = require_cuda()
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    print("\nLoading feature_engineered_v2.csv ...", flush=True)
    feat = pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])
    stations = station_lookup(feat)
    print(f"Stations in lookup: {len(stations)}")

    forecast, x_by_h, live_rmse = build_cache(device, feat, stations)
    stations_meta = build_station_metadata(forecast, stations)

    print("\nWriting parquet files ...", flush=True)
    write_forecast_parquet(forecast)
    stations_meta.to_parquet(STATION_PQ, index=False, compression="zstd")
    print(f"  Wrote {FORECAST_PQ}")
    print(f"  Wrote {STATION_PQ}")

    validate(forecast, stations_meta, x_by_h, live_rmse, device, hashes_before)
    print("\nPHASE 0 complete.")


if __name__ == "__main__":
    main()
