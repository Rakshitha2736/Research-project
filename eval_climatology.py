"""
Per-station, per-season climatology baseline (train-period only).

Statistic: mean rainfall per (station_id, season), using rows with
date_of_record <= TRAIN_END (2022-12-31). Mean (not median) is used because
RMSE/MSE are the project's primary regression metrics and the MSE-optimal
constant predictor for a cell is the arithmetic mean.

Season mapping is imported from eval_seasonal_performance (matches
clean_dataset.csv / seasonal_performance.csv exactly).

Usage (from RainfallPrediction/):
    python eval_climatology.py

Writes:
    reports/tables/climatology_baseline.csv
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from eval_seasonal_performance import (
    SEASON_ORDER,
    month_to_season,
    verify_season_matches_clean_dataset,
)
from src.eval_attention import TRAIN_END, rebuild_test_meta
from src.persistence_baseline import HORIZONS_DEFAULT, data_paths, metrics_mm

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
TABLES = BASE / "reports" / "tables"
FEAT_CSV = DATA / "feature_engineered_v2.csv"
OUT_CSV = TABLES / "climatology_baseline.csv"
MASTER_CSV = TABLES / "master_results.csv"


@dataclass
class ClimatologyLookup:
    """Train-only climatology with explicit fallback hierarchy."""

    cell_mean: dict[tuple[str, str], float]
    cell_n: dict[tuple[str, str], int]
    station_mean: dict[str, float]
    station_n: dict[str, int]
    global_mean: float
    global_n: int
    cells_direct: int
    cells_station_fallback: int
    cells_global_fallback: int
    stations_global_fallback: list[str]


def build_climatology_lookup(train_df: pd.DataFrame) -> ClimatologyLookup:
    """Fit climatology on train rows only (no val/test leakage)."""
    df = train_df.copy()
    df["station_id"] = df["station_id"].astype(str)
    df["season"] = df["date_of_record"].dt.month.map(month_to_season)

    global_mean = float(df["rainfall"].mean())
    global_n = int(len(df))

    station_grp = df.groupby("station_id")["rainfall"]
    station_mean = station_grp.mean().astype(float).to_dict()
    station_n = station_grp.size().astype(int).to_dict()

    cell_grp = df.groupby(["station_id", "season"])["rainfall"]
    cell_mean = {
        (str(sid), str(season)): float(val)
        for (sid, season), val in cell_grp.mean().items()
    }
    cell_n = {
        (str(sid), str(season)): int(val)
        for (sid, season), val in cell_grp.size().items()
    }

    all_stations = sorted(df["station_id"].unique())
    cells_direct = 0
    cells_station_fallback = 0
    cells_global_fallback = 0
    stations_global_fallback: list[str] = []

    for sid in all_stations:
        if station_n.get(sid, 0) == 0:
            stations_global_fallback.append(sid)
        for season in SEASON_ORDER:
            n = cell_n.get((sid, season), 0)
            if n > 0:
                cells_direct += 1
            elif station_n.get(sid, 0) > 0:
                cells_station_fallback += 1
            else:
                cells_global_fallback += 1

    return ClimatologyLookup(
        cell_mean=cell_mean,
        cell_n=cell_n,
        station_mean=station_mean,
        station_n=station_n,
        global_mean=global_mean,
        global_n=global_n,
        cells_direct=cells_direct,
        cells_station_fallback=cells_station_fallback,
        cells_global_fallback=cells_global_fallback,
        stations_global_fallback=stations_global_fallback,
    )


def predict_value(lookup: ClimatologyLookup, station_id: str, season: str) -> tuple[float, str]:
    key = (station_id, season)
    if lookup.cell_n.get(key, 0) > 0:
        return lookup.cell_mean[key], "direct"
    if lookup.station_n.get(station_id, 0) > 0:
        return lookup.station_mean[station_id], "station_fallback"
    return lookup.global_mean, "global_fallback"


def eval_horizon(
    horizon: int,
    feat: pd.DataFrame,
    lookup: ClimatologyLookup,
) -> dict:
    paths = data_paths(horizon, BASE)
    y_test = np.load(paths["y_test"])
    scaler_y = joblib.load(paths["scaler_y"])
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    meta = rebuild_test_meta(feat, horizon)
    if len(meta) != len(y_true):
        raise RuntimeError(
            f"Meta/test length mismatch h={horizon}: meta={len(meta)} vs y={len(y_true)}"
        )

    y_pred = np.empty(len(meta), dtype=np.float64)
    route_counts = {"direct": 0, "station_fallback": 0, "global_fallback": 0}

    for i, row in enumerate(meta):
        target_date = pd.Timestamp(row["target_date"])
        season = month_to_season(int(target_date.month))
        val, route = predict_value(lookup, str(row["station_id"]), season)
        y_pred[i] = val
        route_counts[route] += 1

    m = metrics_mm(y_true, y_pred)
    return {
        "horizon": horizon,
        "n_test": int(len(y_true)),
        **m,
        "test_rows_direct": route_counts["direct"],
        "test_rows_station_fallback": route_counts["station_fallback"],
        "test_rows_global_fallback": route_counts["global_fallback"],
    }


def print_worked_example(lookup: ClimatologyLookup, train_df: pd.DataFrame) -> None:
    """One station×season with direct train-cell statistics."""
    train = train_df.copy()
    train["season"] = train["date_of_record"].dt.month.map(month_to_season)

    best_sid, best_season, best_n = None, None, -1
    for (sid, season), n in lookup.cell_n.items():
        if n > best_n:
            best_sid, best_season, best_n = sid, season, n

    if best_sid is None:
        print("Worked example: no direct cells found.")
        return

    sub = train[(train["station_id"].astype(str) == best_sid) & (train["season"] == best_season)]
    computed_mean = float(sub["rainfall"].mean())
    stored = lookup.cell_mean[(best_sid, best_season)]
    print("\n=== Worked example (direct cell) ===")
    print(f"  station_id: {best_sid}")
    print(f"  season:     {best_season}")
    print(f"  train rows: {best_n} (date_of_record <= {TRAIN_END.date()})")
    print(f"  climatology mean rainfall: {stored:.6f} mm/day")
    print(f"  recomputed from train rows: {computed_mean:.6f} mm/day")
    print(f"  match: {np.isclose(stored, computed_mean)}")


def append_master_results(rows: list[dict]) -> None:
    """Add climatology rows; replace any prior climatology rows on re-run."""
    existing = [
        ln
        for ln in MASTER_CSV.read_text(encoding="utf-8").splitlines()
        if ln and not ln.startswith("Climatology Baseline,")
    ]
    if not existing:
        raise RuntimeError(f"Empty {MASTER_CSV}")

    new_lines = []
    for r in rows:
        new_lines.append(
            ",".join(
                [
                    "Climatology Baseline",
                    str(r["horizon"]),
                    "Not Applicable",
                    "n/a",
                    f"{r['RMSE']:.4f}",
                    f"{r['MAE']:.4f}",
                    f"{r['MSE']:.4f}",
                    f"{r['R2']:.4f}",
                    (
                        "eval_climatology.py; per-station per-season train mean "
                        "(<=2022-12-31); constant ŷ per (station, season)"
                    ),
                ]
            )
        )

    updated = existing + new_lines
    MASTER_CSV.write_text("\n".join(updated) + "\n", encoding="utf-8")


def main() -> None:
    verify_season_matches_clean_dataset()

    feat = pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])
    train_df = feat[feat["date_of_record"] <= TRAIN_END].copy()
    print(f"Train rows (<= {TRAIN_END.date()}): {len(train_df):,}")
    print(f"Unique stations in train: {train_df['station_id'].nunique()}")

    lookup = build_climatology_lookup(train_df)
    total_cells = (
        lookup.cells_direct
        + lookup.cells_station_fallback
        + lookup.cells_global_fallback
    )
    fallback_cells = lookup.cells_station_fallback + lookup.cells_global_fallback
    fallback_frac = fallback_cells / total_cells if total_cells else 1.0

    print("\n=== Climatology cell coverage (station × season) ===")
    print(f"  Total cells:              {total_cells}")
    print(f"  Direct (train n > 0):     {lookup.cells_direct}")
    print(f"  Station-level fallback:   {lookup.cells_station_fallback}")
    print(f"  Global fallback cells:    {lookup.cells_global_fallback}")
    print(f"  Stations on global mean:  {len(lookup.stations_global_fallback)}")
    if lookup.stations_global_fallback:
        print(f"    flagged: {lookup.stations_global_fallback[:5]}...")
    print(f"  Fallback fraction:        {fallback_frac:.4f}")

    test_only_stations = set()
    for h in HORIZONS_DEFAULT:
        meta = rebuild_test_meta(feat, h)
        train_stations = set(lookup.station_n.keys())
        for row in meta:
            sid = str(row["station_id"])
            if sid not in train_stations:
                test_only_stations.add(sid)
    if test_only_stations:
        print(
            f"  Test-only stations (no train rows): {len(test_only_stations)} "
            f"-> {sorted(test_only_stations)}"
        )

    if fallback_frac > 0.5:
        raise SystemExit(
            "STOP: >50% of station×season cells require fallback — "
            "reconsider granularity before proceeding."
        )

    print_worked_example(lookup, train_df)

    result_rows: list[dict] = []
    print("\n=== Climatology baseline metrics (mm/day) ===")
    print(f"{'h':>3} {'n_test':>8} {'RMSE':>10} {'MAE':>10} {'MSE':>12} {'R2':>10}")
    for h in HORIZONS_DEFAULT:
        row = eval_horizon(h, feat, lookup)
        result_rows.append(row)
        print(
            f"{h:3d} {row['n_test']:8d} {row['RMSE']:10.4f} "
            f"{row['MAE']:10.4f} {row['MSE']:12.4f} {row['R2']:10.4f}"
        )

    TABLES.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(result_rows)
    header = (
        "# Climatology = per-station per-season mean rainfall from train period only "
        f"(date_of_record <= {TRAIN_END.date()}).\n"
        "# Season map matches eval_seasonal_performance / seasonal_performance.csv.\n"
        "# Statistic: arithmetic mean (MSE-optimal constant predictor per cell).\n"
    )
    OUT_CSV.write_text(header + df_out.to_csv(index=False), encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")

    append_master_results(result_rows)
    print(f"Appended {len(result_rows)} rows to {MASTER_CSV} (additive only)")


if __name__ == "__main__":
    main()
