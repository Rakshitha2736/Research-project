"""
Offline builder: each usable station's last contiguous 30-day input window.

Mirrors the Phase-0 pattern of build_forecast_cache.py:
  one-off offline script → verified parquet → dashboard reads only.

Windowing reuses the SAME contiguous-segment logic as generate_sequences_v2.py
(calendar day diffs == 1; no gap-fill). For forward inference we need SEQ_LEN=30
observed days only (no observed target), so a segment qualifies if length >= 30.

Writes ONLY:
  reports/dashboard_data/latest_forecast_windows.parquet

Does NOT retrain. Does NOT modify verified result tables or existing dashboard
parquets (forecast_cache / station_metadata).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
TABLES = BASE / "reports" / "tables"
OUT_DIR = BASE / "reports" / "dashboard_data"
FEAT_CSV = DATA / "feature_engineered_v2.csv"
STATION_PQ = OUT_DIR / "station_metadata.parquet"
FORECAST_PQ = OUT_DIR / "forecast_cache.parquet"
OUT_PQ = OUT_DIR / "latest_forecast_windows.parquet"

# Same two stations as dashboard.lib.paths.ZERO_TEST_STATIONS — present in the
# station picker (414 total) but absent from station_metadata.parquet.
# Forward Forecast includes them so the picker always has window metadata.
# Their windows can be far older than the usable-network 548–638 day norm;
# warning_flag marks that extra-stale case (not the network-wide baseline).
ZERO_TEST_STATION_IDS = (
    "Chikkanahalli / Sadali_13.67_77.92_672",
    "Darjeeling_27.05_88.27_2127",
)

# Contiguous window length required for model input (generate_sequences_v2.SEQ_LEN)
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

# Fixed reference calendar date for days_stale (feature brief: 12-08-2026)
REFERENCE_DATE = pd.Timestamp("2026-08-12")
# Removed: STALE_BEFORE = 2023-01-01. That threshold fired on 0/412 usable
# stations and hid the real situation (entire usable network is ~548–638 days
# stale). warning_flag now means EXTRA-stale vs the usable-network ceiling
# (days_stale > max among the 412), e.g. the zero-test Chikkanahalli case.

VERIFIED = [
    TABLES / "master_results.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
    TABLES / "station_wise_error.csv",
    TABLES / "seasonal_performance.csv",
    TABLES / "rain_classification_metrics.csv",
    TABLES / "extreme_rainfall_evaluation.csv",
    TABLES / "attention_extreme_vs_normal.csv",
    FORECAST_PQ,
    STATION_PQ,
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_valid_window_end(g: pd.DataFrame) -> pd.Timestamp | None:
    """Return window_end_date for the station's most recent contiguous 30-day window.

    Logic mirrors generate_sequences_v2.build_sequences segment splitting:
      day_ints → breaks where np.diff != 1 → contiguous segments.
    Unlike sequence generation (which needs SEQ_LEN+1 for an observed target),
    forward inference only needs SEQ_LEN finite feature rows.
    """
    g = g.sort_values("date_of_record").reset_index(drop=True)
    dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
    feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
    n = len(g)
    if n < SEQ_LEN:
        return None

    day_ints = dates.astype("datetime64[D]").astype(np.int64)
    breaks = np.where(np.diff(day_ints) != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))

    best_end: pd.Timestamp | None = None
    for seg_start, seg_end in zip(starts, ends):
        if seg_end - seg_start < SEQ_LEN:
            continue
        # Candidate windows: last SEQ_LEN rows of this contiguous segment
        i = seg_end - SEQ_LEN
        x_seq = feats[i:seg_end]
        if not np.isfinite(x_seq).all():
            # Walk backward within the segment for a finite 30-day block
            found = False
            for j in range(seg_end - SEQ_LEN, seg_start - 1, -1):
                x_seq = feats[j : j + SEQ_LEN]
                if np.isfinite(x_seq).all():
                    best_end = pd.Timestamp(dates[j + SEQ_LEN - 1])
                    found = True
                    break
            if not found:
                continue
        else:
            best_end = pd.Timestamp(dates[seg_end - 1])
    return best_end


def main() -> None:
    hashes_before = {p: sha256_file(p) for p in VERIFIED if p.exists()}

    stations = pd.read_parquet(STATION_PQ, columns=["station_id"])
    usable_ids = set(stations["station_id"].astype(str))
    assert len(usable_ids) == 412, f"Expected 412 parquet stations, got {len(usable_ids)}"
    # Picker scope = 412 test-usable + 2 zero-test supplements (matches load_stations)
    target_ids = usable_ids | set(ZERO_TEST_STATION_IDS)
    assert len(target_ids) == 414, f"Expected 414 picker stations, got {len(target_ids)}"

    print(f"Loading {FEAT_CSV} …")
    df = pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])
    df = df[df["station_id"].astype(str).isin(target_ids)].copy()

    rows: list[dict] = []
    missing: list[str] = []
    for station_id, g in df.groupby("station_id", sort=False):
        end = last_valid_window_end(g)
        if end is None:
            missing.append(str(station_id))
            continue
        days_stale = int((REFERENCE_DATE - end).days)
        rows.append(
            {
                "station_id": str(station_id),
                "window_end_date": end.normalize(),
                "days_stale": days_stale,
                # Filled after usable-network max is known
                "warning_flag": False,
            }
        )

    out = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    if missing:
        raise RuntimeError(
            f"{len(missing)} stations have no contiguous 30-day window: "
            f"{missing[:5]}…"
        )
    if len(out) != 414:
        raise RuntimeError(f"Expected 414 rows (picker scope), got {len(out)}")

    usable_mask = out["station_id"].isin(usable_ids)
    usable_max_stale = int(out.loc[usable_mask, "days_stale"].max())
    usable_min_stale = int(out.loc[usable_mask, "days_stale"].min())
    # Extra-stale only: worse than every usable station (548–638 day norm).
    # Does NOT imply usable stations are only mildly stale — they are all
    # critically stale; the page banner states that as the general case.
    out["warning_flag"] = out["days_stale"] > usable_max_stale

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PQ, index=False)
    print(f"Wrote {OUT_PQ} ({len(out)} stations)")

    print("\n=== latest_forecast_windows.parquet summary ===")
    print(f"stations: {len(out)}")
    print(f"window_end_date min: {out['window_end_date'].min().date()}")
    print(f"window_end_date max: {out['window_end_date'].max().date()}")
    print(f"days_stale min: {out['days_stale'].min()}")
    print(f"days_stale median: {out['days_stale'].median():.1f}")
    print(f"days_stale max: {out['days_stale'].max()}")
    print(
        f"412 usable days_stale: min={usable_min_stale} max={usable_max_stale} "
        f"(network-wide critical staleness — stated in UI banner, not warning_flag)"
    )
    n_warn = int(out["warning_flag"].sum())
    print(
        f"extra-stale warning_flag (days_stale > usable max {usable_max_stale}): {n_warn}"
    )
    if n_warn:
        warn = out.loc[out["warning_flag"]].sort_values("window_end_date")
        print("Extra-stale stations (beyond usable-network norm):")
        for r in warn.itertuples():
            print(
                f"  {r.station_id}  window_end={r.window_end_date.date()}  "
                f"days_stale={r.days_stale}"
            )

    for p, h in hashes_before.items():
        if sha256_file(p) != h:
            raise RuntimeError(f"HASH DRIFT (must not change): {p}")
    print("\nHash check: all 10 verified/frozen files unchanged.")


if __name__ == "__main__":
    main()
