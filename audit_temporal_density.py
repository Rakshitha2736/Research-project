"""
Temporal Density Audit — Phase 5 prep
=====================================
Measures per-station calendar coverage and gap sizes so we can choose a
gap-handling policy:

  A) reindex-and-fill
  B) filter-to-dense-stations
  C) irregular-time features

Usage:
    python audit_temporal_density.py data/processed/feature_engineered.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


def load(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["date_of_record"] = pd.to_datetime(df["date_of_record"], errors="coerce")
    return df


def station_key_column(df: pd.DataFrame) -> str:
    """Prefer disambiguated station_id when present."""
    return "station_id" if "station_id" in df.columns else "station_name"


def per_station_stats(df: pd.DataFrame, station_col: str = "station_name") -> pd.DataFrame:
    rows = []
    for station, g in df.groupby(station_col, sort=False):
        dates = g["date_of_record"].sort_values().drop_duplicates()
        n_obs = len(dates)
        if n_obs == 0:
            continue
        start, end = dates.iloc[0], dates.iloc[-1]
        span_days = (end - start).days + 1
        expected = span_days  # daily calendar
        coverage = n_obs / expected if expected > 0 else np.nan

        gaps = dates.diff().dt.days.dropna() - 1  # 0 = consecutive days
        gap_sizes = gaps[gaps > 0]

        rows.append(
            {
                station_col: station,
                "station_name": g["station_name"].mode().iloc[0]
                if "station_name" in g.columns
                else station,
                "state": g["state"].mode().iloc[0] if "state" in g.columns else None,
                "n_obs": n_obs,
                "start": start,
                "end": end,
                "span_days": span_days,
                "coverage": coverage,
                "n_gaps": int((gaps > 0).sum()),
                "max_gap_days": int(gap_sizes.max()) if len(gap_sizes) else 0,
                "median_gap_days": float(gap_sizes.median()) if len(gap_sizes) else 0.0,
                "mean_gap_days": float(gap_sizes.mean()) if len(gap_sizes) else 0.0,
                "p90_gap_days": float(gap_sizes.quantile(0.9)) if len(gap_sizes) else 0.0,
                "total_missing_days": int(gap_sizes.sum()) if len(gap_sizes) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values("coverage", ascending=False).reset_index(drop=True)


def all_gap_sizes(df: pd.DataFrame, station_col: str = "station_name") -> pd.Series:
    parts = []
    for _, g in df.groupby(station_col, sort=False):
        dates = g["date_of_record"].sort_values().drop_duplicates()
        gaps = dates.diff().dt.days.dropna() - 1
        parts.append(gaps[gaps > 0])
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, ignore_index=True)


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main(path: str) -> None:
    df = load(path)
    station_col = station_key_column(df)
    if station_col not in df.columns:
        raise KeyError(f"Missing station column: {station_col}")
    df = df.dropna(subset=["date_of_record", station_col]).copy()

    print_section("0. INPUT")
    print(f"File: {path}")
    print(f"Station key: {station_col}")
    print(f"Rows: {len(df):,}")
    print(f"Stations: {df[station_col].nunique()}")
    print(f"Date range: {df['date_of_record'].min().date()} -> {df['date_of_record'].max().date()}")

    # Duplicate dates within a station
    print_section("1. DUPLICATE DATES WITHIN STATION")
    dup = df.duplicated(subset=[station_col, "date_of_record"], keep=False)
    print(f"Rows sharing a {station_col}+date key: {dup.sum():,}")
    if dup.any():
        print(df.loc[dup, [station_col, "date_of_record"]].value_counts().head(10))

    stats = per_station_stats(df, station_col=station_col)

    print_section("2. PER-STATION OBSERVATION COUNTS")
    print(stats["n_obs"].describe())

    print_section("3. PER-STATION CALENDAR COVERAGE (obs / span_days)")
    print(stats["coverage"].describe())
    for thr in [0.95, 0.90, 0.80, 0.70, 0.50]:
        n = (stats["coverage"] >= thr).sum()
        print(f"  coverage >= {thr:.0%}: {n} stations ({n / len(stats) * 100:.1f}%)")

    print_section("4. GAP SIZE DISTRIBUTION (all stations, gap > 0 days)")
    gaps = all_gap_sizes(df, station_col=station_col)
    print(f"Total gaps (discontinuities): {len(gaps):,}")
    if len(gaps):
        print(gaps.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
        bins = [1, 2, 3, 7, 14, 30, 90, 180, 365, 10_000]
        labels = ["1", "2", "3-6", "7-13", "14-29", "30-89", "90-179", "180-364", "365+"]
        cut = pd.cut(gaps, bins=bins, right=False, labels=labels)
        print("\nGap size buckets:")
        print(cut.value_counts().sort_index().to_string())

    print_section("5. WORST STATIONS BY MAX GAP")
    cols = [
        station_col,
        "station_name",
        "state",
        "n_obs",
        "span_days",
        "coverage",
        "n_gaps",
        "max_gap_days",
        "median_gap_days",
        "total_missing_days",
    ]
    cols = [c for c in cols if c in stats.columns]
    print(stats.nlargest(15, "max_gap_days")[cols].to_string(index=False))

    print_section("6. BEST STATIONS BY COVERAGE")
    print(stats.nlargest(15, "coverage")[cols].to_string(index=False))

    print_section("7. WORST STATIONS BY COVERAGE")
    print(stats.nsmallest(15, "coverage")[cols].to_string(index=False))

    # Sequence feasibility under SEQ_LEN=30 (need 31 contiguous calendar days)
    print_section("8. SEQUENCE FEASIBILITY (SEQ_LEN=30 -> need 31 contiguous days)")
    seq_len = 30
    need = seq_len + 1
    feasible_counts = []
    for _, g in df.groupby(station_col, sort=False):
        dates = np.sort(g["date_of_record"].dt.normalize().unique())
        if len(dates) == 0:
            feasible_counts.append(0)
            continue
        day_ints = dates.astype("datetime64[D]").astype(np.int64)
        span = np.arange(day_ints[0], day_ints[-1] + 1)
        present = np.isin(span, day_ints)
        if len(present) < need:
            feasible_counts.append(0)
            continue
        c = np.cumsum(present, dtype=np.int32)
        window_sums = c[need - 1 :] - np.concatenate(([0], c[:-need]))
        feasible_counts.append(int((window_sums == need).sum()))
    feasible = pd.Series(feasible_counts, name="n_contiguous_windows")
    print(feasible.describe())
    print(f"Stations with ZERO contiguous {need}-day windows: {(feasible == 0).sum()}")
    print(f"Stations with >= 100 windows: {(feasible >= 100).sum()}")
    print(f"Stations with >= 500 windows: {(feasible >= 500).sum()}")

    print_section("9. POLICY HINTS (read with heatmap)")
    high = (stats["coverage"] >= 0.90).mean()
    med_max_gap = stats["max_gap_days"].median()
    p95_gap = gaps.quantile(0.95) if len(gaps) else np.nan
    print(f"Share of stations with coverage >= 90%: {high * 100:.1f}%")
    print(f"Median station max_gap_days: {med_max_gap:.0f}")
    print(f"95th percentile gap size (all gaps): {p95_gap}")
    print(
        "\nIf most gaps are short (1-3 days) and coverage is high -> reindex-and-fill is viable.\n"
        "If many stations have long / sparse coverage -> filter-to-dense-stations.\n"
        "If gaps are large and irregular for most stations -> irregular-time features."
    )

    out = Path(path).resolve().parent / "temporal_density_by_station.csv"
    stats.to_csv(out, index=False)
    print_section("10. SAVED")
    print(f"Per-station audit table: {out}")
    print("DONE — paste everything above back into the chat.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audit_temporal_density.py path/to/feature_engineered.csv")
        sys.exit(1)
    main(sys.argv[1])
