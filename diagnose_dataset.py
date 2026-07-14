"""
Dataset Diagnostic Script — Phase 4 Prep
=========================================
Run this locally against your cleaned dataset and paste the FULL printed
output back into the chat. Do not summarize it yourself — paste it verbatim
so nothing gets lost or misremembered.

Usage:
    python diagnose_dataset.py path/to/your_clean_dataset.xlsx
    (or .csv — the script auto-detects)
"""

import sys
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def load(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_excel(path)


def main(path: str) -> None:
    df = load(path)

    print("=" * 70)
    print("1. SHAPE")
    print("=" * 70)
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")

    print("\n" + "=" * 70)
    print("2. COLUMN NAMES + DTYPES")
    print("=" * 70)
    print(df.dtypes)

    print("\n" + "=" * 70)
    print("3. FIRST 10 ROWS")
    print("=" * 70)
    print(df.head(10).to_string())

    print("\n" + "=" * 70)
    print("4. MISSING VALUES PER COLUMN")
    print("=" * 70)
    miss = df.isna().sum()
    miss_pct = (miss / len(df) * 100).round(3)
    print(pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct}))

    print("\n" + "=" * 70)
    print("5. DUPLICATE ROWS")
    print("=" * 70)
    print(f"Fully duplicated rows: {df.duplicated().sum():,}")

    # Try to auto-detect a date column
    date_col_candidates = [c for c in df.columns if "date" in c.lower()]
    print("\n" + "=" * 70)
    print("6. DATE COLUMN CHECK")
    print("=" * 70)
    print(f"Candidate date columns found: {date_col_candidates}")
    for c in date_col_candidates:
        try:
            parsed = pd.to_datetime(df[c], errors="coerce")
            print(f"\nColumn '{c}':")
            print(f"  Min date: {parsed.min()}")
            print(f"  Max date: {parsed.max()}")
            print(f"  Unparseable (NaT) count: {parsed.isna().sum()}")
        except Exception as e:
            print(f"  Could not parse '{c}': {e}")

    # Try to auto-detect a station column
    station_col_candidates = [
        c for c in df.columns if "station" in c.lower() or "id" in c.lower()
    ]
    print("\n" + "=" * 70)
    print("7. STATION COLUMN CHECK")
    print("=" * 70)
    print(f"Candidate station columns found: {station_col_candidates}")
    for c in station_col_candidates:
        print(f"\nColumn '{c}':")
        print(f"  Unique values: {df[c].nunique()}")
        print(f"  Sample values: {df[c].dropna().unique()[:10]}")

    # Try to auto-detect lat/lon columns
    geo_col_candidates = [
        c for c in df.columns if any(k in c.lower() for k in ["lat", "lon", "long"])
    ]
    print("\n" + "=" * 70)
    print("8. LATITUDE/LONGITUDE COLUMN CHECK")
    print("=" * 70)
    print(f"Candidate geo columns found: {geo_col_candidates}")
    for c in geo_col_candidates:
        print(f"\nColumn '{c}':")
        print(f"  min={df[c].min()}, max={df[c].max()}, nunique={df[c].nunique()}")

    # Rainfall target column check
    rain_col_candidates = [c for c in df.columns if "rain" in c.lower()]
    print("\n" + "=" * 70)
    print("9. RAINFALL (TARGET) COLUMN CHECK")
    print("=" * 70)
    print(f"Candidate rainfall columns found: {rain_col_candidates}")
    for c in rain_col_candidates:
        print(f"\nColumn '{c}':")
        print(df[c].describe())
        print(f"  Zero-rainfall rows: {(df[c] == 0).sum():,} ({(df[c] == 0).mean()*100:.2f}%)")
        print(f"  Negative values (should be none): {(df[c] < 0).sum()}")

    print("\n" + "=" * 70)
    print("10. NUMERIC COLUMN SUMMARY STATISTICS")
    print("=" * 70)
    print(df.describe(include="number").T)

    print("\n" + "=" * 70)
    print("11. NON-NUMERIC / CATEGORICAL COLUMN SUMMARY")
    print("=" * 70)
    obj_cols = df.select_dtypes(exclude="number").columns.tolist()
    for c in obj_cols:
        print(f"\nColumn '{c}': dtype={df[c].dtype}, nunique={df[c].nunique()}")
        print(f"  Sample values: {df[c].dropna().unique()[:10]}")

    print("\n" + "=" * 70)
    print("12. ROWS PER STATION (if station column detected)")
    print("=" * 70)
    if station_col_candidates:
        c = station_col_candidates[0]
        counts = df[c].value_counts()
        print(f"Using column '{c}'")
        print(f"  Number of stations: {counts.shape[0]}")
        print(f"  Min rows/station: {counts.min()}")
        print(f"  Max rows/station: {counts.max()}")
        print(f"  Mean rows/station: {counts.mean():.1f}")
    else:
        print("No station column auto-detected — skipping.")

    print("\n" + "=" * 70)
    print("DONE — paste everything above back into the chat.")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diagnose_dataset.py path/to/your_dataset.xlsx")
        sys.exit(1)
    main(sys.argv[1])
