"""Paths and constants for the dashboard (parquet-only data layer)."""

from __future__ import annotations

from pathlib import Path

# RainfallPrediction/dashboard/lib/paths.py → RainfallPrediction/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_DATA = PROJECT_ROOT / "reports" / "dashboard_data"
TABLES = PROJECT_ROOT / "reports" / "tables"
STATION_METADATA = DASHBOARD_DATA / "station_metadata.parquet"
FORECAST_CACHE = DASHBOARD_DATA / "forecast_cache.parquet"
# Per-station last contiguous 30-day window for Forward Forecast (additive; not
# used by historical/cache pages). Built by build_latest_forecast_windows.py.
LATEST_FORECAST_WINDOWS = DASHBOARD_DATA / "latest_forecast_windows.parquet"
# Fixed calendar reference for days_stale (feature brief: 12-08-2026)
LATEST_FORECAST_REFERENCE_DATE = "2026-08-12"
# Usable-network (412) days_stale vs LATEST_FORECAST_REFERENCE_DATE:
# min = median = 548, max = 638. Banner states this as the general case.
LATEST_FORECAST_NETWORK_MIN_DAYS_STALE = 548
LATEST_FORECAST_NETWORK_MAX_DAYS_STALE = 638
FEATURE_ENGINEERED_V2 = PROJECT_ROOT / "data" / "processed" / "feature_engineered_v2.csv"
STATION_WISE_ERROR_CSV = TABLES / "station_wise_error.csv"

# Locked evaluation scope (single source of truth for header + Home KPIs)
TEST_PERIOD_START = "2024-01-01"
TEST_PERIOD_END = "2025-02-10"
FORECAST_HORIZONS = (1, 2, 3, 4)
N_USABLE_STATIONS = 412  # stations with test-period samples in station_metadata
N_PRIMARY_MODELS = 3  # LSTM, CNN-LSTM-Temporal, CNN-LSTM+Attention
# Documented raw workbook size (PROJECT_VERIFICATION_REPORT / README) — NOT a KPI.
# Used only to distinguish cleaned feature_engineered_v2 rows from raw input.
RAW_DATASET_ROWS_DOCUMENTED = 970_339

# Session keys shared across pages
SS_STATION_ID = "selected_station_id"
SS_STATION_NAME = "selected_station_name"

# Two stations present in feature_engineered_v2.csv but absent from
# station_metadata.parquet (no contiguous test-period windows).
#
# Provenance: NOT typed from memory. Values were taken from a one-off
# programmatic query of feature_engineered_v2.csv during Phase 1
# (setdiff of feature stations vs station_metadata), then embedded here
# so the map can show all 414 without rewriting Phase 0 parquet.
# load_stations() cross-checks these fields against feature_engineered_v2.csv
# at runtime and raises if they drift.
ZERO_TEST_STATIONS = (
    {
        "station_id": "Chikkanahalli / Sadali_13.67_77.92_672",
        "station_name": "Chikkanahalli / Sadali",
        "latitude": 13.6667,
        "longitude": 77.9167,
        "state": "KA",
        "district": "Chikkaballapura",
        "elevation": 672,
        "n_test_samples_available": 0,
    },
    {
        "station_id": "Darjeeling_27.05_88.27_2127",
        "station_name": "Darjeeling",
        "latitude": 27.05,
        "longitude": 88.2667,
        "state": "WB",
        "district": "Darjeeling",
        "elevation": 2127,
        "n_test_samples_available": 0,
    },
)
