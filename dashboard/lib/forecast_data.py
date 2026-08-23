"""Forecast cache + extreme thresholds + lookback rainfall loaders."""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from .paths import (
    FORECAST_CACHE,
    PROJECT_ROOT,
    TEST_PERIOD_END,
    TEST_PERIOD_START,
)

EXTREME_CSV = PROJECT_ROOT / "reports" / "tables" / "extreme_rainfall_evaluation.csv"
FEAT_CSV = PROJECT_ROOT / "data" / "processed" / "feature_engineered_v2.csv"

MODEL_ORDER = ("LSTM", "CNN-LSTM-Temporal", "CNN-LSTM+Attention")
MODEL_COLORS = {
    "LSTM": "#4c78a8",
    "CNN-LSTM-Temporal": "#f58518",
    "CNN-LSTM+Attention": "#0f6b5c",
}


@st.cache_data(show_spinner="Loading extreme thresholds…")
def load_extreme_thresholds() -> dict[int, float]:
    """Per-horizon 95th-percentile thresholds (mm) from Feature 4 table."""
    df = pd.read_csv(EXTREME_CSV, comment="#")
    out: dict[int, float] = {}
    for h, g in df.groupby("Horizon"):
        out[int(h)] = float(g["Threshold_mm"].iloc[0])
    return out


@st.cache_data(show_spinner="Loading forecast cache…")
def load_station_forecasts(station_id: str) -> pd.DataFrame:
    """All cached forecast rows for one station (all models, horizons, dates)."""
    if not FORECAST_CACHE.exists():
        raise FileNotFoundError(FORECAST_CACHE)
    table = pq.read_table(
        FORECAST_CACHE,
        columns=[
            "station_id",
            "station_name",
            "target_date",
            "horizon",
            "model_name",
            "y_true_mm",
            "y_pred_mm",
            "abs_error_mm",
        ],
        filters=[("station_id", "=", station_id)],
    )
    df = table.to_pandas()
    if df.empty:
        return df
    df["target_date"] = pd.to_datetime(df["target_date"]).dt.normalize()
    df["horizon"] = df["horizon"].astype(int)
    return df


def available_dates_for_station(station_df: pd.DataFrame) -> list[pd.Timestamp]:
    """Sorted unique target dates present in the cache for this station."""
    if station_df.empty:
        return []
    return list(sorted(station_df["target_date"].dropna().unique()))


def slice_forecast(
    station_df: pd.DataFrame,
    target_date: pd.Timestamp,
    horizons: list[int],
) -> pd.DataFrame:
    """Rows for (station already filtered, date, selected horizons), all models."""
    td = pd.Timestamp(target_date).normalize()
    mask = (station_df["target_date"] == td) & (station_df["horizon"].isin(horizons))
    out = station_df.loc[mask].copy()
    out["model_name"] = pd.Categorical(
        out["model_name"], categories=list(MODEL_ORDER), ordered=True
    )
    return out.sort_values(["horizon", "model_name"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading rainfall series…")
def _all_station_rainfall() -> pd.DataFrame:
    raw = pd.read_csv(
        FEAT_CSV,
        usecols=["station_id", "date_of_record", "rainfall"],
        parse_dates=["date_of_record"],
    )
    raw["date_of_record"] = pd.to_datetime(raw["date_of_record"]).dt.normalize()
    return raw


def load_station_rainfall_series(station_id: str) -> pd.DataFrame:
    """Daily rainfall for one station from feature_engineered_v2 (read-only)."""
    raw = _all_station_rainfall()
    g = raw.loc[raw["station_id"] == station_id, ["date_of_record", "rainfall"]].copy()
    return (
        g.sort_values("date_of_record")
        .drop_duplicates("date_of_record")
        .reset_index(drop=True)
    )


def model_input_window_bounds(
    target_date: pd.Timestamp,
    horizon: int,
    n_days: int = 30,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Inclusive [start, end] of the 30-day model input window for (target, h).

    Matches sequence construction: target = window_end + h, so
    window_end = target − h and the window is [window_end − (n_days−1), window_end].
    """
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    td = pd.Timestamp(target_date).normalize()
    window_end = td - pd.Timedelta(days=int(horizon))
    window_start = window_end - pd.Timedelta(days=int(n_days) - 1)
    return window_start, window_end


def lookback_window(
    rainfall: pd.DataFrame,
    target_date: pd.Timestamp,
    horizon: int,
    n_days: int = 30,
) -> pd.DataFrame:
    """Observed rainfall on the true model input dates for (target_date, horizon)."""
    start, end = model_input_window_bounds(target_date, horizon, n_days=n_days)
    mask = (rainfall["date_of_record"] >= start) & (rainfall["date_of_record"] <= end)
    return rainfall.loc[mask].copy()


def test_period_rainfall_summary(station_id: str) -> dict | None:
    """Mean / max / count of observed rainfall in the locked test period.

    Reuses load_station_rainfall_series (feature_engineered_v2.csv). Returns
    None if the station has no rows in [TEST_PERIOD_START, TEST_PERIOD_END].
    """
    rain = load_station_rainfall_series(station_id)
    if rain.empty:
        return None
    start = pd.Timestamp(TEST_PERIOD_START)
    end = pd.Timestamp(TEST_PERIOD_END)
    sub = rain[
        (rain["date_of_record"] >= start) & (rain["date_of_record"] <= end)
    ]
    if sub.empty:
        return None
    return {
        "n_days": int(len(sub)),
        "mean_mm": float(sub["rainfall"].mean()),
        "max_mm": float(sub["rainfall"].max()),
        "period_start": TEST_PERIOD_START,
        "period_end": TEST_PERIOD_END,
        "source": "feature_engineered_v2.csv",
    }
