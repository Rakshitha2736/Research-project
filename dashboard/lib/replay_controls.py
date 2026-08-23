"""Shared station + cache-restricted date (+ horizon) controls for Replay / Explainability."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import streamlit as st

from .forecast_data import available_dates_for_station, load_station_forecasts
from .paths import SS_STATION_ID, TEST_PERIOD_END, TEST_PERIOD_START
from .station_picker import render_cascading_station_picker
from .stations import selected_station_banner
from .style import render_empty_state


def render_station_date_horizon(
    stations: pd.DataFrame,
    *,
    key_prefix: str,
    horizon_mode: Literal["multi", "single"] = "multi",
    default_horizons: list[int] | None = None,
    default_horizon: int = 1,
) -> tuple[str | None, pd.Timestamp | None, list[int], pd.DataFrame | None]:
    """Cascading station picker + cached-only date selectbox + horizon control.

    Returns
    -------
    station_id, target_date, horizons, station_forecast_df
    (None placeholders if selection incomplete.)
    """
    selected_station_banner(stations)

    if not st.session_state.get(SS_STATION_ID):
        st.info("No station selected yet — pick one below (or on Station Explorer).")

    sid = render_cascading_station_picker(
        stations,
        key_prefix=key_prefix,
        show_clear=True,
        title="Station",
    )
    if not sid:
        return None, None, [], None

    station_row = stations.loc[stations["station_id"] == sid].iloc[0]
    if not bool(station_row["has_test_data"]):
        render_empty_state(
            "No test-period forecasts available for this station",
            "This station has no contiguous windows in the locked test period "
            f"({TEST_PERIOD_START} to {TEST_PERIOD_END}), so Forecast Replay / "
            "Explainability cannot load cached predictions here. "
            "Pick a station with test data, or clear the selection.",
        )
        return sid, None, [], None

    station_fc = load_station_forecasts(sid)
    valid_dates = available_dates_for_station(station_fc)
    if not valid_dates:
        render_empty_state(
            "No cached dates",
            "No cached dates found for this station in forecast_cache.parquet.",
        )
        return sid, None, [], station_fc
    date_labels = [d.strftime("%Y-%m-%d") for d in valid_dates]
    date_to_ts = dict(zip(date_labels, valid_dates))

    st.subheader("Target date & horizon")
    st.caption(
        f"Date list is **restricted to {len(valid_dates)} cached test dates** for this "
        "station. Dates outside this list cannot be selected."
    )

    c_date, c_h = st.columns([1, 2])
    with c_date:
        chosen_label = st.selectbox(
            "Target date (cached only)",
            options=date_labels,
            index=0,
            key=f"{key_prefix}_target_date",
        )
        target_date = date_to_ts[chosen_label]

    with c_h:
        if horizon_mode == "multi":
            if default_horizons is None:
                default_horizons = [1, 2, 3, 4]
            horizons = st.multiselect(
                "Horizons",
                options=[1, 2, 3, 4],
                default=default_horizons,
                format_func=lambda h: f"h={h}",
                key=f"{key_prefix}_horizons",
            )
            horizons = sorted(int(h) for h in horizons)
        else:
            h = st.selectbox(
                "Horizon",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(default_horizon)
                if default_horizon in (1, 2, 3, 4)
                else 0,
                format_func=lambda x: f"h={x}",
                key=f"{key_prefix}_horizon",
            )
            horizons = [int(h)]

    return sid, target_date, horizons, station_fc
