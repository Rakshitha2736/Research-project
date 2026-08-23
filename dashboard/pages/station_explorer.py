"""Station Explorer — geospatial analytics workspace (verified data only)."""

from __future__ import annotations

import streamlit as st

from lib.forecast_data import test_period_rainfall_summary
from lib.home_data import station_rmse_lookup
from lib.paths import N_USABLE_STATIONS, SS_STATION_ID
from lib.station_picker import render_cascading_station_picker
from lib.station_rmse_map import render_station_rmse_map
from lib.stations import (
    init_selection_state,
    load_stations,
    selected_station_banner,
    set_selected_station,
)
from lib.style import LABEL_SEED42, inject_base_css, render_empty_state
from lib.ui_components import (
    card_container,
    render_card_header,
    render_detail_grid,
    render_kpi_card,
    render_kpi_row,
    render_page_header,
    render_section_header,
)

inject_base_css()
init_selection_state()

render_page_header(
    "Station Explorer",
    eyebrow="Geographic Analysis",
    subtitle=(
        "Browse the 414 Indian weather stations used in this study. "
        "Search, filter, or click the RMSE map — all update the shared selection "
        "used by Forecast Replay and Explainability."
    ),
    show_status_chips=True,
)

stations = load_stations()
n_total = len(stations)
n_usable = int(stations["has_test_data"].sum())
n_zero = n_total - n_usable

render_kpi_row(
    [
        render_kpi_card(
            label="Stations shown",
            value=str(n_total),
            sublabel="414 = usable + zero-test supplements",
            icon="map-pin",
            accent="#3b82f6",
        ),
        render_kpi_card(
            label="Usable (test data)",
            value=str(n_usable),
            sublabel="station_metadata.parquet",
            icon="activity",
            accent="#22c55e",
            value_accent=True,
        ),
        render_kpi_card(
            label="Zero-test stations",
            value=str(n_zero),
            sublabel="No contiguous test windows",
            icon="alert",
            accent="#f59e0b",
            value_accent=True,
        ),
        render_kpi_card(
            label="Map RMSE source",
            value="Seed-42",
            sublabel=f"Attention · {LABEL_SEED42}",
            icon="layers",
            accent="#a78bfa",
            value_accent=True,
        ),
    ],
    columns=4,
)

selected_station_banner(stations)

render_section_header("Find & inspect", "Search · cascading filters · map selection")
left, right = st.columns([1.0, 1.35], gap="large")

with left:
    with card_container():
        render_card_header(
            "Search & filters",
            caption="Case-insensitive name search + State → District → Station",
        )
        search_q = st.text_input(
            "Search by station name",
            value="",
            placeholder="e.g. Agra, Darjeeling",
            key="explorer_name_search",
        )

        if search_q.strip():
            q = search_q.strip()
            matches = stations[
                stations["station_name"].astype(str).str.contains(q, case=False, na=False)
            ].sort_values("station_name")
            if matches.empty:
                render_empty_state(
                    "No matches",
                    "No stations match your filters. Try a different name substring.",
                )
            else:
                labels = {
                    f"{r.station_name} ({r.state} / {r.district})": r.station_id
                    for r in matches.itertuples()
                }
                opts = ["(pick a match)"] + list(labels.keys())
                pick = st.selectbox(
                    f"{len(matches)} match(es)",
                    opts,
                    key="explorer_search_pick",
                )
                if pick != "(pick a match)":
                    chosen = labels[pick]
                    if chosen != st.session_state.get(SS_STATION_ID):
                        set_selected_station(chosen, stations)
                        st.rerun()

        render_cascading_station_picker(
            stations,
            key_prefix="explorer",
            show_clear=True,
            title="State → District → Station",
        )

    with card_container():
        render_card_header(
            "Selected station",
            caption="Metadata from station_metadata.parquet",
        )
        sid = st.session_state.get(SS_STATION_ID)
        if not sid:
            render_empty_state(
                "No selection",
                "No station selected yet — use search, filters, or click the map.",
            )
        else:
            row = stations.loc[stations["station_id"] == sid]
            if row.empty:
                st.warning("Selected station id is not in the metadata table.")
            else:
                r = row.iloc[0]
                st.markdown(f"### {r['station_name']}")
                render_detail_grid(
                    [
                        (
                            "Station ID",
                            str(r["station_id"])[:48]
                            + ("…" if len(str(r["station_id"])) > 48 else ""),
                        ),
                        ("State", str(r["state"])),
                        ("District", str(r["district"])),
                        ("Elevation (m)", f"{int(r['elevation'])}"),
                        ("Latitude", f"{float(r['latitude']):.4f}"),
                        ("Longitude", f"{float(r['longitude']):.4f}"),
                        ("Test samples", f"{int(r['n_test_samples_available']):,}"),
                        ("Availability", str(r["availability"])),
                    ]
                )

                if not r["has_test_data"]:
                    render_empty_state(
                        "No test-period forecasts available for this station",
                        "No Forecast Replay cache rows and no station-wise RMSE in "
                        "station_wise_error.csv for this station.",
                    )
                else:
                    rmse = station_rmse_lookup(str(sid))
                    r1, r2 = st.columns(2)
                    h1 = rmse.get(1)
                    h4 = rmse.get(4)
                    r1.metric(
                        "RMSE h=1 (Attention, seed-42)",
                        f"{h1:.3f} mm/day" if h1 is not None else "Not in table",
                    )
                    r2.metric(
                        "RMSE h=4 (Attention, seed-42)",
                        f"{h4:.3f} mm/day" if h4 is not None else "Not in table",
                    )
                    st.caption("Source: station_wise_error.csv (Feature 6).")

                st.markdown("##### Historical rainfall (locked test period)")
                summary = test_period_rainfall_summary(str(sid))
                if summary is None:
                    st.caption(
                        "No rainfall rows in the locked test period for this station "
                        "(feature_engineered_v2.csv)."
                    )
                else:
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Days with records", f"{summary['n_days']:,}")
                    s2.metric("Mean rainfall", f"{summary['mean_mm']:.2f} mm/day")
                    s3.metric("Max rainfall", f"{summary['max_mm']:.2f} mm/day")
                    st.caption(
                        f"Source: {summary['source']} · "
                        f"{summary['period_start']} to {summary['period_end']} · "
                        "observed daily rainfall (not model predictions)."
                    )

with right:
    with card_container():
        render_card_header(
            "Station RMSE map",
            caption="Click a marker to select · Feature 6 discrete bins · Plotly native hover",
        )
        render_station_rmse_map(stations, key_prefix="explorer", height=680)
        st.caption(
            f"Map shows {N_USABLE_STATIONS} usable stations from station_wise_error.csv. "
            "The 2 zero-test stations have no RMSE markers — use search/filters to select them."
        )

st.caption(
    f"Showing {len(stations)} stations "
    f"({N_USABLE_STATIONS} with test data + 2 without)."
)
