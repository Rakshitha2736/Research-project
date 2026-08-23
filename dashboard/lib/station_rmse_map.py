"""Shared Feature-6 station RMSE map (discrete bins from home_data).

Used by Station Explorer (Phase 2). Home keeps its own inline map so that
verified Home logic stays untouched; both call sites single-source binning
via load_station_wise_error / RMSE_BIN_*.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .home_data import (
    RMSE_BIN_COLORS,
    RMSE_BIN_LABELS,
    load_station_wise_error,
)
from .paths import SS_STATION_ID
from .stations import set_selected_station


def build_rmse_map_figure(
    horizon: int,
    *,
    selected_station_id: str | None = None,
    height: int = 520,
) -> go.Figure:
    """Discrete-bin RMSE map for Attention seed-42 at h=1 or h=4."""
    err = load_station_wise_error(int(horizon)).copy()
    err["RMSE_bin"] = pd.Categorical(
        err["RMSE_bin"], categories=list(RMSE_BIN_LABELS), ordered=True
    )
    err["marker_size"] = 8
    if selected_station_id:
        err.loc[err["station_id"] == selected_station_id, "marker_size"] = 16

    fig = px.scatter_map(
        err,
        lat="latitude",
        lon="longitude",
        color="RMSE_bin",
        color_discrete_map=dict(RMSE_BIN_COLORS),
        category_orders={"RMSE_bin": list(RMSE_BIN_LABELS)},
        size="marker_size",
        size_max=18,
        hover_name="station_name",
        hover_data={
            "RMSE": ":.3f",
            "n_test_samples": True,
            "RMSE_bin": True,
            "latitude": False,
            "longitude": False,
            "marker_size": False,
        },
        custom_data=["station_id"],
        zoom=4.0,
        center={"lat": 22.8, "lon": 79.0},
        height=height,
    )
    fig.update_layout(
        map_style="carto-darkmatter",
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(
            title="RMSE (mm/day)",
            orientation="h",
            yanchor="bottom",
            y=0.01,
            x=0.01,
            bgcolor="rgba(15,20,25,0.75)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e8eef7",
    )
    fig.update_traces(marker=dict(opacity=0.88))

    # Distinct ring for the selected station (if present in this horizon table)
    if selected_station_id:
        sel = err.loc[err["station_id"] == selected_station_id]
        if not sel.empty:
            r = sel.iloc[0]
            fig.add_trace(
                go.Scattermap(
                    lat=[float(r["latitude"])],
                    lon=[float(r["longitude"])],
                    mode="markers",
                    marker=dict(
                        size=22,
                        color="rgba(0,0,0,0)",
                        opacity=1.0,
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Selected",
                )
            )
            # Plotly map markers don't support stroke reliably across versions;
            # add a bright outer marker underneath via a second filled point.
            fig.add_trace(
                go.Scattermap(
                    lat=[float(r["latitude"])],
                    lon=[float(r["longitude"])],
                    mode="markers",
                    marker=dict(size=14, color="#f5f7fa", opacity=1.0),
                    hovertemplate=(
                        f"<b>{r['station_name']}</b> (selected)<br>"
                        f"RMSE=%{{customdata[0]:.3f}}<extra></extra>"
                    ),
                    customdata=[[float(r["RMSE"])]],
                    showlegend=True,
                    name="Selected station",
                )
            )
    return fig


def render_station_rmse_map(
    stations: pd.DataFrame,
    *,
    key_prefix: str,
    height: int = 520,
) -> None:
    """Radio h=1|h=4 + map; click updates shared station selection."""
    map_h = st.radio(
        "Map horizon",
        options=[1, 4],
        index=0,
        horizontal=True,
        key=f"{key_prefix}_map_horizon",
        help="Feature 6 station_wise_error.csv includes h=1 and h=4 only.",
    )
    st.caption(
        f"Attention, seed-42, h={map_h} · source: `station_wise_error.csv` · "
        "discrete RMSE bins (mm/day), not a continuous gradient."
    )

    selected = st.session_state.get(SS_STATION_ID)
    fig = build_rmse_map_figure(
        int(map_h), selected_station_id=selected, height=height
    )
    event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=f"{key_prefix}_rmse_map",
    )

    points = []
    if event is not None:
        sel = getattr(event, "selection", None)
        if sel is not None:
            points = getattr(sel, "points", None) or []
            if not points and isinstance(sel, dict):
                points = sel.get("points", [])

    if not points:
        return

    pt = points[0]
    sid = None
    if isinstance(pt, dict):
        cd = pt.get("customdata")
        if isinstance(cd, (list, tuple)) and cd:
            # customdata may be [station_id] or nested
            sid = cd[0] if not isinstance(cd[0], (list, tuple)) else cd[0][0]
        elif cd is not None and not isinstance(cd, (list, tuple)):
            sid = cd
    if sid is not None and str(sid) != st.session_state.get(SS_STATION_ID):
        set_selected_station(str(sid), stations)
        st.rerun()
