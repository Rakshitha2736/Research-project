"""Latest-Available Forecast — forward inference beyond each station's last window.

Distinct from all cache-based historical pages: runs on-demand CPU inference
on each station's most recent contiguous 30-day observed window. Predictions
are dated relative to that station's window_end_date (not a shared calendar
"now"). Dataset ends 2025-02-10; this is NOT a real-world current-date forecast.
"""

from __future__ import annotations

import html as html_lib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.comparison_data import MODEL_COLORS, PRIMARY_MODELS
from lib.imd_rainfall import classify_imd_rainfall
from lib.live_inference import HORIZONS, run_station_inference
from lib.paths import (
    LATEST_FORECAST_NETWORK_MAX_DAYS_STALE,
    LATEST_FORECAST_NETWORK_MIN_DAYS_STALE,
    LATEST_FORECAST_REFERENCE_DATE,
    LATEST_FORECAST_WINDOWS,
)
from lib.station_picker import render_cascading_station_picker
from lib.stations import init_selection_state, load_stations
from lib.style import inject_base_css
from lib.ui_components import (
    apply_plotly_theme,
    card_container,
    render_card_header,
    render_html,
    render_insight_card,
    render_kpi_card,
    render_kpi_row,
    render_page_header,
    render_section_header,
    render_status_panel,
)

# Framing banner — network-wide staleness is the general case (not conditional).
FORWARD_BANNER = (
    f"Every station's most recent available data is at least "
    f"{LATEST_FORECAST_NETWORK_MIN_DAYS_STALE} days old "
    f"(as of {LATEST_FORECAST_REFERENCE_DATE}) — the dataset ends 2025-02-10 "
    "and no more recent data exists in this project. This feature demonstrates "
    "the model's forward-inference mechanism using the best available data, "
    "not a current or recent-condition forecast. Model accuracy on this "
    "out-of-sample extrapolation is unverified — see Model Comparison for "
    "tested accuracy on known historical data."
)

inject_base_css()
init_selection_state()

render_page_header(
    "Latest-Available Forecast",
    eyebrow="Forward Forecast",
    subtitle=(
        "Genuine forward model inference beyond each station's last complete "
        "30-day observed window. Target dates are window_end + horizon — they "
        "differ by station and are not tied to the wall-clock calendar."
    ),
    show_status_chips=False,
)

render_html(
    '<div class="disclaimer-box" role="note">'
    '<span class="label">Network-Wide Stale Data — Forward Mechanism Demo</span>'
    f"{html_lib.escape(FORWARD_BANNER)}</div>"
)

render_insight_card(
    "How to read these dates",
    f"All {412} usable stations are critically stale "
    f"({LATEST_FORECAST_NETWORK_MIN_DAYS_STALE}–"
    f"{LATEST_FORECAST_NETWORK_MAX_DAYS_STALE} days vs "
    f"{LATEST_FORECAST_REFERENCE_DATE}). Horizon h=1..4 predictions are dated "
    "window_end_date + h. A separate extra-stale flag applies only when a "
    "station exceeds that usable-network ceiling (e.g. zero-test stations). "
    "IMD intensity badges are the published IMD daily rainfall classes — "
    "not the project's Feature-4 95th-percentile statistical extreme threshold.",
    tone="caveat",
)


@st.cache_data(show_spinner="Loading latest-window metadata…")
def load_latest_windows() -> pd.DataFrame:
    if not LATEST_FORECAST_WINDOWS.exists():
        raise FileNotFoundError(
            f"Missing {LATEST_FORECAST_WINDOWS}. Run build_latest_forecast_windows.py first."
        )
    df = pd.read_parquet(LATEST_FORECAST_WINDOWS)
    df["window_end_date"] = pd.to_datetime(df["window_end_date"])
    return df


stations = load_stations()
windows = load_latest_windows()

with card_container():
    render_card_header(
        "Station selection",
        caption="Uses each station's own last contiguous 30-day window",
    )
    sid = render_cascading_station_picker(
        stations,
        key_prefix="latest_fc",
        title=None,
    )

if not sid:
    render_status_panel(
        "Select a station",
        "Choose a station to run on-demand CPU inference on its latest-available "
        "30-day observed window.",
        tone="info",
    )
    st.stop()

win_row = windows.loc[windows["station_id"] == sid]
if win_row.empty:
    st.error("No latest-window metadata for this station.")
    st.stop()

win_row = win_row.iloc[0]
window_end = pd.Timestamp(win_row["window_end_date"]).normalize()
days_stale = int(win_row["days_stale"])
warning_flag = bool(win_row["warning_flag"])
station_row = stations.loc[stations["station_id"] == sid].iloc[0]
station_name = str(station_row["station_name"])

render_section_header(
    "Last observed window",
    f"{station_name} · contiguous 30-day input ending on window_end_date",
)

render_kpi_row(
    [
        render_kpi_card(
            label="Last observed data",
            value=str(window_end.date()),
            sublabel=(
                f"{days_stale} days before {LATEST_FORECAST_REFERENCE_DATE}"
            ),
            icon="clock",
            # Amber for all (network-wide critical staleness); red if extra-stale
            accent="#ef4444" if warning_flag else "#f59e0b",
            value_accent=True,
        ),
        render_kpi_card(
            label="Days stale",
            value=str(days_stale),
            sublabel=(
                "extra-stale vs network"
                if warning_flag
                else (
                    f"network norm "
                    f"{LATEST_FORECAST_NETWORK_MIN_DAYS_STALE}–"
                    f"{LATEST_FORECAST_NETWORK_MAX_DAYS_STALE}d"
                )
            ),
            icon="activity",
            accent="#ef4444" if warning_flag else "#f59e0b",
        ),
        render_kpi_card(
            label="Station",
            value=station_name[:18],
            sublabel=f"{station_row['state']} / {station_row['district']}",
            icon="map-pin",
            accent="#22c55e",
            value_accent=True,
        ),
        render_kpi_card(
            label="h=1 target date",
            value=str((window_end + pd.Timedelta(days=1)).date()),
            sublabel="window_end + 1 (not a shared calendar day)",
            icon="trending-up",
            accent="#a78bfa",
        ),
    ],
    columns=4,
)

if warning_flag:
    render_status_panel(
        "Extra-stale station (beyond usable-network norm)",
        f"This station's last contiguous 30-day window ends on {window_end.date()} "
        f"({days_stale} days before {LATEST_FORECAST_REFERENCE_DATE}). That is "
        f"meaningfully older than the usable-network ceiling of "
        f"{LATEST_FORECAST_NETWORK_MAX_DAYS_STALE} days "
        f"(usable stations: {LATEST_FORECAST_NETWORK_MIN_DAYS_STALE}–"
        f"{LATEST_FORECAST_NETWORK_MAX_DAYS_STALE} days stale). "
        "The network-wide banner above already covers the ~1.5-year baseline; "
        "this flag marks an even more severe case — interpret with extra caution.",
        tone="danger",
    )

with st.spinner("Running on-demand CPU inference (seed=42 checkpoints)…"):
    result = run_station_inference(sid, prefer_cuda=False)

render_section_header(
    "Horizon forecasts",
    "Predicted rainfall (mm) · IMD intensity category · three primary models",
)

for h in HORIZONS:
    target = result.target_dates[h]
    cols = st.columns(3)
    for i, model_name in enumerate(PRIMARY_MODELS):
        mm = float(result.predictions[(model_name, h)])
        cat = classify_imd_rainfall(mm)
        with cols[i]:
            with card_container():
                render_card_header(
                    f"h={h} · {target.date()}",
                    caption=f"{model_name} · IMD: {cat.name}",
                )
                color = MODEL_COLORS[model_name]
                render_html(
                    f'<div style="padding:0.35rem 0 0.15rem 0;">'
                    f'<div style="font-size:1.55rem;font-weight:700;color:{color};">'
                    f"{mm:.2f} <span style='font-size:0.85rem;font-weight:500;'>mm</span>"
                    f"</div>"
                    f'<div style="margin-top:0.45rem;display:inline-block;padding:0.2rem 0.55rem;'
                    f"border-radius:6px;background:{cat.color}22;border:1px solid {cat.color}66;"
                    f'color:{cat.color};font-size:0.78rem;font-weight:600;letter-spacing:0.02em;">'
                    f"IMD · {html_lib.escape(cat.name)}</div>"
                    f'<div style="margin-top:0.55rem;font-size:0.75rem;color:#8b9bb4;">'
                    f"Target date = window_end ({window_end.date()}) + {h}"
                    f"</div></div>"
                )

# Grouped chart
render_section_header(
    "Predicted rainfall by horizon",
    "One series per model · colors match Model Comparison",
)

chart_rows = []
for model_name in PRIMARY_MODELS:
    for h in HORIZONS:
        chart_rows.append(
            {
                "Horizon": f"h={h}",
                "Target date": str(result.target_dates[h].date()),
                "Model": model_name,
                "Predicted mm": result.predictions[(model_name, h)],
            }
        )
chart_df = pd.DataFrame(chart_rows)

with card_container():
    render_card_header(
        "Model predictions (mm)",
        caption="Grouped by horizon · on-demand seed=42 CPU inference",
    )
    fig = go.Figure()
    for model_name in PRIMARY_MODELS:
        sub = chart_df.loc[chart_df["Model"] == model_name]
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=sub["Horizon"],
                y=sub["Predicted mm"],
                marker_color=MODEL_COLORS[model_name],
                customdata=np.stack([sub["Target date"], sub["Model"]], axis=-1),
                hovertemplate=(
                    "%{customdata[1]}<br>%{x} · target %{customdata[0]}"
                    "<br>%{y:.2f} mm<extra></extra>"
                ),
            )
        )
    apply_plotly_theme(fig, height=380)
    fig.update_layout(
        barmode="group",
        margin=dict(l=44, r=16, t=28, b=48),
        xaxis_title="Forecast horizon",
        yaxis_title="Predicted rainfall (mm)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "IMD intensity classes (No Rain / Light / Moderate / Heavy / Very Heavy / "
        "Extremely Heavy) are shown on the cards above. They are distinct from the "
        "project's Feature-4 95th-percentile statistical extreme threshold used on "
        "Historical Analysis / Forecast Replay."
    )

# Attention mini-chart (C2: duplicate Explainability bar pattern; do not import/edit that page)
render_section_header(
    "Attention weights (CNN-LSTM+Attention)",
    "This forward pass only · day −1 = most recent observed day in the window",
)

attn_h = st.selectbox(
    "Attention horizon",
    options=list(HORIZONS),
    index=0,
    format_func=lambda h: f"h={h} · target {result.target_dates[h].date()}",
    key="latest_fc_attn_h",
)
attn = result.attention_by_horizon[int(attn_h)]
# Chronological α: index 0 = oldest (−30) … 29 = newest (−1) — same as Explainability
day_offsets = np.arange(-30, 0)
sample_df = pd.DataFrame({"day_offset": day_offsets, "weight": attn})

with card_container():
    render_card_header(
        f"Attention for this forward pass — {station_name}, h={attn_h}",
        caption="On-demand CNN-LSTM+Attention · seed=42 · CPU",
    )
    colors = [
        "#14b8a6" if d == -1 else ("#f59e0b" if d == -30 else "#3b82f6")
        for d in sample_df["day_offset"]
    ]
    fig_a = go.Figure(
        data=[
            go.Bar(
                x=sample_df["day_offset"],
                y=sample_df["weight"],
                marker_color=colors,
                hovertemplate="Day %{x}<br>α=%{y:.4f}<extra></extra>",
            )
        ]
    )
    fig_a.add_vline(
        x=-1, line_dash="dot", line_color="#14b8a6", annotation_text="most recent (−1)"
    )
    fig_a.add_vline(
        x=-30, line_dash="dot", line_color="#f59e0b", annotation_text="oldest (−30)"
    )
    apply_plotly_theme(fig_a, height=320)
    fig_a.update_layout(
        margin=dict(l=44, r=16, t=28, b=48),
        xaxis_title="Days within input window (−1 = most recent = window_end)",
        yaxis_title="Attention weight α",
        xaxis=dict(dtick=1, tickmode="linear", range=[-30.6, -0.4]),
        showlegend=False,
    )
    st.plotly_chart(fig_a, use_container_width=True)
