"""Forecast Replay — historical forecast investigation console."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.forecast_data import (
    MODEL_ORDER,
    load_extreme_thresholds,
    load_station_rainfall_series,
    lookback_window,
    model_input_window_bounds,
    slice_forecast,
)
from lib.replay_controls import render_station_date_horizon
from lib.stations import init_selection_state, load_stations
from lib.style import LABEL_SEED42, inject_base_css, render_disclaimer
from lib.ui_components import (
    apply_plotly_theme,
    card_container,
    render_card_header,
    render_html,
    render_kpi_card,
    render_kpi_row,
    render_page_header,
    render_section_header,
)

inject_base_css()
init_selection_state()

render_page_header(
    "Forecast Replay",
    eyebrow="Historical Forecasts",
    subtitle=(
        "Replay cached historical forecasts for a station and test-period date. "
        f"No live model inference — numbers come from forecast_cache.parquet "
        f"({LABEL_SEED42})."
    ),
    show_status_chips=True,
)

render_disclaimer()

stations = load_stations()

with card_container():
    render_card_header(
        "Replay controls",
        caption="Station · cached target date · horizons",
    )
    sid, target_date, horizons, station_fc = render_station_date_horizon(
        stations,
        key_prefix="replay",
        horizon_mode="multi",
        default_horizons=[1, 2, 3, 4],
    )

if not sid or target_date is None or station_fc is None:
    st.stop()
if not horizons:
    st.warning("Select at least one horizon.")
    st.stop()

station_row = stations.loc[stations["station_id"] == sid].iloc[0]
view = slice_forecast(station_fc, target_date, horizons)
if view.empty:
    st.warning(
        "No cache rows for this station/date/horizon combination "
        "(date may not exist at every horizon)."
    )
    st.stop()

# Extreme flag
thresholds = load_extreme_thresholds()
extreme_notes: list[str] = []
for h in horizons:
    sub = view.loc[view["horizon"] == h]
    if sub.empty:
        continue
    y_true = float(sub["y_true_mm"].iloc[0])
    thr = float(thresholds[h])
    if y_true > thr:
        extreme_notes.append(
            f"h={h}: actual {y_true:.2f} mm exceeds Feature-4 95th-pct threshold "
            f"({thr:.1f} mm)"
        )

if extreme_notes:
    import html as html_lib

    body = (
        "Observed target-day rainfall exceeds the locked test-set 95th percentile "
        "for at least one selected horizon. Feature 4 showed all models perform "
        "notably worse on these days — large errors here are expected context, "
        "not a UI glitch. " + " · ".join(extreme_notes)
    )
    render_html(
        '<div class="extreme-flag-box" role="note">'
        '<span class="label">Extreme rainfall event</span>'
        f"{html_lib.escape(body)}</div>"
    )

render_section_header("Forecast summary", f"{station_row['station_name']} · {LABEL_SEED42}")

att = view.loc[view["model_name"] == "CNN-LSTM+Attention"]
summary_src = att if not att.empty else view
h0 = horizons[0]
sum_h = summary_src.loc[summary_src["horizon"] == h0]
if not sum_h.empty:
    row0 = sum_h.iloc[0]
    render_kpi_row(
        [
            render_kpi_card(
                label="Actual rainfall",
                value=f"{float(row0['y_true_mm']):.2f} mm",
                sublabel=f"Target date {target_date.date()}",
                icon="activity",
                accent="#3b82f6",
            ),
            render_kpi_card(
                label="Predicted",
                value=f"{float(row0['y_pred_mm']):.2f} mm",
                sublabel=str(row0["model_name"]),
                icon="trending-up",
                accent="#14b8a6",
                value_accent=True,
            ),
            render_kpi_card(
                label="Abs error",
                value=f"{float(row0['abs_error_mm']):.2f} mm",
                sublabel="|pred − actual|",
                icon="alert",
                accent="#f59e0b",
                value_accent=True,
            ),
            render_kpi_card(
                label="Horizon",
                value=f"h={h0}",
                sublabel=f"Station: {station_row['station_name']}",
                icon="clock",
                accent="#a78bfa",
                value_accent=True,
            ),
        ],
        columns=4,
    )
    st.caption(
        f"Summary tiles show {row0['model_name']} at h={h0} · {LABEL_SEED42}."
    )

tcol, ccol = st.columns([0.95, 1.15], gap="medium")

with tcol:
    with card_container():
        render_card_header(
            "Model comparison table",
            caption=f"forecast_cache.parquet · {LABEL_SEED42}",
        )
        table = view[
            ["horizon", "model_name", "y_pred_mm", "y_true_mm", "abs_error_mm"]
        ].rename(
            columns={
                "horizon": "Horizon",
                "model_name": "Model",
                "y_pred_mm": "Predicted (mm)",
                "y_true_mm": "Actual (mm)",
                "abs_error_mm": "Abs Error (mm)",
            }
        )
        table["Predicted (mm)"] = table["Predicted (mm)"].map(lambda x: round(float(x), 4))
        table["Actual (mm)"] = table["Actual (mm)"].map(lambda x: round(float(x), 4))
        table["Abs Error (mm)"] = table["Abs Error (mm)"].map(lambda x: round(float(x), 4))
        st.dataframe(table, use_container_width=True, hide_index=True)

with ccol:
    with card_container():
        render_card_header(
            "Actual vs predicted",
            caption="Grouped by horizon · all selected models",
        )
        chart_df = view.copy()
        chart_df["Horizon"] = chart_df["horizon"].map(lambda h: f"h={h}")
        pred_part = chart_df[["Horizon", "model_name", "y_pred_mm"]].rename(
            columns={"y_pred_mm": "mm", "model_name": "Model"}
        )
        pred_part["Series"] = "Predicted"
        act_part = chart_df[["Horizon", "model_name", "y_true_mm"]].rename(
            columns={"y_true_mm": "mm", "model_name": "Model"}
        )
        act_part["Series"] = "Actual"
        long_df = pd.concat([pred_part, act_part], ignore_index=True)

        fig = px.bar(
            long_df,
            x="Model",
            y="mm",
            color="Series",
            barmode="group",
            facet_col="Horizon",
            category_orders={
                "Model": list(MODEL_ORDER),
                "Series": ["Actual", "Predicted"],
                "Horizon": [f"h={h}" for h in horizons],
            },
            color_discrete_map={"Actual": "#64748b", "Predicted": "#14b8a6"},
            labels={"mm": "Rainfall (mm)", "Model": ""},
            height=420,
        )
        apply_plotly_theme(fig, height=420)
        fig.update_layout(
            margin=dict(l=24, r=16, t=48, b=28),
            legend_title_text="",
            bargap=0.25,
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        st.plotly_chart(fig, use_container_width=True)

with card_container():
    render_card_header(
        "30-day model input window (observed rainfall)",
        caption=(
            f"Exact 30 calendar days used as model input · "
            f"Station: {station_row['station_name']}"
        ),
    )
    rain = load_station_rainfall_series(sid)
    for h in horizons:
        win_start, win_end = model_input_window_bounds(target_date, h)
        ctx = lookback_window(rain, target_date, horizon=h, n_days=30)
        st.markdown(f"**h={h}** · input window ends {h} day(s) before the target date")
        if ctx.empty:
            st.warning(
                f"No rainfall observations found for the h={h} input window "
                f"({win_start.date()} → {win_end.date()})."
            )
            continue
        fig_ctx = go.Figure()
        fig_ctx.add_trace(
            go.Scatter(
                x=ctx["date_of_record"],
                y=ctx["rainfall"],
                mode="lines+markers",
                name="Observed rainfall",
                line=dict(color="#14b8a6", width=2),
                marker=dict(size=5),
            )
        )
        fig_ctx.add_vline(
            x=win_end,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text=f"h={h} window end",
            annotation_position="top",
        )
        apply_plotly_theme(fig_ctx, height=280)
        fig_ctx.update_layout(
            margin=dict(l=44, r=16, t=32, b=40),
            yaxis_title="Rainfall (mm)",
            xaxis_title="Date",
            showlegend=False,
        )
        st.plotly_chart(fig_ctx, use_container_width=True)
        st.caption(
            f"Input window: {win_start.date()} → {win_end.date()}  ·  "
            f"{len(ctx)} daily observations  ·  "
            f"target date {target_date.date()} not included (prediction target)."
        )
