"""Home page — premium analytics landing (verified data only)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib.attention_data import chrono_to_day_labels, load_mean_attention_profile
from lib.comparison_data import MODEL_COLORS, PRIMARY_MODELS, load_primary_rmse_bars
from lib.home_data import (
    RMSE_BIN_COLORS,
    RMSE_BIN_LABELS,
    build_comparison_kpis,
    build_performance_highlights,
    count_cleaned_records,
    load_extreme_home_summary,
    load_network_rainfall_tail,
    load_station_wise_error,
)
from lib.paths import (
    FORECAST_HORIZONS,
    N_PRIMARY_MODELS,
    N_USABLE_STATIONS,
    RAW_DATASET_ROWS_DOCUMENTED,
    TEST_PERIOD_END,
    TEST_PERIOD_START,
)
from lib.stations import init_selection_state, load_stations, selected_station_banner
from lib.style import LABEL_3SEED, LABEL_SEED42, inject_base_css, render_disclaimer
from lib.ui_components import (
    apply_plotly_theme,
    card_container,
    render_card_header,
    render_highlight_items,
    render_kpi_card,
    render_kpi_row,
    render_mini_metrics,
    render_page_header,
    render_section_header,
    render_section_label,
)

inject_base_css()
init_selection_state()

render_page_header(
    "Rainfall Forecasting Dashboard",
    eyebrow="Research Analytics",
    subtitle=(
        "Attention-Based Temporal CNN-LSTM for Precipitation Forecasting Across India — "
        f"historical replay of the locked test period "
        f"({TEST_PERIOD_START} → {TEST_PERIOD_END})."
    ),
    show_status_chips=True,
)

render_disclaimer()

cleaned_n = count_cleaned_records()
kpis = build_comparison_kpis()
pct = kpis["h4_pct_reduction"]
pct_label = f"{abs(pct):.1f}%" if pct > 0 else f"+{abs(pct):.1f}%"
pct_delta = (
    f"RMSE ↓ vs Temporal at h=4 · significant={kpis['h4_significant']}"
    if pct > 0
    else f"RMSE ↑ vs Temporal at h=4 · significant={kpis['h4_significant']}"
)

render_section_label("At a glance")
render_kpi_row(
    [
        render_kpi_card(
            label="Cleaned records",
            value=f"{cleaned_n:,}",
            sublabel=f"feature_engineered_v2.csv · raw ~{RAW_DATASET_ROWS_DOCUMENTED:,}",
            icon="database",
            accent="#3b82f6",
        ),
        render_kpi_card(
            label="Usable stations",
            value=str(N_USABLE_STATIONS),
            sublabel="With test-period samples",
            icon="map-pin",
            accent="#22c55e",
            value_accent=True,
        ),
        render_kpi_card(
            label="Forecast horizons",
            value=str(len(FORECAST_HORIZONS)),
            sublabel="h=" + ", ".join(str(h) for h in FORECAST_HORIZONS),
            icon="clock",
            accent="#a78bfa",
            value_accent=True,
        ),
        render_kpi_card(
            label="Primary models",
            value=str(N_PRIMARY_MODELS),
            sublabel="LSTM · Temporal · Attention",
            icon="layers",
            accent="#f97316",
            value_accent=True,
        ),
        render_kpi_card(
            label="Best verified gain",
            value=pct_label,
            sublabel="Attention vs CNN-LSTM-Temporal · ablation_study.csv",
            icon="trending-up",
            accent="#14b8a6",
            value_accent=True,
            delta=pct_delta,
        ),
    ]
)

st.caption(
    f"Interactive replay uses {LABEL_SEED42}. Headline comparison metrics use "
    f"{LABEL_3SEED}. GNN-LSTM and Transformer are out of scope on this page."
)

render_section_header("Core analytics", "Map · model RMSE · verified highlights")
col_map, col_bars, col_hi = st.columns([1.25, 1.15, 0.95], gap="medium")

with col_map:
    with card_container():
        render_card_header(
            "Station RMSE map",
            caption=f"Attention · {LABEL_SEED42} · station_wise_error.csv",
        )
        map_h = st.radio(
            "Map horizon",
            options=[1, 4],
            index=0,
            horizontal=True,
            key="home_map_horizon",
            help="Feature 6 station_wise_error.csv includes h=1 and h=4 only.",
            label_visibility="collapsed",
        )
        st.caption(f"Horizon h={map_h} · discrete RMSE bins (mm/day)")
        err = load_station_wise_error(int(map_h)).copy()
        err["RMSE_bin"] = pd.Categorical(
            err["RMSE_bin"], categories=list(RMSE_BIN_LABELS), ordered=True
        )
        fig_map = px.scatter_map(
            err,
            lat="latitude",
            lon="longitude",
            color="RMSE_bin",
            color_discrete_map=dict(RMSE_BIN_COLORS),
            category_orders={"RMSE_bin": list(RMSE_BIN_LABELS)},
            hover_name="station_name",
            hover_data={
                "RMSE": ":.3f",
                "n_test_samples": True,
                "RMSE_bin": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=3.7,
            center={"lat": 22.8, "lon": 79.0},
            height=420,
        )
        fig_map.update_layout(
            map_style="carto-darkmatter",
            margin=dict(l=0, r=0, t=4, b=0),
            legend=dict(
                title="RMSE (mm/day)",
                orientation="h",
                yanchor="bottom",
                y=0.01,
                x=0.01,
                bgcolor="rgba(11,18,32,0.82)",
                font=dict(size=11),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e8eef7",
        )
        st.plotly_chart(fig_map, use_container_width=True)
        try:
            st.page_link("pages/station_explorer.py", label="Open Station Explorer →")
        except Exception:
            st.caption("Use sidebar → Station Explorer")

with col_bars:
    with card_container():
        render_card_header(
            "Model comparison (RMSE)",
            caption=f"{LABEL_3SEED} · master_results.csv",
        )
        metrics = load_primary_rmse_bars()
        fig_bars = px.bar(
            metrics,
            x="Horizon",
            y="RMSE_mean",
            color="Model",
            error_y="RMSE_std",
            barmode="group",
            category_orders={
                "Model": list(PRIMARY_MODELS),
                "Horizon": list(FORECAST_HORIZONS),
            },
            color_discrete_map={m: MODEL_COLORS[m] for m in PRIMARY_MODELS},
            labels={"RMSE_mean": "RMSE (mm/day)", "Horizon": "Forecast horizon"},
            height=420,
        )
        apply_plotly_theme(fig_bars, height=420)
        fig_bars.update_layout(
            margin=dict(l=40, r=10, t=28, b=36),
            legend_title_text="",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                x=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            bargap=0.22,
            yaxis_title="RMSE (mm/day)",
        )
        fig_bars.update_xaxes(dtick=1, tickmode="linear")
        if pct > 0:
            fig_bars.add_annotation(
                text=f"h=4 Att. vs Temp. · {abs(pct):.1f}% ↓ RMSE",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                showarrow=False,
                font=dict(size=11, color="#4ade80"),
                bgcolor="rgba(20,83,45,0.55)",
                bordercolor="rgba(34,197,94,0.4)",
                borderpad=4,
                xanchor="right",
            )
        st.plotly_chart(fig_bars, use_container_width=True)
        try:
            st.page_link("pages/model_comparison.py", label="Full model comparison →")
        except Exception:
            st.caption("Use sidebar → Model Comparison")

with col_hi:
    with card_container():
        render_card_header(
            "Performance highlights",
            caption="ablation_study.csv · multiseed_robustness_summary.csv",
        )
        bullets = build_performance_highlights()
        items: list[tuple[str, str]] = []
        for i, b in enumerate(bullets):
            tone = (
                "ok"
                if i == 0
                else (
                    "caveat"
                    if any(
                        k in b
                        for k in (
                            "LSTM",
                            "Not established",
                            "does not replicate",
                            "Neither",
                        )
                    )
                    else "warn"
                )
            )
            short = b if len(b) < 240 else b[:237] + "…"
            items.append((tone, short))
        render_highlight_items(items)

render_section_header("Deeper context", "Observed rainfall · extremes · attention")
c1, c2, c3, c4 = st.columns(4, gap="medium")

with c1:
    with card_container():
        render_card_header(
            "Recent rainfall snapshot",
            caption="Network mean/max · last 7 days of test period · observed",
        )
        snap = load_network_rainfall_tail(7)
        snap_view = snap.copy()
        snap_view["Date"] = snap_view["date_of_record"].dt.strftime("%Y-%m-%d")
        snap_view["Mean (mm)"] = snap_view["mean_mm"].round(2)
        snap_view["Max (mm)"] = snap_view["max_mm"].round(2)
        st.dataframe(
            snap_view[["Date", "Mean (mm)", "Max (mm)"]],
            use_container_width=True,
            hide_index=True,
            height=240,
        )
        try:
            st.page_link("pages/historical_analysis.py", label="Historical Analysis →")
        except Exception:
            pass

with c2:
    with card_container():
        render_card_header(
            "Extreme rainfall summary",
            caption="extreme_rainfall_evaluation.csv · 95th-pct split",
        )
        ext = load_extreme_home_summary()
        render_mini_metrics(
            [
                (
                    "red",
                    "h=1 threshold",
                    f"{ext['threshold_h1']:.1f} mm"
                    if ext["threshold_h1"] is not None
                    else "—",
                ),
                (
                    "amber",
                    "Att. better vs Temp.",
                    f"{ext['horizons_att_better_vs_temporal']}/{ext['n_horizons']} h",
                ),
                (
                    "green",
                    "Att. extreme RMSE mean",
                    f"{ext['att_extreme_rmse_mean']:.2f}",
                ),
            ]
        )
        st.caption(
            "Tiles compare Attention vs Temporal extreme-day RMSE across horizons "
            "(lower RMSE = better). All models degrade on extreme days."
        )

with c3:
    with card_container():
        render_card_header(
            "Attention insight",
            caption=f"All-test mean α · Feature 5 · {LABEL_SEED42}",
        )
        mean_h1 = chrono_to_day_labels(load_mean_attention_profile(1))
        mean_h4 = chrono_to_day_labels(load_mean_attention_profile(4))
        fig_attn = go.Figure()
        fig_attn.add_trace(
            go.Scatter(
                x=mean_h1["day_offset"],
                y=mean_h1["weight"],
                mode="lines",
                name="h=1",
                line=dict(color="#14b8a6", width=2.2),
            )
        )
        fig_attn.add_trace(
            go.Scatter(
                x=mean_h4["day_offset"],
                y=mean_h4["weight"],
                mode="lines",
                name="h=4",
                line=dict(color="#a78bfa", width=2.2),
            )
        )
        apply_plotly_theme(fig_attn, height=240)
        fig_attn.update_layout(
            margin=dict(l=36, r=8, t=8, b=32),
            legend=dict(orientation="h", y=1.15, x=0, bgcolor="rgba(0,0,0,0)"),
            xaxis_title="Day (−1 = most recent)",
            yaxis_title="Mean α",
            showlegend=True,
        )
        st.plotly_chart(fig_attn, use_container_width=True)
        st.caption(
            "h=1 peaks on day −1; h=4 is near-uniform. "
            "See Explainability for shared y-axis comparison."
        )
        try:
            st.page_link("pages/explainability.py", label="Explainability →")
        except Exception:
            pass

with c4:
    with card_container():
        render_card_header("Quick actions", caption="Jump to analysis pages")
        for path, label, hint in [
            ("pages/station_explorer.py", "Station Explorer", "Map · search · filters"),
            ("pages/forecast_replay.py", "Forecast Replay", "Cached historical forecasts"),
            ("pages/model_comparison.py", "Model Comparison", "3-seed research metrics"),
            ("pages/historical_analysis.py", "Historical Analysis", "Seasonal & extreme"),
            ("pages/explainability.py", "Explainability", "30-day attention profiles"),
        ]:
            try:
                st.page_link(path, label=label)
            except Exception:
                st.markdown(f"**{label}**")
            st.caption(hint)

with card_container():
    render_card_header(
        "Station selection",
        caption="Shared across Forecast Replay and Explainability",
    )
    stations = load_stations()
    selected_station_banner(stations)
    st.caption("Pick a station on Station Explorer, then open Forecast Replay.")
