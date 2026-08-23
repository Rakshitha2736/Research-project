"""Historical Analysis — seasonal and extreme-day research view."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from lib.comparison_data import MODEL_COLORS, PRIMARY_MODELS, load_seasonal_h4
from lib.forecast_data import load_extreme_thresholds
from lib.paths import PROJECT_ROOT
from lib.style import inject_base_css
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

TABLES = PROJECT_ROOT / "reports" / "tables"
SEASONAL_CSV = TABLES / "seasonal_performance.csv"
EXTREME_CSV = TABLES / "extreme_rainfall_evaluation.csv"

render_page_header(
    "Historical Analysis",
    eyebrow="Seasonal & Extreme",
    subtitle=(
        "Seasonal and extreme-day breakdowns from the locked test evaluation "
        "(2024-01-01 to 2025-02-10; seed-42 inference tables — no live model runs)."
    ),
    show_status_chips=True,
)

seasonal = pd.read_csv(SEASONAL_CSV, comment="#")
seasonal = seasonal[seasonal["Model"].isin(PRIMARY_MODELS)].copy()
seasonal["Horizon"] = seasonal["Horizon"].astype(int)
extreme = pd.read_csv(EXTREME_CSV, comment="#")
extreme = extreme[extreme["Model"].isin(PRIMARY_MODELS)].copy()
thr = load_extreme_thresholds()

# Compact verified KPIs (display only)
h4 = load_seasonal_h4()
lstm_h4_best = True
if not h4.empty:
    # Check LSTM numerically lowest RMSE in every season at h=4
    for season, g in h4.groupby("Season"):
        best = g.loc[g["RMSE"].idxmin(), "Model"]
        if best != "LSTM":
            lstm_h4_best = False
            break

render_kpi_row(
    [
        render_kpi_card(
            label="Seasonal table",
            value="seed-42",
            sublabel="seasonal_performance.csv",
            icon="clock",
            accent="#3b82f6",
        ),
        render_kpi_card(
            label="Extreme split",
            value="95th pct",
            sublabel="extreme_rainfall_evaluation.csv",
            icon="alert",
            accent="#ef4444",
            value_accent=True,
        ),
        render_kpi_card(
            label="h=1 threshold",
            value=f"{thr[1]:.1f} mm" if 1 in thr else "—",
            sublabel="Extreme = above threshold",
            icon="activity",
            accent="#f59e0b",
            value_accent=True,
        ),
        render_kpi_card(
            label="h=4 seasonal note",
            value="LSTM lowest" if lstm_h4_best else "See table",
            sublabel="Across seasons at h=4 (seed-42)",
            icon="layers",
            accent="#22c55e",
            value_accent=True,
        ),
    ],
    columns=4,
)

render_section_header("Seasonal performance", "Winter · Summer · Monsoon · Post-monsoon")

with card_container():
    render_card_header(
        "Seasonal RMSE (seed 42)",
        caption="seasonal_performance.csv · seasons match original EDA column definitions",
    )
    horizons = st.multiselect(
        "Show horizons",
        options=[1, 4],
        default=[1, 4],
        format_func=lambda h: f"h={h}",
        key="hist_horizons",
    )
    if not horizons:
        st.warning("Select at least one horizon.")
        st.stop()

    plot_df = seasonal[seasonal["Horizon"].isin(horizons)]
    fig = px.bar(
        plot_df,
        x="Season",
        y="RMSE",
        color="Model",
        facet_col="Horizon",
        barmode="group",
        category_orders={
            "Season": ["Winter", "Summer", "Monsoon", "Post-monsoon"],
            "Model": list(PRIMARY_MODELS),
        },
        color_discrete_map={m: MODEL_COLORS[m] for m in PRIMARY_MODELS},
        labels={"RMSE": "RMSE (mm)"},
        height=460,
    )
    apply_plotly_theme(fig, height=460)
    fig.update_layout(
        margin=dict(l=44, r=16, t=48, b=40),
        legend_title_text="",
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

render_html(
    '<div class="honesty-box" role="note">'
    '<span class="label">h=4 seasonal note</span>'
    "LSTM remains numerically best across all four seasons at h=4 "
    "(see table below). Attention's significant gains are vs Temporal, not vs LSTM."
    "</div>"
)

with card_container():
    render_card_header("h=4 seasonal RMSE table", caption="seasonal_performance.csv · seed-42")
    pivot = (
        h4.pivot_table(index="Season", columns="Model", values="RMSE", aggfunc="first")
        .reindex(columns=list(PRIMARY_MODELS))
        .round(3)
    )
    st.dataframe(pivot, use_container_width=True)

render_section_header(
    "Extreme rainfall analysis",
    "95th-percentile split of observed target rainfall",
)

with card_container():
    render_card_header(
        "Extreme vs Normal days",
        caption="extreme_rainfall_evaluation.csv · all models degrade sharply on extreme days",
    )
    st.write(
        "Thresholds (mm): "
        + " · ".join(f"h={h}: **{thr[h]:.1f}**" for h in sorted(thr))
    )

    fig_e = px.bar(
        extreme,
        x="Horizon",
        y="RMSE",
        color="Model",
        facet_col="Subset",
        barmode="group",
        category_orders={
            "Model": list(PRIMARY_MODELS),
            "Subset": ["Normal", "Extreme"],
            "Horizon": [1, 2, 3, 4],
        },
        color_discrete_map={m: MODEL_COLORS[m] for m in PRIMARY_MODELS},
        labels={"RMSE": "RMSE (mm)"},
        height=460,
    )
    apply_plotly_theme(fig_e, height=460)
    fig_e.update_layout(
        margin=dict(l=44, r=16, t=48, b=40),
        legend_title_text="",
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig_e.update_xaxes(dtick=1)
    st.plotly_chart(fig_e, use_container_width=True)

with st.expander("Full extreme-evaluation table", expanded=False):
    st.dataframe(extreme, use_container_width=True, hide_index=True)
