"""Explainability — attention interpretability dashboard (weights unchanged)."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from lib.attention_data import (
    chrono_to_day_labels,
    horizon_strategy_caption,
    load_attention_summary_table,
    load_mean_attention_profile,
    load_sample_attention,
)
from lib.forecast_data import model_input_window_bounds
from lib.replay_controls import render_station_date_horizon
from lib.stations import init_selection_state, load_stations
from lib.style import LABEL_SEED42, inject_base_css, render_disclaimer
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
)

inject_base_css()
init_selection_state()

render_page_header(
    "Explainability",
    eyebrow="Attention Profiles",
    subtitle=(
        "How the CNN-LSTM+Attention model weights the 30-day lookback window. "
        f"No model inference — weights from forecast_cache.parquet and Feature 5 "
        f"arrays ({LABEL_SEED42})."
    ),
    show_status_chips=True,
)

render_disclaimer()

render_html(
    '<div class="honesty-box" role="note">'
    '<span class="label">Day-index convention (locked for this project)</span>'
    "Day <b>−1</b> = <b>most recent</b> day in the lookback window · "
    "day <b>−30</b> = <b>oldest</b> day. "
    "The lookback window ends at <b>target − h</b> (not target − 1 when h&gt;1). "
    "This matches Feature 5 / attention_extreme_vs_normal.csv "
    "(day-position 1 = most recent … 30 = oldest).</div>"
)

mean_h1 = chrono_to_day_labels(load_mean_attention_profile(1))
mean_h4 = chrono_to_day_labels(load_mean_attention_profile(4))
a1 = float(mean_h1.loc[mean_h1.day_offset == -1, "weight"].iloc[0])
a4 = float(mean_h4.loc[mean_h4.day_offset == -1, "weight"].iloc[0])

render_kpi_row(
    [
        render_kpi_card(
            label="h=1 mean α (day −1)",
            value=f"{a1:.3f}",
            sublabel="Strong recency focus",
            icon="activity",
            accent="#14b8a6",
            value_accent=True,
        ),
        render_kpi_card(
            label="h=4 mean α (day −1)",
            value=f"{a4:.4f}",
            sublabel="Near-uniform profile",
            icon="brain",
            accent="#a78bfa",
            value_accent=True,
        ),
        render_kpi_card(
            label="Lookback window",
            value="30 days",
            sublabel="day −30 … day −1",
            icon="clock",
            accent="#3b82f6",
        ),
        render_kpi_card(
            label="Source",
            value="Seed-42",
            sublabel=LABEL_SEED42,
            icon="layers",
            accent="#f97316",
            value_accent=True,
        ),
    ],
    columns=4,
)

render_section_header(
    "Typical attention: h=1 vs h=4",
    "All-test mean profiles · shared y-axis so magnitude gap is visible",
)

with card_container():
    render_card_header(
        "All-test mean attention profiles",
        caption=f"Feature 5 attention_weights_h{{1,4}}_seed42.npy · {LABEL_SEED42}",
    )
    y_max = float(max(mean_h1["weight"].max(), mean_h4["weight"].max()) * 1.05)
    y_range = [0.0, y_max]

    cap1, cap4 = st.columns(2)
    with cap1:
        render_insight_card("h=1 attention strategy", horizon_strategy_caption(1), tone="ok")
    with cap4:
        render_insight_card(
            "h=4 attention strategy", horizon_strategy_caption(4), tone="caveat"
        )

    col_a, col_b = st.columns(2)

    def _mean_bar_figure(df, *, title: str, y_range: list[float]) -> go.Figure:
        colors = [
            "#14b8a6" if d == -1 else ("#f59e0b" if d == -30 else "#64748b")
            for d in df["day_offset"]
        ]
        fig = go.Figure(
            data=[
                go.Bar(
                    x=df["day_offset"],
                    y=df["weight"],
                    marker_color=colors,
                    hovertemplate="Day %{x}<br>mean α=%{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.add_vline(
            x=-1,
            line_dash="dot",
            line_color="#14b8a6",
            annotation_text="most recent (−1)",
        )
        fig.add_vline(
            x=-30,
            line_dash="dot",
            line_color="#f59e0b",
            annotation_text="oldest (−30)",
        )
        apply_plotly_theme(fig, height=380)
        fig.update_layout(
            title=title,
            margin=dict(l=44, r=12, t=48, b=48),
            xaxis_title="Day offset (−1 = most recent, −30 = oldest)",
            yaxis_title="Mean attention weight α",
            xaxis=dict(dtick=5, tickmode="linear", range=[-30.6, -0.4]),
            yaxis=dict(range=y_range, fixedrange=True),
            showlegend=False,
        )
        return fig

    with col_a:
        st.plotly_chart(
            _mean_bar_figure(
                mean_h1,
                title=f"h=1 mean (day −1 α={a1:.3f})",
                y_range=y_range,
            ),
            use_container_width=True,
        )
    with col_b:
        st.plotly_chart(
            _mean_bar_figure(
                mean_h4,
                title=f"h=4 mean (day −1 α={a4:.4f})",
                y_range=y_range,
            ),
            use_container_width=True,
        )
    st.caption(
        f"Shared y-axis: 0 – {y_max:.4f}. "
        "Independent auto-scaling would hide that h=4 weights are ~50× smaller."
    )

with card_container():
    render_card_header(
        "What this means",
        caption="Interpretation supported by Feature 5 / project findings only",
    )
    render_insight_card(
        "Research interpretation",
        "The attention profile shows how strongly the model weights historical days "
        "when generating its forecast. At h=1, attention concentrates on the most "
        "recent day (day −1). At h=4, attention is more evenly distributed across "
        "the 30-day window — a distinct strategy from h=1.",
        tone="caveat",
    )

render_section_header("Per-sample attention", "Station / date / horizon from cached forecasts")

with card_container():
    render_card_header(
        "Selection controls",
        caption="Restricted to cached test dates",
    )
    stations = load_stations()
    sid, target_date, horizons, _station_fc = render_station_date_horizon(
        stations,
        key_prefix="explain",
        horizon_mode="single",
        default_horizon=1,
    )

if not sid or target_date is None or not horizons:
    st.stop()

horizon = int(horizons[0])
station_name = stations.loc[stations["station_id"] == sid].iloc[0]["station_name"]
station_meta = stations.loc[stations["station_id"] == sid].iloc[0]
win_start, win_end = model_input_window_bounds(target_date, horizon)

render_insight_card(
    f"h={horizon} attention strategy",
    horizon_strategy_caption(horizon),
    tone="ok" if horizon == 1 else "caveat",
)
st.caption(
    f"Model input window: {win_start.date()} → {win_end.date()} "
    f"(window end = target − {horizon}; day −1 = {win_end.date()})."
)

sample_w = load_sample_attention(sid, target_date, horizon)
if sample_w is None:
    st.warning(
        "No Attention weights in the cache for this station / date / horizon "
        "(row may be missing at this horizon)."
    )
    st.stop()

mean_w = load_mean_attention_profile(horizon)
sample_df = chrono_to_day_labels(sample_w)
mean_df = chrono_to_day_labels(mean_w)

sample_peak_day = int(sample_df.loc[sample_df["weight"].idxmax(), "day_offset"])
mean_peak_day = int(mean_df.loc[mean_df["weight"].idxmax(), "day_offset"])
mean_day_minus1 = float(mean_df.loc[mean_df["day_offset"] == -1, "weight"].iloc[0])

render_kpi_row(
    [
        render_kpi_card(
            label="Station",
            value=str(station_name)[:18],
            sublabel=f"{station_meta['state']} / {station_meta['district']}",
            icon="map-pin",
            accent="#22c55e",
            value_accent=True,
        ),
        render_kpi_card(
            label="Target date",
            value=str(target_date.date()),
            sublabel=f"Horizon h={horizon}",
            icon="clock",
            accent="#3b82f6",
        ),
        render_kpi_card(
            label="Sample peak day",
            value=f"day {sample_peak_day}",
            sublabel="This forecast",
            icon="activity",
            accent="#14b8a6",
            value_accent=True,
        ),
        render_kpi_card(
            label="Typical peak day",
            value=f"day {mean_peak_day}",
            sublabel=f"All-test mean · day −1 α={mean_day_minus1:.4f}",
            icon="brain",
            accent="#a78bfa",
            value_accent=True,
        ),
    ],
    columns=4,
)

with card_container():
    render_card_header(
        f"Attention for this forecast — {station_name}, {target_date.date()}, h={horizon}",
        caption=f"forecast_cache.parquet · {LABEL_SEED42}",
    )
    colors = [
        "#14b8a6" if d == -1 else ("#f59e0b" if d == -30 else "#3b82f6")
        for d in sample_df["day_offset"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sample_df["day_offset"],
            y=sample_df["weight"],
            marker_color=colors,
            name="This forecast",
            hovertemplate="Day %{x}<br>α=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=-1, line_dash="dot", line_color="#14b8a6", annotation_text="most recent (−1)"
    )
    fig.add_vline(
        x=-30, line_dash="dot", line_color="#f59e0b", annotation_text="oldest (−30)"
    )
    apply_plotly_theme(fig, height=400)
    fig.update_layout(
        margin=dict(l=44, r=16, t=28, b=48),
        xaxis_title=(
            f"Days within input window (−1 = most recent = target−{horizon}, "
            "−30 = oldest)"
        ),
        yaxis_title="Attention weight α",
        xaxis=dict(dtick=1, tickmode="linear", range=[-30.6, -0.4]),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2, gap="medium")

with c1:
    with card_container():
        render_card_header(
            f"Typical attention at h={horizon}",
            caption=f"Feature 5 · {LABEL_SEED42}",
        )
        colors2 = [
            "#14b8a6" if d == -1 else ("#f59e0b" if d == -30 else "#64748b")
            for d in mean_df["day_offset"]
        ]
        fig2 = go.Figure()
        fig2.add_trace(
            go.Bar(
                x=mean_df["day_offset"],
                y=mean_df["weight"],
                marker_color=colors2,
                name="All-test mean",
                hovertemplate="Day %{x}<br>mean α=%{y:.4f}<extra></extra>",
            )
        )
        fig2.add_vline(
            x=-1, line_dash="dot", line_color="#14b8a6", annotation_text="most recent (−1)"
        )
        fig2.add_vline(
            x=-30, line_dash="dot", line_color="#f59e0b", annotation_text="oldest (−30)"
        )
        apply_plotly_theme(fig2, height=340)
        fig2.update_layout(
            margin=dict(l=44, r=12, t=20, b=48),
            xaxis_title="Day offset",
            yaxis_title="Mean α",
            xaxis=dict(dtick=1, tickmode="linear", range=[-30.6, -0.4]),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

with c2:
    with card_container():
        render_card_header("This forecast vs typical profile", caption="Overlay comparison")
        fig3 = make_subplots(specs=[[{"secondary_y": False}]])
        fig3.add_trace(
            go.Scatter(
                x=mean_df["day_offset"],
                y=mean_df["weight"],
                mode="lines",
                name="All-test mean",
                line=dict(color="#94a3b8", width=2),
            )
        )
        fig3.add_trace(
            go.Scatter(
                x=sample_df["day_offset"],
                y=sample_df["weight"],
                mode="lines+markers",
                name="This forecast",
                line=dict(color="#14b8a6", width=2),
                marker=dict(size=5),
            )
        )
        apply_plotly_theme(fig3, height=340)
        fig3.update_layout(
            margin=dict(l=44, r=12, t=20, b=48),
            xaxis_title="Day offset",
            yaxis_title="Attention weight α",
            legend=dict(orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(dtick=5, range=[-30.6, -0.4]),
        )
        st.plotly_chart(fig3, use_container_width=True)

with st.expander("Feature 5 Extreme vs Normal attention summary", expanded=False):
    st.caption(
        "From attention_extreme_vs_normal.csv. "
        "Peak_Day_Position: 1 = most recent … 30 = oldest."
    )
    summary = load_attention_summary_table()
    st.dataframe(
        summary[summary["Horizon"].astype(int) == horizon],
        use_container_width=True,
        hide_index=True,
    )
