"""Model Comparison — research evaluation laboratory (visual redesign only)."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from lib.comparison_data import (
    MODEL_COLORS,
    PRIMARY_MODELS,
    format_mean_std,
    load_ablation_summary,
    load_gnn_secondary,
    load_primary_rmse_bars,
    load_seasonal_h4,
)
from lib.home_data import build_comparison_kpis
from lib.style import (
    LABEL_3SEED,
    LABEL_SEED42,
    inject_base_css,
    render_dual_number_note,
)
from lib.ui_components import (
    apply_plotly_theme,
    card_container,
    render_card_header,
    render_finding_block,
    render_html,
    render_kpi_card,
    render_kpi_row,
    render_page_header,
    render_section_header,
)

inject_base_css()

HONESTY_CAVEAT = (
    "While CNN-LSTM+Attention significantly outperforms "
    "CNN-LSTM-Temporal at h=2 and h=4, it has NOT established a consistent "
    "advantage over plain LSTM. Attention-vs-LSTM is not significant at "
    "h=1, h=2, and h=4. At h=3, LSTM significantly outperforms Attention in "
    "2 of 3 seeds (13 and 123); seed 42 alone shows the reverse — a "
    "seed-dependent finding, not a resolved Attention win (see "
    "significance_results.csv / ablation_study.csv). LSTM remains numerically "
    "best across all seasons at h=4 (seasonal_performance.csv)."
)

render_page_header(
    "Model Comparison",
    eyebrow="Research Results",
    subtitle=(
        "Headline comparison is limited to LSTM, CNN-LSTM-Temporal, and "
        f"CNN-LSTM+Attention ({LABEL_3SEED}). GNN-LSTM is a secondary "
        "spatial investigation and is not shown in the primary chart."
    ),
    show_status_chips=True,
)

render_dual_number_note()
render_html(
    '<div class="honesty-box" role="note">'
    '<span class="label">Scientific caveat — read before interpreting charts</span>'
    f"{HONESTY_CAVEAT}</div>"
)

kpis = build_comparison_kpis()
pct = kpis["h4_pct_reduction"]
label_val = f"~{abs(pct):.1f}% ↓" if pct > 0 else f"~{abs(pct):.1f}% ↑"
hs = ", ".join(f"h={h}" for h in kpis["sig_horizons"]) or "none"

render_section_header("Research results at a glance", LABEL_3SEED)
render_kpi_row(
    [
        render_kpi_card(
            label="Best Att. vs Temporal (h=4)",
            value=label_val,
            sublabel=(
                f"Δ={kpis['h4_delta_rmse']:.4f} mm · significant={kpis['h4_significant']} "
                "· ablation_study.csv"
            ),
            icon="trending-up",
            accent="#14b8a6",
            value_accent=True,
        ),
        render_kpi_card(
            label="Horizons Att.>Temp significant",
            value=str(kpis["n_sig_horizons"]),
            sublabel=f"α=0.05: {hs} · significance_results.csv",
            icon="activity",
            accent="#3b82f6",
            value_accent=True,
        ),
        render_kpi_card(
            label="Vs plain LSTM",
            value="Not established",
            sublabel=kpis["caveat"],
            icon="alert",
            accent="#f59e0b",
            value_accent=True,
        ),
        render_kpi_card(
            label="Primary models",
            value="3",
            sublabel="LSTM · Temporal · Attention",
            icon="layers",
            accent="#a78bfa",
            value_accent=True,
        ),
    ],
    columns=4,
)

with card_container():
    render_card_header(
        "Test-set RMSE by model and horizon",
        caption=f"master_results.csv · {LABEL_3SEED} · error bars = 3-seed std · lower is better",
    )
    metrics = load_primary_rmse_bars()
    fig = px.bar(
        metrics,
        x="Horizon",
        y="RMSE_mean",
        color="Model",
        error_y="RMSE_std",
        barmode="group",
        category_orders={"Model": list(PRIMARY_MODELS), "Horizon": [1, 2, 3, 4]},
        color_discrete_map={m: MODEL_COLORS[m] for m in PRIMARY_MODELS},
        labels={"RMSE_mean": "RMSE (mm)", "Horizon": "Forecast horizon"},
        height=460,
    )
    apply_plotly_theme(fig, height=460)
    fig.update_layout(
        margin=dict(l=48, r=16, t=28, b=44),
        legend_title_text="",
        legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
        bargap=0.2,
        yaxis_title="RMSE (mm / day)",
    )
    fig.update_xaxes(dtick=1, tickmode="linear")
    st.plotly_chart(fig, use_container_width=True)

# Key findings from verified ablation/significance (no invented claims)
abl = load_ablation_summary()
att_abl = abl[abl["Model"] == "CNN-LSTM+Attention"].set_index("Horizon")
temp_abl = abl[abl["Model"] == "CNN-LSTM-Temporal"].set_index("Horizon")
finding_rows: list[tuple[str, str]] = []
for h in kpis["sig_horizons"]:
    delta = float(att_abl.loc[h, "Delta_RMSE_vs_previous_stage"])
    temp_rmse = float(temp_abl.loc[h, "RMSE_mean"])
    pct_h = 100.0 * delta / temp_rmse
    direction = "RMSE reduction" if delta < 0 else "RMSE increase"
    finding_rows.append(
        (f"h={h} Att. vs Temporal", f"~{abs(pct_h):.1f}% {direction} (Δ={delta:.4f})")
    )
finding_rows.append(
    ("Significant horizons (α=0.05)", hs if hs != "none" else "none")
)
finding_rows.append(("Attention vs plain LSTM", "Not established (see caveat)"))

c_find, c_table = st.columns([1.0, 1.2], gap="medium")
with c_find:
    with card_container():
        render_card_header("Key findings", caption="Verified ablation + significance only")
        render_finding_block(
            "Attention vs CNN-LSTM-Temporal",
            finding_rows,
            footnote="Source: ablation_study.csv · significance_results.csv",
        )

with c_table:
    with card_container():
        render_card_header(
            "Primary metrics (RMSE, MAE, R²)",
            caption=f"master_results.csv · {LABEL_3SEED}",
        )
        st.caption(
            "LSTM MAE/R² for h≥2 are recorded as Not Available in the verified table."
        )
        show = metrics.copy()
        show["RMSE"] = show.apply(
            lambda r: format_mean_std(r["RMSE_mean"], r["RMSE_std"]), axis=1
        )
        show["MAE"] = show.apply(
            lambda r: format_mean_std(r["MAE_mean"], r["MAE_std"]), axis=1
        )
        show["R²"] = show.apply(
            lambda r: format_mean_std(r["R2_mean"], r["R2_std"]), axis=1
        )
        st.dataframe(
            show[["Model", "Horizon", "RMSE", "MAE", "R²"]],
            use_container_width=True,
            hide_index=True,
            height=320,
        )

with card_container():
    render_card_header(
        f"h=4 seasonal RMSE ({LABEL_SEED42})",
        caption="seasonal_performance.csv · supports the honesty caveat",
    )
    st.caption(
        f"LSTM is numerically best in every season at h=4. "
        f"These seed-42 figures are not interchangeable with the {LABEL_3SEED} "
        "headline RMSE above."
    )
    seasonal = load_seasonal_h4()
    season_pivot = (
        seasonal.pivot_table(
            index="Season", columns="Model", values="RMSE", aggfunc="first"
        )
        .reindex(columns=list(PRIMARY_MODELS))
        .round(3)
    )
    st.dataframe(season_pivot, use_container_width=True)

with card_container():
    render_card_header(
        "Ablation summary (stage-wise deltas)",
        caption=f"ablation_study.csv · {LABEL_3SEED} · Negative Δ = lower RMSE (better)",
    )
    st.caption(
        "Δ vs previous stage: Temporal vs LSTM, then Attention vs Temporal."
    )
    abl_view = abl.rename(
        columns={
            "RMSE_mean": "RMSE (3-seed mean)",
            "MAE_mean": "MAE (3-seed mean)",
            "R2_mean": "R² (3-seed mean)",
            "Delta_RMSE_vs_previous_stage": "Δ RMSE vs previous stage",
            "Significant_vs_previous_stage": "Significant vs previous?",
            "Delta_RMSE_vs_LSTM": "Δ RMSE vs LSTM",
            "Significant_vs_LSTM": "Significant vs LSTM?",
        }
    )
    st.dataframe(abl_view, use_container_width=True, hide_index=True)
    st.markdown(
        """
- **LSTM** — baseline stage (Δ vs previous = 0).
- **CNN-LSTM-Temporal** — Δ vs previous = Temporal − LSTM.
- **CNN-LSTM+Attention** — Δ vs previous = Attention − Temporal.
"""
    )

st.markdown(
    '<p class="secondary-gnn-note">Secondary material (not a headline claim)</p>',
    unsafe_allow_html=True,
)
with st.expander("Secondary Investigation: GNN-LSTM", expanded=False):
    st.caption(
        "GNN-LSTM is a secondary spatial investigation on an irregular station graph — "
        "not part of the primary temporal-attention claim."
    )
    gnn_cmp, gnn_sig = load_gnn_secondary()
    st.markdown(f"##### GNN vs LSTM RMSE ({LABEL_3SEED})")
    gnn_show = gnn_cmp.rename(
        columns={
            "RMSE_mean": "GNN-LSTM RMSE",
            "RMSE_std": "GNN std",
            "LSTM_RMSE": "LSTM RMSE",
            "LSTM_std": "LSTM std",
            "Delta_RMSE_GNN_minus_LSTM": "Δ (GNN − LSTM)",
        }
    )[
        [
            "Horizon",
            "GNN-LSTM RMSE",
            "GNN std",
            "LSTM RMSE",
            "LSTM std",
            "Δ (GNN − LSTM)",
        ]
    ]
    st.dataframe(gnn_show, use_container_width=True, hide_index=True)
    st.markdown("##### Significance: GNN_vs_LSTM (significance_results.csv)")
    st.dataframe(
        gnn_sig[
            [
                "Forecast_Horizon",
                "DM_p_value",
                "Bootstrap_95CI_lo",
                "Bootstrap_95CI_hi",
                "Significant_at_0.05",
                "Notes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Positive Δ / CI above 0 means GNN has higher RMSE than LSTM "
        "(GNN worse on this metric in the locked eval)."
    )
