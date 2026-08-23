"""Home-page data helpers (verified tables only, read-only, no inference)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .paths import (
    FEATURE_ENGINEERED_V2,
    STATION_WISE_ERROR_CSV,
    TABLES,
)

ABLATION_CSV = TABLES / "ablation_study.csv"
SIGNIFICANCE_CSV = TABLES / "significance_results.csv"

# Fixed discrete RMSE bins (mm/day) for Feature 6 station map — not continuous.
RMSE_BIN_EDGES = (0.0, 6.0, 8.0, 10.0, 15.0, float("inf"))
RMSE_BIN_LABELS = (
    "< 6",
    "6 – 8",
    "8 – 10",
    "10 – 15",
    "≥ 15",
)
RMSE_BIN_COLORS = {
    "< 6": "#1b9e77",
    "6 – 8": "#5ab4a0",
    "8 – 10": "#e6a23c",
    "10 – 15": "#e07b39",
    "≥ 15": "#c0392b",
}


@st.cache_data(show_spinner="Counting cleaned records…")
def count_cleaned_records() -> int:
    """Live row count of feature_engineered_v2.csv (cleaned records)."""
    if not FEATURE_ENGINEERED_V2.exists():
        raise FileNotFoundError(FEATURE_ENGINEERED_V2)
    # Fast line count; subtract header. Does not load the full CSV into memory.
    with open(FEATURE_ENGINEERED_V2, "rb") as f:
        n = sum(1 for _ in f) - 1
    return int(n)


@st.cache_data(show_spinner="Loading station RMSE map…")
def load_station_wise_error(horizon: int) -> pd.DataFrame:
    """Per-station RMSE from Feature 6 table (Attention seed-42; h=1 or h=4)."""
    if horizon not in (1, 4):
        raise ValueError(f"station_wise_error only has horizons 1 and 4, got {horizon}")
    if not STATION_WISE_ERROR_CSV.exists():
        raise FileNotFoundError(STATION_WISE_ERROR_CSV)
    df = pd.read_csv(STATION_WISE_ERROR_CSV, comment="#")
    out = df[df["horizon"].astype(int) == int(horizon)].copy()
    out["RMSE_bin"] = pd.cut(
        out["RMSE"],
        bins=list(RMSE_BIN_EDGES),
        labels=list(RMSE_BIN_LABELS),
        right=False,
        include_lowest=True,
    ).astype(str)
    # pd.cut with right=False: last edge inf → "≥ 15"; coerce any NaN labels
    out.loc[out["RMSE_bin"].isin(["nan", "NaN"]), "RMSE_bin"] = "≥ 15"
    return out.reset_index(drop=True)


def build_performance_highlights() -> list[str]:
    """Auto bullets from ablation_study.csv + significance_results.csv (live %)."""
    abl = pd.read_csv(ABLATION_CSV)
    sig = pd.read_csv(SIGNIFICANCE_CSV)

    bullets: list[str] = []

    # Attention vs Temporal significance (verified DM tests)
    att_sig = sig[sig["Comparison"] == "Attention_vs_Temporal"].copy()
    att_sig["Forecast_Horizon"] = att_sig["Forecast_Horizon"].astype(int)
    sig_yes = sorted(
        att_sig.loc[att_sig["Significant_at_0.05"] == "Yes", "Forecast_Horizon"].tolist()
    )
    sig_no = sorted(
        att_sig.loc[att_sig["Significant_at_0.05"] == "No", "Forecast_Horizon"].tolist()
    )

    # Live % RMSE change: Attention vs Temporal from ablation deltas
    att_rows = abl[abl["Model"] == "CNN-LSTM+Attention"].copy()
    temp_rows = abl[abl["Model"] == "CNN-LSTM-Temporal"].set_index("Horizon")
    pct_bits: list[str] = []
    for _, r in att_rows.iterrows():
        h = int(r["Horizon"])
        if r["Significant_vs_previous_stage"] != "Yes":
            continue
        temp_rmse = float(temp_rows.loc[h, "RMSE_mean"])
        delta = float(r["Delta_RMSE_vs_previous_stage"])
        pct = 100.0 * delta / temp_rmse
        # Negative delta = lower RMSE = improvement
        if delta < 0:
            pct_bits.append(
                f"h={h} (~{abs(pct):.1f}% RMSE reduction vs CNN-LSTM-Temporal)"
            )
        else:
            pct_bits.append(
                f"h={h} (~{abs(pct):.1f}% RMSE increase vs CNN-LSTM-Temporal)"
            )

    if sig_yes and pct_bits:
        bullets.append(
            "Attention significantly improves over CNN-LSTM-Temporal at "
            + " and ".join(f"h={h}" for h in sig_yes)
            + ": "
            + "; ".join(pct_bits)
            + f" — sourced from `ablation_study.csv` ΔRMSE and "
            f"`significance_results.csv` Attention_vs_Temporal "
            f"(Significant_at_0.05=Yes)."
        )
    elif sig_yes:
        bullets.append(
            "Attention significantly improves over CNN-LSTM-Temporal at "
            + " and ".join(f"h={h}" for h in sig_yes)
            + " (`significance_results.csv`, Attention_vs_Temporal)."
        )

    if sig_no:
        bullets.append(
            "Attention vs CNN-LSTM-Temporal is not significant at "
            + " and ".join(f"h={h}" for h in sig_no)
            + " (`significance_results.csv`)."
        )

    # Honesty: Attention vs LSTM
    att_lstm = sig[sig["Comparison"] == "Attention_vs_LSTM"]
    if not att_lstm.empty:
        row = att_lstm.iloc[0]
        bullets.append(
            "Not established to outperform plain LSTM: Attention_vs_LSTM was "
            f"formally tested at h={int(row['Forecast_Horizon'])} only and was "
            f"not significant (Significant_at_0.05={row['Significant_at_0.05']}; "
            "`significance_results.csv`). Ablation 3-seed means also show "
            "Attention RMSE above LSTM at every horizon."
        )
    else:
        bullets.append(
            "Not established to outperform plain LSTM "
            "(`significance_results.csv` / `ablation_study.csv`)."
        )

    return bullets


def station_rmse_lookup(station_id: str) -> dict[int, float | None]:
    """Per-station Attention seed-42 RMSE at h=1 and h=4, or None if absent."""
    out: dict[int, float | None] = {1: None, 4: None}
    for h in (1, 4):
        df = load_station_wise_error(h)
        hit = df.loc[df["station_id"] == station_id]
        if not hit.empty:
            out[h] = float(hit.iloc[0]["RMSE"])
    return out


def build_comparison_kpis() -> dict:
    """Three research-result KPIs for Model Comparison (live from verified CSVs).

    Cards:
      1) best Attention-vs-Temporal % RMSE reduction (prefers significant h=4)
      2) count + list of horizons where Attention_vs_Temporal is significant
      3) honesty one-liner (not established vs LSTM)
    """
    abl = pd.read_csv(ABLATION_CSV)
    sig = pd.read_csv(SIGNIFICANCE_CSV)

    att_sig = sig[sig["Comparison"] == "Attention_vs_Temporal"].copy()
    att_sig["Forecast_Horizon"] = att_sig["Forecast_Horizon"].astype(int)
    sig_yes = sorted(
        att_sig.loc[att_sig["Significant_at_0.05"] == "Yes", "Forecast_Horizon"].tolist()
    )

    att_rows = abl[abl["Model"] == "CNN-LSTM+Attention"].set_index("Horizon")
    temp_rows = abl[abl["Model"] == "CNN-LSTM-Temporal"].set_index("Horizon")

    # Headline effect: h=4 Attention vs Temporal (matches Home highlight formula)
    h4_delta = float(att_rows.loc[4, "Delta_RMSE_vs_previous_stage"])
    h4_temp = float(temp_rows.loc[4, "RMSE_mean"])
    h4_pct = 100.0 * h4_delta / h4_temp
    h4_sig = str(att_rows.loc[4, "Significant_vs_previous_stage"]) == "Yes"

    att_lstm = sig[sig["Comparison"] == "Attention_vs_LSTM"]
    if not att_lstm.empty:
        caveat = (
            "Not established to outperform plain LSTM "
            f"(Attention_vs_LSTM tested at h={int(att_lstm.iloc[0]['Forecast_Horizon'])} "
            f"only; Significant_at_0.05={att_lstm.iloc[0]['Significant_at_0.05']})."
        )
    else:
        caveat = "Not established to outperform plain LSTM."

    return {
        "h4_pct_reduction": abs(h4_pct) if h4_delta < 0 else -abs(h4_pct),
        "h4_delta_rmse": h4_delta,
        "h4_significant": h4_sig,
        "sig_horizons": sig_yes,
        "n_sig_horizons": len(sig_yes),
        "caveat": caveat,
    }


@st.cache_data(show_spinner="Loading network rainfall snapshot…")
def load_network_rainfall_tail(n_days: int = 7) -> pd.DataFrame:
    """Last N calendar days of locked test period: network mean/max rainfall.

    Derived from feature_engineered_v2.csv observed rainfall only (presentation).
    """
    from .paths import FEATURE_ENGINEERED_V2, TEST_PERIOD_END, TEST_PERIOD_START

    if not FEATURE_ENGINEERED_V2.exists():
        raise FileNotFoundError(FEATURE_ENGINEERED_V2)
    raw = pd.read_csv(
        FEATURE_ENGINEERED_V2,
        usecols=["date_of_record", "rainfall"],
        parse_dates=["date_of_record"],
    )
    start = pd.Timestamp(TEST_PERIOD_START)
    end = pd.Timestamp(TEST_PERIOD_END)
    raw["date_of_record"] = pd.to_datetime(raw["date_of_record"]).dt.normalize()
    mask = (raw["date_of_record"] >= start) & (raw["date_of_record"] <= end)
    daily = (
        raw.loc[mask]
        .groupby("date_of_record", as_index=False)
        .agg(mean_mm=("rainfall", "mean"), max_mm=("rainfall", "max"), n_stations=("rainfall", "count"))
        .sort_values("date_of_record")
    )
    return daily.tail(int(n_days)).reset_index(drop=True)


@st.cache_data(show_spinner="Loading extreme summary…")
def load_extreme_home_summary() -> dict:
    """Compact extreme-day RMSE facts from extreme_rainfall_evaluation.csv."""
    from .paths import PROJECT_ROOT

    path = PROJECT_ROOT / "reports" / "tables" / "extreme_rainfall_evaluation.csv"
    df = pd.read_csv(path, comment="#")
    thr = {}
    for h, g in df.groupby("Horizon"):
        thr[int(h)] = float(g["Threshold_mm"].iloc[0])

    att = df[(df["Model"] == "CNN-LSTM+Attention") & (df["Subset"] == "Extreme")]
    # Mean Attention extreme RMSE across horizons (display only)
    att_ext_mean = float(att["RMSE"].mean()) if not att.empty else float("nan")
    # Count horizons where Attention extreme RMSE < Temporal extreme RMSE
    better = 0
    worse = 0
    for h in sorted(df["Horizon"].unique()):
        sub = df[(df["Horizon"] == h) & (df["Subset"] == "Extreme")]
        a = sub.loc[sub["Model"] == "CNN-LSTM+Attention", "RMSE"]
        t = sub.loc[sub["Model"] == "CNN-LSTM-Temporal", "RMSE"]
        if a.empty or t.empty:
            continue
        if float(a.iloc[0]) < float(t.iloc[0]):
            better += 1
        else:
            worse += 1
    return {
        "threshold_h1": thr.get(1),
        "threshold_h4": thr.get(4),
        "att_extreme_rmse_mean": att_ext_mean,
        "horizons_att_better_vs_temporal": better,
        "horizons_att_worse_vs_temporal": worse,
        "n_horizons": len(thr),
    }
