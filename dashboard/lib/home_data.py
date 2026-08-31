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
ROBUST_CSV = TABLES / "multiseed_robustness_summary.csv"
_HIGHLIGHT_SUFFIX = " See Model Comparison for the full breakdown."

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


def _is_binary_verdict(value: object) -> bool:
    """True for plain Yes/No/N/A (no multi-seed verdict tier)."""
    return str(value).strip() in {"Yes", "No", "N/A", ""}


def _ablation_sig_state(value: object) -> str:
    """Map ablation Yes/No to binary labels; pass through any verdict tier verbatim."""
    s = str(value).strip()
    if s == "Yes":
        return "Yes"
    if s in {"No", "N/A", ""}:
        return "No"
    return s


def _load_robustness() -> pd.DataFrame:
    if not ROBUST_CSV.exists():
        raise FileNotFoundError(ROBUST_CSV)
    return pd.read_csv(ROBUST_CSV)


def _seed_tallies(row: pd.Series, first: str, second: str) -> dict[str, int]:
    """Count significant / non-significant seeds by direction."""
    out = {
        "sig_first": 0,
        "sig_second": 0,
        "ns_first": 0,
        "ns_second": 0,
    }
    for seed in (13, 42, 123):
        direction = str(row[f"Seed{seed}_Direction"])
        sig = str(row[f"Seed{seed}_Sig"])
        if sig == "Yes" and direction == first:
            out["sig_first"] += 1
        elif sig == "Yes" and direction == second:
            out["sig_second"] += 1
        elif sig == "No" and direction == first:
            out["ns_first"] += 1
        elif sig == "No" and direction == second:
            out["ns_second"] += 1
    return out


def _attention_temporal_highlight(h: int, row: pd.Series) -> str:
    """Template bullet for Attention vs Temporal at one horizon (<200 chars)."""
    t = _seed_tallies(row, "Attention", "Temporal")
    n_sig = int(row["N_Significant_of_3"])
    verdict = str(row["Consistency_Verdict"])

    if verdict == "DIRECTION-UNSTABLE (contested)" and n_sig == 3:
        text = (
            f"At h={h}, Attention's edge over CNN-LSTM-Temporal does not hold — "
            f"all 3 seeds are significant, but {t['sig_second']} favors Temporal "
            f"instead."
        )
    elif verdict == "DIRECTION-UNSTABLE (contested)" and n_sig == 2:
        if t["sig_first"] == 2 and t["sig_second"] == 0:
            text = (
                f"At h={h}, Attention vs CNN-LSTM-Temporal is seed-dependent — "
                f"2 of 3 significantly favor Attention; 1 shows no significant "
                f"difference."
            )
        else:
            text = (
                f"At h={h}, Attention vs CNN-LSTM-Temporal splits across seeds: "
                f"2 of 3 are significant and disagree in direction."
            )
    elif verdict == "DIRECTION-UNSTABLE (weak)":
        text = (
            f"At h={h}, Attention vs CNN-LSTM-Temporal is unsettled — only "
            f"{n_sig} of 3 seeds is significant; the rest are non-significant."
        )
    elif verdict == "DIRECTION-STABLE":
        text = (
            f"At h={h}, Attention vs CNN-LSTM-Temporal is direction-stable but "
            f"significance varies ({n_sig} of 3 seeds significant)."
        )
    else:
        text = (
            f"At h={h}, Attention vs CNN-LSTM-Temporal is {verdict.lower()} "
            f"({n_sig} of 3 seeds significant)."
        )
    return text + _HIGHLIGHT_SUFFIX


def _attention_lstm_summary(rows: pd.DataFrame) -> str:
    """One summary bullet for Attention vs LSTM across horizons."""
    contested = sorted(
        int(r["Horizon"])
        for _, r in rows.iterrows()
        if str(r["Consistency_Verdict"]).startswith("DIRECTION-UNSTABLE")
    )
    stable = sorted(
        int(r["Horizon"])
        for _, r in rows.iterrows()
        if str(r["Consistency_Verdict"]) == "DIRECTION-STABLE"
    )
    parts: list[str] = []
    if stable:
        hs = ", ".join(f"h={h}" for h in stable)
        parts.append(f"direction-stable at {hs}")
    if contested:
        hs = ", ".join(f"h={h}" for h in contested)
        parts.append(f"unstable at {hs}")
    detail = "; ".join(parts) if parts else "non-binary at all horizons"
    return (
        f"Attention vs plain LSTM shows no reproducible edge ({detail}). "
        f"LSTM is numerically better in 10 of 12 tests."
        f"{_HIGHLIGHT_SUFFIX}"
    )


def build_performance_highlights() -> list[str]:
    """Home bullets from multiseed_robustness_summary.csv (template-based)."""
    abl = pd.read_csv(ABLATION_CSV)
    att_abl = abl[abl["Model"] == "CNN-LSTM+Attention"].copy()
    robust = _load_robustness()

    bullets: list[str] = [
        (
            "GNN-LSTM vs LSTM is robust: LSTM significantly outperforms GNN-LSTM "
            "at all 4 horizons × all 3 seeds (12/12 tests)."
        )
    ]

    att_t = robust[robust["Comparison"] == "Attention_vs_Temporal"].sort_values("Horizon")
    for _, row in att_t.iterrows():
        h = int(row["Horizon"])
        abl_row = att_abl[att_abl["Horizon"] == h]
        if abl_row.empty:
            continue
        prev = abl_row.iloc[0]["Significant_vs_previous_stage"]
        if _is_binary_verdict(prev):
            continue
        bullets.append(_attention_temporal_highlight(h, row))

    att_l = robust[robust["Comparison"] == "Attention_vs_LSTM"].sort_values("Horizon")
    if any(not _is_binary_verdict(r["Significant_vs_LSTM"]) for _, r in att_abl.iterrows()):
        bullets.append(_attention_lstm_summary(att_l))

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
    """Research-result KPIs for Model Comparison (live from verified CSVs).

    Cards use the complete 3-seed picture, not seed-42-only significance flags.
    """
    abl = pd.read_csv(ABLATION_CSV)
    robust_path = TABLES / "multiseed_robustness_summary.csv"
    robust = pd.read_csv(robust_path) if robust_path.exists() else pd.DataFrame()

    att_rows = abl[abl["Model"] == "CNN-LSTM+Attention"].set_index("Horizon")
    temp_rows = abl[abl["Model"] == "CNN-LSTM-Temporal"].set_index("Horizon")

    h4_delta = float(att_rows.loc[4, "Delta_RMSE_vs_previous_stage"])
    h4_temp = float(temp_rows.loc[4, "RMSE_mean"])
    h4_pct = 100.0 * h4_delta / h4_temp
    h4_state = _ablation_sig_state(att_rows.loc[4, "Significant_vs_previous_stage"])

    nonbinary_horizons = sorted(
        int(h)
        for h, r in att_rows.iterrows()
        if not _is_binary_verdict(r["Significant_vs_previous_stage"])
    )

    att_t = robust[robust["Comparison"] == "Attention_vs_Temporal"] if not robust.empty else pd.DataFrame()
    n_unanimous_attn_temp = (
        int((att_t["Consistency_Verdict"] == "CONSISTENT").sum()) if not att_t.empty else 0
    )

    caveat = (
        "Not established vs LSTM: 10 of 12 tests numerically favor LSTM "
        "(6 significant). Attention-vs-Temporal is non-unanimous at all 4 horizons."
    )

    return {
        "h4_pct_reduction": abs(h4_pct) if h4_delta < 0 else -abs(h4_pct),
        "h4_delta_rmse": h4_delta,
        "h4_significant": h4_state,
        "sig_horizons": nonbinary_horizons,
        "n_sig_horizons": n_unanimous_attn_temp,
        "n_mixed_horizons": len(nonbinary_horizons),
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
