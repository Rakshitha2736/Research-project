"""Loaders for Model Comparison page (verified tables only, read-only)."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from .paths import PROJECT_ROOT

TABLES = PROJECT_ROOT / "reports" / "tables"
MASTER_CSV = TABLES / "master_results.csv"
ABLATION_CSV = TABLES / "ablation_study.csv"
SIGNIFICANCE_CSV = TABLES / "significance_results.csv"
SEASONAL_CSV = TABLES / "seasonal_performance.csv"

PRIMARY_MODELS = (
    "LSTM",
    "CNN-LSTM-Temporal",
    "CNN-LSTM+Attention",
)

MODEL_COLORS = {
    "LSTM": "#4c78a8",
    "CNN-LSTM-Temporal": "#f58518",
    "CNN-LSTM+Attention": "#0f6b5c",
    "GNN-LSTM": "#9aa3b2",
}


def parse_mean_std(cell: str) -> tuple[float | None, float | None]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None, None
    s = str(cell).strip()
    if not s or s.lower() in {"not available", "nan", "n/a", "na"}:
        return None, None
    m = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[±+-]\s*"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
        s,
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), None
    except ValueError:
        return None, None


def _normalize_primary_model(name: str) -> str | None:
    n = str(name)
    if "GNN" in n or "Transformer" in n or "Persistence" in n or "ARIMA" in n:
        return None
    if "Attention" in n and "mean" in n:
        return "CNN-LSTM+Attention"
    if "Temporal" in n and "mean" in n:
        return "CNN-LSTM-Temporal"
    if n.startswith("LSTM") and "mean" in n:
        return "LSTM"
    return None


@st.cache_data(show_spinner="Loading research metrics…")
def load_primary_rmse_bars() -> pd.DataFrame:
    """RMSE / MAE / R² mean±std for primary models at h=1..4 from master_results.

    LSTM h≥2 MAE/R² cells are "Not Available" in the verified table; those
    become null means and are formatted as "Not Available" in the UI.
    """
    raw = pd.read_csv(MASTER_CSV)
    rows: list[dict] = []
    for _, r in raw.iterrows():
        model = _normalize_primary_model(r["Model"])
        if model is None:
            continue
        rmse_mean, rmse_std = parse_mean_std(r["RMSE"])
        if rmse_mean is None or rmse_std is None:
            continue
        mae_mean, mae_std = parse_mean_std(r["MAE"])
        r2_mean, r2_std = parse_mean_std(r["R2"])
        rows.append(
            {
                "Model": model,
                "Horizon": int(r["Forecast_Horizon"]),
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std,
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "R2_mean": r2_mean,
                "R2_std": r2_std,
            }
        )
    df = pd.DataFrame(rows)
    df["Model"] = pd.Categorical(df["Model"], categories=list(PRIMARY_MODELS), ordered=True)
    return df.sort_values(["Horizon", "Model"]).reset_index(drop=True)


def format_mean_std(mean: float | None, std: float | None) -> str:
    """Display helper for master_results mean±std cells (incl. Not Available)."""
    if mean is None or (isinstance(mean, float) and pd.isna(mean)):
        return "Not Available"
    if std is None or (isinstance(std, float) and pd.isna(std)):
        return f"{mean:.4f}"
    return f"{mean:.4f} ± {std:.4f}"


@st.cache_data(show_spinner="Loading ablation summary…")
def load_ablation_summary() -> pd.DataFrame:
    abl = pd.read_csv(ABLATION_CSV)
    out = abl[
        [
            "Horizon",
            "Model",
            "RMSE_mean",
            "MAE_mean",
            "R2_mean",
            "Delta_RMSE_vs_previous_stage",
            "Significant_vs_previous_stage",
            "Delta_RMSE_vs_LSTM",
            "Significant_vs_LSTM",
        ]
    ].copy()
    out["Model"] = pd.Categorical(
        out["Model"], categories=list(PRIMARY_MODELS), ordered=True
    )
    return out.sort_values(["Horizon", "Model"]).reset_index(drop=True)


@st.cache_data(show_spinner="Loading secondary GNN metrics…")
def load_gnn_secondary() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GNN mean±std RMSE + GNN_vs_LSTM significance rows (secondary only)."""
    master = pd.read_csv(MASTER_CSV)
    gnn_rows = []
    for _, r in master.iterrows():
        if "GNN-LSTM (mean" not in str(r["Model"]):
            continue
        mean, std = parse_mean_std(r["RMSE"])
        if mean is None:
            continue
        gnn_rows.append(
            {
                "Model": "GNN-LSTM",
                "Horizon": int(r["Forecast_Horizon"]),
                "RMSE_mean": mean,
                "RMSE_std": std if std is not None else float("nan"),
            }
        )
    gnn_rmse = pd.DataFrame(gnn_rows).sort_values("Horizon")

    # Pair with LSTM for a compact comparison table
    primary = load_primary_rmse_bars()
    lstm = primary[primary["Model"] == "LSTM"][["Horizon", "RMSE_mean", "RMSE_std"]].rename(
        columns={"RMSE_mean": "LSTM_RMSE", "RMSE_std": "LSTM_std"}
    )
    gnn_cmp = gnn_rmse.merge(lstm, on="Horizon", how="left")
    gnn_cmp["Delta_RMSE_GNN_minus_LSTM"] = gnn_cmp["RMSE_mean"] - gnn_cmp["LSTM_RMSE"]

    sig = pd.read_csv(SIGNIFICANCE_CSV)
    gnn_sig = sig[sig["Comparison"] == "GNN_vs_LSTM"].copy()
    return gnn_cmp, gnn_sig


@st.cache_data(show_spinner="Loading seasonal RMSE…")
def load_seasonal_h4() -> pd.DataFrame:
    """Seed-42 seasonal RMSE at h=4 (supports the honesty caveat)."""
    df = pd.read_csv(SEASONAL_CSV, comment="#")
    df = df[df["Horizon"].astype(int) == 4].copy()
    df = df[df["Model"].isin(PRIMARY_MODELS)]
    return df.sort_values(["Season", "Model"]).reset_index(drop=True)
