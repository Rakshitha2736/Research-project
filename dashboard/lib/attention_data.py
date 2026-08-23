"""Attention weight helpers for the Explainability page (no GPU / no model)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from .paths import FORECAST_CACHE, PROJECT_ROOT

ATTN_CSV = PROJECT_ROOT / "reports" / "tables" / "attention_extreme_vs_normal.csv"
ATTN_NPY = PROJECT_ROOT / "data" / "processed"

# Chronological storage in cache / Feature-5 .npy: index 0 = oldest, 29 = most recent.
# Display axis used in this project: day -30 (oldest) … day -1 (most recent).
DAY_OFFSETS = list(range(-30, 0))  # -30 .. -1


def chrono_to_day_labels(weights_chrono: np.ndarray) -> pd.DataFrame:
    """Map chronological α[0=oldest…29=newest] → day −30…−1 for plotting."""
    w = np.asarray(weights_chrono, dtype=np.float64).ravel()
    if w.shape != (30,):
        raise ValueError(f"Expected 30 attention weights, got shape {w.shape}")
    return pd.DataFrame(
        {
            "day_offset": DAY_OFFSETS,
            "weight": w,
            "role": ["oldest" if d == -30 else ("most recent" if d == -1 else "") for d in DAY_OFFSETS],
        }
    )


@st.cache_data(show_spinner="Loading attention profiles…")
def load_mean_attention_profile(horizon: int) -> np.ndarray:
    """All-test-sample mean α from Feature 5 attention_weights_h*_seed42.npy.

    Same arrays used for attention_extreme_vs_normal.csv / mean-profile figures.
    Chronological: index 0 = oldest, 29 = most recent.
    """
    path = ATTN_NPY / f"attention_weights_h{horizon}_seed42.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    attn = np.load(path)
    mean_w = attn.mean(axis=0)
    return mean_w.astype(np.float64)


@st.cache_data(show_spinner="Loading attention summary…")
def load_attention_summary_table() -> pd.DataFrame:
    return pd.read_csv(ATTN_CSV, comment="#")


def load_sample_attention(
    station_id: str,
    target_date: pd.Timestamp,
    horizon: int,
) -> np.ndarray | None:
    """Per-sample 30-d attention vector from forecast_cache (Attention model only)."""
    td = str(pd.Timestamp(target_date).normalize().date())
    table = pq.read_table(
        FORECAST_CACHE,
        columns=["attention_weights", "model_name", "target_date", "horizon"],
        filters=[
            ("station_id", "=", station_id),
            ("model_name", "=", "CNN-LSTM+Attention"),
            ("horizon", "=", int(horizon)),
            ("target_date", "=", td),
        ],
    )
    if table.num_rows == 0:
        return None
    # One row expected
    val = table.column("attention_weights")[0].as_py()
    if val is None:
        return None
    return np.asarray(val, dtype=np.float64)


def horizon_strategy_caption(horizon: int) -> str:
    if horizon == 1:
        return (
            "Attention concentrates heavily on the most recent day (day −1), "
            "similar to a persistence-style strategy."
        )
    if horizon == 4:
        return (
            "Attention is more evenly distributed, with a mild emphasis on the "
            "oldest day in the window — a distinct strategy from h=1."
        )
    return (
        "At this horizon the attention pattern is less clearly characterized in "
        "this study (intermediate between the sharp recency focus at h=1 and the "
        "near-uniform / oldest-leaning profile at h=4)."
    )
