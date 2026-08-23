"""
Historical pages (Home, Station Explorer, Forecast Replay, Historical Analysis,
Model Comparison, Explainability) read only from precomputed parquet caches —
no inference.

The Forward Forecast page is the sole exception: it runs on-demand CPU-only
model inference on each station's most recent available data.

Run from RainfallPrediction/:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_FAVICON = Path(__file__).resolve().parent / "assets" / "favicon.png"

st.set_page_config(
    page_title="Rainfall Forecast Dashboard",
    page_icon=str(_FAVICON) if _FAVICON.exists() else "🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from lib.stations import init_selection_state  # noqa: E402
from lib.style import inject_base_css, render_sidebar_brand  # noqa: E402

# CSS + branded sidebar chrome (idempotent). Pages also call inject_base_css.
inject_base_css()
render_sidebar_brand()
init_selection_state()

# --- Multipage navigation (routing unchanged for historical pages) ---
home = st.Page("pages/home.py", title="Home", icon=":material/home:", default=True)
station_explorer = st.Page(
    "pages/station_explorer.py",
    title="Station Explorer",
    icon=":material/map:",
)
forecast_replay = st.Page(
    "pages/forecast_replay.py",
    title="Forecast Replay",
    icon=":material/timeline:",
)
historical = st.Page(
    "pages/historical_analysis.py",
    title="Historical Analysis",
    icon=":material/bar_chart:",
)
comparison = st.Page(
    "pages/model_comparison.py",
    title="Model Comparison",
    icon=":material/compare_arrows:",
)
explainability = st.Page(
    "pages/explainability.py",
    title="Explainability",
    icon=":material/psychology:",
)
latest_available = st.Page(
    "pages/latest_available_forecast.py",
    title="Latest-Available Forecast",
    icon=":material/forward:",
)

pg = st.navigation(
    {
        "Overview": [home, station_explorer],
        "Analysis": [forecast_replay, historical, comparison, explainability],
        "Forward Forecast": [latest_available],
    }
)
pg.run()
