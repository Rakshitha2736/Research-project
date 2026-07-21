"""
Dataset loading and cleaning helpers.

Full end-to-end cleaning (missing-value strategies, station sort) lives in
`run_pipeline.py` step_clean. This module provides reusable path constants
and lightweight helpers used by notebooks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DATASET_PATH = DATA_DIR / "raw" / "india_weather_rainfall_data.xlsx"
CLEAN_DATASET_PATH = DATA_DIR / "processed" / "clean_dataset.csv"
FEATURE_ENGINEERED_PATH = DATA_DIR / "processed" / "feature_engineered_v2.csv"

# Backward-compatible aliases
DATASET_PATH = RAW_DATASET_PATH


def load_dataset(path: str | Path = RAW_DATASET_PATH) -> pd.DataFrame:
    """Load the raw Excel rainfall dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place india_weather_rainfall_data.xlsx under data/raw/."
        )
    return pd.read_excel(path)


def inspect_dataset(df: pd.DataFrame) -> dict:
    """Return a compact inspection summary for notebooks / diagnostics."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numerical_features": df.select_dtypes(include=["number"]).columns.tolist(),
        "categorical_features": df.select_dtypes(exclude=["number"]).columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight date parse + station/date sort.

    For production cleaning (drop missing rainfall, station-wise fills),
    use `run_pipeline.py` or the preprocessing notebook.
    """
    df = df.copy()
    if "date_of_record" in df.columns:
        df["date_of_record"] = pd.to_datetime(df["date_of_record"], errors="coerce")
        sort_cols = [c for c in ["station_name", "date_of_record"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def save_clean_dataset(df: pd.DataFrame, output_path: str | Path = CLEAN_DATASET_PATH) -> Path:
    """Write a cleaned dataframe to CSV under data/processed/."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
