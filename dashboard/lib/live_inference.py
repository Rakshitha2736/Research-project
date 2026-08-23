"""
On-demand CPU inference for the Latest-Available Forecast page.

Runs seed=42 checkpoints (LSTM, CNN-LSTM-Temporal, CNN-LSTM+Attention) on each
station's last contiguous 30-day observed window from feature_engineered_v2.csv.

Windowing mirrors generate_sequences_v2.py (contiguous calendar segments only;
no gap-fill). Uses existing MinMax scalers — does not fit new ones.

Default device is CPU. CUDA is used only if explicitly available and requested;
AMP/autocast is intentionally NOT used (single-sample precision).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"
FEAT_CSV = DATA / "feature_engineered_v2.csv"

SEQ_LEN = 30
SEED = 42
FEATURE_COLS = [
    "avg_temp",
    "min_temp",
    "max_temp",
    "wind_speed",
    "air_pressure",
    "rainfall",
    "doy_sin",
    "doy_cos",
]

MODEL_SPECS = (
    ("LSTM", "lstm"),
    ("CNN-LSTM-Temporal", "temporal"),
    ("CNN-LSTM+Attention", "attention"),
)
HORIZONS = (1, 2, 3, 4)


def resolve_device(*, prefer_cuda: bool = False) -> torch.device:
    """Default CPU. CUDA only when explicitly preferred and available."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ckpt_path(model_key: str, horizon: int) -> Path:
    if model_key == "attention":
        return MODELS / f"cnn_lstm_attention_h{horizon}_seed{SEED}.pt"
    if model_key == "temporal":
        return MODELS / f"cnn_lstm_temporal_h{horizon}_seed{SEED}.pt"
    if model_key == "lstm":
        if horizon == 1:
            return MODELS / f"lstm_baseline_v2_seed{SEED}.pt"
        return MODELS / f"lstm_h{horizon}_seed{SEED}.pt"
    raise ValueError(model_key)


def scaler_y_path(horizon: int) -> Path:
    if horizon == 1:
        return MODELS / "minmax_scaler_y_v2.joblib"
    return MODELS / f"minmax_scaler_y_h{horizon}.joblib"


def build_model(model_key: str) -> torch.nn.Module:
    if model_key == "attention":
        return CNNLSTMAttention(n_features=8)
    if model_key == "temporal":
        return CNNLSTMTemporalBaseline(n_features=8, use_pooling=False)
    if model_key == "lstm":
        return LSTMBaseline(input_size=8, hidden_size=64, num_layers=2)
    raise ValueError(model_key)


def find_last_valid_window(
    g: pd.DataFrame,
) -> tuple[pd.Timestamp, np.ndarray] | None:
    """Most recent contiguous SEQ_LEN-day finite feature window for one station.

    Segment splitting matches generate_sequences_v2.build_sequences:
    breaks where consecutive calendar-day diffs != 1. No gap-fill.
    """
    g = g.sort_values("date_of_record").reset_index(drop=True)
    dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
    feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
    n = len(g)
    if n < SEQ_LEN:
        return None

    day_ints = dates.astype("datetime64[D]").astype(np.int64)
    breaks = np.where(np.diff(day_ints) != 1)[0] + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [n]))

    best: tuple[pd.Timestamp, np.ndarray] | None = None
    for seg_start, seg_end in zip(starts, ends):
        if seg_end - seg_start < SEQ_LEN:
            continue
        for j in range(seg_end - SEQ_LEN, seg_start - 1, -1):
            x_seq = feats[j : j + SEQ_LEN]
            if np.isfinite(x_seq).all():
                end = pd.Timestamp(dates[j + SEQ_LEN - 1])
                best = (end, x_seq.astype(np.float64))
                break
    return best


@lru_cache(maxsize=1)
def _load_feat_frame() -> pd.DataFrame:
    return pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])


@lru_cache(maxsize=1)
def _load_x_scaler():
    return joblib.load(MODELS / "minmax_scaler_v2.joblib")


@lru_cache(maxsize=4)
def _load_y_scaler(horizon: int):
    return joblib.load(scaler_y_path(horizon))


@lru_cache(maxsize=12)
def _load_model(model_key: str, horizon: int, device_str: str) -> torch.nn.Module:
    device = torch.device(device_str)
    model = build_model(model_key)
    ckpt = ckpt_path(model_key, horizon)
    # Checkpoints store {"model_state_dict": ...} (same as build_forecast_cache)
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@dataclass(frozen=True)
class StationForecastResult:
    station_id: str
    window_end_date: pd.Timestamp
    # predictions[(model_name, horizon)] = mm
    predictions: dict[tuple[str, int], float]
    # target_dates[horizon] = window_end + horizon
    target_dates: dict[int, pd.Timestamp]
    # attention_by_horizon[h] = (30,) weights for CNN-LSTM+Attention
    attention_by_horizon: dict[int, np.ndarray]


def load_station_raw_window(
    station_id: str,
) -> tuple[pd.Timestamp, np.ndarray]:
    """Return (window_end_date, raw X shape (30, 8)) for a station."""
    df = _load_feat_frame()
    g = df.loc[df["station_id"].astype(str) == str(station_id)]
    if g.empty:
        raise ValueError(f"Station not found in feature_engineered_v2: {station_id}")
    found = find_last_valid_window(g)
    if found is None:
        raise ValueError(f"No contiguous {SEQ_LEN}-day window for station {station_id}")
    return found


@torch.no_grad()
def run_station_inference(
    station_id: str,
    *,
    prefer_cuda: bool = False,
) -> StationForecastResult:
    """Forward-pass all primary models × horizons for one station's last window."""
    device = resolve_device(prefer_cuda=prefer_cuda)
    window_end, x_raw = load_station_raw_window(station_id)
    assert x_raw.shape == (SEQ_LEN, len(FEATURE_COLS))

    scaler_x = _load_x_scaler()
    # Scaler expects (n_samples, n_features); flatten timesteps then reshape
    x_scaled = scaler_x.transform(x_raw).astype(np.float32)
    x_t = torch.from_numpy(x_scaled).unsqueeze(0).to(device)  # (1, 30, 8)

    predictions: dict[tuple[str, int], float] = {}
    attention_by_horizon: dict[int, np.ndarray] = {}
    target_dates = {
        h: window_end + pd.Timedelta(days=h) for h in HORIZONS
    }

    for model_name, model_key in MODEL_SPECS:
        for h in HORIZONS:
            model = _load_model(model_key, h, str(device))
            if model_key == "attention":
                pred_s, attn = model(x_t, return_attention=True)
                attention_by_horizon[h] = (
                    attn.detach().float().cpu().numpy().reshape(SEQ_LEN)
                )
            else:
                pred_s = model(x_t)
            pred_np = pred_s.detach().float().cpu().numpy().reshape(-1, 1)
            y_mm = float(_load_y_scaler(h).inverse_transform(pred_np).ravel()[0])
            predictions[(model_name, h)] = y_mm

    return StationForecastResult(
        station_id=str(station_id),
        window_end_date=pd.Timestamp(window_end).normalize(),
        predictions=predictions,
        target_dates=target_dates,
        attention_by_horizon=attention_by_horizon,
    )
