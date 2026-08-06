"""Helpers for conditioned attention analysis (no architecture changes)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader
from src.model import CNNLSTMAttention

SEQ_LEN = 30
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
TARGET_COL = "rainfall"

TRAIN_END = pd.Timestamp("2022-12-31")
VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2025-02-10")

MONSOON_MONTHS = {6, 7, 8, 9}  # JJAS


def split_name(target_date: pd.Timestamp) -> str | None:
    if target_date <= TRAIN_END:
        return "train"
    if VAL_START <= target_date <= VAL_END:
        return "val"
    if TEST_START <= target_date <= TEST_END:
        return "test"
    return None


def paths_for_horizon(base: Path, horizon: int) -> dict[str, Path]:
    """Map horizon → X/y/scaler/checkpoint naming conventions."""
    data = base / "data" / "processed"
    models = base / "models"
    if horizon == 1:
        return {
            "X_test": data / "X_test_v2.npy",
            "y_test": data / "y_test_v2.npy",
            "scaler_y": models / "minmax_scaler_y_v2.joblib",
            "ckpt": models / "cnn_lstm_attention_h1_seed{seed}.pt",
            "attn_cache": data / "attention_weights_h1_seed{seed}.npy",
        }
    return {
        "X_test": data / f"X_test_h{horizon}.npy",
        "y_test": data / f"y_test_h{horizon}.npy",
        "scaler_y": models / f"minmax_scaler_y_h{horizon}.joblib",
        "ckpt": models / f"cnn_lstm_attention_h{horizon}_seed{{seed}}.pt",
        "attn_cache": data / f"attention_weights_h{horizon}_seed{{seed}}.npy",
    }


def rebuild_test_meta(df: pd.DataFrame, horizon: int) -> list[dict]:
    """Rebuild test-split metas in the same order as sequence generation.

    Mirrors generate_sequences_v2 / generate_sequences_multihorizon window
    logic but only stores light metadata (no X arrays).
    """
    meta: list[dict] = []
    for station_id, g in df.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        feats = g[FEATURE_COLS].to_numpy(dtype=np.float64)
        targets = g[TARGET_COL].to_numpy(dtype=np.float64)
        n = len(g)
        if n < SEQ_LEN:
            continue

        day_ints_all = dates.astype("datetime64[D]").astype(np.int64)
        date_to_idx: dict[int, int] = {int(d): i for i, d in enumerate(day_ints_all)}

        breaks = np.where(np.diff(day_ints_all) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))

        for seg_start, seg_end in zip(starts, ends):
            if seg_end - seg_start < SEQ_LEN:
                continue
            for i in range(seg_start, seg_end - SEQ_LEN + 1):
                window_dates = dates[i : i + SEQ_LEN]
                window_end = pd.Timestamp(window_dates[-1])
                target_date = window_end + pd.Timedelta(days=horizon)
                target_day_int = int(np.datetime64(target_date, "D").astype(np.int64))

                win_ints = window_dates.astype("datetime64[D]").astype(np.int64)
                if not np.all(np.diff(win_ints) == 1):
                    continue

                target_idx = date_to_idx.get(target_day_int)
                if target_idx is None:
                    continue

                y_val = targets[target_idx]
                x_seq = feats[i : i + SEQ_LEN]
                if not np.isfinite(x_seq).all() or not np.isfinite(y_val):
                    continue

                if split_name(target_date) != "test":
                    continue

                meta.append(
                    {
                        "station_id": station_id,
                        "target_date": str(target_date.date()),
                        "window_end_date": str(window_end.date()),
                        "horizon": horizon,
                    }
                )
    return meta


@torch.no_grad()
def collect_attention_and_preds(
    model: CNNLSTMAttention,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (attn [N,30], pred_scaled [N]). Chronological: idx0=oldest."""
    model.eval()
    attn_chunks: list[torch.Tensor] = []
    pred_chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            pred, attn = model(xb, return_attention=True)
        attn_chunks.append(attn.float())
        pred_chunks.append(pred.float())
    attn = torch.cat(attn_chunks, dim=0).cpu().numpy()
    pred = torch.cat(pred_chunks, dim=0).cpu().numpy()
    return attn, pred


def load_attention_model(
    ckpt_path: Path, device: torch.device
) -> CNNLSTMAttention:
    model = CNNLSTMAttention(n_features=8).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def attention_entropy(attn: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-sample entropy of α over the 30-day window. Shape (N,)."""
    a = np.clip(attn, eps, 1.0)
    return -np.sum(a * np.log(a), axis=1)


def profile_summary(mean_w: np.ndarray) -> dict[str, float | int]:
    """Summarize a mean attention profile (length 30, oldest→newest)."""
    assert mean_w.shape == (SEQ_LEN,)
    peak_idx_chrono = int(np.argmax(mean_w))  # 0=oldest
    peak_day = SEQ_LEN - peak_idx_chrono  # 1=most recent … 30=oldest
    return {
        "peak_day_position": peak_day,
        "recent_7_share": float(mean_w[-7:].sum()),
        "oldest_7_share": float(mean_w[:7].sum()),
        "entropy_of_mean": float(attention_entropy(mean_w.reshape(1, -1))[0]),
    }


def mean_profile(attn: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is None:
        return attn.mean(axis=0)
    if not np.any(mask):
        return np.full(SEQ_LEN, np.nan, dtype=np.float64)
    return attn[mask].mean(axis=0)


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """95% bootstrap CI for mean(a) - mean(b) on 1-D scalars."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    obs = float(a.mean() - b.mean())
    diffs = np.empty(n_boot, dtype=np.float64)
    na, nb = len(a), len(b)
    for i in range(n_boot):
        sa = a[rng.integers(0, na, size=na)]
        sb = b[rng.integers(0, nb, size=nb)]
        diffs[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, float(lo), float(hi)


def bootstrap_profile_diff_ci(
    attn: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    n_boot: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep 95% CI for mean_α(A) - mean_α(B)."""
    rng = np.random.default_rng(seed)
    aa = attn[mask_a]
    bb = attn[mask_b]
    na, nb = len(aa), len(bb)
    diffs = np.empty((n_boot, SEQ_LEN), dtype=np.float64)
    for i in range(n_boot):
        sa = aa[rng.integers(0, na, size=na)]
        sb = bb[rng.integers(0, nb, size=nb)]
        diffs[i] = sa.mean(axis=0) - sb.mean(axis=0)
    lo = np.percentile(diffs, 2.5, axis=0)
    hi = np.percentile(diffs, 97.5, axis=0)
    return lo.astype(np.float64), hi.astype(np.float64)


def is_monsoon(dates: Iterable[pd.Timestamp] | pd.Series | np.ndarray) -> np.ndarray:
    s = pd.to_datetime(pd.Series(dates))
    return s.dt.month.isin(MONSOON_MONTHS).to_numpy()


def run_inference_bundle(
    base: Path,
    horizon: int,
    seed: int,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reuse_attn_cache: bool = True,
) -> dict:
    """Load data + model; return attn, y_true_mm, y_pred_mm, abs_err."""
    import joblib

    paths = paths_for_horizon(base, horizon)
    X_test = np.load(paths["X_test"])
    y_test = np.load(paths["y_test"])
    scaler_y = joblib.load(paths["scaler_y"])
    ckpt = Path(str(paths["ckpt"]).format(seed=seed))
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    cache = Path(str(paths["attn_cache"]).format(seed=seed))
    dummy = np.zeros(len(X_test), dtype=np.float32)
    loader = make_loader(X_test, dummy, batch_size=batch_size, shuffle=False)
    model = load_attention_model(ckpt, device)

    if reuse_attn_cache and cache.exists():
        attn = np.load(cache)
        if attn.shape[0] != len(X_test):
            attn, pred_s = collect_attention_and_preds(model, loader, device)
            np.save(cache, attn)
        else:
            # still need preds
            pred_chunks = []
            model.eval()
            with torch.no_grad():
                for xb, _ in loader:
                    xb = xb.to(device, non_blocking=True)
                    with autocast("cuda"):
                        pred_chunks.append(model(xb).float())
            pred_s = torch.cat(pred_chunks, dim=0).cpu().numpy()
    else:
        attn, pred_s = collect_attention_and_preds(model, loader, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, attn)

    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    abs_err = np.abs(y_true - y_pred)

    assert attn.shape == (len(X_test), SEQ_LEN)
    return {
        "attn": attn,
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_err": abs_err,
        "n": len(X_test),
    }
