"""Rainfall-specific evaluation metrics (thresholds + intensity bins)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

DEFAULT_THRESHOLDS_MM = (0.1, 1.0, 5.0, 10.0)
# IMD heavy / very heavy / extremely heavy lower bounds (24 h accumulated rainfall).
IMD_EVENT_THRESHOLDS_MM = (35.6, 64.4, 124.4)
FULL_THRESHOLDS_MM = tuple(sorted(set(DEFAULT_THRESHOLDS_MM + IMD_EVENT_THRESHOLDS_MM)))
DEFAULT_INTENSITY_EDGES_MM = (0.0, 0.1, 1.0, 5.0, 10.0, np.inf)


def contingency(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict[str, int]:
    """Binary rain event contingency at threshold (mm).

    Observed/predicted "rain" if value >= threshold.
    """
    yt = np.asarray(y_true, dtype=np.float64) >= threshold
    yp = np.asarray(y_pred, dtype=np.float64) >= threshold
    hits = int(np.sum(yt & yp))
    misses = int(np.sum(yt & ~yp))
    false_alarms = int(np.sum(~yt & yp))
    correct_neg = int(np.sum(~yt & ~yp))
    return {
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
        "correct_negatives": correct_neg,
        "n": int(len(y_true)),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def threshold_skill(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> dict[str, float | int]:
    """POD, FAR, CSI, Bias, HSS, Accuracy at one threshold."""
    c = contingency(y_true, y_pred, threshold)
    h, m, f, n_cn = c["hits"], c["misses"], c["false_alarms"], c["correct_negatives"]
    n = c["n"]

    pod = _safe_div(h, h + m)  # hit rate / recall
    far = _safe_div(f, h + f)  # false alarm ratio
    csi = _safe_div(h, h + m + f)  # critical success index / threat score
    bias = _safe_div(h + f, h + m)  # frequency bias

    # Heidke Skill Score
    expected = ((h + m) * (h + f) + (f + n_cn) * (m + n_cn)) / n if n > 0 else float("nan")
    hss = _safe_div((h + n_cn) - expected, n - expected) if n > 0 else float("nan")

    acc = _safe_div(h + n_cn, n)
    return {
        "threshold_mm": float(threshold),
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "Bias": bias,
        "HSS": hss,
        "Accuracy": acc,
        "hits": h,
        "misses": m,
        "false_alarms": f,
        "correct_negatives": n_cn,
        "n": n,
        "n_obs_event": h + m,
        "n_pred_event": h + f,
    }


def threshold_skill_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS_MM,
) -> list[dict[str, float | int]]:
    return [threshold_skill(y_true, y_pred, float(t)) for t in thresholds]


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return float("nan"), float("nan")
    err = yt - yp
    return float(np.sqrt(np.mean(err**2))), float(np.mean(np.abs(err)))


def intensity_bin_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    edges: Sequence[float] = DEFAULT_INTENSITY_EDGES_MM,
) -> list[dict[str, float | int | str]]:
    """RMSE/MAE stratified by observed rainfall intensity bins.

    Bins are [edges[i], edges[i+1]). Last edge may be +inf.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    rows: list[dict[str, float | int | str]] = []
    e = list(edges)
    for i in range(len(e) - 1):
        lo, hi = float(e[i]), float(e[i + 1])
        if np.isinf(hi):
            mask = yt >= lo
            label = f"[{lo:g}, inf)"
        else:
            mask = (yt >= lo) & (yt < hi)
            label = f"[{lo:g}, {hi:g})"
        n = int(mask.sum())
        rmse, mae = rmse_mae(yt[mask], yp[mask]) if n else (float("nan"), float("nan"))
        rows.append(
            {
                "bin": label,
                "bin_lo_mm": lo,
                "bin_hi_mm": hi if not np.isinf(hi) else float("inf"),
                "n": n,
                "RMSE": rmse,
                "MAE": mae,
                "mean_obs_mm": float(yt[mask].mean()) if n else float("nan"),
                "mean_pred_mm": float(yp[mask].mean()) if n else float("nan"),
            }
        )
    return rows


def tolerance_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, tol_mm: float
) -> float:
    """Fraction of samples with |y - ŷ| <= tol_mm."""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp) <= tol_mm))
