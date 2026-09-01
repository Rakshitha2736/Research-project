"""IMD rainfall intensity tiers for stratified evaluation (Phase 6).

Boundaries match dashboard/lib/imd_rainfall.py / published IMD daily scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

MIN_TIER_SAMPLES = 100

# (display name, mask on observed mm/day)
IMD_TIER_SPECS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("No Rain", lambda y: y < 0.1),
    ("Light", lambda y: (y >= 0.1) & (y <= 7.5)),
    ("Moderate", lambda y: (y >= 7.6) & (y <= 35.5)),
    ("Heavy", lambda y: (y >= 35.6) & (y <= 64.4)),
    ("Very Heavy", lambda y: (y >= 64.5) & (y <= 124.4)),
    ("Extremely Heavy", lambda y: y >= 124.5),
)


@dataclass
class MergedTier:
    name: str
    mask: np.ndarray
    source_tiers: tuple[str, ...]


def raw_tier_counts(y_true: np.ndarray) -> dict[str, int]:
    y = np.asarray(y_true, dtype=np.float64)
    return {name: int(fn(y).sum()) for name, fn in IMD_TIER_SPECS}


def merge_tiers(y_true: np.ndarray, min_samples: int = MIN_TIER_SAMPLES) -> tuple[list[MergedTier], list[str]]:
    """Merge sparse IMD tiers per horizon until each has >= min_samples.

    Top sparse tier merges into the adjacent less-severe tier; bottom sparse tier
    merges into the adjacent more-severe tier. Repeats until stable.
    """
    y = np.asarray(y_true, dtype=np.float64)
    tiers: list[MergedTier] = [
        MergedTier(name=name, mask=fn(y).copy(), source_tiers=(name,))
        for name, fn in IMD_TIER_SPECS
    ]
    decisions: list[str] = []

    changed = True
    while changed:
        changed = False
        if len(tiers) < 2:
            break

        # Most severe tier too sparse -> merge into less-severe neighbor below.
        if int(tiers[-1].mask.sum()) < min_samples:
            n = int(tiers[-1].mask.sum())
            upper, lower = tiers[-1], tiers[-2]
            decisions.append(
                f"Merged '{upper.name}' (n={n} < {min_samples}) into '{lower.name}' "
                f"-> '{lower.name} + {upper.name}'"
            )
            lower.mask = lower.mask | upper.mask
            lower.name = f"{lower.name} + {upper.name}"
            lower.source_tiers = lower.source_tiers + upper.source_tiers
            tiers.pop()
            changed = True
            continue

        # Least severe tier too sparse -> merge into more-severe neighbor above.
        if int(tiers[0].mask.sum()) < min_samples:
            n = int(tiers[0].mask.sum())
            lower, upper = tiers[0], tiers[1]
            decisions.append(
                f"Merged '{lower.name}' (n={n} < {min_samples}) into '{upper.name}' "
                f"-> '{lower.name} + {upper.name}'"
            )
            upper.mask = lower.mask | upper.mask
            upper.name = f"{lower.name} + {upper.name}"
            upper.source_tiers = lower.source_tiers + upper.source_tiers
            tiers.pop(0)
            changed = True

    if len(tiers) < 2:
        raise SystemExit(
            "STOP: fewer than 2 usable IMD tiers after merging at this horizon — "
            "IMD boundaries may be a poor fit for this dataset."
        )
    return tiers, decisions


def tier_continuous_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    yt = np.asarray(y_true, dtype=np.float64)[mask]
    yp = np.asarray(y_pred, dtype=np.float64)[mask]
    n = int(len(yt))
    if n == 0:
        return {"N_samples": 0, "RMSE": float("nan"), "MAE": float("nan"), "Mean_Error_Bias": float("nan")}
    err = yt - yp
    return {
        "N_samples": n,
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "Mean_Error_Bias": float(np.mean(yp - yt)),
    }
