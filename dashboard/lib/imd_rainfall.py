"""IMD standard daily rainfall intensity classification.

India Meteorological Department (IMD) rainfall intensity categories for
24-hour accumulated rainfall. These are published IMD thresholds, not
project-invented bins.

IMPORTANT: Do NOT conflate with this project's Feature-4 statistical
"extreme" definition (test-set 95th-percentile threshold in
extreme_rainfall_evaluation.csv). IMD intensity classes and the project's
percentile-extreme flag are different concepts.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IMDCategory:
    name: str
    # Inclusive lower bound in mm; None = open (use next upper)
    low_mm: float
    # Inclusive upper bound in mm; None = no upper bound
    high_mm: float | None
    color: str


# IMD standard daily rainfall intensity scale (mm / 24 h)
IMD_CATEGORIES: tuple[IMDCategory, ...] = (
    IMDCategory("No Rain", 0.0, 0.0, "#64748b"),
    IMDCategory("Light", 0.1, 7.5, "#38bdf8"),
    IMDCategory("Moderate", 7.6, 35.5, "#22c55e"),
    IMDCategory("Heavy", 35.6, 64.4, "#f59e0b"),
    IMDCategory("Very Heavy", 64.5, 124.4, "#f97316"),
    IMDCategory("Extremely Heavy", 124.5, None, "#ef4444"),
)


def classify_imd_rainfall(mm: float) -> IMDCategory:
    """Map a daily rainfall amount (mm) to the IMD intensity category."""
    if not np_isfinite(mm) or mm < 0.1:
        # Non-finite / negative (unconstrained model) / trace < 0.1 → No Rain
        return IMD_CATEGORIES[0]
    for cat in IMD_CATEGORIES[1:]:
        if cat.high_mm is None:
            if mm >= cat.low_mm:
                return cat
        elif cat.low_mm <= mm <= cat.high_mm:
            return cat
    return IMD_CATEGORIES[-1]


def np_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))
