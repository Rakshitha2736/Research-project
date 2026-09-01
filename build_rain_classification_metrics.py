"""
Binary rain/no-rain classification metrics at threshold=1.0 mm.

No retraining. No fresh inference.

Raw inverse-transformed prediction arrays are NOT saved for LSTM /
Temporal / Attention. Contingency counts (hits/misses/false_alarms/
correct_negatives) for seed=42 at threshold 1.0 mm already exist in
reports/tables/threshold_skill.csv from the prior eval_threshold_skill.py
run — Precision/Recall/F1 are derived from those counts only.

Writes:
  reports/tables/rain_classification_metrics.csv
  reports/figures/rain_classification_f1.png
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
TABLES = BASE / "reports" / "tables"
FIGURES = BASE / "reports" / "figures"

SKILL_CSV = TABLES / "threshold_skill.csv"
OUT_CSV = TABLES / "rain_classification_metrics.csv"
OUT_FIG = FIGURES / "rain_classification_f1.png"

# Hash-monitored verified sources (must not be written)
VERIFIED = [
    TABLES / "threshold_skill.csv",
    TABLES / "threshold_skill_summary.csv",
    TABLES / "master_results.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
]

THRESHOLD_MM = 1.0
SEED = 42
HORIZONS = (1, 2, 3, 4)

MODEL_MAP = {
    "lstm": "LSTM",
    "temporal": "CNN-LSTM-Temporal",
    "attention": "CNN-LSTM+Attention",
}
MODEL_ORDER = ["LSTM", "CNN-LSTM-Temporal", "CNN-LSTM+Attention"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_raw_prediction_arrays() -> list[str]:
    """Return list of missing (model, horizon) where raw pred arrays absent."""
    missing: list[str] = []
    # Canonical places we would look for saved mm predictions — none exist.
    patterns = [
        "data/processed/*pred*",
        "data/processed/y_pred*",
        "models/*pred*.npy",
        "reports/**/y_pred*",
    ]
    found_any = False
    for pat in patterns:
        if list(BASE.glob(pat)):
            found_any = True
            break
    if not found_any:
        for model in MODEL_ORDER:
            for h in HORIZONS:
                missing.append(f"{model} h={h}")
    return missing


def build_metrics(skill: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, label in MODEL_MAP.items():
        for h in HORIZONS:
            m = skill[
                (skill["model"] == key)
                & (skill["horizon"] == h)
                & (skill["seed"] == SEED)
                & (skill["threshold_mm"] == THRESHOLD_MM)
            ]
            if len(m) != 1:
                raise ValueError(
                    f"expected one threshold_skill row for "
                    f"{key} h={h} seed={SEED} thr={THRESHOLD_MM}, got {len(m)}"
                )
            r = m.iloc[0]
            tp = int(r["hits"])
            fn = int(r["misses"])
            fp = int(r["false_alarms"])
            tn = int(r["correct_negatives"])
            n = int(r["n"])
            n_rain = int(r["n_obs_event"])

            precision = tp / (tp + fp) if (tp + fp) else float("nan")
            recall = tp / (tp + fn) if (tp + fn) else float("nan")
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else float("nan")
            )
            base_rate = 100.0 * n_rain / n if n else float("nan")

            # Cross-check against POD / (1-FAR) from same row
            pod = float(r["POD"])
            far = float(r["FAR"])
            if abs(recall - pod) > 1e-6:
                raise AssertionError(f"Recall != POD for {label} h={h}")
            if abs(precision - (1.0 - far)) > 1e-6:
                raise AssertionError(f"Precision != 1-FAR for {label} h={h}")

            rows.append(
                {
                    "Model": label,
                    "Horizon": h,
                    "Threshold_mm": THRESHOLD_MM,
                    "Precision": round(precision, 6),
                    "Recall": round(recall, 6),
                    "F1": round(f1, 6),
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                    "Rain_Day_Base_Rate_Pct": round(base_rate, 4),
                }
            )
    return pd.DataFrame(rows)


def plot_f1(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "LSTM": "#2F4F4F",
        "CNN-LSTM-Temporal": "#3D7A8C",
        "CNN-LSTM+Attention": "#C45C26",
    }
    labels = {
        "LSTM": "LSTM",
        "CNN-LSTM-Temporal": "CNN-LSTM-Temporal",
        "CNN-LSTM+Attention": "CNN-LSTM+Attention",
    }

    x = np.arange(len(HORIZONS), dtype=float)
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for i, model in enumerate(MODEL_ORDER):
        vals = [
            float(df[(df["Horizon"] == h) & (df["Model"] == model)].iloc[0]["F1"])
            for h in HORIZONS
        ]
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            vals,
            width,
            label=labels[model],
            color=colors[model],
            edgecolor="none",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("F1 (rain class)")
    ax.set_title(
        f"Rain/no-rain F1 by model and horizon (threshold = {THRESHOLD_MM:.1f} mm)"
    )
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


def main() -> None:
    hashes_before = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print("BEFORE hashes (verified sources):")
    for k, v in hashes_before.items():
        print(f"  {k}: {v}")

    missing_preds = check_raw_prediction_arrays()
    print()
    print("=" * 72)
    print("PREDICTION SOURCE AUDIT")
    print("=" * 72)
    print(
        "Raw inverse-transformed prediction arrays (.npy/.npz with y_pred): "
        "NOT FOUND for any model/horizon."
    )
    print(
        f"Flagged as unavailable for fresh-array reuse ({len(missing_preds)}/12):"
    )
    for item in missing_preds:
        print(f"  - {item}")
    print(
        "Fallback (no fresh inference): derive Precision/Recall/F1/TP/FP/TN/FN "
        f"from reports/tables/threshold_skill.csv contingency counts "
        f"(seed={SEED}, threshold_mm={THRESHOLD_MM})."
    )
    print(f"Binarization rule: rain = value >= {THRESHOLD_MM:.1f} mm.")

    skill = pd.read_csv(SKILL_CSV)
    metrics = build_metrics(skill)
    TABLES.mkdir(parents=True, exist_ok=True)
    header = (
        "# This table reports F1, not CSI. For CSI/POD/FAR/Bias, see threshold_skill.csv.\n"
        "# Derived from threshold_skill.csv contingency counts (seed=42, threshold_mm=1.0).\n"
    )
    OUT_CSV.write_text(header + metrics.to_csv(index=False), encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")

    plot_f1(metrics)

    hashes_after = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print()
    print("=" * 72)
    print("RAIN CLASSIFICATION METRICS (full table)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(metrics.to_string(index=False))

    # Base-rate consistency across models per horizon
    print()
    print("Rain-day base rate by horizon (should match across models):")
    for h in HORIZONS:
        rates = metrics.loc[metrics["Horizon"] == h, "Rain_Day_Base_Rate_Pct"].tolist()
        print(f"  h={h}: {rates}")

    print()
    print("=" * 72)
    print("INTEGRITY")
    print("=" * 72)
    all_ok = True
    for k, v0 in hashes_before.items():
        v1 = hashes_after[k]
        ok = v0 == v1
        all_ok = all_ok and ok
        print(f"  {k}: {'UNCHANGED' if ok else 'CHANGED'}")
    assert all_ok
    print("CONFIRM: no verified source CSV modified; no model retrained; no fresh inference.")
    print(
        "NOTE: metrics derived from already-saved contingency counts in "
        "threshold_skill.csv (not from re-running checkpoints)."
    )


if __name__ == "__main__":
    main()
