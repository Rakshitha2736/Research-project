"""Consolidate existing significance_results.csv rows into a 12-row robustness table.

Does not recompute DM/bootstrap. Reformats verified seed-level rows only.
Writes:
  reports/tables/multiseed_robustness_summary.csv
  reports/figures/multiseed_robustness_summary.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

BASE = Path(__file__).resolve().parent
SIG = BASE / "reports" / "tables" / "significance_results.csv"
OUT_CSV = BASE / "reports" / "tables" / "multiseed_robustness_summary.csv"
OUT_PNG = BASE / "reports" / "figures" / "multiseed_robustness_summary.png"

COMPARISONS = ("GNN_vs_LSTM", "Attention_vs_Temporal", "Attention_vs_LSTM")
HORIZONS = (1, 2, 3, 4)
SEEDS = (13, 42, 123)

# Comparison A_vs_B: CI = RMSE_A - RMSE_B. Negative CI => A better.
FIRST = {
    "GNN_vs_LSTM": "GNN",
    "Attention_vs_Temporal": "Attention",
    "Attention_vs_LSTM": "Attention",
}
SECOND = {
    "GNN_vs_LSTM": "LSTM",
    "Attention_vs_Temporal": "Temporal",
    "Attention_vs_LSTM": "LSTM",
}


def direction_from_ci(comp: str, lo: float, hi: float) -> str:
    first, second = FIRST[comp], SECOND[comp]
    if hi < 0:
        return first
    if lo > 0:
        return second
    mid = (lo + hi) / 2.0
    if mid < 0:
        return first
    if mid > 0:
        return second
    return "tie"


def seed_pattern(comp: str, direction: str, sig: str) -> str:
    first = FIRST[comp]
    second = SECOND[comp]
    if sig == "Yes" and direction == first:
        return "first_sig"
    if sig == "Yes" and direction == second:
        return "second_sig"
    if sig == "No":
        return "ns"
    return "ambig"


def consistency_verdict(patterns: list[str]) -> str:
    # patterns ordered seed 13, 42, 123 — same scheme as prior multi-seed reports
    if len(set(patterns)) == 1:
        return "CONSISTENT"
    if patterns[1] != patterns[0] and patterns[0] == patterns[2]:
        return "INCONSISTENT"
    return "MIXED"


def cell_code(comp: str, direction: str, sig: str) -> int:
    """0 dark green, 1 light green, 2 light red, 3 dark red (first-named = green)."""
    first = FIRST[comp]
    favors_first = direction == first
    if sig == "Yes" and favors_first:
        return 0
    if sig == "No" and favors_first:
        return 1
    if sig == "No" and not favors_first:
        return 2
    return 3


def main() -> None:
    sig = pd.read_csv(SIG)
    rows: list[dict] = []
    grid: list[list[int]] = []
    ylabels: list[str] = []

    for comp in COMPARISONS:
        for h in HORIZONS:
            sub = sig[
                (sig["Comparison"] == comp) & (sig["Forecast_Horizon"].astype(int) == h)
            ]
            by_seed = {}
            patterns = []
            rec: dict = {"Comparison": comp, "Horizon": h}
            for seed in SEEDS:
                r = sub[sub["Seeds_Used"].astype(int) == seed]
                if len(r) != 1:
                    raise SystemExit(f"expected 1 row for {comp} h={h} seed={seed}, got {len(r)}")
                r = r.iloc[0]
                lo = float(r["Bootstrap_95CI_lo"])
                hi = float(r["Bootstrap_95CI_hi"])
                direction = direction_from_ci(comp, lo, hi)
                yesno = str(r["Significant_at_0.05"])
                rec[f"Seed{seed}_Direction"] = direction
                rec[f"Seed{seed}_Sig"] = yesno
                by_seed[seed] = (direction, yesno)
                patterns.append(seed_pattern(comp, direction, yesno))
            rec["Consistency_Verdict"] = consistency_verdict(patterns)
            rec["N_Significant_of_3"] = sum(
                1 for s in SEEDS if rec[f"Seed{s}_Sig"] == "Yes"
            )
            rows.append(rec)
            grid.append([cell_code(comp, *by_seed[s]) for s in SEEDS])
            ylabels.append(f"{comp}  h={h}")

    out = pd.DataFrame(rows)
    cols = [
        "Comparison",
        "Horizon",
        "Seed13_Direction",
        "Seed13_Sig",
        "Seed42_Direction",
        "Seed42_Sig",
        "Seed123_Direction",
        "Seed123_Sig",
        "Consistency_Verdict",
        "N_Significant_of_3",
    ]
    out = out[cols]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(out.to_csv(index=False))

    cmap = ListedColormap(["#1b7f3a", "#9ccc6b", "#f0a0a0", "#c0392b"])
    arr = np.array(grid, dtype=int)
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.imshow(arr, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["seed 13", "seed 42", "seed 123"])
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    for i, rec in enumerate(rows):
        for j, seed in enumerate(SEEDS):
            d = rec[f"Seed{seed}_Direction"]
            s = rec[f"Seed{seed}_Sig"]
            mark = "sig" if s == "Yes" else "ns"
            ax.text(
                j,
                i,
                f"{d}\n{mark}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cell_code(rec["Comparison"], d, s) in (0, 3) else "#1a1a1a",
            )
    ax.set_title(
        "Multi-seed robustness (first-named model = green; opposite = red)",
        fontsize=11,
    )
    ax.set_xlabel("Green = first-named model better (GNN / Attention); red = opposite (LSTM / Temporal)")
    legend = [
        Patch(facecolor="#1b7f3a", label="Significant, first-named better"),
        Patch(facecolor="#9ccc6b", label="Non-significant, lean first-named"),
        Patch(facecolor="#f0a0a0", label="Non-significant, lean opposite"),
        Patch(facecolor="#c0392b", label="Significant, opposite better"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
