"""Consolidate existing significance_results.csv rows into a 12-row robustness table.

Does not recompute DM/bootstrap. Reformats verified seed-level rows only.
Writes:
  reports/tables/multiseed_robustness_summary.csv
  reports/figures/multiseed_robustness_summary.png
"""

from __future__ import annotations

from pathlib import Path
import inspect

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


def consistency_verdict(directions: list[str], sigs: list[str]) -> str:
    """3-tier rule (4 labels). `directions`/`sigs` are length-3, seeds 13, 42, 123.

    CONSISTENT: all 3 seeds agree on direction AND all 3 (or none) are significant.
    DIRECTION-STABLE: all 3 agree on direction, but significance is mixed (1 or 2 of 3).
    DIRECTION-UNSTABLE (contested): directions disagree AND >=2 seeds are significant.
    DIRECTION-UNSTABLE (weak): directions disagree AND <2 seeds are significant.
    """
    n_sig = sum(1 for s in sigs if s == "Yes")
    same_direction = len(set(directions)) == 1
    if same_direction:
        if n_sig in (0, 3):
            return "CONSISTENT"
        return "DIRECTION-STABLE"
    if n_sig >= 2:
        return "DIRECTION-UNSTABLE (contested)"
    return "DIRECTION-UNSTABLE (weak)"


VERDICT_COLORS = {
    "CONSISTENT": "#1b4f9e",
    "DIRECTION-STABLE": "#d4a017",
    "DIRECTION-UNSTABLE (contested)": "#8b1a1a",
    "DIRECTION-UNSTABLE (weak)": "#e07b39",
}
VERDICT_CODES = {
    "CONSISTENT": 0,
    "DIRECTION-STABLE": 1,
    "DIRECTION-UNSTABLE (contested)": 2,
    "DIRECTION-UNSTABLE (weak)": 3,
}


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
            rec: dict = {"Comparison": comp, "Horizon": h}
            dirs: list[str] = []
            sigs: list[str] = []
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
                dirs.append(direction)
                sigs.append(yesno)
            rec["Consistency_Verdict"] = consistency_verdict(dirs, sigs)
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
    print("RULE (consistency_verdict):")
    print(inspect.getsource(consistency_verdict))
    print(out.to_csv(index=False))
    a1 = out[(out["Comparison"] == "Attention_vs_LSTM") & (out["Horizon"] == 1)].iloc[0]
    a2 = out[(out["Comparison"] == "Attention_vs_LSTM") & (out["Horizon"] == 2)].iloc[0]
    t2 = out[(out["Comparison"] == "Attention_vs_Temporal") & (out["Horizon"] == 2)].iloc[0]
    a3 = out[(out["Comparison"] == "Attention_vs_LSTM") & (out["Horizon"] == 3)].iloc[0]
    print(
        "PAIR CHECK Attn_vs_LSTM h=1 vs h=2:",
        a1["Consistency_Verdict"],
        a2["Consistency_Verdict"],
        "MATCH" if a1["Consistency_Verdict"] == a2["Consistency_Verdict"] else "FAIL",
    )
    print(
        "PAIR CHECK Attn_vs_Temporal h=2 vs Attn_vs_LSTM h=3:",
        t2["Consistency_Verdict"],
        a3["Consistency_Verdict"],
        "MATCH" if t2["Consistency_Verdict"] == a3["Consistency_Verdict"] else "FAIL",
    )

    seed_cmap = ListedColormap(["#1b7f3a", "#9ccc6b", "#f0a0a0", "#c0392b"])
    verdict_cmap = ListedColormap(
        [
            VERDICT_COLORS["CONSISTENT"],
            VERDICT_COLORS["DIRECTION-STABLE"],
            VERDICT_COLORS["DIRECTION-UNSTABLE (contested)"],
            VERDICT_COLORS["DIRECTION-UNSTABLE (weak)"],
        ]
    )
    seed_arr = np.array(grid, dtype=int)
    verdict_arr = np.array(
        [[VERDICT_CODES[r["Consistency_Verdict"]]] for r in rows], dtype=int
    )
    fig, (ax_s, ax_v) = plt.subplots(
        1, 2, figsize=(10.4, 7.2), gridspec_kw={"width_ratios": [3.2, 1.35]}
    )
    ax_s.imshow(seed_arr, cmap=seed_cmap, vmin=0, vmax=3, aspect="auto")
    ax_s.set_xticks([0, 1, 2])
    ax_s.set_xticklabels(["seed 13", "seed 42", "seed 123"])
    ax_s.set_yticks(range(len(ylabels)))
    ax_s.set_yticklabels(ylabels, fontsize=8)
    ax_s.set_title("Per-seed direction + significance", fontsize=10)
    for i, rec in enumerate(rows):
        for j, seed in enumerate(SEEDS):
            d = rec[f"Seed{seed}_Direction"]
            s = rec[f"Seed{seed}_Sig"]
            mark = "sig" if s == "Yes" else "ns"
            ax_s.text(
                j,
                i,
                f"{d}\n{mark}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if cell_code(rec["Comparison"], d, s) in (0, 3) else "#1a1a1a",
            )

    ax_v.imshow(verdict_arr, cmap=verdict_cmap, vmin=0, vmax=3, aspect="auto")
    ax_v.set_xticks([0])
    ax_v.set_xticklabels(["verdict"])
    ax_v.set_yticks(range(len(ylabels)))
    ax_v.set_yticklabels([])
    ax_v.set_title("4-tier rule", fontsize=10)
    short = {
        "CONSISTENT": "CONSISTENT",
        "DIRECTION-STABLE": "DIR-STABLE",
        "DIRECTION-UNSTABLE (contested)": "UNSTABLE\n(contested)",
        "DIRECTION-UNSTABLE (weak)": "UNSTABLE\n(weak)",
    }
    for i, rec in enumerate(rows):
        v = rec["Consistency_Verdict"]
        ax_v.text(
            0,
            i,
            short[v],
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            fontweight="bold",
        )
    fig.suptitle(
        "Multi-seed robustness — GNN is uniformly CONSISTENT; Attention is not",
        fontsize=11,
        y=0.98,
    )
    legend = [
        Patch(facecolor="#1b7f3a", label="Seed: sig, first-named better"),
        Patch(facecolor="#9ccc6b", label="Seed: ns, lean first-named"),
        Patch(facecolor="#f0a0a0", label="Seed: ns, lean opposite"),
        Patch(facecolor="#c0392b", label="Seed: sig, opposite better"),
        Patch(facecolor=VERDICT_COLORS["CONSISTENT"], label="Verdict: CONSISTENT"),
        Patch(facecolor=VERDICT_COLORS["DIRECTION-STABLE"], label="Verdict: DIRECTION-STABLE"),
        Patch(
            facecolor=VERDICT_COLORS["DIRECTION-UNSTABLE (contested)"],
            label="Verdict: DIRECTION-UNSTABLE (contested)",
        ),
        Patch(
            facecolor=VERDICT_COLORS["DIRECTION-UNSTABLE (weak)"],
            label="Verdict: DIRECTION-UNSTABLE (weak)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=7.5,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
