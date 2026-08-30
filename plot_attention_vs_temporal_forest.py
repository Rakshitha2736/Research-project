"""
Attention vs Temporal-CNN-LSTM multi-horizon forest plot (seed-42 point estimates only).

This plot shows seed-42 point estimates only. See
multiseed_robustness_summary.csv/.png for the complete 3-seed picture, which
shows this result is not consistent across seeds.

Read-only sources:
  - reports/tables/significance_results.csv (Attention_vs_Temporal, seed 42)
  - reports/tables/ablation_study.csv (seed-42 Delta_RMSE vs previous stage)

No training. Does not modify source CSVs.
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

SIG_CSV = TABLES / "significance_results.csv"
ABLATION_CSV = TABLES / "ablation_study.csv"
MASTER_CSV = TABLES / "master_results.csv"
OUT_FIG = FIGURES / "attention_vs_temporal_forest_plot.png"

ALPHA_BONF = 0.05 / 4  # 0.0125
HORIZONS = (1, 2, 3, 4)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_forest_data() -> pd.DataFrame:
    sig = pd.read_csv(SIG_CSV)
    abl = pd.read_csv(ABLATION_CSV)

    att_sig = sig[sig["Comparison"] == "Attention_vs_Temporal"].copy()
    if "Seeds_Used" in att_sig.columns:
        att_sig = att_sig[att_sig["Seeds_Used"].astype(int) == 42]
    att_abl = abl[abl["Model"] == "CNN-LSTM+Attention"].copy()

    rows = []
    for h in HORIZONS:
        s = att_sig[att_sig["Forecast_Horizon"] == h]
        a = att_abl[att_abl["Horizon"] == h]
        if len(s) != 1 or len(a) != 1:
            raise ValueError(f"expected one Attention_vs_Temporal + Attention row for h={h}")
        s = s.iloc[0]
        a = a.iloc[0]
        dm_p = float(s["DM_p_value"])
        ci_lo = float(s["Bootstrap_95CI_lo"])
        ci_hi = float(s["Bootstrap_95CI_hi"])
        marker = float(a["Delta_RMSE_seed42_vs_previous_stage"])
        sig_bonf = dm_p < ALPHA_BONF
        rows.append(
            {
                "horizon": h,
                "marker": marker,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "dm_p": dm_p,
                "sig_uncorrected_0.05": dm_p < 0.05,
                "sig_bonferroni_0.0125": sig_bonf,
                "ci_excludes_0": (ci_lo > 0) or (ci_hi < 0),
            }
        )
    return pd.DataFrame(rows)


def print_step1(df: pd.DataFrame) -> None:
    print("=" * 72)
    print("STEP 1 — Extracted markers + bootstrap CIs (Attention vs Temporal)")
    print("=" * 72)
    for _, r in df.iterrows():
        print(
            f"h={int(r['horizon'])}: marker={r['marker']:+.4f}  "
            f"CI=({r['ci_lo']:+.4f}, {r['ci_hi']:+.4f})"
        )
    print()
    print("8 values summary:")
    print("  markers:", [f"{r['marker']:+.4f}" for _, r in df.iterrows()])
    print(
        "  CIs:    ",
        [f"({r['ci_lo']:+.4f}, {r['ci_hi']:+.4f})" for _, r in df.iterrows()],
    )


def print_step3(df: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print(f"STEP 3 — Bonferroni-corrected significance (alpha={ALPHA_BONF})")
    print("=" * 72)
    print(
        f"{'h':>3}  {'DM_p':>14}  {'<0.05':>6}  {'<0.0125':>8}  "
        f"{'CI excl 0':>9}  {'PASS (Bonf)':>11}"
    )
    print("-" * 72)
    for _, r in df.iterrows():
        pass_bonf = "PASS" if r["sig_bonferroni_0.0125"] else "FAIL"
        print(
            f"{int(r['horizon']):>3}  {r['dm_p']:14.6e}  "
            f"{'Yes' if r['sig_uncorrected_0.05'] else 'No':>6}  "
            f"{'Yes' if r['sig_bonferroni_0.0125'] else 'No':>8}  "
            f"{'Yes' if r['ci_excludes_0'] else 'No':>9}  "
            f"{pass_bonf:>11}"
        )
    passes = [int(r["horizon"]) for _, r in df.iterrows() if r["sig_bonferroni_0.0125"]]
    fails = [int(r["horizon"]) for _, r in df.iterrows() if not r["sig_bonferroni_0.0125"]]
    print()
    print(f"PASS under Bonferroni: h={passes}")
    print(f"FAIL under Bonferroni: h={fails}")
    assert passes == [2, 4], f"expected PASS h=2,4 got {passes}"
    assert fails == [1, 3], f"expected FAIL h=1,3 got {fails}"
    print("CONFIRM: h=2 and h=4 pass; h=1 and h=3 fail under alpha=0.0125.")


def plot_forest(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Match gnn_vs_lstm_forest_plot visual language
    color = "#1a3a5c"
    y = np.arange(len(HORIZONS), dtype=float)[::-1]  # h=1 at top

    fig, ax = plt.subplots(figsize=(8.5, 4.8))

    for i, (_, r) in enumerate(df.iterrows()):
        yi = y[i]
        xerr_lo = r["marker"] - r["ci_lo"]
        xerr_hi = r["ci_hi"] - r["marker"]
        ax.errorbar(
            r["marker"],
            yi,
            xerr=[[xerr_lo], [xerr_hi]],
            fmt="none",
            ecolor=color,
            elinewidth=1.6,
            capsize=4,
            capthick=1.4,
            zorder=2,
        )
        if r["sig_bonferroni_0.0125"]:
            ax.plot(
                r["marker"],
                yi,
                marker="o",
                markersize=9,
                color=color,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.4,
                linestyle="None",
                zorder=3,
            )
        else:
            ax.plot(
                r["marker"],
                yi,
                marker="o",
                markersize=9,
                color=color,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                linestyle="None",
                zorder=3,
            )
        # Grayscale-readable annotation
        tag = "sig*" if r["sig_bonferroni_0.0125"] else "n.s."
        ax.text(
            r["ci_hi"] + 0.008,
            yi,
            tag,
            va="center",
            ha="left",
            fontsize=9,
            color=color,
        )

    ax.axvline(0.0, color="#888888", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"h={h}" for h in HORIZONS])
    ax.set_xlabel("Delta RMSE (Attention - Temporal-CNN-LSTM), mm/day")
    ax.set_title(
        "CNN-LSTM+Attention vs CNN-LSTM-Temporal: RMSE Difference by Forecast Horizon"
    )
    ax.grid(axis="x", linestyle=":", linewidth=0.7, color="#cccccc", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: one filled + one hollow entry
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="None",
            markersize=9,
            label="Significant (Bonferroni α=0.0125)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            linestyle="None",
            markersize=9,
            label="Not significant",
        ),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="lower right")

    caption = (
        "Negative = Attention performs better. Filled markers indicate statistical "
        "significance (bootstrap 95% CI excludes zero, Bonferroni-corrected "
        "alpha=0.0125 for 4 comparisons)."
    )
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"\nWrote {OUT_FIG}")


def main() -> None:
    hash_before = {
        "ablation_study.csv": sha256_file(ABLATION_CSV),
        "significance_results.csv": sha256_file(SIG_CSV),
        "master_results.csv": sha256_file(MASTER_CSV),
    }
    print("BEFORE hashes:")
    for k, v in hash_before.items():
        print(f"  {k}: {v}")

    df = extract_forest_data()
    print_step1(df)
    print_step3(df)
    plot_forest(df)

    hash_after = {
        "ablation_study.csv": sha256_file(ABLATION_CSV),
        "significance_results.csv": sha256_file(SIG_CSV),
        "master_results.csv": sha256_file(MASTER_CSV),
    }
    print()
    print("=" * 72)
    print("STEP 4 — Integrity")
    print("=" * 72)
    all_ok = True
    for k in hash_before:
        ok = hash_before[k] == hash_after[k]
        all_ok = all_ok and ok
        print(f"  {k}: {'UNCHANGED' if ok else 'CHANGED'}  ({hash_after[k]})")
    assert all_ok, "source CSV was modified"
    print("CONFIRM: no source CSV modified; no model retrained.")
    print(f"Figure: {OUT_FIG}")


if __name__ == "__main__":
    main()
