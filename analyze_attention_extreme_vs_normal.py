"""
Attention profiles on Extreme vs Normal rainfall days (CNN-LSTM+Attention).

Horizons 1 & 4: reuse saved attention_weights_h*_seed42.npy.
Horizons 2 & 3: CUDA+autocast inference (saves cache), same provenance
convention as Features 3/4.

Writes:
  reports/tables/attention_extreme_vs_normal.csv
  reports/figures/attention_extreme_vs_normal_h{1,2,3,4}.png
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.eval_attention import (
    SEQ_LEN,
    collect_attention_and_preds,
    load_attention_model,
    paths_for_horizon,
    profile_summary,
)
from src.model import CNNLSTMAttention

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
TABLES = BASE / "reports" / "tables"
FIGURES = BASE / "reports" / "figures"

OUT_CSV = TABLES / "attention_extreme_vs_normal.csv"
EXTREME_CSV = TABLES / "extreme_rainfall_evaluation.csv"

SEED = 42
HORIZONS = (1, 2, 3, 4)
# Figures required at minimum for significance-confirmed horizons; generate all.
PLOT_HORIZONS = (1, 2, 3, 4)

VERIFIED = [
    TABLES / "master_results.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
    TABLES / "threshold_skill.csv",
    TABLES / "extreme_rainfall_evaluation.csv",
    TABLES / "rain_classification_metrics.csv",
    DATA / "attention_weights_h1_seed42.npy",
    DATA / "attention_weights_h4_seed42.npy",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_thresholds() -> dict[int, float]:
    df = pd.read_csv(EXTREME_CSV, comment="#")
    thr = {}
    for h in HORIZONS:
        rows = df[(df["Horizon"] == h) & (df["Subset"] == "Extreme")]
        thr[h] = float(rows.iloc[0]["Threshold_mm"])
    return thr


def load_or_infer_attn(
    horizon: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return attn (N,30), y_true_mm, provenance string."""
    paths = paths_for_horizon(BASE, horizon)
    cache = Path(str(paths["attn_cache"]).format(seed=SEED))
    X_test = np.load(paths["X_test"])
    y_test = np.load(paths["y_test"])
    scaler_y = joblib.load(paths["scaler_y"])
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

    if cache.exists():
        attn = np.load(cache)
        if attn.shape != (len(X_test), SEQ_LEN):
            raise ValueError(f"bad cache shape {cache}: {attn.shape}")
        return (
            attn,
            y_true,
            f"reused saved {cache.name} (prior CUDA+autocast inference)",
        )

    # Fresh inference for missing horizons
    ckpt = Path(str(paths["ckpt"]).format(seed=SEED))
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    dummy = np.zeros(len(X_test), dtype=np.float32)
    loader = make_loader(X_test, dummy, batch_size=DEFAULT_BATCH_SIZE, shuffle=False)
    model = load_attention_model(ckpt, device)
    attn, _ = collect_attention_and_preds(model, loader, device)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, attn)
    del model
    torch.cuda.empty_cache()
    return (
        attn,
        y_true,
        (
            f"NEW CUDA+autocast inference via analyze_attention_extreme_vs_normal.py; "
            f"saved {cache.name} (same seed-{SEED} checkpoint; provenance = Feature 3/4 "
            f"convention — distinct run, deterministic eval-mode Attention model)"
        ),
    )


def plot_overlay(
    horizon: int,
    mean_ext: np.ndarray,
    mean_norm: np.ndarray,
    thr_mm: float,
    n_ext: int,
    n_norm: int,
) -> Path:
    """mean_* are chronological (idx0=oldest). Plot day-pos 1=recent … 30=oldest."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    days = np.arange(1, SEQ_LEN + 1)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(
        days,
        mean_norm[::-1],
        color="#3D7A8C",
        linewidth=2.0,
        label=f"Normal (n={n_norm})",
    )
    ax.plot(
        days,
        mean_ext[::-1],
        color="#C45C26",
        linewidth=2.0,
        linestyle="--",
        label=f"Extreme (n={n_ext})",
    )
    ax.set_xlabel("Days before target (1 = most recent, 30 = oldest)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(
        f"Attention: Extreme vs Normal targets (h={horizon}, seed {SEED}, "
        f"thr≥{thr_mm:.1f} mm)"
    )
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0.5, 30.5)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIGURES / f"attention_extreme_vs_normal_h{horizon}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    device = require_cuda()
    TABLES.mkdir(parents=True, exist_ok=True)

    hashes_before = {str(p): sha256_file(p) for p in VERIFIED if p.exists()}
    print("BEFORE hashes (verified sources):")
    for k, v in hashes_before.items():
        print(f"  {Path(k).name}: {v}")

    thresholds = load_thresholds()
    print("\n95th-pct thresholds from extreme_rainfall_evaluation.csv:")
    for h, t in thresholds.items():
        print(f"  h={h}: {t:.6f} mm")

    rows: list[dict] = []
    mw_rows: list[dict] = []
    provenances: dict[int, str] = {}

    for h in HORIZONS:
        attn, y_true, prov = load_or_infer_attn(h, device)
        provenances[h] = prov
        print(f"\nh={h}: attn={attn.shape}  source={prov}")

        thr = thresholds[h]
        extreme = y_true >= thr
        normal = ~extreme
        n_ext = int(extreme.sum())
        n_norm = int(normal.sum())
        assert n_ext + n_norm == len(y_true)

        mean_ext = attn[extreme].mean(axis=0)
        mean_norm = attn[normal].mean(axis=0)
        sum_ext = profile_summary(mean_ext)
        sum_norm = profile_summary(mean_norm)

        for subset, summary, n in (
            ("Extreme", sum_ext, n_ext),
            ("Normal", sum_norm, n_norm),
        ):
            rows.append(
                {
                    "Horizon": h,
                    "Subset": subset,
                    "Peak_Day_Position": int(summary["peak_day_position"]),
                    "Recent_7d_Share": round(float(summary["recent_7_share"]), 6),
                    "Oldest_7d_Share": round(float(summary["oldest_7_share"]), 6),
                    "N_samples": n,
                }
            )

        # Day-30 = oldest = chronological index 0
        w30_ext = attn[extreme, 0]
        w30_norm = attn[normal, 0]
        u_stat, p_val = stats.mannwhitneyu(
            w30_ext, w30_norm, alternative="two-sided"
        )
        mw_rows.append(
            {
                "Horizon": h,
                "U": float(u_stat),
                "p_value": float(p_val),
                "mean_day30_Extreme": float(w30_ext.mean()),
                "mean_day30_Normal": float(w30_norm.mean()),
                "diff_Ext_minus_Norm": float(w30_ext.mean() - w30_norm.mean()),
            }
        )

        if h in PLOT_HORIZONS:
            out = plot_overlay(h, mean_ext, mean_norm, thr, n_ext, n_norm)
            print(f"  Wrote {out}")

    df = pd.DataFrame(rows)
    NOTES = (
        "# NOTES (pandas: read_csv(..., comment=\"#\"))\n"
        "# Attention Extreme vs Normal split uses the same true-target 95th-percentile "
        "thresholds as extreme_rainfall_evaluation.csv.\n"
        "# Day-position convention: 1=most recent ... 30=oldest (matches "
        "attention_weights_h4_mean.png).\n"
        "# Provenance: h=1 and h=4 reuse saved attention_weights_h*_seed42.npy; "
        "h=2 and h=3 from a separate CUDA+autocast inference session "
        "(analyze_attention_extreme_vs_normal.py) on the same seed-42 Attention "
        "checkpoints — distinct run, not literally reused arrays from significance "
        "testing; deterministic in eval mode (no dropout/BN).\n"
    )
    OUT_CSV.write_text(NOTES + df.to_csv(index=False), encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")

    mw = pd.DataFrame(mw_rows)

    print("\n" + "=" * 72)
    print("attention_extreme_vs_normal.csv (full)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df.to_string(index=False))

    print("\n" + "=" * 72)
    print("Mann-Whitney U: day-30 attention (Extreme vs Normal)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(mw.to_string(index=False))

    print("\n" + "=" * 72)
    print("PLAIN-LANGUAGE SUMMARY")
    print("=" * 72)
    for _, r in mw.iterrows():
        h = int(r["Horizon"])
        p = float(r["p_value"])
        d = float(r["diff_Ext_minus_Norm"])
        peak_e = int(df[(df.Horizon == h) & (df.Subset == "Extreme")].iloc[0]["Peak_Day_Position"])
        peak_n = int(df[(df.Horizon == h) & (df.Subset == "Normal")].iloc[0]["Peak_Day_Position"])
        rec_e = float(df[(df.Horizon == h) & (df.Subset == "Extreme")].iloc[0]["Recent_7d_Share"])
        rec_n = float(df[(df.Horizon == h) & (df.Subset == "Normal")].iloc[0]["Recent_7d_Share"])
        old_e = float(df[(df.Horizon == h) & (df.Subset == "Extreme")].iloc[0]["Oldest_7d_Share"])
        old_n = float(df[(df.Horizon == h) & (df.Subset == "Normal")].iloc[0]["Oldest_7d_Share"])
        sig = "significantly different" if p < 0.05 else "NOT significantly different"
        direction = (
            "higher day-30 (oldest) weight on Extreme days"
            if d > 0
            else "lower day-30 (oldest) weight on Extreme days"
        )
        print(
            f"h={h}: peak Extreme=day {peak_e}, Normal=day {peak_n}; "
            f"recent-7 share Ext={rec_e:.4f} vs Norm={rec_n:.4f}; "
            f"oldest-7 Ext={old_e:.4f} vs Norm={old_n:.4f}."
        )
        print(
            f"       Day-30 weight is {sig} (Mann-Whitney p={p:.3e}; "
            f"mean Ext-Norm={d:+.5f}) — {direction}."
        )

    print("\nProvenance by horizon:")
    for h in HORIZONS:
        print(f"  h={h}: {provenances[h]}")

    # Integrity: verified files unchanged; newly written caches for h2/h3 are expected
    hashes_after = {str(p): sha256_file(p) for p in VERIFIED if p.exists()}
    print("\n" + "=" * 72)
    print("INTEGRITY")
    print("=" * 72)
    ok = True
    for k, v0 in hashes_before.items():
        # newly created h2/h3 caches were not in before set
        v1 = hashes_after.get(k)
        if v1 is None:
            continue
        match = v0 == v1
        ok = ok and match
        print(f"  {Path(k).name}: {'UNCHANGED' if match else 'CHANGED'}")
    assert ok
    for h in (2, 3):
        cache = DATA / f"attention_weights_h{h}_seed{SEED}.npy"
        print(f"  NEW cache written: {cache.name} exists={cache.exists()}")
    print(
        "CONFIRM: no prior verified CSV/npy modified; h=2/h=3 attention caches "
        "are new CUDA+autocast artifacts (documented provenance)."
    )


if __name__ == "__main__":
    main()
