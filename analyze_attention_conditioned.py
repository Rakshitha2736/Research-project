"""
Conditioned attention analysis for temporal CNN-LSTM + Bahdanau Attention.

Keeps the architecture frozen. For each horizon:
  - mean α overall
  - wet vs dry (y_true >= rain_threshold mm)
  - JJAS monsoon vs non-monsoon
  - high-error vs low-error (abs error >= 90th percentile)

Writes:
  reports/tables/attention_conditioned_h{h}.csv
  reports/figures/attention_conditioned_h{h}_*.png

Usage (from RainfallPrediction/, CUDA venv):
  python analyze_attention_conditioned.py
  python analyze_attention_conditioned.py --horizons 1 4
  python analyze_attention_conditioned.py --horizons 1 2 3 4 --rain-threshold 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.cuda_setup import require_cuda
from src.eval_attention import (
    SEQ_LEN,
    attention_entropy,
    bootstrap_mean_diff,
    is_monsoon,
    mean_profile,
    profile_summary,
    rebuild_test_meta,
    run_inference_bundle,
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
FIGURES = BASE / "reports" / "figures"
TABLES = BASE / "reports" / "tables"
FEAT_CSV = DATA / "feature_engineered_v2.csv"

# Plot axis convention (matches plot_attention_weights_h4.py):
# day-position 1 = most recent … 30 = oldest
DAYS_BEFORE = np.arange(1, SEQ_LEN + 1)


def _plot_profiles(
    profiles: dict[str, np.ndarray],
    title: str,
    out: Path,
    colors: dict[str, str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    default_colors = {
        "all": "#2c6e8a",
        "wet": "#c45c26",
        "dry": "#6b8f71",
        "monsoon": "#3d5a80",
        "non_monsoon": "#98c1d9",
        "high_error": "#9b2226",
        "low_error": "#588157",
    }
    for name, mean_w in profiles.items():
        if mean_w is None or np.isnan(mean_w).all():
            continue
        # reverse chronological → days-before-target axis
        y = mean_w[::-1]
        color = (colors or default_colors).get(name, None)
        ax.plot(DAYS_BEFORE, y, label=name, linewidth=2.0, color=color)
    ax.set_xlabel("Days before target (1 = most recent, 30 = oldest)")
    ax.set_ylabel("Mean attention weight")
    ax.set_title(title)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlim(0.5, 30.5)
    ax.legend(frameon=False, ncols=2)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()


def _stratum_row(
    name: str,
    horizon: int,
    seed: int,
    mask: np.ndarray,
    attn: np.ndarray,
    ent: np.ndarray,
) -> dict:
    n = int(mask.sum())
    mean_w = mean_profile(attn, mask)
    summary = profile_summary(mean_w) if n > 0 else {
        "peak_day_position": -1,
        "recent_7_share": float("nan"),
        "oldest_7_share": float("nan"),
        "entropy_of_mean": float("nan"),
    }
    return {
        "horizon": horizon,
        "seed": seed,
        "stratum": name,
        "n": n,
        "fraction": float(n / len(mask)) if len(mask) else float("nan"),
        "peak_day_position": summary["peak_day_position"],
        "recent_7_share": summary["recent_7_share"],
        "oldest_7_share": summary["oldest_7_share"],
        "mean_sample_entropy": float(ent[mask].mean()) if n else float("nan"),
        "entropy_of_mean_profile": summary["entropy_of_mean"],
    }


def analyze_horizon(
    horizon: int,
    seed: int,
    device,
    rain_threshold: float,
    high_err_q: float,
    n_boot: int,
    feat_df: pd.DataFrame | None,
) -> pd.DataFrame:
    print(f"\n========== h={horizon} seed={seed} ==========", flush=True)
    bundle = run_inference_bundle(BASE, horizon, seed, device)
    attn = bundle["attn"]
    y_true = bundle["y_true"]
    abs_err = bundle["abs_err"]
    n = bundle["n"]
    ent = attention_entropy(attn)

    # --- masks that do not need dates ---
    wet = y_true >= rain_threshold
    dry = ~wet
    err_thr = float(np.quantile(abs_err, high_err_q))
    high_err = abs_err >= err_thr
    low_err = ~high_err

    # --- monsoon needs target dates (rebuild meta; assert length) ---
    if feat_df is None:
        raise RuntimeError("feature_engineered_v2.csv required for monsoon stratum")
    print("Rebuilding test meta for date alignment (matches sequence order)...", flush=True)
    meta = rebuild_test_meta(feat_df, horizon)
    if len(meta) != n:
        raise RuntimeError(
            f"Meta/test length mismatch h={horizon}: meta={len(meta)} vs X_test={n}. "
            "feature_engineered_v2.csv may be out of sync with sequence arrays."
        )
    dates = pd.to_datetime([m["target_date"] for m in meta])
    monsoon = is_monsoon(dates)
    non_monsoon = ~monsoon

    rows = [
        _stratum_row("all", horizon, seed, np.ones(n, dtype=bool), attn, ent),
        _stratum_row(f"wet_ge_{rain_threshold:g}mm", horizon, seed, wet, attn, ent),
        _stratum_row(f"dry_lt_{rain_threshold:g}mm", horizon, seed, dry, attn, ent),
        _stratum_row("monsoon_JJAS", horizon, seed, monsoon, attn, ent),
        _stratum_row("non_monsoon", horizon, seed, non_monsoon, attn, ent),
        _stratum_row(
            f"high_error_q{int(high_err_q * 100)}", horizon, seed, high_err, attn, ent
        ),
        _stratum_row(
            f"low_error_below_q{int(high_err_q * 100)}", horizon, seed, low_err, attn, ent
        ),
    ]

    # Contrast stats: recent-7 share and entropy
    contrasts = []
    for label, ma, mb in (
        ("wet_minus_dry_recent7", wet, dry),
        ("monsoon_minus_non_recent7", monsoon, non_monsoon),
        ("high_minus_low_err_recent7", high_err, low_err),
        ("wet_minus_dry_entropy", wet, dry),
        ("monsoon_minus_non_entropy", monsoon, non_monsoon),
        ("high_minus_low_err_entropy", high_err, low_err),
    ):
        if "entropy" in label:
            a_vals = ent[ma]
            b_vals = ent[mb]
        else:
            a_vals = attn[ma][:, -7:].sum(axis=1)
            b_vals = attn[mb][:, -7:].sum(axis=1)
        if len(a_vals) == 0 or len(b_vals) == 0:
            continue
        obs, lo, hi = bootstrap_mean_diff(a_vals, b_vals, n_boot=n_boot, seed=42)
        contrasts.append(
            {
                "horizon": horizon,
                "seed": seed,
                "contrast": label,
                "diff": obs,
                "ci_lo": lo,
                "ci_hi": hi,
                "significant_ci_excludes_0": bool(lo > 0 or hi < 0),
                "n_a": int(ma.sum()),
                "n_b": int(mb.sum()),
            }
        )

    # Plots
    FIGURES.mkdir(parents=True, exist_ok=True)
    _plot_profiles(
        {
            "all": mean_profile(attn),
            "wet": mean_profile(attn, wet),
            "dry": mean_profile(attn, dry),
        },
        f"Attention by wet/dry (h={horizon}, seed {seed}, τ={rain_threshold:g} mm)",
        FIGURES / f"attention_conditioned_h{horizon}_wet_dry.png",
    )
    _plot_profiles(
        {
            "monsoon": mean_profile(attn, monsoon),
            "non_monsoon": mean_profile(attn, non_monsoon),
        },
        f"Attention by season (h={horizon}, seed {seed}, JJAS vs rest)",
        FIGURES / f"attention_conditioned_h{horizon}_monsoon.png",
    )
    _plot_profiles(
        {
            "high_error": mean_profile(attn, high_err),
            "low_error": mean_profile(attn, low_err),
        },
        f"Attention by error (h={horizon}, seed {seed}, q{int(high_err_q*100)} abs-err)",
        FIGURES / f"attention_conditioned_h{horizon}_error.png",
    )

    # Entropy histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ent[dry], bins=40, alpha=0.55, label="dry", color="#6b8f71", density=True)
    ax.hist(ent[wet], bins=40, alpha=0.55, label="wet", color="#c45c26", density=True)
    ax.set_xlabel("Attention entropy (nats)")
    ax.set_ylabel("Density")
    ax.set_title(f"Attention entropy: wet vs dry (h={horizon}, seed {seed})")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / f"attention_conditioned_h{horizon}_entropy_wet_dry.png", dpi=150)
    plt.close()

    # Save tables
    TABLES.mkdir(parents=True, exist_ok=True)
    df_strata = pd.DataFrame(rows)
    df_contrasts = pd.DataFrame(contrasts)
    strata_path = TABLES / f"attention_conditioned_h{horizon}.csv"
    contrast_path = TABLES / f"attention_conditioned_contrasts_h{horizon}.csv"
    df_strata.to_csv(strata_path, index=False)
    df_contrasts.to_csv(contrast_path, index=False)

    print(df_strata.to_string(index=False), flush=True)
    print("\nContrasts (bootstrap 95% CI):", flush=True)
    print(df_contrasts.to_string(index=False), flush=True)
    print(f"wrote {strata_path}", flush=True)
    print(f"wrote {contrast_path}", flush=True)
    print(
        f"figures: attention_conditioned_h{horizon}_wet_dry/monsoon/error/entropy_wet_dry.png",
        flush=True,
    )
    return df_strata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Conditioned attention analysis")
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[4],
        help="Horizons to analyze (default: 4). Example: --horizons 1 4",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--rain-threshold",
        type=float,
        default=1.0,
        help="Wet-day threshold in mm (default: 1.0)",
    )
    p.add_argument(
        "--high-err-q",
        type=float,
        default=0.90,
        help="Abs-error quantile for high-error stratum (default: 0.90)",
    )
    p.add_argument("--n-boot", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda()
    print(f"Loading {FEAT_CSV} for monsoon date alignment...", flush=True)
    feat_df = pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])
    for h in args.horizons:
        analyze_horizon(
            horizon=h,
            seed=args.seed,
            device=device,
            rain_threshold=args.rain_threshold,
            high_err_q=args.high_err_q,
            n_boot=args.n_boot,
            feat_df=feat_df,
        )


if __name__ == "__main__":
    main()
