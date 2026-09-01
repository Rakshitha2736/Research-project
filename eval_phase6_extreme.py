"""
Phase 6 — consolidated IMD-tier extreme rainfall evaluation (multi-seed).

Inference only. Reuses seed 13/42/123 checkpoints for LSTM / Temporal / Attention,
persistence via src.persistence_baseline, climatology via eval_climatology lookup.

GNN-LSTM excluded: requires graph adjacency + GNNLSTM loader (not in the flat
eval_extreme_rainfall / eval_threshold_skill path); no shared multi-horizon eval
module exists beyond training scripts.

Writes:
  reports/tables/phase6_extreme_evaluation.csv
  (additive rows appended to master_results.csv for headline IMD-tier RMSE)

Usage (CUDA venv, from RainfallPrediction/):
  python eval_phase6_extreme.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

from eval_climatology import build_climatology_lookup
from eval_seasonal_performance import month_to_season
from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.eval_attention import TRAIN_END, rebuild_test_meta
from src.imd_tiers import (
    MIN_TIER_SAMPLES,
    merge_tiers,
    raw_tier_counts,
    tier_continuous_metrics,
)
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline
from src.persistence_baseline import data_paths, persistence_mm

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
TABLES = BASE / "reports" / "tables"
FEAT_CSV = DATA / "feature_engineered_v2.csv"
OUT_CSV = TABLES / "phase6_extreme_evaluation.csv"
MASTER_CSV = TABLES / "master_results.csv"

SEEDS = (13, 42, 123)
HORIZONS = (1, 2, 3, 4)
# Labels from Phase 6 first report (eval_phase6_extreme.py run, 2026-09-01).
# Regression reference — validation must reproduce these exactly.
PREVIOUSLY_REPORTED_STABILITY: dict[tuple[str, int], str] = {
    ("LSTM", 1): "SEED-SENSITIVE",
    ("CNN-LSTM-Temporal", 1): "SEED-SENSITIVE",
    ("CNN-LSTM+Attention", 1): "SEED-SENSITIVE",
    ("LSTM", 2): "stable",
    ("CNN-LSTM-Temporal", 2): "stable",
    ("CNN-LSTM+Attention", 2): "stable",
    ("LSTM", 3): "stable",
    ("CNN-LSTM-Temporal", 3): "SEED-SENSITIVE",
    ("CNN-LSTM+Attention", 3): "SEED-SENSITIVE",
    ("LSTM", 4): "stable",
    ("CNN-LSTM-Temporal", 4): "stable",
    ("CNN-LSTM+Attention", 4): "SEED-SENSITIVE",
}
SEED_STABILITY_CV_THRESHOLD_PCT = 1.0
BATCH_SIZE = DEFAULT_BATCH_SIZE

DL_MODEL_SPECS = (
    ("LSTM", "lstm"),
    ("CNN-LSTM-Temporal", "temporal"),
    ("CNN-LSTM+Attention", "attention"),
)

# Hash-monitored Phase 1-5 artifacts (must not be modified)
VERIFIED = [
    TABLES / "master_results.csv",
    TABLES / "climatology_baseline.csv",
    TABLES / "extreme_rainfall_evaluation.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ckpt_path(model_key: str, horizon: int, seed: int) -> Path:
    if model_key == "attention":
        return MODELS / f"cnn_lstm_attention_h{horizon}_seed{seed}.pt"
    if model_key == "temporal":
        return MODELS / f"cnn_lstm_temporal_h{horizon}_seed{seed}.pt"
    if model_key == "lstm":
        if horizon == 1:
            return MODELS / f"lstm_baseline_v2_seed{seed}.pt"
        return MODELS / f"lstm_h{horizon}_seed{seed}.pt"
    raise ValueError(model_key)


def build_model(model_key: str, device: torch.device) -> torch.nn.Module:
    if model_key == "attention":
        return CNNLSTMAttention(n_features=8).to(device)
    if model_key == "temporal":
        return CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
    if model_key == "lstm":
        return LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    raise ValueError(model_key)


@torch.no_grad()
def predict_mm_dl(
    model: torch.nn.Module,
    X: np.ndarray,
    y_scaled: np.ndarray,
    scaler_y,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(X, y_scaled, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    chunks: list[torch.Tensor] = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(model(xb).float())
    pred_s = torch.cat(chunks, dim=0).cpu().numpy()
    y_pred = scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    return y_true, y_pred


def load_y_true(horizon: int) -> np.ndarray:
    paths = data_paths(horizon, BASE)
    y_test = np.load(paths["y_test"])
    scaler_y = joblib.load(paths["scaler_y"])
    return scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()


def climatology_predictions(
    horizon: int,
    feat: pd.DataFrame,
    lookup,
) -> tuple[np.ndarray, np.ndarray]:
    paths = data_paths(horizon, BASE)
    y_test = np.load(paths["y_test"])
    scaler_y = joblib.load(paths["scaler_y"])
    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    meta = rebuild_test_meta(feat, horizon)
    if len(meta) != len(y_true):
        raise RuntimeError(f"meta/y length mismatch h={horizon}")

    from eval_climatology import predict_value

    y_pred = np.empty(len(meta), dtype=np.float64)
    for i, row in enumerate(meta):
        target_date = pd.Timestamp(row["target_date"])
        season = month_to_season(int(target_date.month))
        val, _ = predict_value(lookup, str(row["station_id"]), season)
        y_pred[i] = val
    return y_true, y_pred


def print_tier_count_table(horizon_tiers: dict[int, dict]) -> None:
    from src.imd_tiers import IMD_TIER_SPECS

    print("\n" + "=" * 88)
    print("STEP 1 — IMD tier sample counts (observed test rainfall, pre-merge)")
    print("=" * 88)
    tier_names = [n for n, _ in IMD_TIER_SPECS]
    header = f"{'Tier':22}" + "".join(f"h={h:>6}" for h in HORIZONS)
    print(header)
    print("-" * len(header))
    for name in tier_names:
        row = f"{name:22}"
        for h in HORIZONS:
            counts = raw_tier_counts(horizon_tiers[h]["y_true"])
            row += f"{counts[name]:7d}"
        print(row)

    print("\nMerge decisions (>=100 samples rule):")
    for h in HORIZONS:
        decisions = horizon_tiers[h]["merge_decisions"]
        if decisions:
            print(f"  h={h}:")
            for d in decisions:
                print(f"    - {d}")
        else:
            print(f"  h={h}: no merges required (all raw tiers n >= {MIN_TIER_SAMPLES})")


def eval_model_tiers(
    model_label: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    merged_tiers: list,
    seed: int | str,
) -> list[dict]:
    rows = []
    for tier in merged_tiers:
        m = tier_continuous_metrics(y_true, y_pred, tier.mask)
        rows.append(
            {
                "Model": model_label,
                "Horizon": horizon,
                "Tier": tier.name,
                "Seed": seed,
                "N_samples": m["N_samples"],
                "RMSE": round(m["RMSE"], 6),
                "MAE": round(m["MAE"], 6),
                "Mean_Error_Bias": round(m["Mean_Error_Bias"], 6),
                "Source_Tiers": "|".join(tier.source_tiers),
            }
        )
    return rows


def add_summary_rows(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    """Append 3-seed mean±std rows for DL models."""
    agg_rows = []
    dl = per_seed_df[per_seed_df["Seed"].isin(SEEDS)]
    for (model, horizon, tier), g in dl.groupby(["Model", "Horizon", "Tier"], dropna=False):
        if len(g) != len(SEEDS):
            continue
        rmse_vals = g.sort_values("Seed")["RMSE"].astype(float).to_numpy()
        mae_vals = g.sort_values("Seed")["MAE"].astype(float).to_numpy()
        bias_vals = g.sort_values("Seed")["Mean_Error_Bias"].astype(float).to_numpy()
        agg_rows.append(
            {
                "Model": model,
                "Horizon": horizon,
                "Tier": tier,
                "Seed": "mean ± std",
                "N_samples": int(g["N_samples"].iloc[0]),
                "RMSE": f"{rmse_vals.mean():.4f} ± {rmse_vals.std(ddof=1):.4f}",
                "MAE": f"{mae_vals.mean():.4f} ± {mae_vals.std(ddof=1):.4f}",
                "Mean_Error_Bias": f"{bias_vals.mean():.4f} ± {bias_vals.std(ddof=1):.4f}",
                "Source_Tiers": g["Source_Tiers"].iloc[0],
            }
        )
    return pd.concat([per_seed_df, pd.DataFrame(agg_rows)], ignore_index=True)


def seed_stability_verdict(cv_pct: float, threshold_pct: float = SEED_STABILITY_CV_THRESHOLD_PCT) -> str:
    """Classify 3-seed RMSE dispersion as stable or seed-sensitive.

    Returns ``SEED-SENSITIVE`` when ``cv_pct >= threshold_pct``, else ``stable``.

    Justification (threshold_pct=1.0):
      Phase 4 full-test RMSE across seeds showed coefficient of variation ~0.02-0.08%
      on a ~9-10 mm scale (std ~0.02-0.08 mm). IMD Extremely Heavy tier RMSE is
      ~140-167 mm -- an order of magnitude larger in absolute terms but evaluated with
      the same 3 checkpoints. A 1% CV threshold flags dispersion materially above
      the full-test baseline (~10-50x higher relative spread) without treating the
      tiny Phase-4 CV as a hard floor (which would mark every extreme-tier result
      sensitive). Boundary is inclusive at 1.0%: CV == 1.0% -> SEED-SENSITIVE.
    """
    if cv_pct >= threshold_pct:
        return "SEED-SENSITIVE"
    return "stable"


def _dl_per_seed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to DL per-seed rows (tolerates int or str Seed from CSV reload)."""
    dl = df[df["Seed"].astype(str).isin([str(s) for s in SEEDS])].copy()
    dl["Seed"] = dl["Seed"].astype(int)
    return dl


def _severe_tier_name(tiers: list[str] | np.ndarray) -> str:
    severe = [t for t in tiers if "Extremely Heavy" in t]
    return severe[0] if severe else sorted(tiers)[-1]


def compute_seed_stability_verdicts(df: pd.DataFrame) -> list[dict]:
    """Recompute Extremely Heavy tier CV% and verdict for each DL model × horizon."""
    dl = _dl_per_seed_frame(df)
    rows: list[dict] = []
    for h in HORIZONS:
        tier = _severe_tier_name(dl[dl["Horizon"] == h]["Tier"].unique())
        for model, _ in DL_MODEL_SPECS:
            sub = dl[(dl["Model"] == model) & (dl["Horizon"] == h) & (dl["Tier"] == tier)]
            if len(sub) != len(SEEDS):
                rows.append(
                    {
                        "Model": model,
                        "Horizon": h,
                        "CV_pct": float("nan"),
                        "Verdict": "INCOMPLETE",
                    }
                )
                continue
            vals = sub.sort_values("Seed")["RMSE"].astype(float).to_numpy()
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1))
            cv_pct = 100.0 * std / mean if mean > 0 else float("nan")
            rows.append(
                {
                    "Model": model,
                    "Horizon": h,
                    "CV_pct": round(cv_pct, 2),
                    "Verdict": seed_stability_verdict(cv_pct),
                }
            )
    return rows


def validate_seed_stability_verdicts(df: pd.DataFrame) -> bool:
    """Print rule source + 12-row table; confirm MATCH against prior Phase 6 report."""
    import inspect

    print("\n" + "=" * 88)
    print("SEED STABILITY VERDICT RULE (seed_stability_verdict)")
    print("=" * 88)
    print(inspect.getsource(seed_stability_verdict))

    recomputed = compute_seed_stability_verdicts(df)
    print("\n" + "=" * 88)
    print("RECOMPUTED 12-ROW VERDICT TABLE (Extremely Heavy tier)")
    print("=" * 88)
    print(f"{'Model':28} {'h':>2}  {'CV%':>7}  {'Verdict':>16}  {'Prior':>16}  Check")
    print("-" * 88)
    all_match = True
    for row in recomputed:
        key = (row["Model"], int(row["Horizon"]))
        prior = PREVIOUSLY_REPORTED_STABILITY.get(key, "MISSING")
        verdict = row["Verdict"]
        match = verdict == prior
        all_match = all_match and match
        check = "MATCH" if match else "MISMATCH"
        print(
            f"{row['Model']:28} {row['Horizon']:2d}  {row['CV_pct']:7.2f}  "
            f"{verdict:>16}  {prior:>16}  {check}"
        )
    print("-" * 88)
    print(f"Overall: {'ALL MATCH' if all_match else 'MISMATCH DETECTED — STOP'}")
    if not all_match:
        raise SystemExit(
            "STOP: seed_stability_verdict produced at least one label different "
            "from the Phase 6 report. Do not reconcile silently."
        )
    return all_match


def seed_stability_report(df: pd.DataFrame) -> None:
    """Report whether severe-tier RMSE is seed-stable across DL models."""
    print("\n" + "=" * 88)
    print("SEED STABILITY — most severe IMD tier RMSE (DL models, 3 seeds)")
    print("=" * 88)
    dl = _dl_per_seed_frame(df)
    for h in HORIZONS:
        tier = _severe_tier_name(dl[dl["Horizon"] == h]["Tier"].unique())
        print(f"\nh={h} tier='{tier}':")
        print(f"  {'Model':28} {'seed13':>10} {'seed42':>10} {'seed123':>10} {'std':>8} {'cv%':>8}  verdict")
        for model, _ in DL_MODEL_SPECS:
            sub = dl[(dl["Model"] == model) & (dl["Horizon"] == h) & (dl["Tier"] == tier)]
            if len(sub) != 3:
                print(f"  {model:28} incomplete seeds ({len(sub)}/3)")
                continue
            vals = sub.sort_values("Seed")["RMSE"].astype(float).to_numpy()
            std = float(np.std(vals, ddof=1))
            mean = float(np.mean(vals))
            cv = 100.0 * std / mean if mean > 0 else float("nan")
            flag = seed_stability_verdict(cv)
            print(
                f"  {model:28} {vals[0]:10.4f} {vals[1]:10.4f} {vals[2]:10.4f} "
                f"{std:8.4f} {cv:7.2f}%  {flag}"
            )
    validate_seed_stability_verdicts(df)


def append_master_headlines(df: pd.DataFrame, merge_notes: dict[int, list[str]]) -> None:
    """Additive rows: most severe tier RMSE mean±std per DL model + persistence/climatology."""
    lines = [
        ln
        for ln in MASTER_CSV.read_text(encoding="utf-8").splitlines()
        if ln and not ln.startswith("Phase 6 IMD Tier,")
    ]
    new_rows = []
    dl_models = [m for m, _ in DL_MODEL_SPECS]
    for h in HORIZONS:
        sub_h = df[df["Horizon"] == h]
        tiers = sub_h["Tier"].unique()
        severe = [t for t in tiers if "Extremely Heavy" in t]
        tier = severe[0] if severe else sorted(tiers)[-1]
        merge_note = "; ".join(merge_notes[h]) if merge_notes[h] else "no tier merges"

        for model in dl_models + ["Persistence Baseline", "Climatology Baseline"]:
            summary = sub_h[
                (sub_h["Model"] == model)
                & (sub_h["Tier"] == tier)
                & (sub_h["Seed"] == "mean ± std")
            ]
            if not summary.empty:
                rmse_s = str(summary["RMSE"].iloc[0])
                mae_s = str(summary["MAE"].iloc[0])
                seeds = "13,42,123"
            else:
                single = sub_h[
                    (sub_h["Model"] == model)
                    & (sub_h["Tier"] == tier)
                    & (sub_h["Seed"] == "n/a")
                ]
                if single.empty:
                    continue
                rmse_s = f"{float(single['RMSE'].iloc[0]):.4f}"
                mae_s = f"{float(single['MAE'].iloc[0]):.4f}"
                seeds = "n/a"
            new_rows.append(
                ",".join(
                    [
                        "Phase 6 IMD Tier",
                        str(h),
                        f"IMD tier: {tier}",
                        seeds,
                        rmse_s,
                        mae_s,
                        "Not Available",
                        "Not Available",
                        (
                            f"eval_phase6_extreme.py; IMD primary tier scheme; "
                            f"merge: {merge_note}; continuous mean-error bias in phase6_extreme_evaluation.csv"
                        ),
                    ]
                )
            )
    MASTER_CSV.write_text("\n".join(lines + new_rows) + "\n", encoding="utf-8")
    print(f"\nAppended {len(new_rows)} Phase 6 headline rows to {MASTER_CSV}")


def main() -> None:
    hashes_before = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    device = require_cuda()
    TABLES.mkdir(parents=True, exist_ok=True)

    # Step 1 — tier counts and merge plan
    horizon_tiers: dict[int, dict] = {}
    merge_notes: dict[int, list[str]] = {}
    for h in HORIZONS:
        y_true = load_y_true(h)
        merged, decisions = merge_tiers(y_true)
        horizon_tiers[h] = {"y_true": y_true, "merged": merged, "merge_decisions": decisions}
        merge_notes[h] = decisions

    print_tier_count_table(horizon_tiers)

    # Step 2 — multi-seed tier evaluation
    feat = pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])
    train_df = feat[feat["date_of_record"] <= TRAIN_END]
    clim_lookup = build_climatology_lookup(train_df)

    all_rows: list[dict] = []

    for h in HORIZONS:
        paths = data_paths(h, BASE)
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"])
        scaler_y = joblib.load(paths["scaler_y"])
        scaler_x = joblib.load(paths["scaler_x"])
        merged = horizon_tiers[h]["merged"]
        y_true_ref = horizon_tiers[h]["y_true"]

        print(f"\n========== horizon h={h} ==========", flush=True)

        # Persistence (deterministic)
        y_true_p, y_pred_p = persistence_mm(X_test, y_test, scaler_x, scaler_y)
        assert np.allclose(y_true_p, y_true_ref, rtol=0, atol=1e-5)
        all_rows.extend(
            eval_model_tiers("Persistence Baseline", y_true_p, y_pred_p, h, merged, "n/a")
        )

        # Climatology (deterministic)
        y_true_c, y_pred_c = climatology_predictions(h, feat, clim_lookup)
        assert np.allclose(y_true_c, y_true_ref, rtol=0, atol=1e-5)
        all_rows.extend(
            eval_model_tiers("Climatology Baseline", y_true_c, y_pred_c, h, merged, "n/a")
        )

        # DL models × seeds
        for label, key in DL_MODEL_SPECS:
            for seed in SEEDS:
                ckpt = ckpt_path(key, h, seed)
                if not ckpt.exists():
                    raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
                model = build_model(key, device)
                state = torch.load(ckpt, map_location=device, weights_only=False)
                model.load_state_dict(state["model_state_dict"])
                y_true, y_pred = predict_mm_dl(model, X_test, y_test, scaler_y, device)
                if not np.allclose(y_true, y_true_ref, rtol=0, atol=1e-5):
                    raise RuntimeError(f"y_true mismatch {label} h={h} seed={seed}")
                all_rows.extend(eval_model_tiers(label, y_true, y_pred, h, merged, seed))
                del model
                torch.cuda.empty_cache()
                print(f"  {label} seed={seed} done", flush=True)

    per_seed_df = pd.DataFrame(all_rows)
    full_df = add_summary_rows(per_seed_df)

    header = (
        "# Phase 6 — IMD-tier extreme rainfall evaluation (primary tier scheme).\n"
        "# Boundaries: IMD daily scale (see src/imd_tiers.py / dashboard/lib/imd_rainfall.py).\n"
        f"# Merge rule: tiers with < {MIN_TIER_SAMPLES} test samples merged per horizon (see merge log below).\n"
        "# Mean_Error_Bias = mean(predicted - actual) mm/day (continuous; NOT contingency frequency bias).\n"
        "# GNN-LSTM excluded (graph inference path not shared with flat eval scripts).\n"
        "# Secondary cross-check: extreme_rainfall_evaluation.csv (95th-percentile split; different framing).\n"
    )
    for h in HORIZONS:
        if merge_notes[h]:
            header += f"# h={h} merges: " + " | ".join(merge_notes[h]) + "\n"
        else:
            header += f"# h={h} merges: none (all raw IMD tiers n >= {MIN_TIER_SAMPLES})\n"

    OUT_CSV.write_text(header + full_df.to_csv(index=False), encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")

    print("\n" + "=" * 88)
    print("STEP 2 — Multi-seed tier results (per-seed + mean±std)")
    print("=" * 88)
    with pd.option_context("display.max_rows", 200, "display.width", 200):
        print(full_df.to_string(index=False))

    seed_stability_report(per_seed_df)
    append_master_headlines(full_df, merge_notes)

    hashes_after = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print("\n=== INTEGRITY (Phase 1-5 artifacts, except master_results additive rows) ===")
    for k, v0 in hashes_before.items():
        if k == "master_results.csv":
            print(f"  {k}: CHANGED (expected — Phase 6 additive rows)")
            continue
        ok = v0 == hashes_after.get(k)
        print(f"  {k}: {'UNCHANGED' if ok else 'CHANGED'}")
        if not ok:
            raise RuntimeError(f"Unexpected modification of {k}")


if __name__ == "__main__":
    main()
