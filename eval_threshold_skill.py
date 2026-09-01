"""
Threshold / categorical skill + intensity-bin RMSE for flat models.

Architecture frozen — evaluation only on existing checkpoints.
Models: LSTM, CNN-LSTM-Temporal, CNN-LSTM+Attention (+ Persistence).

Usage (CUDA venv, from RainfallPrediction/):
  python eval_threshold_skill.py
  python eval_threshold_skill.py --horizons 1 4 --seeds 42
  python eval_threshold_skill.py --horizons 1 2 3 4 --seeds 13 42 123

Writes:
  reports/tables/threshold_skill.csv
  reports/tables/threshold_skill_summary.csv   (mean±std over seeds)
  reports/tables/intensity_bins.csv
  reports/tables/intensity_bins_summary.csv

Thresholds (default): 0.1/1/5/10 mm (operational) + 35.6/64.4/124.4 mm (IMD event cutoffs).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.persistence_baseline import data_paths, persistence_mm
from src.metrics_rainfall import (
    DEFAULT_INTENSITY_EDGES_MM,
    FULL_THRESHOLDS_MM,
    intensity_bin_metrics,
    threshold_skill_table,
    tolerance_accuracy,
)
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
TABLES = BASE / "reports" / "tables"

SEEDS_DEFAULT = (13, 42, 123)
HORIZONS_DEFAULT = (1, 2, 3, 4)
MODELS_DEFAULT = ("lstm", "temporal", "attention", "persistence")


def ckpt_path(model_name: str, horizon: int, seed: int) -> Path:
    if model_name == "attention":
        return MODELS / f"cnn_lstm_attention_h{horizon}_seed{seed}.pt"
    if model_name == "temporal":
        return MODELS / f"cnn_lstm_temporal_h{horizon}_seed{seed}.pt"
    if model_name == "lstm":
        if horizon == 1:
            return MODELS / f"lstm_baseline_v2_seed{seed}.pt"
        return MODELS / f"lstm_h{horizon}_seed{seed}.pt"
    raise ValueError(model_name)


def build_model(model_name: str, device: torch.device) -> torch.nn.Module:
    if model_name == "attention":
        return CNNLSTMAttention(n_features=8).to(device)
    if model_name == "temporal":
        return CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
    if model_name == "lstm":
        return LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    raise ValueError(model_name)


@torch.no_grad()
def predict_mm(
    model: torch.nn.Module,
    X: np.ndarray,
    y_scaled: np.ndarray,
    scaler_y,
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(X, y_scaled, batch_size=batch_size, shuffle=False)
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


def _agg_mean_std(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(g["seed"].nunique()) if "seed" in g.columns else len(g)
        for c in value_cols:
            vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                row[f"{c}_mean"] = float("nan")
                row[f"{c}_std"] = float("nan")
            elif len(vals) == 1:
                row[f"{c}_mean"] = float(vals[0])
                row[f"{c}_std"] = 0.0
            else:
                row[f"{c}_mean"] = float(vals.mean())
                row[f"{c}_std"] = float(vals.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold + intensity-bin skill eval")
    p.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS_DEFAULT))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS_DEFAULT))
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS_DEFAULT),
        choices=["lstm", "temporal", "attention", "persistence"],
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(FULL_THRESHOLDS_MM),
        help="Operational + IMD event thresholds (mm); default includes 0.1/1/5/10 and 35.6/64.4/124.5",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda()
    TABLES.mkdir(parents=True, exist_ok=True)

    thr_rows: list[dict] = []
    bin_rows: list[dict] = []
    tol_rows: list[dict] = []

    for h in args.horizons:
        paths = data_paths(h)
        print(f"\n========== horizon h={h} ==========", flush=True)
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"])
        scaler_y = joblib.load(paths["scaler_y"])
        scaler_x = joblib.load(paths["scaler_x"])
        assert X_test.shape[-1] == 8

        # Cache y_true once via persistence path (same for all models)
        y_true_ref, _ = persistence_mm(X_test, y_test, scaler_x, scaler_y)

        for model_name in args.models:
            if model_name == "persistence":
                seeds_iter = [None]
            else:
                seeds_iter = args.seeds

            for seed in seeds_iter:
                if model_name == "persistence":
                    y_true, y_pred = persistence_mm(X_test, y_test, scaler_x, scaler_y)
                    seed_label = "n/a"
                    print(f"  model=persistence", flush=True)
                else:
                    ckpt = ckpt_path(model_name, h, int(seed))
                    if not ckpt.exists():
                        print(f"  SKIP missing {ckpt.name}", flush=True)
                        continue
                    print(f"  model={model_name} seed={seed} ({ckpt.name})", flush=True)
                    model = build_model(model_name, device)
                    state = torch.load(ckpt, map_location=device, weights_only=False)
                    model.load_state_dict(state["model_state_dict"])
                    y_true, y_pred = predict_mm(model, X_test, y_test, scaler_y, device)
                    seed_label = str(seed)
                    del model
                    torch.cuda.empty_cache()

                # Sanity: y_true matches reference
                if not np.allclose(y_true, y_true_ref, rtol=0, atol=1e-5):
                    raise RuntimeError(f"y_true mismatch for {model_name} h={h}")

                for skill in threshold_skill_table(y_true, y_pred, args.thresholds):
                    thr_rows.append(
                        {
                            "model": model_name,
                            "horizon": h,
                            "seed": seed_label,
                            **skill,
                        }
                    )

                for b in intensity_bin_metrics(y_true, y_pred, DEFAULT_INTENSITY_EDGES_MM):
                    bin_rows.append(
                        {
                            "model": model_name,
                            "horizon": h,
                            "seed": seed_label,
                            **b,
                        }
                    )

                for tol in (1.0, 2.0, 5.0):
                    tol_rows.append(
                        {
                            "model": model_name,
                            "horizon": h,
                            "seed": seed_label,
                            "tol_mm": tol,
                            "tolerance_accuracy": tolerance_accuracy(y_true, y_pred, tol),
                        }
                    )

    df_thr = pd.DataFrame(thr_rows)
    df_bin = pd.DataFrame(bin_rows)
    df_tol = pd.DataFrame(tol_rows)

    thr_path = TABLES / "threshold_skill.csv"
    bin_path = TABLES / "intensity_bins.csv"
    tol_path = TABLES / "tolerance_accuracy.csv"
    df_thr.to_csv(thr_path, index=False)
    df_bin.to_csv(bin_path, index=False)
    df_tol.to_csv(tol_path, index=False)

    # Summaries over numeric seeds only (exclude persistence from mean±std of DL models)
    dl_thr = df_thr[df_thr["seed"] != "n/a"].copy()
    dl_thr["seed"] = dl_thr["seed"].astype(int)
    summary_thr = _agg_mean_std(
        dl_thr,
        ["model", "horizon", "threshold_mm"],
        ["POD", "FAR", "CSI", "Bias", "HSS", "Accuracy"],
    )
    # Attach persistence (single row) as mean with std=0 for convenience
    pers_thr = df_thr[df_thr["model"] == "persistence"].copy()
    if len(pers_thr):
        pers_sum = pers_thr.rename(
            columns={
                c: f"{c}_mean"
                for c in ["POD", "FAR", "CSI", "Bias", "HSS", "Accuracy"]
            }
        )
        for c in ["POD", "FAR", "CSI", "Bias", "HSS", "Accuracy"]:
            pers_sum[f"{c}_std"] = 0.0
        pers_sum["n_seeds"] = 1
        keep = [
            "model",
            "horizon",
            "threshold_mm",
            "n_seeds",
            "POD_mean",
            "POD_std",
            "FAR_mean",
            "FAR_std",
            "CSI_mean",
            "CSI_std",
            "Bias_mean",
            "Bias_std",
            "HSS_mean",
            "HSS_std",
            "Accuracy_mean",
            "Accuracy_std",
        ]
        summary_thr = pd.concat([summary_thr, pers_sum[keep]], ignore_index=True)

    dl_bin = df_bin[df_bin["seed"] != "n/a"].copy()
    dl_bin["seed"] = dl_bin["seed"].astype(int)
    summary_bin = _agg_mean_std(
        dl_bin,
        ["model", "horizon", "bin"],
        ["RMSE", "MAE", "n"],
    )

    dl_tol = df_tol[df_tol["seed"] != "n/a"].copy()
    dl_tol["seed"] = dl_tol["seed"].astype(int)
    summary_tol = _agg_mean_std(
        dl_tol,
        ["model", "horizon", "tol_mm"],
        ["tolerance_accuracy"],
    )

    sum_thr_path = TABLES / "threshold_skill_summary.csv"
    sum_bin_path = TABLES / "intensity_bins_summary.csv"
    sum_tol_path = TABLES / "tolerance_accuracy_summary.csv"
    summary_thr.to_csv(sum_thr_path, index=False)
    summary_bin.to_csv(sum_bin_path, index=False)
    summary_tol.to_csv(sum_tol_path, index=False)

    print("\n=== Wrote ===", flush=True)
    for p in (thr_path, sum_thr_path, bin_path, sum_bin_path, tol_path, sum_tol_path):
        print(f"  {p}", flush=True)

    # Quick headline: CSI @ 1mm, mean over seeds
    print("\n=== Headline CSI @ 1.0 mm (mean over seeds) ===", flush=True)
    sub = summary_thr[summary_thr["threshold_mm"] == 1.0][
        ["model", "horizon", "CSI_mean", "CSI_std", "POD_mean", "FAR_mean", "HSS_mean"]
    ].sort_values(["horizon", "model"])
    print(sub.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
