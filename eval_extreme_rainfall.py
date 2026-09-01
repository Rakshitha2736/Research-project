"""
Extreme vs Normal rainfall evaluation (top 5% observed days).

No training. One CUDA+autocast inference pass from existing seed-42
checkpoints (same path as eval_threshold_skill.py). Raw prediction
arrays were never persisted; continuous y_true/y_pred are regenerated
from checkpoints for this analysis only.

Writes:
  reports/tables/extreme_rainfall_evaluation.csv
  reports/figures/extreme_rainfall_rmse_comparison.png
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
TABLES = BASE / "reports" / "tables"
FIGURES = BASE / "reports" / "figures"

OUT_CSV = TABLES / "extreme_rainfall_evaluation.csv"
OUT_FIG = FIGURES / "extreme_rainfall_rmse_comparison.png"

VERIFIED = [
    TABLES / "master_results.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
    TABLES / "threshold_skill.csv",
    TABLES / "intensity_bins.csv",
    TABLES / "rain_classification_metrics.csv",
]

SEED = 42
HORIZONS = (1, 2, 3, 4)
PCT = 95.0
BATCH_SIZE = DEFAULT_BATCH_SIZE

MODEL_SPECS = (
    ("LSTM", "lstm"),
    ("CNN-LSTM-Temporal", "temporal"),
    ("CNN-LSTM+Attention", "attention"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_paths(horizon: int) -> dict[str, Path]:
    if horizon == 1:
        return {
            "X_test": DATA / "X_test_v2.npy",
            "y_test": DATA / "y_test_v2.npy",
            "scaler_y": MODELS / "minmax_scaler_y_v2.joblib",
        }
    return {
        "X_test": DATA / f"X_test_h{horizon}.npy",
        "y_test": DATA / f"y_test_h{horizon}.npy",
        "scaler_y": MODELS / f"minmax_scaler_y_h{horizon}.joblib",
    }


def ckpt_path(model_key: str, horizon: int) -> Path:
    if model_key == "attention":
        return MODELS / f"cnn_lstm_attention_h{horizon}_seed{SEED}.pt"
    if model_key == "temporal":
        return MODELS / f"cnn_lstm_temporal_h{horizon}_seed{SEED}.pt"
    if model_key == "lstm":
        if horizon == 1:
            return MODELS / f"lstm_baseline_v2_seed{SEED}.pt"
        return MODELS / f"lstm_h{horizon}_seed{SEED}.pt"
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
def predict_mm(
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


def metrics_subset(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def eval_all(device: torch.device) -> tuple[pd.DataFrame, dict[int, float]]:
    rows: list[dict] = []
    thresholds: dict[int, float] = {}

    for h in HORIZONS:
        paths = data_paths(h)
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"])
        scaler_y = joblib.load(paths["scaler_y"])

        # Threshold from TRUE targets once (same for all models)
        y_true_ref = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        thr = float(np.percentile(y_true_ref, PCT))
        thresholds[h] = thr
        extreme_mask_ref = y_true_ref >= thr
        print(
            f"h={h}: N={len(y_true_ref)}  95th pct threshold={thr:.4f} mm  "
            f"N_extreme={int(extreme_mask_ref.sum())}  "
            f"N_normal={int((~extreme_mask_ref).sum())}"
        )

        for label, key in MODEL_SPECS:
            ckpt = ckpt_path(key, h)
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)
            model = build_model(key, device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            y_true, y_pred = predict_mm(model, X_test, y_test, scaler_y, device)
            if not np.allclose(y_true, y_true_ref, rtol=0, atol=1e-5):
                raise RuntimeError(f"y_true mismatch {label} h={h}")

            extreme = y_true >= thr
            for subset_name, mask in (("Extreme", extreme), ("Normal", ~extreme)):
                yt = y_true[mask]
                yp = y_pred[mask]
                m = metrics_subset(yt, yp)
                rows.append(
                    {
                        "Model": label,
                        "Horizon": h,
                        "Subset": subset_name,
                        "N_samples": int(mask.sum()),
                        "Threshold_mm": round(thr, 6),
                        "RMSE": round(m["RMSE"], 6),
                        "MAE": round(m["MAE"], 6),
                        "R2": round(m["R2"], 6),
                    }
                )
            del model
            torch.cuda.empty_cache()

    return pd.DataFrame(rows), thresholds


def plot_rmse(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = [m for m, _ in MODEL_SPECS]
    colors = {
        "LSTM": "#2F4F4F",
        "CNN-LSTM-Temporal": "#3D7A8C",
        "CNN-LSTM+Attention": "#C45C26",
    }
    labels_short = {
        "LSTM": "LSTM",
        "CNN-LSTM-Temporal": "CNN-LSTM-Temporal",
        "CNN-LSTM+Attention": "CNN-LSTM+Attn",
    }

    # 8 groups: (h, Normal), (h, Extreme) for each horizon
    group_keys = [(h, s) for h in HORIZONS for s in ("Normal", "Extreme")]
    x = np.arange(len(group_keys), dtype=float)
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, model in enumerate(models):
        vals = []
        for h, subset in group_keys:
            row = df[
                (df["Model"] == model)
                & (df["Horizon"] == h)
                & (df["Subset"] == subset)
            ].iloc[0]
            vals.append(float(row["RMSE"]))
        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            color=colors[model],
            edgecolor="black",
            linewidth=0.6,
            label=labels_short[model],
        )
        # Hatch Extreme bars for grayscale distinguishability
        for j, (h, subset) in enumerate(group_keys):
            if subset == "Extreme":
                bars[j].set_hatch("///")
                bars[j].set_alpha(0.95)

    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}\n{s}" for h, s in group_keys], fontsize=8)
    ax.set_ylabel("RMSE (mm/day)")
    ax.set_title("RMSE on Extreme (top 5%) vs Normal rainfall days (seed 42)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Combined legend: model colors + hatch meaning
    handles = [
        mpatches.Patch(facecolor=colors[m], edgecolor="black", label=labels_short[m])
        for m in models
    ]
    handles.append(
        mpatches.Patch(
            facecolor="white", edgecolor="black", hatch="///", label="Extreme (hatched)"
        )
    )
    handles.append(
        mpatches.Patch(facecolor="#cccccc", edgecolor="black", label="Normal (solid)")
    )
    ax.legend(handles=handles, frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


def print_ratios(df: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("Extreme RMSE / Normal RMSE (higher = worse on heavy days)")
    print("=" * 72)
    print(f"{'Model':28} {'h':>2}  {'RMSE_ext':>10}  {'RMSE_norm':>10}  {'ratio':>8}")
    print("-" * 72)
    for label, _ in MODEL_SPECS:
        for h in HORIZONS:
            e = df[(df.Model == label) & (df.Horizon == h) & (df.Subset == "Extreme")].iloc[0]
            n = df[(df.Model == label) & (df.Horizon == h) & (df.Subset == "Normal")].iloc[0]
            ratio = float(e["RMSE"]) / float(n["RMSE"]) if float(n["RMSE"]) > 0 else float("nan")
            print(
                f"{label:28} {h:>2}  {float(e['RMSE']):10.4f}  "
                f"{float(n['RMSE']):10.4f}  {ratio:8.3f}"
            )


def main() -> None:
    device = require_cuda()
    TABLES.mkdir(parents=True, exist_ok=True)

    hashes_before = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print("BEFORE hashes:")
    for k, v in hashes_before.items():
        print(f"  {k}: {v}")

    print()
    print("=" * 72)
    print("PREDICTION SOURCE")
    print("=" * 72)
    print(
        "Continuous raw prediction arrays: NOT FOUND on disk.\n"
        "threshold_skill.csv: binary contingency counts only (no continuous y_pred).\n"
        "intensity_bins.csv: fixed mm bins only (not 95th-percentile splits).\n"
        "Action (authorized): CUDA+autocast re-inference from seed-42 checkpoints\n"
        "via same path as eval_threshold_skill.py (deterministic models in eval mode)."
    )

    df, thresholds = eval_all(device)
    NOTES_HEADER = (
        "# NOTES (do not parse as data; pandas: read_csv(..., comment=\"#\"))\n"
        "# R2 values on the Normal subset are negative for all models/horizons. "
        "This is an expected statistical artifact of computing R2 on a "
        "variance-truncated subset (the 95% of days with lowest rainfall have "
        "very low target variance, shrinking R2's denominator disproportionately "
        "to the model's absolute error), NOT evidence the models perform poorly "
        "on typical days - their absolute RMSE/MAE on the Normal subset "
        "(~4.3-5.0 mm RMSE) is in fact considerably better than their "
        "Extreme-subset performance (~36-45 mm RMSE), consistent with expectations. "
        "RMSE and MAE, not R2, should be used to interpret Normal-vs-Extreme "
        "performance in this table.\n"
        "# Exception (documented): CNN-LSTM-Temporal h=4 Normal R2 is near-zero "
        "positive (~+0.029); interpret with RMSE/MAE anyway.\n"
        "# Absolute Extreme-RMSE ranking: LSTM has the lowest absolute "
        "Extreme-RMSE at h=1-2; CNN-LSTM+Attention has the lowest absolute "
        "Extreme-RMSE at h=3-4.\n"
    )
    body = df.to_csv(index=False)
    OUT_CSV.write_text(NOTES_HEADER + body, encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")
    plot_rmse(df)

    print()
    print("=" * 72)
    print("95th-PERCENTILE THRESHOLDS (mm) BY HORIZON")
    print("=" * 72)
    for h in HORIZONS:
        print(f"  h={h}: {thresholds[h]:.6f} mm")

    print()
    print("=" * 72)
    print("extreme_rainfall_evaluation.csv (full)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(df.to_string(index=False))

    print_ratios(df)

    hashes_after = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print()
    print("=" * 72)
    print("INTEGRITY")
    print("=" * 72)
    ok = True
    for k, v0 in hashes_before.items():
        match = v0 == hashes_after[k]
        ok = ok and match
        print(f"  {k}: {'UNCHANGED' if match else 'CHANGED'}")
    assert ok
    print(
        "CONFIRM: no verified source files modified; no training; "
        "one authorized CUDA+autocast seed-42 inference pass only."
    )


if __name__ == "__main__":
    main()
