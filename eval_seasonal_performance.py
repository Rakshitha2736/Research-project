"""
Seasonal (Winter/Summer/Monsoon/Post-monsoon) performance breakdown
for LSTM, CNN-LSTM-Temporal, and CNN-LSTM+Attention at h=1 and h=4.

No training. Does not modify verified result tables.
CUDA+autocast seed-42 inference; rebuild_test_meta() for target_date.

Season definition matches clean_dataset.csv's original `season` column
(used by EDA before month/season were dropped for doy_sin/cos):
  Winter:        Dec, Jan, Feb, Mar
  Summer:        Apr, May, Jun
  Monsoon:       Jul, Aug, Sep
  Post-monsoon:  Oct, Nov
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.amp import autocast

from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.eval_attention import rebuild_test_meta
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
TABLES = BASE / "reports" / "tables"
FIGURES = BASE / "reports" / "figures"

OUT_CSV = TABLES / "seasonal_performance.csv"
OUT_FIG = FIGURES / "seasonal_rmse_comparison.png"
FEAT_CSV = DATA / "feature_engineered_v2.csv"
CLEAN_CSV = DATA / "clean_dataset.csv"

SEED = 42
HORIZONS = (1, 4)
BATCH_SIZE = DEFAULT_BATCH_SIZE

# Exact original EDA season column → month numbers (1–12)
SEASON_MONTHS: dict[str, frozenset[int]] = {
    "Winter": frozenset({12, 1, 2, 3}),
    "Summer": frozenset({4, 5, 6}),
    "Monsoon": frozenset({7, 8, 9}),
    "Post-monsoon": frozenset({10, 11}),
}
SEASON_ORDER = ("Winter", "Summer", "Monsoon", "Post-monsoon")

MODEL_SPECS = (
    ("LSTM", "lstm"),
    ("CNN-LSTM-Temporal", "temporal"),
    ("CNN-LSTM+Attention", "attention"),
)

VERIFIED = [
    TABLES / "master_results.csv",
    TABLES / "significance_results.csv",
    TABLES / "ablation_study.csv",
    TABLES / "threshold_skill.csv",
    TABLES / "rain_classification_metrics.csv",
    TABLES / "extreme_rainfall_evaluation.csv",
    TABLES / "attention_extreme_vs_normal.csv",
    TABLES / "station_wise_error.csv",
]


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


def month_to_season(month: int) -> str:
    for name, months in SEASON_MONTHS.items():
        if month in months:
            return name
    raise ValueError(f"Unhandled month: {month}")


def verify_season_matches_clean_dataset() -> None:
    """Assert SEASON_MONTHS matches clean_dataset.csv month→season exactly."""
    clean = pd.read_csv(CLEAN_CSV, usecols=["month", "season"])
    month_name_to_num = {
        "January": 1,
        "February": 2,
        "March": 3,
        "April": 4,
        "May": 5,
        "June": 6,
        "July": 7,
        "August": 8,
        "September": 9,
        "October": 10,
        "November": 11,
        "December": 12,
    }
    pairs = clean.drop_duplicates()
    for _, row in pairs.iterrows():
        m = month_name_to_num[row["month"]]
        expected = month_to_season(m)
        if row["season"] != expected:
            raise RuntimeError(
                f"Season map mismatch: {row['month']} → dataset={row['season']} "
                f"vs map={expected}"
            )
    print(
        "SEASON DEFINITION CONFIRMED vs clean_dataset.csv:\n"
        "  Winter: Dec, Jan, Feb, Mar\n"
        "  Summer: Apr, May, Jun\n"
        "  Monsoon: Jul, Aug, Sep\n"
        "  Post-monsoon: Oct, Nov\n"
        "  (Matches original EDA `season` column exactly; NOT IMD JJAS/DJF/MAM/ON.)"
    )


def metrics_subset(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def eval_all(device: torch.device, feat: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    for h in HORIZONS:
        paths = data_paths(h)
        X_test = np.load(paths["X_test"])
        y_test = np.load(paths["y_test"])
        scaler_y = joblib.load(paths["scaler_y"])

        print(f"\n--- h={h}: rebuilding test meta ---", flush=True)
        meta = rebuild_test_meta(feat, h)
        if len(meta) != len(X_test):
            raise RuntimeError(
                f"Meta/test length mismatch h={h}: meta={len(meta)} vs X={len(X_test)}"
            )
        target_dates = pd.to_datetime([m["target_date"] for m in meta])
        seasons = np.array([month_to_season(int(d.month)) for d in target_dates])

        for label, key in MODEL_SPECS:
            ckpt = ckpt_path(key, h)
            assert ckpt.exists(), ckpt
            print(f"--- h={h}: {label} CUDA+autocast ({ckpt.name}) ---", flush=True)
            model = build_model(key, device)
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state["model_state_dict"])
            y_true, y_pred = predict_mm(model, X_test, y_test, scaler_y, device)
            del model
            torch.cuda.empty_cache()

            for season in SEASON_ORDER:
                mask = seasons == season
                m = metrics_subset(y_true[mask], y_pred[mask])
                rows.append(
                    {
                        "Model": label,
                        "Horizon": h,
                        "Season": season,
                        "n_samples": int(mask.sum()),
                        "RMSE": m["RMSE"],
                        "MAE": m["MAE"],
                        "R2": m["R2"],
                        "mean_obs_mm": float(np.mean(y_true[mask])),
                    }
                )

    return pd.DataFrame(rows)


def plot_rmse(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {
        "LSTM": "#4c78a8",
        "CNN-LSTM-Temporal": "#f58518",
        "CNN-LSTM+Attention": "#54a24b",
    }
    labels_short = {
        "LSTM": "LSTM",
        "CNN-LSTM-Temporal": "Temporal",
        "CNN-LSTM+Attention": "Attention",
    }
    models = [m for m, _ in MODEL_SPECS]
    x = np.arange(len(SEASON_ORDER), dtype=float)
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
    for ax, h in zip(axes, HORIZONS):
        for i, model in enumerate(models):
            vals = []
            for season in SEASON_ORDER:
                row = df[
                    (df["Model"] == model)
                    & (df["Horizon"] == h)
                    & (df["Season"] == season)
                ].iloc[0]
                vals.append(float(row["RMSE"]))
            offset = (i - 1) * width
            ax.bar(
                x + offset,
                vals,
                width,
                color=colors[model],
                edgecolor="black",
                linewidth=0.6,
                label=labels_short[model] if h == HORIZONS[0] else None,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(list(SEASON_ORDER), rotation=15, ha="right")
        ax.set_title(f"h={h}")
        ax.set_xlabel("Season")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("RMSE (mm/day)")
    fig.suptitle(
        "Seasonal RMSE by model (India EDA seasons; seed 42)",
        y=1.02,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_FIG}")


def print_summary(df: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("PLAIN-LANGUAGE SUMMARY")
    print("=" * 72)

    for h in HORIZONS:
        sub = df[df["Horizon"] == h]
        print(f"\n--- Horizon h={h} ---")
        # Monsoon vs dryish (Winter as driest proxy; also overall non-monsoon)
        for model in [m for m, _ in MODEL_SPECS]:
            mon = sub[(sub.Model == model) & (sub.Season == "Monsoon")].iloc[0]
            win = sub[(sub.Model == model) & (sub.Season == "Winter")].iloc[0]
            sumr = sub[(sub.Model == model) & (sub.Season == "Summer")].iloc[0]
            post = sub[(sub.Model == model) & (sub.Season == "Post-monsoon")].iloc[0]
            dry_rmse = (win.RMSE + sumr.RMSE + post.RMSE) / 3.0
            print(
                f"  {model}: Monsoon RMSE={mon.RMSE:.3f} "
                f"(mean_obs={mon.mean_obs_mm:.2f} mm) vs "
                f"Winter={win.RMSE:.3f}, Summer={sumr.RMSE:.3f}, "
                f"Post-monsoon={post.RMSE:.3f}; "
                f"Monsoon/mean(non-Monsoon RMSE)={mon.RMSE / dry_rmse:.2f}x"
            )

        print("  Model ranking by RMSE (lower better) per season:")
        name_map = {
            "LSTM": "LSTM",
            "CNN-LSTM-Temporal": "Temporal",
            "CNN-LSTM+Attention": "Attention",
        }
        for season in SEASON_ORDER:
            s = sub[sub.Season == season].sort_values("RMSE")
            short = [
                f"{name_map[r.Model]}({r.RMSE:.3f})" for _, r in s.iterrows()
            ]
            print(f"    {season}: {' < '.join(short)}")


def main() -> None:
    device = require_cuda()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    hashes_before = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print("BEFORE hashes:")
    for k, v in hashes_before.items():
        print(f"  {k}: {v}")

    print()
    verify_season_matches_clean_dataset()

    print()
    print("=" * 72)
    print("PREDICTION SOURCE")
    print("=" * 72)
    print(
        "No saved per-sample y_pred; CUDA+autocast seed-42 inference from\n"
        "existing checkpoints (same protocol as Feature 6 / extreme eval).\n"
        "target_date via rebuild_test_meta(feature_engineered_v2, h).\n"
        "Season from target_date.month using clean_dataset.csv season map."
    )

    print(f"\nLoading {FEAT_CSV.name}...", flush=True)
    feat = pd.read_csv(FEAT_CSV, parse_dates=["date_of_record"])
    df = eval_all(device, feat)

    NOTES = (
        "# NOTES (do not parse as data; pandas: read_csv(..., comment=\"#\"))\n"
        "# Season = Winter/Summer/Monsoon/Post-monsoon from target_date month,\n"
        "# matching clean_dataset.csv original `season` column exactly\n"
        "# (Winter=Dec-Mar, Summer=Apr-Jun, Monsoon=Jul-Sep, Post-monsoon=Oct-Nov).\n"
        "# NOT IMD JJAS (Jun-Sep) or global DJF/MAM/JJA/SON.\n"
        "# Metrics: CUDA+autocast seed=42 inference; no retraining.\n"
    )
    OUT_CSV.write_text(NOTES + df.to_csv(index=False), encoding="utf-8")
    print(f"\nWrote {OUT_CSV}")
    plot_rmse(df)

    print()
    print("=" * 72)
    print("seasonal_performance.csv (full)")
    print("=" * 72)
    with pd.option_context("display.max_columns", None, "display.width", 140, "display.float_format", "{:.6f}".format):
        print(df.to_string(index=False))

    print_summary(df)

    hashes_after = {p.name: sha256_file(p) for p in VERIFIED if p.exists()}
    print()
    print("=" * 72)
    print("INTEGRITY")
    print("=" * 72)
    ok = True
    for k, v0 in hashes_before.items():
        match = v0 == hashes_after[k]
        ok = ok and match
        print(f"  {k}: {'UNCHANGED' if match else 'CHANGED'} ({v0})")
    assert ok
    print(
        "CONFIRM: no existing verified file modified; no training; "
        "new outputs only: seasonal_performance.csv, seasonal_rmse_comparison.png"
    )


if __name__ == "__main__":
    main()
