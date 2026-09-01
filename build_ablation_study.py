"""
Formal ablation study (no retrain).

1) Temporal_vs_LSTM significance at h=1..4 from seed-42 checkpoints (inference only)
2) reports/tables/ablation_study.csv from master_results + significance flags
3) reports/figures/ablation_study_bars.png
4) Integrity check + printed review

Never writes master_results.csv. Appends Temporal_vs_LSTM rows only.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.amp import autocast

from arima_and_significance import diebold_mariano
from src.cuda_setup import DEFAULT_BATCH_SIZE, make_loader, require_cuda
from src.model import CNNLSTMAttention, CNNLSTMTemporalBaseline, LSTMBaseline

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
MODELS = BASE / "models"
TABLES = BASE / "reports" / "tables"
FIGURES = BASE / "reports" / "figures"

MASTER_CSV = TABLES / "master_results.csv"
SIG_CSV = TABLES / "significance_results.csv"
ABLATION_CSV = TABLES / "ablation_study.csv"
FIGURE_PATH = FIGURES / "ablation_study_bars.png"

SEED = 42
N_BOOT = 1000
BOOT_SEED = 42
BATCH_SIZE = DEFAULT_BATCH_SIZE
HORIZONS = (1, 2, 3, 4)

SIG_COLUMNS = [
    "Comparison",
    "Forecast_Horizon",
    "Seeds_Used",
    "DM_p_value",
    "Paired_t_p_value",
    "Bootstrap_95CI_lo",
    "Bootstrap_95CI_hi",
    "CI_Definition",
    "Significant_at_0.05",
    "Notes",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_mean_std(cell: str) -> tuple[float | None, float | None]:
    """Parse '9.3745 ± 0.0408' or return (None, None) for Not Available / blank."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return None, None
    s = str(cell).strip()
    if not s or s.lower() in {"not available", "nan", "n/a", "na"}:
        return None, None
    m = re.match(
        r"^\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*[±+-]\s*"
        r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$",
        s,
    )
    if m:
        return float(m.group(1)), float(m.group(2))
    # bare number (no std)
    try:
        return float(s), None
    except ValueError:
        return None, None


@torch.no_grad()
def predict(model, loader, device) -> np.ndarray:
    model.eval()
    chunks = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        with autocast("cuda"):
            chunks.append(model(xb).float())
    return torch.cat(chunks, dim=0).cpu().numpy()


def bootstrap_rmse_diff_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_resamples: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> tuple[float, float]:
    """95% CI for (RMSE_A - RMSE_B) via paired bootstrap."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        rmse_a = float(np.sqrt(np.mean((yt - pa) ** 2)))
        rmse_b = float(np.sqrt(np.mean((yt - pb) ** 2)))
        diffs[b] = rmse_a - rmse_b
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def paths_for_horizon(h: int) -> dict:
    if h == 1:
        return {
            "X_test": DATA / "X_test_v2.npy",
            "y_test": DATA / "y_test_v2.npy",
            "scaler_y": MODELS / "minmax_scaler_y_v2.joblib",
            "lstm_ckpt": MODELS / f"lstm_baseline_v2_seed{SEED}.pt",
            "temp_ckpt": MODELS / f"cnn_lstm_temporal_h1_seed{SEED}.pt",
            "attn_ckpt": MODELS / f"cnn_lstm_attention_h1_seed{SEED}.pt",
        }
    return {
        "X_test": DATA / f"X_test_h{h}.npy",
        "y_test": DATA / f"y_test_h{h}.npy",
        "scaler_y": MODELS / f"minmax_scaler_y_h{h}.joblib",
        "lstm_ckpt": MODELS / f"lstm_h{h}_seed{SEED}.pt",
        "temp_ckpt": MODELS / f"cnn_lstm_temporal_h{h}_seed{SEED}.pt",
        "attn_ckpt": MODELS / f"cnn_lstm_attention_h{h}_seed{SEED}.pt",
    }


def rmse_mm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def load_preds_mm(
    h: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true, pred_temp, pred_lstm, pred_attn (mm units)."""
    p = paths_for_horizon(h)
    for key, path in p.items():
        if not path.exists():
            raise FileNotFoundError(f"missing {key}: {path}")

    X_test = np.load(p["X_test"])
    y_test = np.load(p["y_test"])
    scaler_y = joblib.load(p["scaler_y"])
    test_loader = make_loader(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

    lstm = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
    lstm_ckpt = torch.load(p["lstm_ckpt"], map_location=device, weights_only=False)
    lstm.load_state_dict(lstm_ckpt["model_state_dict"])

    temp = CNNLSTMTemporalBaseline(n_features=8, use_pooling=False).to(device)
    temp_ckpt = torch.load(p["temp_ckpt"], map_location=device, weights_only=False)
    temp.load_state_dict(temp_ckpt["model_state_dict"])

    attn = CNNLSTMAttention(n_features=8).to(device)
    attn_ckpt = torch.load(p["attn_ckpt"], map_location=device, weights_only=False)
    attn.load_state_dict(attn_ckpt["model_state_dict"])

    pred_lstm_s = predict(lstm, test_loader, device)
    pred_temp_s = predict(temp, test_loader, device)
    pred_attn_s = predict(attn, test_loader, device)

    y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    pred_lstm = scaler_y.inverse_transform(pred_lstm_s.reshape(-1, 1)).ravel()
    pred_temp = scaler_y.inverse_transform(pred_temp_s.reshape(-1, 1)).ravel()
    pred_attn = scaler_y.inverse_transform(pred_attn_s.reshape(-1, 1)).ravel()
    assert len(y_true) == len(pred_lstm) == len(pred_temp) == len(pred_attn)
    return y_true, pred_temp, pred_lstm, pred_attn


def compute_temporal_vs_lstm(
    device: torch.device,
) -> tuple[pd.DataFrame, dict[int, dict[str, float]]]:
    """DM/t/bootstrap for Temporal vs LSTM; also return seed-42 RMSEs for all three models."""
    rows = []
    seed42_rmse: dict[int, dict[str, float]] = {}
    for h in HORIZONS:
        y_true, pred_temp, pred_lstm, pred_attn = load_preds_mm(h, device)
        err_temp = (y_true - pred_temp) ** 2
        err_lstm = (y_true - pred_lstm) ** 2
        dm_p = diebold_mariano(err_temp, err_lstm, h=h)
        tt_p = float(stats.ttest_rel(err_temp, err_lstm).pvalue)
        ci_lo, ci_hi = bootstrap_rmse_diff_ci(y_true, pred_temp, pred_lstm)
        sig = "Yes" if dm_p < 0.05 else "No"
        rows.append(
            {
                "Comparison": "Temporal_vs_LSTM",
                "Forecast_Horizon": h,
                "Seeds_Used": SEED,
                "DM_p_value": f"{dm_p:.6e}",
                "Paired_t_p_value": f"{tt_p:.6e}",
                "Bootstrap_95CI_lo": f"{ci_lo:.4f}",
                "Bootstrap_95CI_hi": f"{ci_hi:.4f}",
                "CI_Definition": "RMSE_temp - RMSE_lstm",
                "Significant_at_0.05": sig,
                "Notes": (
                    f"HAC lag={h - 1} (h={h}); inference-only from seed-42 ckpts; "
                    "build_ablation_study.py"
                ),
            }
        )
        seed42_rmse[h] = {
            "lstm": rmse_mm(y_true, pred_lstm),
            "temp": rmse_mm(y_true, pred_temp),
            "attn": rmse_mm(y_true, pred_attn),
        }
        print(
            f"Temporal_vs_LSTM h={h}: DM p={dm_p:.6e}  paired-t p={tt_p:.6e}  "
            f"CI=({ci_lo:.4f}, {ci_hi:.4f})  sig={sig}  n={len(y_true)}  "
            f"RMSE42 lstm={seed42_rmse[h]['lstm']:.4f} "
            f"temp={seed42_rmse[h]['temp']:.4f} attn={seed42_rmse[h]['attn']:.4f}"
        )
    return pd.DataFrame(rows, columns=SIG_COLUMNS), seed42_rmse


def load_seed42_rmses_only(device: torch.device) -> dict[int, dict[str, float]]:
    """Inference-only seed-42 RMSE point estimates (no DM recomputation)."""
    out: dict[int, dict[str, float]] = {}
    for h in HORIZONS:
        y_true, pred_temp, pred_lstm, pred_attn = load_preds_mm(h, device)
        out[h] = {
            "lstm": rmse_mm(y_true, pred_lstm),
            "temp": rmse_mm(y_true, pred_temp),
            "attn": rmse_mm(y_true, pred_attn),
        }
        print(
            f"seed42 RMSE h={h}: lstm={out[h]['lstm']:.4f} "
            f"temp={out[h]['temp']:.4f} attn={out[h]['attn']:.4f}"
        )
    return out



def append_significance_rows(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Byte-safe append of Temporal_vs_LSTM rows (existing file bytes untouched)."""
    existing = pd.read_csv(SIG_CSV)
    already = existing["Comparison"].eq("Temporal_vs_LSTM").any()
    if already:
        print("Temporal_vs_LSTM rows already present — not appending duplicates.")
        return existing

    # Build CSV lines matching column order; do NOT rewrite existing content.
    lines: list[str] = []
    for _, r in new_rows.iterrows():
        vals = [str(r[c]) for c in SIG_COLUMNS]
        # Quote Notes if it contains commas (it does)
        quoted = []
        for c, v in zip(SIG_COLUMNS, vals):
            if c == "Notes" or "," in v:
                quoted.append('"' + v.replace('"', '""') + '"')
            else:
                quoted.append(v)
        lines.append(",".join(quoted))

    raw = SIG_CSV.read_bytes()
    if not raw.endswith(b"\n"):
        raw = raw + b"\n"
    addition = ("\n".join(lines) + "\n").encode("utf-8")
    SIG_CSV.write_bytes(raw + addition)
    print(f"Appended {len(new_rows)} Temporal_vs_LSTM rows to {SIG_CSV}")
    return pd.read_csv(SIG_CSV)


def master_mean_std_lookup(master: pd.DataFrame) -> dict:
    """Map (model_key, horizon) -> parsed RMSE/MAE/R2 mean/std from mean±std rows."""
    out: dict[tuple[str, int], dict] = {}

    def put(model_key: str, row: pd.Series) -> None:
        h = int(row["Forecast_Horizon"])
        rmse_m, rmse_s = parse_mean_std(row["RMSE"])
        mae_m, mae_s = parse_mean_std(row["MAE"])
        r2_m, r2_s = parse_mean_std(row["R2"])
        out[(model_key, h)] = {
            "RMSE_mean": rmse_m,
            "RMSE_std": rmse_s,
            "MAE_mean": mae_m,
            "MAE_std": mae_s,
            "R2_mean": r2_m,
            "R2_std": r2_s,
        }

    for _, row in master.iterrows():
        name = str(row["Model"])
        if name == "LSTM v2 (mean ± std)" and int(row["Forecast_Horizon"]) == 1:
            put("LSTM", row)
        elif name == "LSTM (mean ± std)" and int(row["Forecast_Horizon"]) in (2, 3, 4):
            put("LSTM", row)
        elif name == "CNN-LSTM-Temporal (mean ± std)":
            put("CNN-LSTM-Temporal", row)
        elif name == "CNN-LSTM+Attention (mean ± std)":
            put("CNN-LSTM+Attention", row)
    return out


def sig_flag(sig_df: pd.DataFrame, comparison: str, h: int) -> str | None:
    m = sig_df[
        (sig_df["Comparison"] == comparison) & (sig_df["Forecast_Horizon"] == h)
    ]
    if m.empty:
        return None
    return str(m.iloc[0]["Significant_at_0.05"])


def fmt_num(v: float | None, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return f"{v:.{digits}f}"


def fmt_optional_metric(v: float | None, digits: int = 4) -> str:
    """MAE/R2: explicit Not Available when master has no aggregate (not a measured zero)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "Not Available"
    return f"{v:.{digits}f}"


H3_TEMPORAL_NOTE = (
    "3-seed mean shows Temporal slightly worse than LSTM (+0.0304 RMSE), "
    "but seed-42 specifically shows Temporal significantly better - this discrepancy "
    "reflects seed-to-seed variance at h=3 and should be reported as a limitation of "
    "single-seed significance testing in the thesis, not as a confirmed directional finding."
)

H1_ATTENTION_NOTE = (
    "3-seed mean shows Attention slightly worse than Temporal (+0.0140 RMSE), "
    "but seed-42 shows Attention better (-0.0326) - seed-to-seed variance; "
    "treat as a limitation of single-seed significance testing, not a confirmed "
    "directional finding. Consistent with non-significant test result (CI includes 0)."
)

H3_ATTENTION_NOTE = (
    "3-seed mean shows Attention slightly worse than Temporal (+0.0150 RMSE), "
    "but seed-42 shows Attention better (-0.0056) - seed-to-seed variance; "
    "treat as a limitation of single-seed significance testing, not a confirmed "
    "directional finding. Consistent with non-significant test result (CI includes 0)."
)


def _sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def seed_direction_mismatch(delta_3seed: float, delta_seed42: float) -> str:
    """Yes if sign(3-seed Δ vs previous) disagrees with sign(seed-42 Δ vs previous)."""
    return "Yes" if _sign(delta_3seed) != _sign(delta_seed42) else "No"


def build_ablation_csv(
    master: pd.DataFrame,
    sig_df: pd.DataFrame,
    seed42_rmse: dict[int, dict[str, float]],
) -> pd.DataFrame:
    metrics = master_mean_std_lookup(master)
    rows = []
    for h in HORIZONS:
        lstm = metrics[("LSTM", h)]
        temp = metrics[("CNN-LSTM-Temporal", h)]
        attn = metrics[("CNN-LSTM+Attention", h)]
        r42 = seed42_rmse[h]

        temp_vs_lstm = sig_flag(sig_df, "Temporal_vs_LSTM", h)
        attn_vs_temp = sig_flag(sig_df, "Attention_vs_Temporal", h)
        attn_vs_lstm = sig_flag(sig_df, "Attention_vs_LSTM", h)

        delta_temp = (
            None
            if lstm["RMSE_mean"] is None or temp["RMSE_mean"] is None
            else temp["RMSE_mean"] - lstm["RMSE_mean"]
        )
        delta_attn_lstm = (
            None
            if lstm["RMSE_mean"] is None or attn["RMSE_mean"] is None
            else attn["RMSE_mean"] - lstm["RMSE_mean"]
        )
        delta_attn_temp = (
            None
            if temp["RMSE_mean"] is None or attn["RMSE_mean"] is None
            else attn["RMSE_mean"] - temp["RMSE_mean"]
        )

        d42_temp_lstm = r42["temp"] - r42["lstm"]
        d42_attn_lstm = r42["attn"] - r42["lstm"]
        d42_attn_temp = r42["attn"] - r42["temp"]

        # 1. LSTM baseline
        rows.append(
            {
                "Horizon": h,
                "Model": "LSTM",
                "RMSE_mean": fmt_num(lstm["RMSE_mean"]),
                "RMSE_std": fmt_num(lstm["RMSE_std"]),
                "MAE_mean": fmt_optional_metric(lstm["MAE_mean"]),
                "MAE_std": fmt_optional_metric(lstm["MAE_std"]),
                "R2_mean": fmt_optional_metric(lstm["R2_mean"]),
                "R2_std": fmt_optional_metric(lstm["R2_std"]),
                "Delta_RMSE_vs_LSTM": "0",
                "Delta_RMSE_vs_previous_stage": "0",
                "Delta_RMSE_seed42_vs_LSTM": "0",
                "Delta_RMSE_seed42_vs_previous_stage": "0",
                "Significant_vs_LSTM": "N/A",
                "Significant_vs_previous_stage": "N/A",
                "Seed_Direction_Mismatch": seed_direction_mismatch(0.0, 0.0),
                "Notes": "",
            }
        )
        # 2. CNN-LSTM-Temporal
        assert delta_temp is not None
        note_temp = H3_TEMPORAL_NOTE if h == 3 else ""
        rows.append(
            {
                "Horizon": h,
                "Model": "CNN-LSTM-Temporal",
                "RMSE_mean": fmt_num(temp["RMSE_mean"]),
                "RMSE_std": fmt_num(temp["RMSE_std"]),
                "MAE_mean": fmt_num(temp["MAE_mean"]),
                "MAE_std": fmt_num(temp["MAE_std"]),
                "R2_mean": fmt_num(temp["R2_mean"]),
                "R2_std": fmt_num(temp["R2_std"]),
                "Delta_RMSE_vs_LSTM": fmt_num(delta_temp),
                "Delta_RMSE_vs_previous_stage": fmt_num(delta_temp),
                "Delta_RMSE_seed42_vs_LSTM": fmt_num(d42_temp_lstm),
                "Delta_RMSE_seed42_vs_previous_stage": fmt_num(d42_temp_lstm),
                "Significant_vs_LSTM": temp_vs_lstm if temp_vs_lstm else "",
                "Significant_vs_previous_stage": temp_vs_lstm if temp_vs_lstm else "",
                "Seed_Direction_Mismatch": seed_direction_mismatch(
                    delta_temp, d42_temp_lstm
                ),
                "Notes": note_temp,
            }
        )
        # 3. CNN-LSTM+Attention
        assert delta_attn_lstm is not None and delta_attn_temp is not None
        sig_vs_lstm = (
            attn_vs_lstm if attn_vs_lstm is not None else "Not tested"
        )
        note_attn = ""
        if h == 1:
            note_attn = H1_ATTENTION_NOTE
        elif h == 3:
            note_attn = H3_ATTENTION_NOTE
        rows.append(
            {
                "Horizon": h,
                "Model": "CNN-LSTM+Attention",
                "RMSE_mean": fmt_num(attn["RMSE_mean"]),
                "RMSE_std": fmt_num(attn["RMSE_std"]),
                "MAE_mean": fmt_num(attn["MAE_mean"]),
                "MAE_std": fmt_num(attn["MAE_std"]),
                "R2_mean": fmt_num(attn["R2_mean"]),
                "R2_std": fmt_num(attn["R2_std"]),
                "Delta_RMSE_vs_LSTM": fmt_num(delta_attn_lstm),
                "Delta_RMSE_vs_previous_stage": fmt_num(delta_attn_temp),
                "Delta_RMSE_seed42_vs_LSTM": fmt_num(d42_attn_lstm),
                "Delta_RMSE_seed42_vs_previous_stage": fmt_num(d42_attn_temp),
                "Significant_vs_LSTM": sig_vs_lstm,
                "Significant_vs_previous_stage": (
                    attn_vs_temp if attn_vs_temp else ""
                ),
                "Seed_Direction_Mismatch": seed_direction_mismatch(
                    delta_attn_temp, d42_attn_temp
                ),
                "Notes": note_attn,
            }
        )

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(ABLATION_CSV, index=False)
    print(f"Wrote {ABLATION_CSV}")
    return df


def plot_ablation_bars(ablation: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    models = ["LSTM", "CNN-LSTM-Temporal", "CNN-LSTM+Attention"]
    # Distinct, non-purple palette
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

    for i, model in enumerate(models):
        means, stds = [], []
        for h in HORIZONS:
            row = ablation[(ablation["Horizon"] == h) & (ablation["Model"] == model)].iloc[0]
            means.append(float(row["RMSE_mean"]))
            stds.append(float(row["RMSE_std"]) if str(row["RMSE_std"]).strip() else 0.0)
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3,
            label=labels[model],
            color=colors[model],
            edgecolor="none",
            error_kw={"elinewidth": 1.0, "capthick": 1.0},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("RMSE (mm/day)")
    ax.set_title("Ablation study: RMSE by model and horizon")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=150)
    plt.close(fig)
    print(f"Wrote {FIGURE_PATH}")


def print_review(ablation: pd.DataFrame, sig_df: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print("ABLATION STUDY TABLE (FINAL)")
    print("=" * 88)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        120,
    ):
        print(ablation.to_string(index=False))

    flagged = ablation[ablation["Seed_Direction_Mismatch"] == "Yes"][
        [
            "Horizon",
            "Model",
            "Delta_RMSE_vs_previous_stage",
            "Delta_RMSE_seed42_vs_previous_stage",
            "Seed_Direction_Mismatch",
        ]
    ]
    print("\n" + "=" * 88)
    print(
        "Seed_Direction_Mismatch == Yes "
        "(3-seed delta vs previous vs seed-42 delta vs previous)"
    )
    print("=" * 88)
    if flagged.empty:
        print("(none)")
    else:
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(flagged.to_string(index=False))
        expected = {(1, "CNN-LSTM+Attention"), (3, "CNN-LSTM-Temporal")}
        got = {
            (int(r.Horizon), str(r.Model))
            for r in flagged.itertuples(index=False)
        }
        print(f"\nFlagged rows (n={len(got)}): {sorted(got)}")
        if got == expected:
            print(
                "CONFIRM: flagged exactly h=1 Attention and h=3 Temporal; no others."
            )
        else:
            extra = got - expected
            missing = expected - got
            if extra:
                print(
                    "NOTE: automated check also flags additional row(s) beyond the "
                    f"two previously noted: {sorted(extra)}"
                )
            if missing:
                print(f"WARNING: expected flags missing: {sorted(missing)}")

    print("\n" + "=" * 88)
    print("Temporal_vs_LSTM SIGNIFICANCE ROWS")
    print("=" * 88)
    tv = sig_df[sig_df["Comparison"] == "Temporal_vs_LSTM"].copy()
    cols = [
        "Forecast_Horizon",
        "DM_p_value",
        "Paired_t_p_value",
        "Bootstrap_95CI_lo",
        "Bootstrap_95CI_hi",
        "Significant_at_0.05",
    ]
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(tv[cols].to_string(index=False))


def rmse_fingerprint(df: pd.DataFrame) -> tuple:
    """Comparable rounded (Horizon, Model, RMSE_mean, RMSE_std) tuples."""
    rows = []
    for _, r in df.iterrows():
        def _f(v: object) -> str:
            try:
                return f"{float(v):.4f}"
            except (TypeError, ValueError):
                return str(v).strip()

        rows.append((int(r["Horizon"]), str(r["Model"]).strip(), _f(r["RMSE_mean"]), _f(r["RMSE_std"])))
    return tuple(rows)


def main() -> None:
    device = require_cuda()
    TABLES.mkdir(parents=True, exist_ok=True)

    # --- Integrity snapshot BEFORE any writes ---
    master_hash_before = sha256_file(MASTER_CSV)
    sig_bytes_before = SIG_CSV.read_bytes()
    sig_hash_before = hashlib.sha256(sig_bytes_before).hexdigest()
    n_sig_lines_before = sig_bytes_before.count(b"\n")
    print(f"BEFORE  master_results.csv SHA-256: {master_hash_before}")
    print(f"BEFORE  significance_results.csv SHA-256: {sig_hash_before}")
    print(f"BEFORE  significance line endings count: {n_sig_lines_before}")

    old_ablation = None
    if ABLATION_CSV.exists():
        old_ablation = pd.read_csv(ABLATION_CSV)

    # Step 1: append Temporal_vs_LSTM only if missing; always get seed-42 RMSEs
    existing_sig = pd.read_csv(SIG_CSV)
    if existing_sig["Comparison"].eq("Temporal_vs_LSTM").any():
        print("Temporal_vs_LSTM rows present — skipping DM recompute; loading seed-42 RMSEs.")
        seed42_rmse = load_seed42_rmses_only(device)
        sig_df = existing_sig
    else:
        new_rows, seed42_rmse = compute_temporal_vs_lstm(device)
        sig_df = append_significance_rows(new_rows)

    # Step 2: ablation CSV
    master = pd.read_csv(MASTER_CSV)
    ablation = build_ablation_csv(master, sig_df, seed42_rmse)

    # Step 3: regenerate figure only if underlying RMSE values changed
    if old_ablation is not None and {"RMSE_mean", "RMSE_std"}.issubset(old_ablation.columns):
        if rmse_fingerprint(old_ablation) == rmse_fingerprint(ablation):
            print(
                f"RMSE mean/std unchanged vs prior ablation_study.csv — "
                f"skipping figure regen ({FIGURE_PATH.name})."
            )
        else:
            print("RMSE values changed — regenerating bar chart.")
            plot_ablation_bars(ablation)
    else:
        plot_ablation_bars(ablation)

    # Integrity AFTER
    master_hash_after = sha256_file(MASTER_CSV)
    sig_bytes_after = SIG_CSV.read_bytes()
    prefix = sig_bytes_before if sig_bytes_before.endswith(b"\n") else sig_bytes_before + b"\n"
    if b"Temporal_vs_LSTM" in sig_bytes_before:
        assert sig_bytes_after == sig_bytes_before, (
            "significance file changed but Temporal_vs_LSTM was already present"
        )
        existing_rows_ok = True
    else:
        assert sig_bytes_after.startswith(prefix), (
            "Existing significance_results.csv bytes were altered (must append only)"
        )
        existing_rows_ok = True

    assert master_hash_after == master_hash_before, (
        "master_results.csv was modified (must remain byte-identical)"
    )

    print("\n" + "=" * 88)
    print("INTEGRITY CONFIRMATION")
    print("=" * 88)
    print(f"master_results.csv unchanged: YES  ({master_hash_after})")
    print(
        f"significance_results.csv existing bytes unchanged: "
        f"{'YES' if existing_rows_ok else 'NO'} (append-only)"
    )
    print("No model was retrained (inference-only from seed-42 checkpoints).")

    print_review(ablation, sig_df)


if __name__ == "__main__":
    main()
