"""
End-to-end RainfallPrediction pipeline.

Usage (from RainfallPrediction/, preferably with the CUDA .venv):
    python run_pipeline.py
    python run_pipeline.py --force
    python run_pipeline.py --skip-train
    python run_pipeline.py --from sequences

Steps:
  1. clean      - raw Excel -> clean_dataset.csv
  2. features   - DOY encoding + station_id -> feature_engineered_v2.csv
  3. audit      - temporal density report (optional)
  4. sequences  - contiguous windows -> X_*_v2.npy / y_*_v2.npy
  5. scale_y    - MinMaxScaler on train y (skipped if already scaled by seq script)
  6. train      - LSTM baseline v2 (seed 42, CUDA)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "raw" / "india_weather_rainfall_data.xlsx"
PROCESSED = BASE / "data" / "processed"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
FIGURES = REPORTS / "figures"

CLEAN_CSV = PROCESSED / "clean_dataset.csv"
FEAT_CSV = PROCESSED / "feature_engineered.csv"
FEAT_V2_CSV = PROCESSED / "feature_engineered_v2.csv"
MISSING_CSV = PROCESSED / "missing_values_summary.csv"

STEPS = ["clean", "features", "audit", "sequences", "scale_y", "train"]


def log(msg: str) -> None:
    print(f"\n=== {msg} ===", flush=True)


def _resolve_python() -> str:
    """Prefer the project .venv (CUDA build) over system CPU-only Python."""
    candidates = [
        BASE.parent / ".venv" / "Scripts" / "python.exe",  # Research Project/.venv
        BASE / ".venv" / "Scripts" / "python.exe",
        Path(r"D:\project\Research Project\.venv\Scripts\python.exe"),
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    return sys.executable


def run_script(script: str, *args: str) -> None:
    python = _resolve_python()
    if "Programs\\Python" in python.replace("/", "\\"):
        print(
            "WARNING: Using system Python (often CPU-only). "
            "Activate D:\\project\\Research Project\\.venv for CUDA training.",
            flush=True,
        )
    cmd = [python, str(BASE / script), *args]
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(BASE), check=True)


def step_clean(force: bool) -> None:
    log("1/6 CLEAN")
    if CLEAN_CSV.exists() and not force:
        print(f"Skip (exists): {CLEAN_CSV}")
        return
    if not RAW.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW}\n"
            "Place india_weather_rainfall_data.xlsx under data/raw/"
        )

    print(f"Loading {RAW} ...")
    df = pd.read_excel(RAW)
    missing = df.isnull().sum()
    missing_df = pd.DataFrame(
        {
            "Missing Values": missing,
            "Percentage": (missing / len(df)) * 100,
        }
    ).sort_values("Percentage", ascending=False)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    missing_df.to_csv(MISSING_CSV)

    df["date_of_record"] = pd.to_datetime(df["date_of_record"])
    df = df.sort_values(["station_name", "date_of_record"]).reset_index(drop=True)
    before = len(df)
    df = df.dropna(subset=["rainfall"]).reset_index(drop=True)
    print(f"Dropped {before - len(df):,} rows with missing rainfall; remaining {len(df):,}")

    for col in ["min_temp", "max_temp"]:
        df[col] = df.groupby("station_name")[col].transform(
            lambda s: s.interpolate(method="linear", limit_direction="both")
        )
        df[col] = df.groupby("station_name")[col].transform(lambda s: s.fillna(s.median()))

    for col in ["wind_speed", "air_pressure"]:
        df[col] = df.groupby("station_name")[col].transform(lambda s: s.fillna(s.median()))

    for col in ["min_temp", "max_temp", "wind_speed", "air_pressure"]:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    df = df.sort_values(["station_name", "date_of_record"]).reset_index(drop=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"Saved {CLEAN_CSV} shape={df.shape}")


def step_features(force: bool) -> None:
    log("2/6 FEATURES + station_id")
    if FEAT_V2_CSV.exists() and not force:
        print(f"Skip (exists): {FEAT_V2_CSV}")
        return
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"Missing {CLEAN_CSV} — run clean step first")

    df = pd.read_csv(CLEAN_CSV)
    df["date_of_record"] = pd.to_datetime(df["date_of_record"])
    n_before = len(df)
    assert df["date_of_record"].isna().sum() == 0

    day_of_year = df["date_of_record"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * day_of_year / 366)
    df["doy_cos"] = np.cos(2 * np.pi * day_of_year / 366)
    df = df.drop(columns=["month", "season"], errors="ignore")

    assert len(df) == n_before
    assert df[["doy_sin", "doy_cos"]].isna().sum().sum() == 0
    df.to_csv(FEAT_CSV, index=False)
    print(f"Saved {FEAT_CSV}")

    # Disambiguate colliding station_name with lat/lon/elevation
    df["station_id"] = (
        df["station_name"].astype(str)
        + "_"
        + df["latitude"].round(2).astype(str)
        + "_"
        + df["longitude"].round(2).astype(str)
        + "_"
        + df["elevation"].astype(int).astype(str)
    )
    n_dup = int(df.groupby(["station_id", "date_of_record"]).size().gt(1).sum())
    if n_dup:
        raise RuntimeError(f"station_id still has {n_dup} duplicate date groups")
    df.to_csv(FEAT_V2_CSV, index=False)
    print(
        f"Saved {FEAT_V2_CSV} | stations: "
        f"name={df['station_name'].nunique()} id={df['station_id'].nunique()} | "
        f"dup(station_id,date)={n_dup}"
    )


def step_audit(force: bool) -> None:
    log("3/6 TEMPORAL AUDIT")
    out = REPORTS / "temporal_density_audit_v2.txt"
    if out.exists() and not force:
        print(f"Skip (exists): {out}")
        return
    if not FEAT_V2_CSV.exists():
        raise FileNotFoundError(f"Missing {FEAT_V2_CSV}")
    REPORTS.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(BASE / "audit_temporal_density.py"), str(FEAT_V2_CSV)],
        cwd=str(BASE),
        capture_output=True,
        text=True,
        check=True,
    )
    out.write_text(result.stdout, encoding="utf-8")
    print(result.stdout.split("DONE")[0][-500:] if "DONE" in result.stdout else result.stdout[-800:])
    print(f"Saved {out}")


def step_sequences(force: bool) -> None:
    log("4/6 SEQUENCES (v2 — 8 features with past rainfall)")
    targets = [
        PROCESSED / f
        for f in ("X_train_v2.npy", "X_val_v2.npy", "X_test_v2.npy", "y_train_v2.npy")
    ]
    if all(p.exists() for p in targets) and not force:
        print("Skip (X/y v2 arrays exist)")
        return
    run_script("generate_sequences_v2.py")


def step_scale_y(force: bool) -> None:
    log("5/6 SCALE Y (train-only MinMax) — v2 arrays")
    y_path = PROCESSED / "y_train_v2.npy"
    scaler_path = MODELS / "minmax_scaler_y_v2.joblib"
    if not y_path.exists():
        raise FileNotFoundError("y_train_v2.npy missing — run sequences first")

    y_train = np.load(y_path)
    if scaler_path.exists() and not force and float(y_train.max()) <= 1.05:
        print(f"Skip (y_v2 looks scaled, scaler exists): max(y_train_v2)={y_train.max():.4f}")
        return

    y_val = np.load(PROCESSED / "y_val_v2.npy")
    y_test = np.load(PROCESSED / "y_test_v2.npy")

    if float(y_train.max()) <= 1.05 and force:
        raise RuntimeError(
            "y_v2 arrays appear already scaled. Re-run with --force including sequences "
            "(or delete y_*_v2.npy and re-run from --from sequences) before scale_y."
        )

    print(f"y_train_v2 BEFORE: min={y_train.min():.4f} max={y_train.max():.4f}")
    scaler = MinMaxScaler()
    y_train_s = scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = scaler.transform(y_val.reshape(-1, 1)).ravel().astype(np.float32)
    y_test_s = scaler.transform(y_test.reshape(-1, 1)).ravel().astype(np.float32)
    print(f"y_train_v2 AFTER:  min={y_train_s.min():.4f} max={y_train_s.max():.4f}")

    np.save(PROCESSED / "y_train_v2.npy", y_train_s)
    np.save(PROCESSED / "y_val_v2.npy", y_val_s)
    np.save(PROCESSED / "y_test_v2.npy", y_test_s)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Saved {scaler_path}")


def step_train(force: bool) -> None:
    log("6/6 TRAIN LSTM (v2 baseline — 8 features)")
    ckpt = MODELS / "lstm_baseline_v2_seed42.pt"
    if ckpt.exists() and not force:
        print(f"Skip (exists): {ckpt}")
        print("Use --force to retrain.")
        return
    FIGURES.mkdir(parents=True, exist_ok=True)
    # Ensure v2 sequences exist; generate_sequences_v2 writes X_*_v2.npy
    if not (PROCESSED / "X_train_v2.npy").exists():
        run_script("generate_sequences_v2.py")
    run_script("train_lstm_baseline_v2.py")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RainfallPrediction end-to-end pipeline")
    p.add_argument(
        "--from",
        dest="start_from",
        choices=STEPS,
        default="clean",
        help="First step to run (default: clean)",
    )
    p.add_argument("--force", action="store_true", help="Rebuild artifacts even if they exist")
    p.add_argument("--skip-train", action="store_true", help="Stop before LSTM training")
    p.add_argument("--skip-audit", action="store_true", help="Skip temporal density audit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_idx = STEPS.index(args.start_from)
    force = args.force

    print(f"Pipeline start_from={args.start_from} force={force}")
    print(f"Project root: {BASE}")

    runners = {
        "clean": lambda: step_clean(force),
        "features": lambda: step_features(force),
        "audit": lambda: None if args.skip_audit else step_audit(force),
        "sequences": lambda: step_sequences(force),
        "scale_y": lambda: step_scale_y(force),
        "train": lambda: None if args.skip_train else step_train(force),
    }

    for i, name in enumerate(STEPS):
        if i < start_idx:
            continue
        runners[name]()

    log("PIPELINE COMPLETE")
    print("Key outputs:")
    print(f"  {FEAT_V2_CSV}")
    print(f"  {PROCESSED / 'X_train_v2.npy'}")
    print(f"  {MODELS / 'minmax_scaler_y_v2.joblib'}")
    print(f"  {MODELS / 'lstm_baseline_v2_seed42.pt'}")


if __name__ == "__main__":
    main()
