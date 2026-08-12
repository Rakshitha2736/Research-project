"""Independent empirical dataset audit — write findings to JSON."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

BASE = Path(r"D:\project\Research Project\RainfallPrediction")
RAW = BASE / "data" / "raw" / "india_weather_rainfall_data.xlsx"
CLEAN = BASE / "data" / "processed" / "clean_dataset.csv"
FEAT = BASE / "data" / "processed" / "feature_engineered_v2.csv"
OUT = BASE / "reports" / "tables" / "dataset_audit_findings.json"


def safe_desc(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce")
    return {
        "count": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "min": None if s.notna().sum() == 0 else float(s.min()),
        "p01": None if s.notna().sum() == 0 else float(s.quantile(0.01)),
        "p50": None if s.notna().sum() == 0 else float(s.median()),
        "p99": None if s.notna().sum() == 0 else float(s.quantile(0.99)),
        "max": None if s.notna().sum() == 0 else float(s.max()),
        "mean": None if s.notna().sum() == 0 else float(s.mean()),
        "std": None if s.notna().sum() == 0 else float(s.std()),
        "neg": int((s < 0).sum()) if s.notna().sum() else 0,
        "zero": int((s == 0).sum()) if s.notna().sum() else 0,
    }


def main() -> None:
    findings: dict = {}

    wb = openpyxl.load_workbook(RAW, read_only=True)
    props = wb.properties
    findings["excel_meta"] = {
        "sheets": wb.sheetnames,
        "creator": props.creator,
        "title": props.title,
        "subject": props.subject,
        "description": props.description,
        "keywords": props.keywords,
        "category": props.category,
        "lastModifiedBy": props.lastModifiedBy,
        "created": str(props.created),
        "modified": str(props.modified),
        "bytes": RAW.stat().st_size,
    }
    wb.close()

    print("Loading RAW...", flush=True)
    raw = pd.read_excel(RAW)
    findings["raw"] = {
        "shape": list(raw.shape),
        "columns": list(raw.columns),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "missing": {c: int(raw[c].isna().sum()) for c in raw.columns},
        "missing_pct": {
            c: float(raw[c].isna().mean() * 100) for c in raw.columns
        },
        "full_dup_rows": int(raw.duplicated().sum()),
        "head": raw.head(5).astype(str).to_dict(orient="records"),
    }

    raw["date_of_record"] = pd.to_datetime(raw["date_of_record"], errors="coerce")
    findings["raw"]["date_min"] = str(raw["date_of_record"].min())
    findings["raw"]["date_max"] = str(raw["date_of_record"].max())
    findings["raw"]["date_nat"] = int(raw["date_of_record"].isna().sum())
    findings["raw"]["n_station_name"] = int(raw["station_name"].nunique())
    findings["raw"]["n_state"] = int(raw["state"].nunique())
    findings["raw"]["n_district"] = int(raw["district"].nunique())

    print("Loading CLEAN...", flush=True)
    clean = pd.read_csv(CLEAN, parse_dates=["date_of_record"])
    findings["clean"] = {
        "shape": list(clean.shape),
        "columns": list(clean.columns),
        "date_min": str(clean["date_of_record"].min().date()),
        "date_max": str(clean["date_of_record"].max().date()),
        "years_span": float(
            (clean["date_of_record"].max() - clean["date_of_record"].min()).days / 365.25
        ),
        "n_station_name": int(clean["station_name"].nunique()),
        "full_dup_rows": int(clean.duplicated().sum()),
        "missing": {c: int(clean[c].isna().sum()) for c in clean.columns},
    }

    print("Loading FEAT...", flush=True)
    feat = pd.read_csv(FEAT, parse_dates=["date_of_record"])
    findings["feat"] = {
        "shape": list(feat.shape),
        "n_station_id": int(feat["station_id"].nunique()),
        "n_station_name": int(feat["station_name"].nunique()),
        "n_state": int(feat["state"].nunique()),
        "states": sorted(feat["state"].dropna().unique().tolist()),
        "date_min": str(feat["date_of_record"].min().date()),
        "date_max": str(feat["date_of_record"].max().date()),
        "years_span": float(
            (feat["date_of_record"].max() - feat["date_of_record"].min()).days / 365.25
        ),
        "dup_station_date": int(
            feat.duplicated(subset=["station_id", "date_of_record"]).sum()
        ),
    }

    # Geographic checks
    lat, lon, elev = feat["latitude"], feat["longitude"], feat["elevation"]
    findings["geo"] = {
        "lat": safe_desc(lat),
        "lon": safe_desc(lon),
        "elev": safe_desc(elev),
        "lat_outside_india": int(((lat < 6) | (lat > 38)).sum()),
        "lon_outside_india": int(((lon < 68) | (lon > 98)).sum()),
        "elev_zero": int((elev == 0).sum()),
        "elev_neg": int((elev < 0).sum()),
        "unique_coords": int(feat[["latitude", "longitude"]].drop_duplicates().shape[0]),
        "stations_per_unique_coord": float(
            feat["station_id"].nunique()
            / max(1, feat[["latitude", "longitude"]].drop_duplicates().shape[0])
        ),
    }

    # Colliding station_name with different coords
    name_coord = (
        feat.groupby("station_name")[["latitude", "longitude", "elevation"]]
        .nunique()
        .reset_index()
    )
    colliding = name_coord[
        (name_coord["latitude"] > 1)
        | (name_coord["longitude"] > 1)
        | (name_coord["elevation"] > 1)
    ]
    findings["station_name_collisions"] = {
        "n_names_with_multiple_coords": int(len(colliding)),
        "examples": colliding.head(15).to_dict(orient="records"),
        "delta_name_to_id": int(
            feat["station_id"].nunique() - feat["station_name"].nunique()
        ),
    }

    # Physical plausibility on CLEAN (pre-imputation target exists; temps may be filled)
    num_cols = [
        "avg_temp",
        "min_temp",
        "max_temp",
        "wind_speed",
        "air_pressure",
        "rainfall",
        "elevation",
    ]
    findings["distributions_clean"] = {c: safe_desc(clean[c]) for c in num_cols if c in clean}

    # Impossible conditions
    findings["impossible"] = {
        "rainfall_neg": int((clean["rainfall"] < 0).sum()),
        "rainfall_gt_500": int((clean["rainfall"] > 500).sum()),
        "rainfall_gt_1000": int((clean["rainfall"] > 1000).sum()),
        "rainfall_max": float(clean["rainfall"].max()),
        "min_gt_max_temp": int((clean["min_temp"] > clean["max_temp"]).sum()),
        "avg_outside_min_max": int(
            (
                (clean["avg_temp"] < clean["min_temp"] - 0.5)
                | (clean["avg_temp"] > clean["max_temp"] + 0.5)
            ).sum()
        ),
        "temp_lt_m40": int((clean[["avg_temp", "min_temp", "max_temp"]] < -40).any(axis=1).sum()),
        "temp_gt_60": int((clean[["avg_temp", "min_temp", "max_temp"]] > 60).any(axis=1).sum()),
        "wind_neg": int((clean["wind_speed"] < 0).sum()),
        "wind_gt_100": int((clean["wind_speed"] > 100).sum()),
        "pressure_lt_800": int((clean["air_pressure"] < 800).sum()),
        "pressure_gt_1100": int((clean["air_pressure"] > 1100).sum()),
        "pressure_units_hint_mean": float(clean["air_pressure"].mean()),
    }

    # Same checks on RAW rainfall/temp before cleaning
    findings["impossible_raw"] = {
        "rainfall_neg": int((raw["rainfall"].dropna() < 0).sum()),
        "rainfall_gt_500": int((raw["rainfall"].dropna() > 500).sum()),
        "rainfall_max": float(raw["rainfall"].max(skipna=True)),
        "min_gt_max_temp": int(
            (raw["min_temp"].notna() & raw["max_temp"].notna() & (raw["min_temp"] > raw["max_temp"])).sum()
        ),
        "avg_outside_minmax": int(
            (
                raw["avg_temp"].notna()
                & raw["min_temp"].notna()
                & raw["max_temp"].notna()
                & (
                    (raw["avg_temp"] < raw["min_temp"] - 0.5)
                    | (raw["avg_temp"] > raw["max_temp"] + 0.5)
                )
            ).sum()
        ),
    }

    # Spatial representation by state
    state_counts = (
        feat.groupby("state")["station_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    findings["spatial"] = {
        "stations_per_state": state_counts.to_dict(),
        "top5_share": float(state_counts.head(5).sum() / state_counts.sum()),
        "bottom10_states": state_counts.tail(10).to_dict(),
        "n_states": int(len(state_counts)),
    }

    # Temporal density by year
    year_counts = feat["date_of_record"].dt.year.value_counts().sort_index()
    findings["temporal_by_year"] = {
        str(int(k)): int(v) for k, v in year_counts.items()
    }

    # Row drop impact
    findings["cleaning_impact"] = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(clean)),
        "dropped": int(len(raw) - len(clean)),
        "drop_pct": float((1 - len(clean) / len(raw)) * 100),
        "raw_rainfall_missing_pct": float(raw["rainfall"].isna().mean() * 100),
    }

    # Encoding / garbled names (replacement char or ?)
    garbled = feat["station_name"].astype(str).str.contains(r"[?\uFFFD]", regex=True)
    findings["garbled_station_names"] = {
        "n_rows": int(garbled.sum()),
        "n_unique_names": int(feat.loc[garbled, "station_name"].nunique()),
        "examples": feat.loc[garbled, "station_name"].drop_duplicates().head(20).tolist(),
    }

    # Duplicate near-stations (haversine same-day identical rainfall correlation proxy)
    # Check identical series between station IDs sharing same rounded lat/lon
    coord_groups = feat.groupby(
        [feat["latitude"].round(2), feat["longitude"].round(2)]
    )["station_id"].nunique()
    findings["near_duplicate_coords"] = {
        "coords_with_multiple_station_ids": int((coord_groups > 1).sum()),
        "max_ids_per_coord": int(coord_groups.max()),
    }

    # Season / month columns in raw (ignored?)
    findings["ignored_raw_columns"] = [
        c
        for c in raw.columns
        if c
        not in {
            "avg_temp",
            "min_temp",
            "max_temp",
            "wind_speed",
            "air_pressure",
            "rainfall",
            "date_of_record",
            "station_name",
            "latitude",
            "longitude",
            "elevation",
            "state",
            "district",
        }
    ]

    # Leakage evidence from sequence metadata
    meta_path = BASE / "data" / "processed" / "sequence_metadata_v2.json"
    if meta_path.exists():
        findings["sequence_metadata_v2"] = json.loads(meta_path.read_text(encoding="utf-8"))

    # Scaler extrema check: train vs all
    import joblib

    sx = joblib.load(BASE / "models" / "minmax_scaler_v2.joblib")
    sy = joblib.load(BASE / "models" / "minmax_scaler_y_v2.joblib")
    Xtr = np.load(BASE / "data" / "processed" / "X_train_v2.npy", mmap_mode="r")
    Xva = np.load(BASE / "data" / "processed" / "X_val_v2.npy", mmap_mode="r")
    Xte = np.load(BASE / "data" / "processed" / "X_test_v2.npy", mmap_mode="r")
    ytr = np.load(BASE / "data" / "processed" / "y_train_v2.npy")
    yva = np.load(BASE / "data" / "processed" / "y_val_v2.npy")
    yte = np.load(BASE / "data" / "processed" / "y_test_v2.npy")

    # Inverse transform train/val/test y and compare max
    ytr_mm = sy.inverse_transform(ytr.reshape(-1, 1)).ravel()
    yva_mm = sy.inverse_transform(yva.reshape(-1, 1)).ravel()
    yte_mm = sy.inverse_transform(yte.reshape(-1, 1)).ravel()
    findings["scaler_audit"] = {
        "x_data_min": sx.data_min_.tolist(),
        "x_data_max": sx.data_max_.tolist(),
        "y_data_min": float(sy.data_min_[0]),
        "y_data_max": float(sy.data_max_[0]),
        "y_train_mm_max": float(ytr_mm.max()),
        "y_val_mm_max": float(yva_mm.max()),
        "y_test_mm_max": float(yte_mm.max()),
        "val_exceeds_train_y_max": bool(yva_mm.max() > ytr_mm.max() + 1e-6),
        "test_exceeds_train_y_max": bool(yte_mm.max() > ytr_mm.max() + 1e-6),
        "X_shapes": {
            "train": list(Xtr.shape),
            "val": list(Xva.shape),
            "test": list(Xte.shape),
        },
        "y_scaled_ranges": {
            "train": [float(ytr.min()), float(ytr.max())],
            "val": [float(yva.min()), float(yva.max())],
            "test": [float(yte.min()), float(yte.max())],
        },
    }

    # Chronological split check via feature dates and sample counts by year in targets
    # Reconstruct contiguous windows count by target year from feat (light check)
    findings["split_documented"] = {
        "train_end": "2022-12-31",
        "val": "2023",
        "test": "2024-01-01 to 2025-02-10",
    }

    # Rainfall sparsity / zero inflation
    rain = feat["rainfall"]
    findings["rainfall_regime"] = {
        "zero_pct": float((rain == 0).mean() * 100),
        "lt_01_pct": float((rain < 0.1).mean() * 100),
        "lt_1_pct": float((rain < 1).mean() * 100),
        "gte_10_pct": float((rain >= 10).mean() * 100),
        "gte_50_pct": float((rain >= 50).mean() * 100),
        "jjas_share_of_total_rain": float(
            rain[feat["date_of_record"].dt.month.isin([6, 7, 8, 9])].sum() / rain.sum() * 100
        ),
    }

    # avg_temp vs (min+max)/2 consistency
    mid = (feat["min_temp"] + feat["max_temp"]) / 2
    diff = (feat["avg_temp"] - mid).abs()
    findings["avg_vs_minmax_mid"] = {
        "mean_abs_diff": float(diff.mean()),
        "p95_abs_diff": float(diff.quantile(0.95)),
        "gt_2C_pct": float((diff > 2).mean() * 100),
        "gt_5C_pct": float((diff > 5).mean() * 100),
    }

    OUT.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}", flush=True)
    print(json.dumps({k: findings[k] for k in ["excel_meta", "cleaning_impact", "impossible", "geo", "spatial"]}, indent=2)[:4000])


if __name__ == "__main__":
    main()
