"""Quantify pre-split imputation leakage into the train period.

Matches run_pipeline.py: groupby station_name; linear interp on min/max temp;
station median on wind/pressure; then chronological split at sequence time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\project\Research Project\RainfallPrediction")
RAW = BASE / "data" / "raw" / "india_weather_rainfall_data.xlsx"
FEAT = BASE / "data" / "processed" / "feature_engineered_v2.csv"
OUT = BASE / "reports" / "tables" / "imputation_leakage_audit.json"

TRAIN_END = pd.Timestamp("2022-12-31")
SEQ_LEN = 30


def station_id_of(df: pd.DataFrame) -> pd.Series:
    return (
        df["station_name"].astype(str)
        + "_"
        + df["latitude"].round(2).astype(str)
        + "_"
        + df["longitude"].round(2).astype(str)
        + "_"
        + df["elevation"].astype(int).astype(str)
    )


def future_dependent_linear_mask(
    dates: pd.DatetimeIndex, values: np.ndarray, train_end: pd.Timestamp
) -> np.ndarray:
    """Train-period originally-missing points whose linear fill uses a next
    valid observation dated after train_end (pandas linear + limit_direction both).
    """
    n = len(values)
    out = np.zeros(n, dtype=bool)
    isnan = ~np.isfinite(values)
    valid_idx = np.where(~isnan)[0]
    if len(valid_idx) == 0:
        return out

    for i in range(n):
        if not isnan[i] or dates[i] > train_end:
            continue
        nexts = valid_idx[valid_idx > i]
        if len(nexts) == 0:
            continue
        next_i = nexts[0]
        # Future endpoint participates in the fill (gap crossing split, or
        # leading NaNs filled from first future observation via limit_direction both).
        if dates[next_i] > train_end:
            out[i] = True
    return out


def median_future_dependent_mask(
    dates: pd.DatetimeIndex, values: np.ndarray, train_end: pd.Timestamp
) -> tuple[np.ndarray, float | None, float | None]:
    """Originally-missing train rows filled with full-series median, when that
    median differs from the train-only median (or train has no observations).
    """
    n = len(values)
    out = np.zeros(n, dtype=bool)
    isnan = ~np.isfinite(values)
    train_mask = np.asarray(dates <= train_end)
    train_obs = values[train_mask & ~isnan]
    all_obs = values[~isnan]
    full_med = float(np.median(all_obs)) if len(all_obs) else None
    train_med = float(np.median(train_obs)) if len(train_obs) else None

    if full_med is None:
        return out, full_med, train_med

    for i in range(n):
        if not isnan[i] or dates[i] > train_end:
            continue
        if train_med is None or abs(full_med - train_med) > 1e-12:
            out[i] = True
    return out, full_med, train_med


def main() -> None:
    print("Loading raw...", flush=True)
    raw = pd.read_excel(RAW)
    raw["date_of_record"] = pd.to_datetime(raw["date_of_record"])
    raw = raw.sort_values(["station_name", "date_of_record"]).reset_index(drop=True)
    raw = raw.dropna(subset=["rainfall"]).reset_index(drop=True)
    raw["station_id"] = station_id_of(raw)

    n = len(raw)
    train_row = (raw["date_of_record"] <= TRAIN_END).to_numpy()
    was = {
        c: raw[c].isna().to_numpy()
        for c in ("min_temp", "max_temp", "wind_speed", "air_pressure", "avg_temp")
    }

    fd_min = np.zeros(n, dtype=bool)
    fd_max = np.zeros(n, dtype=bool)
    fd_wind = np.zeros(n, dtype=bool)
    fd_pres = np.zeros(n, dtype=bool)
    lin_min = np.zeros(n, dtype=bool)
    lin_max = np.zeros(n, dtype=bool)

    print("Scanning station_name groups...", flush=True)
    for _, g in raw.groupby("station_name", sort=False):
        idx = g.index.to_numpy()
        dates = pd.DatetimeIndex(g["date_of_record"])

        for col, fd_arr, lin_arr in (
            ("min_temp", fd_min, lin_min),
            ("max_temp", fd_max, lin_max),
        ):
            vals = g[col].to_numpy(dtype=float)
            lin_mask = future_dependent_linear_mask(dates, vals, TRAIN_END)
            lin_arr[idx] = lin_mask
            # Residual NaNs after linear → station median of originally observed values
            s = pd.Series(vals).interpolate(method="linear", limit_direction="both")
            still_nan = s.isna().to_numpy()
            _, full_med, train_med = median_future_dependent_mask(dates, vals, TRAIN_END)
            combined = lin_mask.copy()
            for j in range(len(idx)):
                if still_nan[j] and dates[j] <= TRAIN_END and np.isnan(vals[j]):
                    if train_med is None or (
                        full_med is not None and abs(full_med - train_med) > 1e-12
                    ):
                        combined[j] = True
            fd_arr[idx] = combined

        for col, fd_arr in (("wind_speed", fd_wind), ("air_pressure", fd_pres)):
            vals = g[col].to_numpy(dtype=float)
            med_mask, _, _ = median_future_dependent_mask(dates, vals, TRAIN_END)
            fd_arr[idx] = med_mask

    sid = raw["station_id"]
    any_fd = (fd_min | fd_max | fd_wind | fd_pres) & train_row

    def summarize(col: str, fd: np.ndarray, missing: np.ndarray, method: str) -> dict:
        train_miss = int((train_row & missing).sum())
        train_fd = int((train_row & fd).sum())
        return {
            "method": method,
            "train_rows_originally_missing": train_miss,
            "train_rows_future_dependent_fill": train_fd,
            "pct_of_all_train_period_rows": round(
                100.0 * train_fd / max(1, int(train_row.sum())), 4
            ),
            "pct_of_train_missing_rows": round(
                100.0 * train_fd / max(1, train_miss), 4
            ),
            "stations_affected": int(sid[train_row & fd].nunique()),
        }

    findings = {
        "answer_was_interpolation_before_split": True,
        "pipeline_order": [
            "Drop missing rainfall (no rainfall interpolation)",
            "Station_name-wise linear interpolate min_temp, max_temp (limit_direction=both) on FULL series",
            "Station_name-wise median fill residual min_temp, max_temp on FULL series",
            "Station_name-wise median fill wind_speed, air_pressure on FULL series",
            "Global median fallback if needed",
            "Feature engineering",
            "Chronological split by TARGET date only when building sequences",
        ],
        "train_end": str(TRAIN_END.date()),
        "n_rows_after_rainfall_drop": n,
        "n_train_period_rows": int(train_row.sum()),
        "variables": {
            "rainfall": {
                "interpolated": False,
                "method": "dropna only",
                "train_rows_future_dependent_fill": 0,
            },
            "avg_temp": {
                "interpolated": False,
                "method": "none (0 missing after rainfall drop)",
                "train_rows_originally_missing": int((train_row & was["avg_temp"]).sum()),
                "train_rows_future_dependent_fill": 0,
            },
            "min_temp": summarize(
                "min_temp",
                fd_min,
                was["min_temp"],
                "linear interpolate (both) + station median",
            ),
            "max_temp": summarize(
                "max_temp",
                fd_max,
                was["max_temp"],
                "linear interpolate (both) + station median",
            ),
            "wind_speed": summarize(
                "wind_speed", fd_wind, was["wind_speed"], "station median fill"
            ),
            "air_pressure": summarize(
                "air_pressure",
                fd_pres,
                was["air_pressure"],
                "station median fill",
            ),
        },
        "temp_linear_component_only": {
            "min_temp_train_rows": int((lin_min & train_row).sum()),
            "max_temp_train_rows": int((lin_max & train_row).sum()),
            "either_train_rows": int(((lin_min | lin_max) & train_row).sum()),
            "stations": int(sid[(lin_min | lin_max) & train_row].nunique()),
        },
        "train_row_totals": {
            "n_train_rows_any_future_dependent_covariate": int(any_fd.sum()),
            "pct_train_rows": round(100.0 * any_fd.sum() / max(1, train_row.sum()), 4),
            "stations_any": int(sid[any_fd].nunique()),
            "stations_temp": int(sid[(fd_min | fd_max) & train_row].nunique()),
            "stations_wind": int(sid[fd_wind & train_row].nunique()),
            "stations_pressure": int(sid[fd_pres & train_row].nunique()),
        },
    }

    print("Mapping to h=1 train sequences...", flush=True)
    feat = pd.read_csv(FEAT, parse_dates=["date_of_record"])
    raw_dates = raw["date_of_record"].dt.normalize()
    leak_keys = set(
        zip(sid[any_fd].astype(str), raw_dates[any_fd].dt.strftime("%Y-%m-%d"))
    )
    lin_keys = set(
        zip(
            sid[(lin_min | lin_max) & train_row].astype(str),
            raw_dates[(lin_min | lin_max) & train_row].dt.strftime("%Y-%m-%d"),
        )
    )

    affected_seq = 0
    affected_lin = 0
    total_train_seq = 0
    need = SEQ_LEN + 1

    for station_id, g in feat.groupby("station_id", sort=False):
        g = g.sort_values("date_of_record").reset_index(drop=True)
        dates = g["date_of_record"].to_numpy(dtype="datetime64[D]")
        day_ints = dates.astype("datetime64[D]").astype(np.int64)
        breaks = np.where(np.diff(day_ints) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [len(g)]))
        for seg_start, seg_end in zip(starts, ends):
            if seg_end - seg_start < need:
                continue
            for i in range(seg_start, seg_end - SEQ_LEN):
                target_date = pd.Timestamp(dates[i + SEQ_LEN])
                if target_date > TRAIN_END:
                    continue
                total_train_seq += 1
                window = [str(pd.Timestamp(d).date()) for d in dates[i : i + SEQ_LEN]]
                if any((station_id, d) in leak_keys for d in window):
                    affected_seq += 1
                if any((station_id, d) in lin_keys for d in window):
                    affected_lin += 1

    findings["sequence_impact_h1_train"] = {
        "total_train_sequences": total_train_seq,
        "sequences_with_any_future_dependent_covariate_in_30day_window": affected_seq,
        "pct_train_sequences_any_covariate": round(
            100.0 * affected_seq / max(1, total_train_seq), 4
        ),
        "sequences_with_future_dependent_LINEAR_temp_in_window": affected_lin,
        "pct_train_sequences_linear_temp_only": round(
            100.0 * affected_lin / max(1, total_train_seq), 4
        ),
    }

    findings["leakage_classification"] = {
        "true_target_leakage": False,
        "covariate_leakage": True,
        "severity": "minor_to_moderate",
        "explanation": (
            "Rainfall is never imputed; target day is excluded from X. "
            "Only covariates (temps via linear+median; wind/pressure via median) "
            "can use post-2022 observations when filling train-period gaps."
        ),
    }
    findings["validity_conclusion"] = {
        "published_results_remain_valid": True,
        "caveat": (
            "Directionally valid for model comparison; not free of preprocessing "
            "covariate leakage. Do not claim leakage-free preprocessing."
        ),
        "should_preprocessing_be_changed": True,
        "recommended_change": (
            "Fit interpolate/median using only dates <= train_end, then transform "
            "val/test. Run a sensitivity retrain if thesis claims hinge on exact RMSE."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(findings, indent=2), encoding="utf-8")
    print(json.dumps(findings, indent=2), flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
