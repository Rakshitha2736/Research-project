"""External validity audit: physical/geographic plausibility of Indian weather panel."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"D:\project\Research Project\RainfallPrediction")
FEAT = BASE / "data" / "processed" / "feature_engineered_v2.csv"
OUT = BASE / "reports" / "tables" / "external_validity_audit.json"

# Slightly generous India bbox (includes AN islands / Kashmir ambiguity)
LAT_MIN, LAT_MAX = 6.0, 37.5
LON_MIN, LON_MAX = 68.0, 97.5


def station_meta(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            state=("state", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            elevation=("elevation", "first"),
            n_obs=("rainfall", "size"),
            mean_daily_rain=("rainfall", "mean"),
            total_rain=("rainfall", "sum"),
            date_min=("date_of_record", "min"),
            date_max=("date_of_record", "max"),
        )
    )


def mean_annual_rainfall(df: pd.DataFrame, station_ids: list[str]) -> pd.DataFrame:
    """Mean of calendar-year totals; also scale mean_daily*365.25."""
    sub = df[df["station_id"].isin(station_ids)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["year"] = sub["date_of_record"].dt.year
    yearly = (
        sub.groupby(["station_id", "year"], as_index=False)
        .agg(
            year_total=("rainfall", "sum"),
            n_days=("rainfall", "size"),
            station_name=("station_name", "first"),
            state=("state", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
            elevation=("elevation", "first"),
        )
    )
    yearly["complete_enough"] = yearly["n_days"] >= 180
    rows = []
    for sid, g in yearly.groupby("station_id"):
        g180 = g[g["complete_enough"]]
        rows.append(
            {
                "station_id": sid,
                "station_name": g["station_name"].iloc[0],
                "state": g["state"].iloc[0],
                "latitude": float(g["latitude"].iloc[0]),
                "longitude": float(g["longitude"].iloc[0]),
                "elevation": float(g["elevation"].iloc[0]),
                "mean_annual_all_years": float(g["year_total"].mean()),
                "mean_annual_ge180days": float(g180["year_total"].mean())
                if len(g180)
                else np.nan,
                "n_years": int(g["year"].nunique()),
                "n_years_ge180": int(g180.shape[0]),
            }
        )
    summary = pd.DataFrame(rows)
    daily = (
        sub.groupby("station_id")["rainfall"]
        .mean()
        .rename("mean_daily")
        .reset_index()
    )
    summary = summary.merge(daily, on="station_id")
    summary["scaled_annual_mean_daily_x365"] = summary["mean_daily"] * 365.25
    return summary.sort_values("scaled_annual_mean_daily_x365", ascending=False)


def _longest_identical_calendar_run(
    dates: pd.DatetimeIndex, ra: np.ndarray, rb: np.ndarray
) -> dict:
    """Longest consecutive calendar-day runs with equal rainfall.

    Returns both any-value runs (incl. all zeros) and runs that contain at least
    one non-zero rainfall day (stronger synthetic/copy signal).
    """
    eq = ra == rb
    n_eq = int(eq.sum())
    if n_eq == 0:
        return {
            "max_run_any": 0,
            "max_run_nonzero": 0,
            "best_start_any": None,
            "best_end_any": None,
            "best_start_nz": None,
            "best_end_nz": None,
            "n_eq": 0,
        }
    day = dates.to_numpy().astype("datetime64[D]").astype(np.int64)
    max_any = max_nz = 0
    run = 0
    run_start = None
    run_has_nz = False
    best_any = (None, None)
    best_nz = (None, None)

    def close_run(end_k: int):
        nonlocal max_any, max_nz, best_any, best_nz
        if run <= 0:
            return
        if run > max_any:
            max_any = run
            best_any = (run_start, end_k)
        if run_has_nz and run > max_nz:
            max_nz = run
            best_nz = (run_start, end_k)

    for k in range(len(eq)):
        cont = eq[k] and (run == 0 or day[k] == day[k - 1] + 1)
        if cont:
            if run == 0:
                run_start = k
                run_has_nz = False
            run += 1
            if ra[k] != 0:
                run_has_nz = True
        else:
            close_run(k - 1)
            if eq[k]:
                run = 1
                run_start = k
                run_has_nz = ra[k] != 0
            else:
                run = 0
                run_start = None
                run_has_nz = False
    close_run(len(eq) - 1)
    return {
        "max_run_any": int(max_any),
        "max_run_nonzero": int(max_nz),
        "best_start_any": best_any[0],
        "best_end_any": best_any[1],
        "best_start_nz": best_nz[0],
        "best_end_nz": best_nz[1],
        "n_eq": n_eq,
    }


def find_identical_rainfall_runs(df: pd.DataFrame, min_run: int = 30) -> list[dict]:
    """Find pairs of distinct station_ids with >= min_run consecutive shared dates
    having identical rainfall values.

    Candidate generation (to avoid 85k blind pairs):
    - all pairs within 50 km, OR
    - pairs sharing the same station_name prefix before ' /', OR
    - pairs with identical rounded lat/lon to 1 decimal
    Then verify runs on the full shared date intersection.
    """
    meta = (
        df.groupby("station_id", as_index=False)
        .agg(
            station_name=("station_name", "first"),
            latitude=("latitude", "first"),
            longitude=("longitude", "first"),
        )
        .set_index("station_id")
    )
    maps: dict[str, pd.Series] = {}
    for sid, g in df.groupby("station_id", sort=False):
        s = g.drop_duplicates("date_of_record").set_index("date_of_record")["rainfall"]
        maps[sid] = s.sort_index()

    sids = list(maps.keys())
    lat = meta.loc[sids, "latitude"].to_numpy()
    lon = meta.loc[sids, "longitude"].to_numpy()
    names = meta.loc[sids, "station_name"].astype(str).to_numpy()

    def haversine_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        p = np.pi / 180
        a = (
            np.sin((lat2 - lat1) * p / 2) ** 2
            + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
        )
        return 2 * r * np.arcsin(np.sqrt(a))

    candidates: set[tuple[str, str]] = set()
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            a, b = sids[i], sids[j]
            close = haversine_km(lat[i], lon[i], lat[j], lon[j]) <= 50.0
            same_city = names[i].split(" /")[0].strip().lower() == names[j].split(" /")[
                0
            ].strip().lower()
            same_deg = round(lat[i], 1) == round(lat[j], 1) and round(lon[i], 1) == round(
                lon[j], 1
            )
            if close or same_city or same_deg:
                candidates.add((a, b))

    # Also add a random sample of distant pairs as negative-control scan:
    # hash fingerprint — stations with unusually high exact-match rate on
    # overlapping wet days. Use rainfall signature: count of exact (date,rain)
    # collisions via inverted index for non-zero rain days only.
    from collections import defaultdict

    rain_index: dict[tuple[str, float], list[str]] = defaultdict(list)
    for sid, s in maps.items():
        for dt, val in s.items():
            if val == 0:
                continue
            rain_index[(str(pd.Timestamp(dt).date()), float(val))].append(sid)
    pair_hits: dict[tuple[str, str], int] = defaultdict(int)
    for stations in rain_index.values():
        if len(stations) < 2 or len(stations) > 30:
            continue
        uniq = sorted(set(stations))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                pair_hits[(uniq[i], uniq[j])] += 1
    for (a, b), hits in pair_hits.items():
        if hits >= min_run:
            candidates.add((a, b) if a < b else (b, a))

    print(f"  candidate pairs to verify: {len(candidates)}", flush=True)
    findings: list[dict] = []
    for a, b in candidates:
        sa, sb = maps[a], maps[b]
        common = sa.index.intersection(sb.index)
        if len(common) < min_run:
            continue
        dates = common.sort_values()
        ra = sa.loc[dates].to_numpy()
        rb = sb.loc[dates].to_numpy()
        stats = _longest_identical_calendar_run(dates, ra, rb)
        # Primary suspicious signal: long identical run containing non-zero rain
        if stats["max_run_nonzero"] >= min_run:
            findings.append(
                {
                    "station_a": a,
                    "station_b": b,
                    "max_consecutive_identical_rain_days_nonzero_run": int(
                        stats["max_run_nonzero"]
                    ),
                    "max_consecutive_identical_including_zeros": int(stats["max_run_any"]),
                    "run_start": str(dates[stats["best_start_nz"]].date())
                    if stats["best_start_nz"] is not None
                    else None,
                    "run_end": str(dates[stats["best_end_nz"]].date())
                    if stats["best_end_nz"] is not None
                    else None,
                    "n_shared_dates": int(len(common)),
                    "n_shared_exact_equal_rain": int(stats["n_eq"]),
                    "distance_km": float(
                        haversine_km(
                            meta.loc[a, "latitude"],
                            meta.loc[a, "longitude"],
                            meta.loc[b, "latitude"],
                            meta.loc[b, "longitude"],
                        )
                    ),
                }
            )
    findings.sort(
        key=lambda d: -d["max_consecutive_identical_rain_days_nonzero_run"]
    )
    return findings


def main() -> None:
    print("Loading", FEAT, flush=True)
    df = pd.read_csv(FEAT, parse_dates=["date_of_record"])
    meta = station_meta(df)
    global_mean_daily = float(df["rainfall"].mean())
    results: dict = {"n_rows": len(df), "n_stations": int(meta.shape[0]), "global_mean_daily_mm": global_mean_daily}

    # ---- STEP 1 ----
    print("STEP 1: geographic bounds...", flush=True)
    outside = meta[
        (meta["latitude"] < LAT_MIN)
        | (meta["latitude"] > LAT_MAX)
        | (meta["longitude"] < LON_MIN)
        | (meta["longitude"] > LON_MAX)
    ]
    results["step1_geographic_bounds"] = {
        "bbox_used": {"lat": [LAT_MIN, LAT_MAX], "lon": [LON_MIN, LON_MAX]},
        "n_outside": int(len(outside)),
        "outside_stations": outside[
            ["station_id", "station_name", "state", "latitude", "longitude"]
        ].to_dict(orient="records"),
        "lat_range": [float(meta["latitude"].min()), float(meta["latitude"].max())],
        "lon_range": [float(meta["longitude"].min()), float(meta["longitude"].max())],
        "pass": len(outside) == 0,
    }

    # ---- STEP 2 ----
    print("STEP 2: known extremes...", flush=True)
    # Meghalaya / Cherrapunji region
    mega = meta[
        (meta["state"] == "ML")
        | (
            (meta["latitude"].between(24.8, 25.8))
            & (meta["longitude"].between(91.0, 92.5))
        )
    ]
    # Western Rajasthan desert
    raj_desert = meta[
        (meta["state"] == "RJ")
        & (meta["latitude"].between(25.5, 29.5))
        & (meta["longitude"].between(69.5, 75.5))
    ]
    # Also pull named stations if present
    named_wet = meta[
        meta["station_name"].astype(str).str.contains(
            "Cherrapunji|Mawsynram|Cherra", case=False, na=False
        )
    ]
    named_dry = meta[
        meta["station_name"].astype(str).str.contains(
            "Jaisalmer|Bikaner|Jodhpur|Barmer", case=False, na=False
        )
    ]

    wet_ids = sorted(set(mega["station_id"]) | set(named_wet["station_id"]))
    dry_ids = sorted(set(raj_desert["station_id"]) | set(named_dry["station_id"]))

    wet_stats = mean_annual_rainfall(df, wet_ids) if wet_ids else pd.DataFrame()
    dry_stats = mean_annual_rainfall(df, dry_ids) if dry_ids else pd.DataFrame()

    # Dataset-wide station mean annual (scaled) for comparison
    all_scaled = meta.copy()
    all_scaled["scaled_annual"] = all_scaled["mean_daily_rain"] * 365.25
    dataset_median_scaled_annual = float(all_scaled["scaled_annual"].median())
    dataset_mean_scaled_annual = float(all_scaled["scaled_annual"].mean())

    wet_pass = False
    dry_pass = False
    wet_note = ""
    dry_note = ""
    if len(wet_stats):
        wet_means = wet_stats["scaled_annual_mean_daily_x365"]
        wet_pass = bool((wet_means > dataset_mean_scaled_annual * 1.5).any())
        wet_note = (
            f"Wet-region stations scaled annual mean range "
            f"{float(wet_means.min()):.0f}–{float(wet_means.max()):.0f} mm vs dataset "
            f"mean {dataset_mean_scaled_annual:.0f} mm"
        )
    else:
        wet_note = "No Meghalaya / Cherrapunji-region station found"
        wet_pass = False

    if len(dry_stats):
        dry_means = dry_stats["scaled_annual_mean_daily_x365"]
        dry_pass = bool((dry_means < dataset_mean_scaled_annual * 0.5).any())
        dry_note = (
            f"Dry-region stations scaled annual mean range "
            f"{float(dry_means.min()):.0f}–{float(dry_means.max()):.0f} mm vs dataset "
            f"mean {dataset_mean_scaled_annual:.0f} mm"
        )
    else:
        dry_note = "No western Rajasthan desert station found"
        dry_pass = False

    results["step2_known_extremes"] = {
        "dataset_mean_scaled_annual_mm": dataset_mean_scaled_annual,
        "dataset_median_scaled_annual_mm": dataset_median_scaled_annual,
        "meghalaya_or_cherrapunji_region": wet_stats.round(3).to_dict(orient="records")
        if len(wet_stats)
        else [],
        "rajasthan_desert_region": dry_stats.round(3).to_dict(orient="records")
        if len(dry_stats)
        else [],
        "wet_notably_higher_than_dataset_mean": wet_pass,
        "dry_notably_lower_than_dataset_mean": dry_pass,
        "wet_note": wet_note,
        "dry_note": dry_note,
        "pass": bool(wet_pass and dry_pass),
        "caveat": (
            "Annual totals use observed days only; sparse years understate true annual "
            "rainfall. scaled_annual_mean_daily_x365 = mean_daily * 365.25 is the "
            "primary cross-station intensity metric."
        ),
    }

    # ---- STEP 3 ----
    print("STEP 3: seasonal plausibility...", flush=True)
    monthly = (
        df.groupby(df["date_of_record"].dt.month)["rainfall"]
        .mean()
        .reindex(range(1, 13))
    )
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    by_month = {month_names[m - 1]: float(monthly.loc[m]) for m in range(1, 13)}
    jjas = float(monthly.loc[[6, 7, 8, 9]].mean())
    djf = float(monthly.loc[[12, 1, 2]].mean())
    ratio = jjas / djf if djf > 0 else float("inf")
    results["step3_seasonal"] = {
        "mean_daily_mm_by_month": by_month,
        "jjas_mean_daily_mm": jjas,
        "djf_mean_daily_mm": djf,
        "jjas_over_djf_ratio": ratio,
        "pass": ratio >= 3.0,  # multi-fold threshold
        "note": f"JJAS/DJF ratio = {ratio:.2f}x (pass if >= 3)",
    }

    # ---- STEP 4 ----
    print("STEP 4: temperature vs elevation...", flush=True)
    # 5 highest / 5 lowest elevation stations; compare mean avg_temp
    # Also control for latitude: compare within similar lat bands if possible
    elev_sorted = meta.sort_values("elevation")
    low5 = elev_sorted.head(5)
    # Prefer elev>0 for "low" if many zeros — still use absolute lowest nonzero if possible
    nonzero = elev_sorted[elev_sorted["elevation"] > 0]
    if len(nonzero) >= 5:
        low5 = nonzero.head(5)
    high5 = elev_sorted.tail(5)

    def station_mean_temp(sids):
        sub = df[df["station_id"].isin(sids)]
        return float(sub["avg_temp"].mean())

    high_ids = high5["station_id"].tolist()
    low_ids = low5["station_id"].tolist()
    high_temp = station_mean_temp(high_ids)
    low_temp = station_mean_temp(low_ids)

    # Latitude-controlled: match each high-elev station to low-elev stations within ±3° lat
    controlled = []
    for _, hs in high5.iterrows():
        candidates = meta[
            (meta["elevation"] <= meta["elevation"].quantile(0.2))
            & (meta["latitude"].between(hs["latitude"] - 3, hs["latitude"] + 3))
            & (meta["station_id"] != hs["station_id"])
        ]
        if len(candidates) == 0:
            candidates = meta.nsmallest(20, "elevation")
        # pick closest lat among candidates
        candidates = candidates.copy()
        candidates["lat_dist"] = (candidates["latitude"] - hs["latitude"]).abs()
        pick = candidates.nsmallest(3, "lat_dist")
        ht = station_mean_temp([hs["station_id"]])
        lt = station_mean_temp(pick["station_id"].tolist())
        controlled.append(
            {
                "high_station": hs["station_id"],
                "high_name": hs["station_name"],
                "high_elev": float(hs["elevation"]),
                "high_lat": float(hs["latitude"]),
                "high_mean_avg_temp": ht,
                "low_comparators": pick["station_id"].tolist(),
                "low_mean_avg_temp": lt,
                "high_colder": ht < lt,
                "delta_C": ht - lt,
            }
        )
    controlled_pass = all(c["high_colder"] for c in controlled)

    results["step4_temperature_elevation"] = {
        "highest_elevation_stations": high5[
            ["station_id", "station_name", "state", "elevation", "latitude", "longitude"]
        ].to_dict(orient="records"),
        "lowest_nonzero_elevation_stations": low5[
            ["station_id", "station_name", "state", "elevation", "latitude", "longitude"]
        ].to_dict(orient="records"),
        "mean_avg_temp_high5": high_temp,
        "mean_avg_temp_low5": low_temp,
        "high5_colder_than_low5": high_temp < low_temp,
        "delta_high_minus_low_C": high_temp - low_temp,
        "latitude_controlled_checks": controlled,
        "pass": bool((high_temp < low_temp) and controlled_pass),
    }

    # ---- STEP 5 ----
    print("STEP 5: duplicate rainfall sequences (may take a few minutes)...", flush=True)
    dup_runs = find_identical_rainfall_runs(df, min_run=30)
    # Attach names
    name_map = meta.set_index("station_id")["station_name"].to_dict()
    for r in dup_runs:
        r["name_a"] = name_map.get(r["station_a"])
        r["name_b"] = name_map.get(r["station_b"])

    results["step5_duplicate_sequences"] = {
        "min_run_days": 30,
        "criterion": (
            "FAIL if two distinct station_ids share >=30 consecutive calendar days "
            "with identical rainfall AND the matching run contains at least one "
            "non-zero day (all-zero dry spells are expected and ignored)"
        ),
        "n_pairs_with_ge30_consecutive_identical_nonzero_rain_runs": len(dup_runs),
        "top_pairs": dup_runs[:25],
        "pass": len(dup_runs) == 0,
    }

    # ---- STEP 6 summary ----
    steps = {
        "geographic_bounds": results["step1_geographic_bounds"]["pass"],
        "known_extremes": results["step2_known_extremes"]["pass"],
        "seasonal_monsoon": results["step3_seasonal"]["pass"],
        "temperature_elevation": results["step4_temperature_elevation"]["pass"],
        "no_copied_rainfall_runs": results["step5_duplicate_sequences"]["pass"],
    }
    n_pass = sum(1 for v in steps.values() if v)
    if n_pass == 5:
        verdict = "PASS — appears physically/geographically plausible as Indian weather data"
    elif n_pass >= 3:
        verdict = (
            "MIXED — broadly plausible Indian weather patterns, but one or more "
            "external checks failed and need disclosure"
        )
    else:
        verdict = "FAIL — does not convincingly look like genuine Indian weather geography"

    results["step6_summary"] = {
        "checks": steps,
        "n_passed": n_pass,
        "n_total": 5,
        "overall_verdict": verdict,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Plain-language printout
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDITY AUDIT — PLAIN SUMMARY")
    print("=" * 70)
    print(f"STEP 1 Geographic bounds: {'PASS' if steps['geographic_bounds'] else 'FAIL'}")
    print(
        f"  lat [{results['step1_geographic_bounds']['lat_range'][0]:.3f}, "
        f"{results['step1_geographic_bounds']['lat_range'][1]:.3f}], "
        f"lon [{results['step1_geographic_bounds']['lon_range'][0]:.3f}, "
        f"{results['step1_geographic_bounds']['lon_range'][1]:.3f}], "
        f"outside={results['step1_geographic_bounds']['n_outside']}"
    )
    print(f"STEP 2 Known extremes: {'PASS' if steps['known_extremes'] else 'FAIL'}")
    print(" ", results["step2_known_extremes"]["wet_note"])
    print(" ", results["step2_known_extremes"]["dry_note"])
    if results["step2_known_extremes"]["meghalaya_or_cherrapunji_region"]:
        print("  Wet stations:")
        for r in results["step2_known_extremes"]["meghalaya_or_cherrapunji_region"]:
            print(
                f"    {r['station_name']} ({r['latitude']},{r['longitude']}) "
                f"scaled_ann~={r['scaled_annual_mean_daily_x365']:.0f} mm"
            )
    if results["step2_known_extremes"]["rajasthan_desert_region"]:
        print("  Dry stations:")
        for r in results["step2_known_extremes"]["rajasthan_desert_region"][:12]:
            print(
                f"    {r['station_name']} ({r['latitude']},{r['longitude']}) "
                f"scaled_ann~={r['scaled_annual_mean_daily_x365']:.0f} mm"
            )
    print(f"STEP 3 Seasonal monsoon: {'PASS' if steps['seasonal_monsoon'] else 'FAIL'}")
    print(
        f"  JJAS={results['step3_seasonal']['jjas_mean_daily_mm']:.2f} mm/day, "
        f"DJF={results['step3_seasonal']['djf_mean_daily_mm']:.2f} mm/day, "
        f"ratio={results['step3_seasonal']['jjas_over_djf_ratio']:.2f}x"
    )
    print("  Monthly means:", {k: round(v, 2) for k, v in by_month.items()})
    print(f"STEP 4 Temp vs elevation: {'PASS' if steps['temperature_elevation'] else 'FAIL'}")
    print(
        f"  high5 mean avg_temp={high_temp:.2f}C (elev "
        f"{high5['elevation'].min():.0f}-{high5['elevation'].max():.0f}m), "
        f"low5={low_temp:.2f}C (elev {low5['elevation'].min():.0f}-{low5['elevation'].max():.0f}m)"
    )
    print(
        f"STEP 5 Duplicate rainfall runs: {'PASS' if steps['no_copied_rainfall_runs'] else 'FAIL'}"
    )
    print(
        f"  pairs with >=30 consecutive identical NON-ZERO-containing rain days: {len(dup_runs)}"
    )
    if dup_runs[:5]:
        for r in dup_runs[:5]:
            print(
                f"    {r['name_a']} vs {r['name_b']}: "
                f"{r['max_consecutive_identical_rain_days_nonzero_run']} days "
                f"({r['run_start']}→{r['run_end']}), dist={r['distance_km']:.1f}km"
            )
    print("-" * 70)
    print(f"OVERALL: {verdict}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
