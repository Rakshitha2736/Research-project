"""Station metadata helpers (read-only parquet + zero-test supplements)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .paths import (
    FEATURE_ENGINEERED_V2,
    SS_STATION_ID,
    SS_STATION_NAME,
    STATION_METADATA,
    ZERO_TEST_STATIONS,
)

_META_COMPARE_COLS = (
    "station_id",
    "station_name",
    "latitude",
    "longitude",
    "state",
    "district",
    "elevation",
)


def assert_zero_test_stations_match_source(parquet_ids: set[str]) -> None:
    """Cross-check embedded ZERO_TEST_STATIONS against feature_engineered_v2.csv.

    Ensures:
      1) Exactly those station_ids are in the feature file but missing from parquet
      2) station_name / lat / lon / state / district / elevation match the source
    """
    if not FEATURE_ENGINEERED_V2.exists():
        raise FileNotFoundError(
            f"Cannot cross-check ZERO_TEST_STATIONS: missing {FEATURE_ENGINEERED_V2}"
        )

    feat = pd.read_csv(
        FEATURE_ENGINEERED_V2,
        usecols=list(_META_COMPARE_COLS),
    )
    feat_meta = feat.drop_duplicates(subset=["station_id"]).set_index(
        "station_id", drop=False
    )
    feat_ids = set(feat_meta["station_id"].astype(str))
    expected_missing = feat_ids - set(parquet_ids)
    embedded_ids = {str(s["station_id"]) for s in ZERO_TEST_STATIONS}

    if expected_missing != embedded_ids:
        raise AssertionError(
            "ZERO_TEST_STATIONS drift vs feature_engineered_v2.csv: "
            f"feature−parquet={sorted(expected_missing)} "
            f"embedded={sorted(embedded_ids)}"
        )

    for spec in ZERO_TEST_STATIONS:
        sid = str(spec["station_id"])
        if sid not in feat_meta.index:
            raise AssertionError(
                f"ZERO_TEST_STATIONS station_id not in feature_engineered_v2.csv: {sid}"
            )
        src = feat_meta.loc[sid]
        for col in ("station_name", "state", "district"):
            if str(src[col]) != str(spec[col]):
                raise AssertionError(
                    f"ZERO_TEST_STATIONS[{sid}].{col} mismatch: "
                    f"embedded={spec[col]!r} source={src[col]!r}"
                )
        if int(src["elevation"]) != int(spec["elevation"]):
            raise AssertionError(
                f"ZERO_TEST_STATIONS[{sid}].elevation mismatch: "
                f"embedded={spec['elevation']!r} source={src['elevation']!r}"
            )
        for col in ("latitude", "longitude"):
            if abs(float(src[col]) - float(spec[col])) > 1e-6:
                raise AssertionError(
                    f"ZERO_TEST_STATIONS[{sid}].{col} mismatch: "
                    f"embedded={spec[col]!r} source={src[col]!r}"
                )


@st.cache_data(show_spinner="Loading station metadata…")
def load_stations() -> pd.DataFrame:
    """All 414 stations: 412 from parquet + 2 zero-test supplements."""
    if not STATION_METADATA.exists():
        raise FileNotFoundError(
            f"Missing {STATION_METADATA}. Run build_forecast_cache.py first."
        )
    base = pd.read_parquet(STATION_METADATA)
    assert_zero_test_stations_match_source(set(base["station_id"].astype(str)))
    extra = pd.DataFrame(list(ZERO_TEST_STATIONS))
    # Guard against accidental double-inclusion if parquet is later refreshed
    extra = extra[~extra["station_id"].isin(base["station_id"])]
    df = pd.concat([base, extra], ignore_index=True)
    df["has_test_data"] = df["n_test_samples_available"].fillna(0).astype(int) > 0
    df["availability"] = df["has_test_data"].map(
        {True: "Has test data", False: "No test-period data available"}
    )
    return df.sort_values(["state", "district", "station_name"]).reset_index(drop=True)


def init_selection_state() -> None:
    if SS_STATION_ID not in st.session_state:
        st.session_state[SS_STATION_ID] = None
    if SS_STATION_NAME not in st.session_state:
        st.session_state[SS_STATION_NAME] = None


def set_selected_station(station_id: str | None, stations: pd.DataFrame) -> None:
    if station_id is None or station_id == "":
        st.session_state[SS_STATION_ID] = None
        st.session_state[SS_STATION_NAME] = None
        return
    match = stations.loc[stations["station_id"] == station_id]
    if match.empty:
        st.session_state[SS_STATION_ID] = None
        st.session_state[SS_STATION_NAME] = None
        return
    row = match.iloc[0]
    st.session_state[SS_STATION_ID] = str(row["station_id"])
    st.session_state[SS_STATION_NAME] = str(row["station_name"])


def selected_station_banner(stations: pd.DataFrame) -> None:
    from .ui_components import render_station_pill

    sid = st.session_state.get(SS_STATION_ID)
    if not sid:
        render_station_pill("No station selected yet", empty=True)
        return
    row = stations.loc[stations["station_id"] == sid]
    if row.empty:
        render_station_pill("No station selected yet", empty=True)
        return
    r = row.iloc[0]
    note = "" if r["has_test_data"] else " — no test-period data available"
    render_station_pill(
        f"Selected: {r['station_name']} ({r['state']} / {r['district']}){note}"
    )

                    