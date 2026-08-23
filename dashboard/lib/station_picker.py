"""Reusable cascading State → District → Station picker."""

from __future__ import annotations

import streamlit as st

from .paths import SS_STATION_ID
from .stations import set_selected_station


def render_cascading_station_picker(
    stations,
    *,
    key_prefix: str,
    show_clear: bool = True,
    title: str | None = "Select by location",
) -> str | None:
    """Render State→District→Station filters; sync selection to session_state.

    Parameters
    ----------
    key_prefix:
        Unique prefix for widget keys so multiple pages can host the picker.
    show_clear:
        Whether to show a Clear selection button.
    title:
        Optional subheader text (None to skip).

    Returns
    -------
    Current selected station_id (or None).
    """
    if title:
        st.subheader(title)

    col_s, col_d, col_st = st.columns(3)
    states = sorted(stations["state"].dropna().unique().tolist())
    current_id = st.session_state.get(SS_STATION_ID)

    cur_state = None
    cur_district = None
    if current_id:
        hit = stations.loc[stations["station_id"] == current_id]
        if not hit.empty:
            cur_state = hit.iloc[0]["state"]
            cur_district = hit.iloc[0]["district"]

    with col_s:
        state_options = ["(all states)"] + states
        state_index = 0
        if cur_state in states:
            state_index = states.index(cur_state) + 1
        state_choice = st.selectbox(
            "State",
            state_options,
            index=state_index,
            key=f"{key_prefix}_state",
        )

    filtered = stations
    if state_choice != "(all states)":
        filtered = filtered[filtered["state"] == state_choice]

    districts = sorted(filtered["district"].dropna().unique().tolist())
    with col_d:
        district_options = ["(all districts)"] + districts
        district_index = 0
        if cur_district in districts:
            district_index = districts.index(cur_district) + 1
        district_choice = st.selectbox(
            "District",
            district_options,
            index=min(district_index, len(district_options) - 1),
            key=f"{key_prefix}_district",
        )

    if district_choice != "(all districts)":
        filtered = filtered[filtered["district"] == district_choice]

    station_rows = filtered.sort_values("station_name")
    if station_rows.empty:
        from .style import render_empty_state

        render_empty_state(
            "No stations match your filters",
            "Try another state or district, or clear the selection and start over.",
        )
        if show_clear:
            c1, _ = st.columns([1, 4])
            with c1:
                if st.button(
                    "Clear selection",
                    use_container_width=True,
                    key=f"{key_prefix}_clear_empty",
                ):
                    set_selected_station(None, stations)
                    st.rerun()
        return st.session_state.get(SS_STATION_ID)

    station_labels = {
        f"{r.station_name} ({r.district})": r.station_id
        for r in station_rows.itertuples()
    }
    label_list = ["(no selection)"] + list(station_labels.keys())
    default_label_idx = 0
    if current_id is not None:
        for i, (lab, sid) in enumerate(station_labels.items(), start=1):
            if sid == current_id:
                default_label_idx = i
                break

    with col_st:
        station_label = st.selectbox(
            "Station",
            label_list,
            index=min(default_label_idx, len(label_list) - 1),
            key=f"{key_prefix}_station",
        )

    if station_label != "(no selection)":
        chosen_id = station_labels[station_label]
        if chosen_id != st.session_state.get(SS_STATION_ID):
            set_selected_station(chosen_id, stations)

    if show_clear:
        c1, _ = st.columns([1, 4])
        with c1:
            if st.button(
                "Clear selection",
                use_container_width=True,
                key=f"{key_prefix}_clear",
            ):
                set_selected_station(None, stations)
                st.rerun()

    return st.session_state.get(SS_STATION_ID)
