"""
Build a symmetrized kNN station adjacency graph (haversine km, k=8).

Saves:
  data/processed/station_graph_edges.csv
  data/processed/station_id_to_index.json

Prints diagnostics only.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "processed"
INPUT_CSV = DATA / "feature_engineered_v2.csv"
EDGES_OUT = DATA / "station_graph_edges.csv"
INDEX_OUT = DATA / "station_id_to_index.json"

K = 8
EARTH_RADIUS_KM = 6371.0
REMOTE_KM = 300.0


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise haversine distance matrix (km). lat/lon in degrees."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def connected_components(n: int, undirected_adj: list[set[int]]) -> list[list[int]]:
    """BFS connected components on an undirected adjacency list."""
    seen = np.zeros(n, dtype=bool)
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        q: deque[int] = deque([start])
        seen[start] = True
        comp = [start]
        while q:
            u = q.popleft()
            for v in undirected_adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps


def main() -> None:
    df = pd.read_csv(INPUT_CSV, usecols=["station_id", "latitude", "longitude"])

    # station_id embeds rounded lat/lon; assert no conflicting coordinates
    nuniq = df.groupby("station_id")[["latitude", "longitude"]].nunique().max().max()
    assert int(nuniq) == 1, f"Inconsistent lat/lon within station_id (max nunique={nuniq})"

    stations = (
        df.drop_duplicates(subset=["station_id"], keep="first")
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    n = len(stations)
    assert n == 414, f"Expected 414 stations, got {n}"

    station_ids = stations["station_id"].tolist()
    lat = stations["latitude"].to_numpy(dtype=np.float64)
    lon = stations["longitude"].to_numpy(dtype=np.float64)

    dist = haversine_km(lat, lon)
    np.fill_diagonal(dist, np.inf)  # exclude self

    # k=8 nearest neighbors (argpartition is O(n) per row)
    knn_idx = np.argpartition(dist, K, axis=1)[:, :K]

    # Directed edges from kNN, then symmetrize by union
    edge_map: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in knn_idx[i]:
            d = float(dist[i, j])
            edge_map[(i, int(j))] = d
            # reverse edge even if j does not list i in its top-k
            edge_map[(int(j), i)] = d

    # Rebuild undirected adjacency for connectivity / nearest-neighbor diagnostics
    undirected: list[set[int]] = [set() for _ in range(n)]
    edge_dists: list[float] = []
    for (i, j), d in edge_map.items():
        undirected[i].add(j)
        edge_dists.append(d)

    edge_dists_arr = np.asarray(edge_dists, dtype=np.float64)
    # Each undirected pair appears twice in directed edge_map; nearest-neighbor
    # distance for station i is min directed distance from i.
    nn_dist = np.full(n, np.inf, dtype=np.float64)
    for (i, j), d in edge_map.items():
        if d < nn_dist[i]:
            nn_dist[i] = d

    remote = [
        (station_ids[i], float(nn_dist[i]))
        for i in range(n)
        if nn_dist[i] > REMOTE_KM
    ]
    remote.sort(key=lambda x: -x[1])

    comps = connected_components(n, undirected)
    comps_sorted = sorted(comps, key=len, reverse=True)

    # Persist artifacts
    rows = [
        {
            "source": station_ids[i],
            "target": station_ids[j],
            "distance_km": d,
        }
        for (i, j), d in sorted(edge_map.items())
    ]
    pd.DataFrame(rows).to_csv(EDGES_OUT, index=False)

    id_to_index = {sid: i for i, sid in enumerate(station_ids)}
    with open(INDEX_OUT, "w", encoding="utf-8") as f:
        json.dump(id_to_index, f, indent=2)

    # --- Diagnostics only ---
    print(f"Total nodes: {n}")
    print(f"Total directed edges (after symmetrization): {len(edge_map)}")
    print(f"Min edge distance (km):    {edge_dists_arr.min():.2f}")
    print(f"Median edge distance (km): {np.median(edge_dists_arr):.2f}")
    print(f"Max edge distance (km):    {edge_dists_arr.max():.2f}")
    print(f"Stations with nearest neighbor > {REMOTE_KM:.0f} km: {len(remote)}")
    if remote:
        for sid, d in remote:
            print(f"  {sid}: {d:.2f} km")
    else:
        print("  (none)")
    print(f"Connected components: {len(comps_sorted)}")
    if len(comps_sorted) == 1:
        print("Graph is one connected component.")
    else:
        print("Isolated subgraphs present. Component sizes:")
        for c in comps_sorted:
            print(f"  size={len(c)}")


if __name__ == "__main__":
    main()
