"""Recompute GNN vs LSTM Diebold-Mariano p-values with HAC lag=h-1.

Uses existing seed-42 checkpoints + paired masked predictions (no retraining).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from arima_and_significance import diebold_mariano
from multiseed_multihorizon import paired_preds_h1, paired_preds_multi
from src.cuda_setup import require_cuda
from train_gnn_lstm import build_and_save_adjacency

BASE = Path(__file__).resolve().parent
MODELS = BASE / "models"

EXPECTED = {
    1: 7.18e-08,
    2: 1.279e-25,
    3: 6.900e-32,
    4: 6.975e-40,
}


def main() -> None:
    device = require_cuda()
    adj_path = MODELS / "adjacency_norm.pt"
    if adj_path.exists():
        adjacency = torch.load(adj_path, map_location="cpu", weights_only=False)
    else:
        adjacency = build_and_save_adjacency()

    print("horizon | DM(h=horizon) | DM(h=1 lag0-stale) | paired-t | n | matches_stated?")
    print("-" * 100)
    results = {}
    for h in (1, 2, 3, 4):
        if h == 1:
            y_true, g_pred, l_pred = paired_preds_h1(device, adjacency)
        else:
            y_true, g_pred, l_pred = paired_preds_multi(h, device, adjacency)
        err_g = (y_true - g_pred) ** 2
        err_l = (y_true - l_pred) ** 2
        dm_correct = diebold_mariano(err_g, err_l, h=h)
        dm_stale = diebold_mariano(err_g, err_l, h=1)
        tt_p = float(stats.ttest_rel(err_g, err_l).pvalue)
        stated = EXPECTED[h]
        # relative order-of-magnitude match (stated are approx)
        ratio = dm_correct / stated if stated > 0 else float("nan")
        match = 0.5 <= ratio <= 2.0  # within factor of 2 of stated approx
        results[h] = {
            "dm": dm_correct,
            "dm_stale": dm_stale,
            "tt": tt_p,
            "n": len(y_true),
            "match": match,
            "ratio": ratio,
        }
        print(
            f"h={h} | {dm_correct:.6e} | {dm_stale:.6e} | {tt_p:.6e} | "
            f"n={len(y_true)} | match_stated={match} (ratio={ratio:.3f})"
        )

    # write compact json for patching CSV
    out = BASE / "reports" / "tables" / "gnn_vs_lstm_dm_recomputed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    import json

    json.dump(
        {str(h): {k: (float(v) if not isinstance(v, bool) else v) for k, v in d.items()}
         for h, d in results.items()},
        open(out, "w", encoding="utf-8"),
        indent=2,
    )
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
