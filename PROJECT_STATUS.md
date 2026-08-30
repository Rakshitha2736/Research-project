# Project Status

**Project:** Rainfall Prediction using Deep Learning
**Date:** 2026-07-29
**Status:** Implementation phase complete — ready for thesis/paper writing

---

## Completed Modules

| Module | Status | Notes |
|--------|--------|-------|
| Data cleaning | ✅ Complete | 712,785 rows, 414 stations |
| EDA + temporal audit | ✅ Complete | Notebooks 01, 02; audit scripts |
| Feature engineering | ✅ Complete | 8 features, doy_sin/cos, station_id |
| Sequence generation (h=1) | ✅ Complete | v2, 30-day contiguous windows |
| Sequence generation (h=2,3,4) | ✅ Complete | Multi-horizon |
| Graph construction | ✅ Complete | 414 nodes, 3,856 edges |
| Graph batch preparation | ✅ Complete | All horizons |
| LSTM v2 baseline (h=1, multi-seed) | ✅ Complete | Seeds 13, 42, 123 |
| LSTM multi-horizon (h=2,3,4) | ✅ Complete | 3 seeds × 3 horizons |
| GNN-LSTM (h=1, multi-seed) | ✅ Complete | Seeds 13, 42, 123 |
| GNN-LSTM multi-horizon (h=2,3,4) | ✅ Complete | 3 seeds × 3 horizons |
| CNN-LSTM-Temporal ablation (h=1) | ✅ Complete | Seed 42 |
| Transformer ablation (h=1) | ✅ Complete | Seed 42 |
| Statistical significance tests | ✅ Complete | DM, paired-t, bootstrap CI |
| ARIMA baseline | ✅ Complete | 30-station sample |
| Persistence baseline | ✅ Complete | RMSE 11.56 |
| Pipeline automation | ✅ Complete | run_pipeline.py |
| Independent verification | ✅ Complete | 92/100 integrity score |

## Pending Modules

| Module | Status | Priority |
|--------|--------|----------|
| None | — | — |

All implementation modules are complete. No further coding is required.

---

## Project Completion: 100%

All planned experiments have been executed, evaluated, and verified.

---

## Implemented Models

| # | Model | Architecture | Horizons | Seeds |
|---|-------|-------------|----------|-------|
| 1 | LSTM v2 | 2-layer LSTM (64) → FC | h=1,2,3,4 | 13, 42, 123 |
| 2 | GNN-LSTM | 2-layer GCN (8→16→32) + LSTM (64) → FC | h=1,2,3,4 | 13, 42, 123 |
| 3 | CNN-LSTM-Temporal | Conv1d (16→32) + LSTM (64) → FC | h=1 | 42 |
| 4 | Transformer Encoder | Pre-norm encoder (d=64, 4-head, 2-layer) → FC | h=1 | 42 |
| 5 | Persistence | y_pred = last observed rainfall | h=1 | n/a |
| 6 | ARIMA | Rolling 1-step-ahead | h=1 | n/a |

---

## Evaluation Metrics

RMSE, MAE, MSE, R² (all in mm/day). Statistical tests: Diebold-Mariano, paired t-test, bootstrap 95% CI.

---

## Generated Outputs

### Model Checkpoints (30 files)
- `lstm_baseline_v2_seed{13,42,123}.pt` (h=1)
- `lstm_h{2,3,4}_seed{13,42,123}.pt` (9 files)
- `gnn_lstm_seed{13,42,123}.pt` (h=1)
- `gnn_lstm_h{2,3,4}_seed{13,42,123}.pt` (9 files)
- `cnn_lstm_temporal_h1_seed42.pt`
- `transformer_h1_seed42.pt`
- `lstm_baseline_seed42.pt` (v1, historical)

### Scalers (7 files)
- `minmax_scaler_v2.joblib`, `minmax_scaler_y_v2.joblib`
- `minmax_scaler_y_h{2,3,4}.joblib`
- `minmax_scaler.joblib`, `minmax_scaler_y.joblib` (v1)

### Figures (13 files)
- Training curves: LSTM, GNN-LSTM, temporal CNN-LSTM, Transformer
- EDA: correlation matrix, rainfall distribution, monthly/seasonal rainfall
- Station density heatmap, forest plot (GNN vs LSTM)

### Tables
- `reports/tables/master_results.csv` — consolidated results

### Logs
- Monitor logs for multi-seed and multi-horizon runs
- Training logs for LSTM baseline

---

## Research Contribution

1. **Systematic comparison** of LSTM vs GNN-LSTM for Indian rainfall forecasting across 4 forecast horizons
2. **Statistical rigor:** multi-seed evaluation (3 seeds), DM test, paired t-test, bootstrap CI at every horizon
3. **Finding:** Simple per-station LSTM consistently outperforms spatial GNN-LSTM in **all 12** multi-seed tests (4 horizons × 3 seeds). Graph convolution does not add predictive value on this sparse, irregular station network. This is the more robustly evidenced negative architectural result.
4. **Finding:** CNN-LSTM+Attention does not produce a reproducible improvement over CNN-LSTM-Temporal (Mixed at all 4 horizons) or over plain LSTM (LSTM better in 10/12 tests). Seed-42 “significant at h=2/h=4 vs Temporal” does not survive unanimous replication.
5. **Methodological contribution:** single-seed significance testing is materially unreliable here (Attention vs LSTM at h=3 reverses between seed 42 and seeds 13/123). Multi-seed evaluation is a minimum standard for this problem.
6. **Supplementary ablations:** Transformer (h=1, seed 42) confirms LSTM competitiveness; temporal CNN-LSTM is the controlled non-attention comparator (3 seeds, all horizons).

---

## Known Limitations

1. **Irregular station coverage:** stations have varying temporal density (mean ~0.816 coverage)
2. **GNN graph construction:** distance-based edges (≤300 km) may not capture optimal spatial relationships
3. **No spatial CNN:** irregular station layout prevents 2D grid-based spatial convolution (base paper approach)
4. **Single dataset:** results may not generalize to other regions or climate zones
5. **AMP sensitivity:** evaluation metrics vary by ~0.0005 depending on autocast settings
6. **Ablation models:** Transformer remains single-seed / h=1 only; CNN-LSTM-Temporal and CNN-LSTM+Attention are 3-seed at h=1–4. Attention significance claims must be cited from `multiseed_robustness_summary.csv`, not seed-42-only rows.

---

## Future Work

1. Paper-style tables and ablation study write-up
2. Attention visualization for Transformer model
3. Alternative graph construction methods (correlation-based, learned adjacency)
4. Rain/no-rain classification metrics (tolerance accuracy)
5. Ensemble methods combining LSTM variants
6. Extended evaluation on other geographic regions
