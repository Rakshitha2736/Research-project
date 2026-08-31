# Rainfall Prediction using Deep Learning

**University research project** — comparing LSTM, GNN-LSTM, temporal CNN-LSTM (and CNN-LSTM+Attention), and Transformer architectures for multi-horizon daily rainfall forecasting at Indian meteorological stations.

---

## Project Overview

| Item | Detail |
|------|--------|
| Task | Daily rainfall regression (mm/day) |
| Input window | 30 contiguous calendar days → predict day 31 (h=1), day 32 (h=2), … day 34 (h=4) |
| Models | LSTM, GNN-LSTM, CNN-LSTM-Temporal, CNN-LSTM+Attention, Transformer Encoder |
| Seeds | 13, 42, 123 (multi-seed for LSTM, GNN-LSTM, CNN-LSTM-Temporal, CNN-LSTM+Attention) |
| Split | Train ≤ 2022 · Val 2023 · Test 2024–Feb 2025 |
| Evaluation | All metrics reported in mm/day (inverse-transformed) |

### Headline Results (h=1, canonical CUDA+autocast eval)

| Model | RMSE (mean ± std) | Seeds |
|-------|-------------------|-------|
| **LSTM v2** | **9.3745 ± 0.0408** | 3-seed |
| GNN-LSTM | 9.7476 ± 0.0441 | 3-seed |
| Transformer | 9.4355 | seed 42 |
| CNN-LSTM-Temporal | 9.4835 | seed 42 |
| Persistence baseline | 11.5559 | n/a |
| Climatology baseline | 10.7917 | n/a |

**Classical baselines (h=1, full test set):** Verified in Phase 5 — `src/persistence_baseline.py`, `eval_climatology.py`; see `reports/tables/climatology_baseline.csv` and `master_results.csv`. Climatology (per-station, per-season train mean) has **positive R² (0.174)** and **beats persistence** (RMSE 10.79 vs 11.56) but **loses to every DL model** (LSTM 9.37). *Correction:* an earlier ad hoc climatology figure (RMSE ≈ 12.26, R² ≈ −0.07) was cited in project discussion but **never saved as a reproducible script**; it is superseded by the Phase 5 implementation above.

LSTM outperforms GNN-LSTM at all four horizons and all three seeds (12/12 Diebold-Mariano tests; CIs strictly positive).

**Headline (complete 3-seed evidence).** Neither candidate extension — spatial GNN-LSTM nor CNN-LSTM+Attention — produces a reproducible improvement over a plain LSTM baseline. GNN-LSTM is unconditionally worse (LSTM better in all 12 tests). Attention vs temporal CNN-LSTM looked significant at h=2 and h=4 under seed 42 alone; that result is Mixed across three seeds (neither horizon is unanimous). Attention vs LSTM: LSTM is numerically better in 10 of 12 tests (significant in 6). Single-seed significance can reverse (Attention vs LSTM at h=3: seed 42 vs seeds 13/123). See `reports/tables/multiseed_robustness_summary.csv` and FINAL_AUDIT.md.

**Forest-plot note:** `reports/figures/attention_vs_temporal_forest_plot.png` shows seed-42 point estimates only. See `multiseed_robustness_summary.csv` / `.png` for the complete 3-seed picture, which shows this result is not consistent across seeds.

---

## Dataset

- **Source:** India weather/rainfall Excel (`data/raw/india_weather_rainfall_data.xlsx`)
- **Raw rows:** ~970,339 × 15 columns
- **After cleaning:** 712,785 rows, 414 disambiguated stations (`station_id`)
- **Date range:** 2015-01-01 to 2025-02-10
- **Columns:** `date_of_record`, `station_name`, `state`, `district`, `avg_temp`, `min_temp`, `max_temp`, `wind_speed`, `air_pressure`, `elevation`, `latitude`, `longitude`, `rainfall`

---

## Preprocessing

1. **Drop missing rainfall** rows (~257k rows removed)
2. **Station-wise interpolation** for `min_temp`, `max_temp` (linear, bidirectional)
3. **Station-wise median fill** for `wind_speed`, `air_pressure`
4. **Global median fallback** for any remaining NaNs
5. **Station disambiguation:** 406 `station_name` → 414 `station_id` using lat/lon/elevation

---

## Feature Engineering

Eight features used by all models:

| # | Feature | Source |
|---|---------|--------|
| 1 | `avg_temp` | Raw |
| 2 | `min_temp` | Raw (cleaned) |
| 3 | `max_temp` | Raw (cleaned) |
| 4 | `wind_speed` | Raw (cleaned) |
| 5 | `air_pressure` | Raw (cleaned) |
| 6 | `rainfall` | Raw (past days only — target excluded from window) |
| 7 | `doy_sin` | `sin(2π · day_of_year / 366)` |
| 8 | `doy_cos` | `cos(2π · day_of_year / 366)` |

---

## Sequence Generation

- **Window:** 30 contiguous calendar days (no gap-fill; only real observations)
- **Target:** rainfall at day `window_end + h` (h = 1, 2, 3, or 4)
- **No target leakage:** target day is excluded from the input window
- **Scaling:** MinMaxScaler fitted on training data only (both X and y)
- **Split:** chronological by **target date** (train ≤ 2022, val = 2023, test = 2024+)

| Horizon | Train samples | Val samples | Test samples |
|--------:|-------------:|------------:|-------------:|
| 1 | 270,109 | 149,720 | 141,263 |
| 2 | 269,211 | 149,718 | 140,653 |
| 3 | 268,655 | 149,716 | 140,293 |
| 4 | 268,147 | 149,714 | 139,908 |

---

## Models Implemented

### 1. LSTM Baseline (v2)
- 2-layer LSTM (hidden=64), FC → 1 output
- Per-station flat sequences, batch=256, Adam lr=1e-3, early stopping patience=15
- AMP (autocast + GradScaler) for training

### 2. GNN-LSTM
- 2-layer GCN (8→16→32) with per-date masked adjacency (D⁻⁰·⁵ A D⁻⁰·⁵)
- Shared-weight GCN encodes all 414 stations per timestep
- Per-station LSTM (hidden=64) on GCN output → FC → 1
- Graph: 414 nodes, 3,856 edges (distance-based, ≤300 km)
- Batch=1 date/step, patience=30

### 3. CNN-LSTM-Temporal (controlled comparator)
- 2-layer Conv1d over time axis (16→32 channels, kernel=3) + 2-layer LSTM (hidden=64)
- Supplementary ablation; NOT the base paper's spatial CNN (irregular stations prevent 2D grid)
- Seeds 13, 42, 123 at h=1–4

### 4. CNN-LSTM+Attention (primary temporal extension)
- Additive Bahdanau attention over LSTM hidden states on the temporal CNN-LSTM backbone
- Seeds 13, 42, 123 at h=1–4
- Does **not** produce a unanimous improvement over CNN-LSTM-Temporal or over plain LSTM (see Headline above)

### 5. Transformer Encoder (ablation)
- Pre-norm Transformer encoder (d_model=64, nhead=4, 2 layers, GELU, dim_ff=256)
- Learnable positional embeddings, last-timestep FC readout

---

## Evaluation Metrics

All metrics computed in **mm/day** after inverse-transforming scaled predictions:

| Metric | Formula |
|--------|---------|
| **RMSE** | √(mean((y − ŷ)²)) |
| **MAE** | mean(\|y − ŷ\|) |
| **MSE** | mean((y − ŷ)²) |
| **R²** | 1 − SS_res / SS_tot |

Statistical significance: Diebold-Mariano test, paired t-test, bootstrap 95% CI (1000 resamples).

### Classical baselines (Phase 5 — verified)

| Baseline | h=1 RMSE | h=1 R² | Script |
|----------|----------|--------|--------|
| Climatology | 10.7917 | 0.1741 | `eval_climatology.py` |
| Persistence | 11.5559 | 0.0529 | `src/persistence_baseline.py` |

Climatology beats persistence but all DL models beat climatology. *Correction:* an earlier RMSE ≈ 12.26 / R² ≈ −0.07 climatology figure was never a reproducible script — superseded by Phase 5 (`climatology_baseline.csv`).

---

## Multi-Horizon Results (LSTM vs GNN-LSTM, canonical eval)

| Horizon | LSTM RMSE (mean±std) | GNN RMSE (mean±std) | DM p-value | Bootstrap 95% CI |
|--------:|----------------------|---------------------|------------|------------------|
| 1 | 9.3745 ± 0.0408 | 9.7476 ± 0.0441 | 7.18e-08 | (0.1820, 0.3775) |
| 2 | 10.2295 ± 0.0184 | 10.4880 ± 0.0880 | 1.279e-25 | (0.2825, 0.4019) |
| 3 | 10.4892 ± 0.0187 | 10.7174 ± 0.0628 | 6.900e-32 | (0.2250, 0.3081) |
| 4 | 10.5841 ± 0.0178 | 10.9702 ± 0.0326 | 6.975e-40 | (0.3107, 0.4082) |

All confidence intervals strictly positive → GNN RMSE is significantly higher than LSTM at every horizon. This GNN result now has 3-seed confirmation at every horizon (12/12; `multiseed_robustness_summary.csv`) and is the more robustly evidenced of the two negative architectural findings.

---

## Project Structure

```
RainfallPrediction/
├── data/
│   ├── raw/                              # Raw Excel dataset
│   └── processed/                        # Cleaned CSV, .npy arrays, scalers metadata
├── models/                               # .pt checkpoints, MinMax scalers (.joblib)
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_sequence_generation.ipynb
├── reports/
│   ├── figures/                          # Training curves, EDA plots
│   ├── tables/                           # master_results.csv, significance_results.csv, multiseed_robustness_summary.csv
│   └── logs/                             # (reserved for future logs)
├── results/                              # (reserved)
├── src/
│   ├── model.py                          # All model architectures
│   ├── cuda_setup.py                     # GPU/DataLoader helpers
│   └── preprocess.py                     # Path constants, data loading helpers
├── docs/
│   └── PROJECT_REVIEW_LEARNING_GUIDE.md  # Detailed explanatory guide
├── run_pipeline.py                       # End-to-end data pipeline (clean→features→sequences→train)
├── generate_sequences_v2.py              # h=1 sequence builder (v2, 8 features)
├── generate_sequences_multihorizon.py    # h=2,3,4 sequence builder
├── build_station_graph.py                # Station graph construction (edges, adjacency)
├── build_graph_batches.py                # Graph-format batches for GNN (h=1)
├── build_graph_batches_multihorizon.py   # Graph-format batches for GNN (h=2,3,4)
├── train_lstm_baseline_v2.py             # LSTM v2 single-seed training
├── train_lstm_baseline_v2_multiseed.py   # LSTM v2 multi-seed (13, 42, 123)
├── train_lstm_multihorizon.py            # LSTM h=2,3,4 training
├── train_gnn_lstm.py                     # GNN-LSTM h=1 training + evaluation
├── train_gnn_lstm_multihorizon.py        # GNN-LSTM h=2,3,4 training
├── train_cnn_lstm_temporal_h1.py         # temporal CNN-LSTM ablation (h=1)
├── train_transformer_h1.py              # Transformer ablation (h=1)
├── multiseed_multihorizon.py             # Full multi-seed multi-horizon eval + significance
├── multiseed_gnn_significance.py         # GNN significance testing
├── arima_and_significance.py             # ARIMA baseline + DM test implementation
├── audit_temporal_density.py             # Temporal coverage analysis
├── diagnose_dataset.py                   # Dataset diagnostic utility
├── requirements.txt
├── REPRODUCE.md                          # Full reproduction instructions
├── PROJECT_STATUS.md                     # Project completion status
├── PROJECT_VERIFICATION_REPORT.md        # Independent pipeline verification (92/100)
└── FINAL_AUDIT.md                        # Final repository audit
```

---

## How to Run

See [REPRODUCE.md](REPRODUCE.md) for complete step-by-step instructions.

### Quick start

```powershell
cd "D:\project\Research Project\RainfallPrediction"
& "D:\project\Research Project\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& "D:\project\Research Project\.venv\Scripts\python.exe" run_pipeline.py
```

### Train specific models

```powershell
# LSTM v2 multi-seed
python train_lstm_baseline_v2_multiseed.py

# GNN-LSTM (requires graph data)
python train_gnn_lstm.py

# Full multi-horizon multi-seed + significance
python multiseed_multihorizon.py

# Ablation models (h=1 only)
python train_cnn_lstm_temporal_h1.py
python train_transformer_h1.py
```

---

## Requirements

See `requirements.txt`. Key dependencies: pandas, numpy, matplotlib, seaborn, scikit-learn, PyTorch (CUDA build), scipy, statsmodels, joblib, openpyxl.

For GPU training, install a **CUDA** build of PyTorch matching your driver (this project uses `torch 2.13.0+cu126` on RTX 2050).

---

## License / Academic Use

Prepared for university project review. Cite the base paper and dataset source as required by your institution.
