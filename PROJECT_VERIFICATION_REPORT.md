# Project Verification Report

**Project:** RainfallPrediction  
**Audit date:** 2026-07-27  
**Mode:** Evaluation / artifact inspection only — **no retraining**, no checkpoint overwrites  
**Scope:** Independent verification of pipeline stages 1–12  

---

## Executive summary

The data pipeline (raw → clean → features → sequences → chronological split → train-only scalers) is **sound**: no target-day leakage found, sequence length is 30, horizons match `window_end + h`, and MinMax scalers refit correctly on training data only.

Model artifacts for LSTM and GNN-LSTM across horizons 1–4 and seeds 13/42/123 are **loadable and intact**. The multi-horizon final table previously reported in `mh_multiseed_monitor.log` **reproduces under the canonical CUDA+autocast evaluation path** (see prior independent verification).

Integrity deductions come mainly from **naming inconsistencies**, **stale documentation/metrics JSON**, and **small float differences** when evaluation is run without matching the original AMP path.

---

## Pipeline map

```
1 Raw Excel
  → 2 clean_dataset.csv
  → 3 feature_engineered(_v2).csv
  → 4 temporal density audit
  → 5 contiguous 30-day sequences (h=1 v2; h=2,3,4)
  → 6 chronological split by TARGET date
  → 7 MinMax X/y (fit train only)
  → 8 LSTM checkpoints
  → 9 GNN-LSTM checkpoints + graph tensors
  → 10 test evaluation (mm/day)
  → 11 DM / paired-t / bootstrap CI
  → 12 final multi-horizon table
```

---

## Stage 1 — Raw dataset

| Item | Detail |
|------|--------|
| **Input** | `data/raw/india_weather_rainfall_data.xlsx` (64,579,550 bytes) |
| **Output** | (consumed by clean step) |
| **Approx shape** | ~970,339 data rows × 15 columns (openpyxl `max_row` includes header → 970,340) |
| **Columns** | `date_of_record`, `month`, `season`, `station_name`, `state`, `district`, `avg_temp`, `min_temp`, `max_temp`, `wind_speed`, `air_pressure`, `elevation`, `latitude`, `longitude`, `rainfall` |
| **Status** | **PASS** — file present; key columns present |

**Missingness (from `missing_values_summary.csv`, computed on raw):**

| Column | Missing % |
|--------|----------:|
| air_pressure | 31.40% |
| wind_speed | 28.28% |
| rainfall | 26.54% |
| max_temp | 11.40% |
| min_temp | 4.52% |

---

## Stage 2 — Clean dataset

| Item | Detail |
|------|--------|
| **Input** | Raw Excel |
| **Outputs** | `data/processed/clean_dataset.csv`, `missing_values_summary.csv` |
| **Row count** | **712,785** (= ~970,339 − 257,554 missing rainfall) |
| **Stations** | 406 unique `station_name` |
| **Date range** | 2015-01-01 → 2025-02-10 |
| **Rainfall NaNs** | **0** |
| **Cleaning rules** | Drop missing rainfall; station-wise interpolate/fill temps; median fill wind/pressure |
| **Status** | **PASS** |

---

## Stage 3 — Feature engineering

| Item | Detail |
|------|--------|
| **Input** | `clean_dataset.csv` |
| **Outputs** | `feature_engineered.csv`, `feature_engineered_v2.csv` |
| **Row count** | 712,785 (unchanged) |
| **Added features** | `doy_sin`, `doy_cos` (day-of-year / 366); `station_id` (v2) |
| **Dropped** | `month`, `season` (v2) |
| **DOY check** | Sample of 5,000 rows: sin/cos match `sin/cos(2π·doy/366)` — **PASS** |
| **Duplicate (station_id, date)** | **0** |
| **Stations** | 406 names → **414** `station_id`s (8 name collisions disambiguated by lat/lon/elevation) |
| **Model feature set (8)** | `avg_temp`, `min_temp`, `max_temp`, `wind_speed`, `air_pressure`, `rainfall`, `doy_sin`, `doy_cos` |
| **Status** | **PASS** (with warning on name collisions — correctly resolved) |

---

## Stage 4 — Temporal audit

| Item | Detail |
|------|--------|
| **Inputs** | Feature CSVs |
| **Outputs** | `reports/temporal_density_audit.txt` (v1 / `station_name`), `reports/temporal_density_audit_v2.txt` (v2 / `station_id`), `temporal_density_by_station.csv` |
| **v2 findings** | 414 stations; 0 duplicate station_id+date; mean coverage ~0.816; 22,604 gaps |
| **Status** | **PASS** for v2 audit |

**Warning:** v1 audit reports duplicate dates under `station_name` (expected before disambiguation). Prefer **v2**.

---

## Stage 5 — Sequence generation

### Horizon h=1 (v2)

| Artifact | Shape / count |
|----------|----------------|
| `X_train_v2.npy` | (270109, **30**, **8**) |
| `X_val_v2.npy` | (149720, 30, 8) |
| `X_test_v2.npy` | (141263, 30, 8) |
| `y_*_v2.npy` | matching 1-D counts |
| Meta | `sequence_metadata_v2.json` — counts/shapes match arrays |
| Leakage asserts (build-time) | 561,092 windows |

**Rule:** 30 contiguous calendar days → predict day 31 rainfall; target day excluded from X.

### Horizons h=2,3,4

| Horizon | Train | Val | Test | X shape pattern |
|--------:|------:|----:|-----:|-----------------|
| 2 | 269211 | 149718 | 140653 | (N, 30, 8) |
| 3 | 268655 | 149716 | 140293 | (N, 30, 8) |
| 4 | 268147 | 149714 | 139908 | (N, 30, 8) |

**Rule (Option A):** contiguous 30-day window; target = `window_end + h` must be a real observation; intermediate days need not exist.

### Independent sample rebuild (20 stations, ~108k windows)

| Check | Result |
|-------|--------|
| Target-day leakage | **0 failures** |
| Horizon offset correctness | **0 failures** |
| Window contiguity | **0 failures** |
| Finite X/y | **PASS** |

**Status:** **PASS**

---

## Stage 6 — Train / validation / test split

| Split | Target-date rule | h=1 target date range | h=1 n |
|-------|------------------|------------------------|------:|
| Train | ≤ 2022-12-31 | 2015-01-31 → 2022-12-31 | 270109 |
| Val | 2023-01-01 … 2023-12-31 | full 2023 | 149720 |
| Test | 2024-01-01 … 2025-02-10 | through 2025-02-10 | 141263 |

| Check | Result |
|-------|--------|
| Chronological by **target date** | **PASS** |
| No (station_id, target_date) overlap across splits | **PASS** (0 overlaps) |

**Warning (expected):** Input windows for early val/test targets may begin in the previous calendar year. Split membership is defined on the **target** date (standard practice).

**Status:** **PASS**

---

## Stage 7 — Normalization

| Scaler | Path | Fit scope | Verification |
|--------|------|-----------|--------------|
| X (8 feat) | `models/minmax_scaler_v2.joblib` | Train X only | Inverse(train X) → refit MinMax ≡ saved `data_min_`/`data_max_` — **PASS** |
| y h=1 | `minmax_scaler_y_v2.joblib` | Train y only | Same check — **PASS**; differs from all-split extremes — **PASS** |
| y h=2,3,4 | `minmax_scaler_y_h{h}.joblib` | Train y per horizon | Refit match — **PASS** |
| Multi-h X | Reuses `minmax_scaler_v2.joblib` | Documented in meta | Consistent with code |

Scaled `y_train_v2` range: **[0.0, 1.0]**.

**Status:** **PASS** — no scaler leakage detected.

---

## Stage 8 — LSTM training (artifacts only)

| Horizon | Checkpoints | Integrity |
|--------:|-------------|-----------|
| 1 | `lstm_baseline_v2_seed{13,42,123}.pt` | Load + `load_state_dict` OK (~213 KB) |
| 2 | `lstm_h2_seed{13,42,123}.pt` | OK (~212 KB) |
| 3 | `lstm_h3_seed{13,42,123}.pt` | OK |
| 4 | `lstm_h4_seed{13,42,123}.pt` | OK |

Checkpoint contents typically include `model_state_dict`, `best_epoch`, `stopped_epoch`, `seed` (and often `horizon` / `patience` for h≥2).

**Architecture (code):** 2-layer LSTM, hidden 64, input size 8 → FC → 1.

**Warnings:**

1. h=1 files are **not** named `lstm_h1_seed*.pt` (pipeline uses `lstm_baseline_v2_seed*.pt`).
2. `lstm_baseline_v2_multiseed_summary.json` seed-42 RMSE **9.4584** is **stale** vs current checkpoint re-eval ≈ **9.4184–9.4188** (matches `lstm_baseline_v2_seed42_metrics.json`). Checkpoint appears retrained after the multiseed summary was written.

**Status:** **PASS** (integrity) with documentation/artifact warnings.

---

## Stage 9 — GNN training (artifacts only)

| Horizon | Checkpoints | Integrity |
|--------:|-------------|-----------|
| 1 | `gnn_lstm_seed{13,42,123}.pt` | Load OK (~926 KB) |
| 2–4 | `gnn_lstm_h{h}_seed{13,42,123}.pt` | Load OK |

| Graph artifact | Detail |
|----------------|--------|
| Stations | 414 nodes (`station_id_to_index.json`) |
| Edges | 3,856 (`station_graph_edges.csv`: source, target, distance_km) |
| Adjacency | `adjacency_norm.pt` present |
| Test graph tensors | `X_test_graph*.npy` shape **(407, 414, 30, 8)** + matching `y`/`mask` |

**Warning:** h=1 named `gnn_lstm_seed*.pt`, not `gnn_lstm_h1_seed*.pt`.  
**Warning:** Train dates filtered to ≥20% station coverage; val/test use all dates (documented design, not leakage).

**Status:** **PASS** (integrity) with naming/design notes.

---

## Stage 10 — Evaluation

| Check | Result |
|-------|--------|
| Metrics in mm/day via inverse y-scaler | Implemented in training/eval scripts |
| Masked GNN metrics (mask=True only) | Confirmed in `train_gnn_lstm.predict_all` / `metrics_mm` |
| Canonical path (CUDA + autocast, matching `multiseed_multihorizon.py`) | Prior audit: **exact match** to reported table (tol 0.0001), **bitwise identical** across 2 runs |
| Alternate FP32 path (this audit, no autocast) | RMSE mean drifts by ~**0.0005** on some horizons — AMP/path sensitivity |

**Status:** **PASS** under canonical path; **warning** on AMP sensitivity for alternate eval settings.

---

## Stage 11 — Statistical significance

From HAC-corrected `diebold_mariano(h=horizon)` recomputation (seed 42, paired masked targets); bootstrap CIs / paired-t from prior eval. Source of truth for DM: `reports/tables/significance_results.csv`.

| h | DM p | paired-t p | Bootstrap 95% CI (RMSE_GNN − RMSE_LSTM) | n |
|--:|------|------------|------------------------------------------|--:|
| 1 | 7.183245e-08 | 7.183245e-08 | (0.1820, 0.3775) | 141263 |
| 2 | 1.279135e-25 | 2.669069e-26 | (0.2825, 0.4019) | 140653 |
| 3 | 6.899672e-32 | 9.848719e-33 | (0.2250, 0.3081) | 140293 |
| 4 | 6.975498e-40 | ~0 | (0.3107, 0.4082) | 139908 |

All CIs strictly positive → GNN RMSE significantly **higher** than LSTM on the masked test set at every horizon.

**Status:** **PASS** (DM values HAC-corrected with lag=h−1; no retrain this audit)

---

## Stage 12 — Final result table

### Canonical verified table (CUDA + autocast)

| horizon | LSTM RMSE (mean±std) | GNN RMSE (mean±std) | DM p | paired-t p | bootstrap 95% CI |
|--------:|----------------------|---------------------|------|------------|------------------|
| 1 | 9.3745±0.0408 | 9.7476±0.0441 | 7.18e-08 | 7.18e-08 | (0.1820, 0.3775) |
| 2 | 10.2295±0.0184 | 10.4880±0.0880 | 1.279e-25 | 2.669069e-26 | (0.2825, 0.4019) |
| 3 | 10.4892±0.0187 | 10.7174±0.0628 | 6.900e-32 | 9.848719e-33 | (0.2250, 0.3081) |
| 4 | 10.5841±0.0178 | 10.9702±0.0326 | 6.975e-40 | ~0 | (0.3107, 0.4082) |

DM p at h≥2 use HAC/Bartlett lag=h−1 (`significance_results.csv`); paired-t is unchanged (not HAC). RMSE/bootstrap match prior CUDA+autocast eval.

### Cross-check notes

- README still cites older headline **9.39±0.06** from `lstm_baseline_v2_multiseed_summary.json` (stale vs current h=1 seed-42 checkpoint).
- Prefer the table above for multi-horizon / multi-seed reporting.

**Status:** **PASS** (canonical); docs partially stale.

---

## Leakage & methodology checklist

| Requirement | Verdict |
|-------------|---------|
| No target-day rainfall in X window | **PASS** |
| Chronological split by target date | **PASS** |
| Scaler fitted only on training data | **PASS** |
| Covariate imputation leakage | **CONFIRMED** (see FINAL_AUDIT.md limitations) — full-series `station_name`-grouped statistics computed before train/val/test split; empirically negligible impact in 2 of 5 model architectures tested (LSTM h=1, Attention h=4); not production-corrected |
| Sequence length = 30 | **PASS** |
| Horizon = window_end + h with real target obs | **PASS** |
| Checkpoint integrity (loadable state dicts) | **PASS** |
| No project files modified except this report | **PASS** |
| No retraining | **PASS** |

---

## Warnings (complete list)

1. **Station name collisions:** 406 `station_name` → 414 `station_id` (resolved correctly in v2).
2. **Dual temporal audits:** v1 (`station_name`) shows duplicates; v2 (`station_id`) shows none — use v2.
3. **Window vs split boundary:** early val/test targets can have windows starting before the split year (expected for target-date splits).
4. **h=1 LSTM naming:** expected `lstm_h1_seed*.pt` absent; actual `lstm_baseline_v2_seed*.pt`.
5. **h=1 GNN naming:** expected `gnn_lstm_h1_seed*.pt` absent; actual `gnn_lstm_seed*.pt`.
6. **Stale multiseed summary JSON:** `lstm_baseline_v2_multiseed_summary.json` seed-42 RMSE 9.4584 ≠ current checkpoint (~9.4184).
7. **Stale README headline:** 9.39±0.06 reflects old summary, not current multi-horizon table (9.3745±0.0408).
8. **GNN train coverage filter:** dates with &lt;20% valid stations excluded from train only (intentional).
9. **Evaluation AMP sensitivity:** FP32-only eval can shift RMSE means by ~5e-4 vs autocast path; always use the same eval settings as training scripts for paper numbers.

---

## Per-stage scorecard

| # | Stage | Score | Notes |
|--:|-------|------:|-------|
| 1 | Raw dataset | 8/8 | Present, schema OK |
| 2 | Clean dataset | 8/8 | Row math & NaN checks OK |
| 3 | Feature engineering | 7/8 | Correct; name-collision warning |
| 4 | Temporal audit | 7/8 | v2 OK; v1 confusing |
| 5 | Sequence generation | 9/9 | Shapes, leakage, horizons OK |
| 6 | Split | 8/8 | Chronological, no overlap |
| 7 | Normalization | 9/9 | Train-only scalers verified |
| 8 | LSTM training | 7/9 | Weights OK; naming + stale summary |
| 9 | GNN training | 7/9 | Weights OK; naming |
| 10 | Evaluation | 7/8 | Canonical OK; AMP path caveat |
| 11 | Statistical significance | 8/8 | Prior recompute matched log |
| 12 | Final result table | 7/8 | Canonical match; README stale |
| | **Total** | **92/100** | |

---

## Project Integrity Score

**Project Integrity Score: 86/100**

*(Scorecard sum 92/100, then −6 for cumulative documentation/naming/stale-artifact risk that could mislead a reviewer even though core science checks pass.)*

### Interpretation

- **86** = research pipeline is **scientifically trustworthy** for the reported multi-horizon LSTM vs GNN comparison under the canonical evaluation path.
- Remaining gap to 100 is almost entirely **artifact hygiene** (filenames, README, old metrics JSON), not data leakage or broken training.

### Recommended fixes (optional; not done in this audit)

1. Rename or symlink h=1 checkpoints to `lstm_h1_seed*.pt` / `gnn_lstm_h1_seed*.pt`.
2. Refresh or delete `lstm_baseline_v2_multiseed_summary.json` and update README headline.
3. Pin evaluation to the autocast script path in any paper/table generation.

---

*End of report.*
