# Rainfall Prediction — Complete Project Review Learning Guide

**Purpose:** Teach you every technical detail of *your* project so you can explain it to your mentor without memorizing scripts.  
**Source of truth:** Your actual `RainfallPrediction/` codebase (locked **v2** baseline).  
**How to use:** Read in order. After each part, say the explanation out loud in your own words.

---

# PART 1 — Project Introduction

## 1.1 Project title

**Simple:**  
**Rainfall Prediction using Deep Learning** — predicting tomorrow’s rainfall (in millimetres) from the last 30 days of weather observations at Indian meteorological stations.

**Technical:**  
A supervised time-series regression problem: given a contiguous 30-day multivariate window of station weather features, predict scalar next-day rainfall (mm/day). Implemented and compared: a 2-layer LSTM baseline, a temporal CNN-LSTM, and a CNN-LSTM+Attention extension (see Part 17 for results).

---

## 1.2 What the project does

**Simple analogy:**  
Imagine each weather station keeps a daily diary: temperature, wind, pressure, rain, etc. You give the model the last 30 diary pages and ask: *“How much rain will fall tomorrow?”*

**What it actually does:**
1. Loads Indian station weather Excel data.
2. Cleans missing values carefully.
3. Builds seasonal features (`doy_sin`, `doy_cos`) and unique station IDs.
4. Cuts **contiguous** 30-day sequences (no fake filled days).
5. Scales features/targets with MinMax (fit on train only).
6. Trains an LSTM to predict day-31 rainfall.
7. Reports RMSE / MAE / R² in real units (mm/day).

**Expected output of one prediction:** one number — rainfall in **mm/day** for the next calendar day at that station.

---

## 1.3 Real-world problem

**Simple:**  
Rainfall is irregular, seasonal, and station-specific. Farmers, cities, and disaster teams need early warning of wet vs dry days. Traditional methods struggle when patterns are nonlinear and depend on recent history.

**Technical problem statement:**  
Map  
\[
X_{t-29:t} \in \mathbb{R}^{30 \times F} \;\rightarrow\; y_{t+1} \in \mathbb{R}
\]  
where \(F=8\) (v2 features), \(y\) is rainfall (mm/day), under chronological train/val/test splits and **no target leakage**.

---

## 1.4 Why rainfall prediction is important

| Stakeholder | Why it matters |
|-------------|----------------|
| Agriculture | Irrigation, sowing, harvest timing |
| Flood / disaster mgmt | Heavy-rain risk awareness (**NOT demonstrated** in this project — no flood-specific validation) |
| Water resources | Reservoir planning |
| Urban planning | Drainage, traffic, public safety |
| Climate science | Understanding local rainfall variability |

Even a modest improvement over naive baselines (e.g. “tomorrow = today”) is scientifically and practically useful.

---

## 1.5 Who benefits

- **You (researcher):** reproducible ML pipeline + paper-ready baseline.
- **Mentors / examiners:** clear methodology; target-date leakage checks and train-only scaling (covariate imputation limitation: see FINAL_AUDIT.md).
- **Downstream users (future):** decision support if models mature.
- **Scientific community:** station-level Indian rainfall DL baseline with honest metrics.

---

## 1.6 Expected output

| Level | Output |
|-------|--------|
| Model | Scalar rainfall prediction (mm/day) |
| Training | Checkpoint `.pt`, training curve PNG |
| Evaluation | MSE, RMSE, MAE, R² on test set |
| Locked result (v2, 3 seeds) | RMSE **9.39 ± 0.06** mm/day, R² **0.375 ± 0.008** |
| Comparison | Persistence baseline RMSE ≈ **11.56** mm/day |

---

## 1.7 Why Deep Learning is used

**Simple:**  
Rainfall depends on complex combinations of recent weather. Deep learning can learn those patterns from data without you hand-writing every rule.

**Technical reasons:**
- Nonlinear relationships (temp × humidity-like proxies × season).
- Temporal dependence (yesterday’s rain informs today).
- Large tabular time series (~700k+ cleaned rows) — enough data for neural nets.
- Extended in this project to CNN-LSTM-Attention (temporal) and GNN-LSTM (spatial) - see Part 17 and the project's GNN evaluation for results.

**Not because “DL is trendy”** — because sequential nonlinear regression fits the data structure.

---

## 1.8 Why LSTM is suitable

**Simple:**  
LSTM is a neural network designed for sequences. It remembers useful past information and forgets noise — like a careful student taking notes over 30 days.

**Technical:**
- Input is ordered: day 1 → day 30.
- Hidden state \(h_t\) and cell state \(c_t\) carry memory across time.
- Gates (forget/input/output) control information flow — good for weather memory over weeks.
- Matches base paper Section 3.3.1 style baseline (scalar target per station).

**Why not only classical ML (Random Forest on flattened days)?**  
Possible as a baseline, but RF does not natively model ordered temporal dependence the way LSTM does. You *did* compare to **persistence** (predict yesterday’s rain).

---

## 1.9 Why the future model is CNN–LSTM–Attention

**Simple story (remember this for viva):**
1. **CNN** looks at short local patterns in the 30-day window (e.g. sudden cooling + pressure drop).
2. **LSTM** models longer temporal evolution across those CNN features.
3. **Attention** learns *which days* mattered most for tomorrow’s rain (monsoon burst day vs dry day).

**Technical motivation:**
- CNN: local temporal filters on multivariate sequences.
- LSTM: sequential dependency modeling.
- Attention: weighted focus over timesteps → better interpretability + often better skill on sparse rain events.

Your current LSTM is the **locked baseline**. CNN-LSTM+Attention (additive Bahdanau attention over LSTM hidden states) is **implemented**, trained across seeds {13,42,123} and horizons h=1–4, and statistically compared against LSTM and CNN-LSTM-Temporal (see FINAL_AUDIT.md / `ablation_study.csv` / `significance_results.csv`). Result: significantly better than CNN-LSTM-Temporal at h=2 and h=4; not shown to outperform plain LSTM.

---

# PART 2 — Overall Project Workflow

## 2.1 Complete workflow (locked v2 path)

```
Raw Excel (india_weather_rainfall_data.xlsx)
        │
        ▼
┌───────────────────┐
│  1. Data Cleaning │  run_pipeline.step_clean / notebook 01
└─────────┬─────────┘
          ▼
  clean_dataset.csv + missing_values_summary.csv
          │
          ▼
┌──────────────────────┐
│ 2. Feature Engineering│  doy_sin/cos, drop month/season, station_id
└──────────┬───────────┘
           ▼
  feature_engineered.csv
  feature_engineered_v2.csv   ← locked input for sequences
           │
           ▼
┌────────────────────┐
│ 3. EDA (notebooks) │  distributions, monthly/season, correlation
└──────────┬─────────┘
           ▼
  reports/figures/*.png
           │
           ▼
┌─────────────────────┐
│ 4. Temporal Audit   │  audit_temporal_density.py
└──────────┬──────────┘
           ▼
  temporal_density_by_station.csv
  reports/temporal_density_audit_v2.txt
           │
           ▼
┌──────────────────────┐
│ 5. Sequence Generation│  generate_sequences_v2.py
│    (contiguous 30-day)│
└──────────┬───────────┘
           ▼
  X_*_v2.npy, y_*_v2.npy
  sequence_metadata_v2.json
  minmax_scaler_v2.joblib
  minmax_scaler_y_v2.joblib
           │
           ▼
┌────────────────────┐
│ 6. Scaling (train) │  already inside seq script; pipeline step_scale_y safety net
└──────────┬─────────┘
           ▼
┌────────────────────┐
│ 7. Model Training  │  train_lstm_baseline_v2.py (+ multiseed)
└──────────┬─────────┘
           ▼
  lstm_baseline_v2_seed*.pt
  metrics JSON + training/pred plots
           │
           ▼
┌────────────────────┐
│ 8. Evaluation      │  inverse-transform y → mm/day metrics
└──────────┬─────────┘
           ▼
┌────────────────────────────┐
│ 9. FUTURE: CNN-LSTM-Attn   │  not implemented yet
└────────────────────────────┘
```

Orchestrator: `python run_pipeline.py` runs steps **clean → features → audit → sequences → scale_y → train**.

---

## 2.2 Stage-by-stage (why / what / I/O)

### Stage 1 — Data Cleaning
| | |
|--|--|
| **Why exists** | Raw Excel has missing rainfall/temps/wind/pressure; models cannot train on NaNs in target. |
| **Necessary?** | Yes — scientific integrity: never invent the target. |
| **Internally** | Parse dates; sort by station+date; drop NaN rainfall; interpolate/fill features station-wise. |
| **Input** | `data/raw/india_weather_rainfall_data.xlsx` |
| **Output** | `data/processed/clean_dataset.csv`, `missing_values_summary.csv` |

### Stage 2 — Feature Engineering
| | |
|--|--|
| **Why** | Month/season as text/categories are weak; cyclical day-of-year encodes seasonality; `station_id` fixes name collisions. |
| **Input** | `clean_dataset.csv` |
| **Output** | `feature_engineered.csv`, `feature_engineered_v2.csv` |

### Stage 3 — EDA
| | |
|--|--|
| **Why** | Understand rainfall skew, seasonality, correlations before modeling. |
| **Input** | cleaned / feature CSVs |
| **Output** | plots under `reports/figures/` |

### Stage 4 — Temporal Audit
| | |
|--|--|
| **Why** | Stations have gaps. Random 30-row slices would cross missing days → fake continuity. |
| **Input** | `feature_engineered_v2.csv` |
| **Output** | audit report + `temporal_density_by_station.csv` |
| **Decision** | Use **contiguous calendar windows only**; do **not** gap-fill for sequences. |

### Stage 5 — Sequence Generation
| | |
|--|--|
| **Why** | LSTM needs tensors `(N, 30, 8)`, not raw CSV rows. |
| **Input** | `feature_engineered_v2.csv` |
| **Output** | `X_{train,val,test}_v2.npy`, `y_*_v2.npy`, scalers, metadata |

### Stage 6 — Scaling
| | |
|--|--|
| **Why** | Features have different units (°C, m/s, hPa, mm). MinMax helps Adam converge. |
| **Critical rule** | Fit scaler on **train only**; transform val/test. |

### Stage 7 — Training
| | |
|--|--|
| **Why** | Learn mapping from 30-day windows → next-day rain. |
| **Input** | NPY + y-scaler |
| **Output** | `.pt` checkpoint, metrics, figures |

### Stage 8 — Prediction & Evaluation
| | |
|--|--|
| **Why** | Scaled MSE is not human-interpretable; report mm/day. |
| **How** | `inverse_transform` predictions and true y, then RMSE/MAE/R². |

### Stage 9 — CNN-LSTM+Attention (implemented)
| | |
|--|--|
| **Why** | Improve local pattern extraction + focus on informative days. |
| **Status** | Implemented: seeds {13,42,123}, horizons h=1–4; significantly better than CNN-LSTM-Temporal at h=2 and h=4; not shown to outperform plain LSTM (see `ablation_study.csv` / `significance_results.csv`). |

---

# PART 3 — Complete Folder Structure

```
RainfallPrediction/
├── data/
│   ├── raw/              # original Excel only
│   ├── processed/        # cleaned CSVs, NPY, metadata
│   └── external/         # placeholder for future external data
├── models/               # scalers (.joblib) + checkpoints (.pt) + metrics JSON
├── reports/
│   └── figures/          # EDA + training plots
├── results/              # placeholder (mostly empty)
├── notebooks/            # exploratory / reproducible analysis
├── src/                  # reusable Python modules
├── docs/                 # this learning guide
├── run_pipeline.py       # orchestrator
├── generate_sequences*.py
├── train_lstm_baseline*.py
├── audit_temporal_density.py
├── diagnose_dataset.py
├── requirements.txt
└── README.md
```

## 3.1 `data/raw/`
| Question | Answer |
|----------|--------|
| Why exists | Immutable source of truth — never overwrite raw data. |
| Files | `india_weather_rainfall_data.xlsx` (~65 MB) |
| Created when | You place the Excel file manually |
| Consumed by | `run_pipeline.step_clean`, notebook 01, `diagnose_dataset.py` |

## 3.2 `data/processed/`
| Question | Answer |
|----------|--------|
| Why | Intermediate scientific artifacts: clean tables → model-ready tensors |
| Key files | `clean_dataset.csv`, `feature_engineered(_v2).csv`, `X_*_v2.npy`, `y_*_v2.npy`, `sequence_metadata_v2.json`, `temporal_density_by_station.csv`, `missing_values_summary.csv` |
| Created by | pipeline / seq scripts / audit |
| Consumed by | later pipeline steps, training, notebooks |

## 3.3 `models/`
| Question | Answer |
|----------|--------|
| Why | Persist fitted scalers and trained weights for reproducibility |
| Files | `minmax_scaler_v2.joblib`, `minmax_scaler_y_v2.joblib`, `lstm_baseline_v2_seed{13,42,123}.pt`, `*_metrics.json`, multiseed summary |
| Created by | `generate_sequences_v2.py`, train scripts |
| Consumed by | evaluation, re-inference, reports |

## 3.4 `reports/` (+ `figures/`)
| Question | Answer |
|----------|--------|
| Why | Human-readable audit logs and plots for review slides |
| Files | temporal audit txt, train logs, PNGs |
| Created by | audit, EDA notebook, train scripts, pipeline |

## 3.5 `results/`
Placeholder for future aggregated experiment tables. Currently mostly empty (`.gitkeep`). Metrics today live mainly under `models/` JSON.

## 3.6 `src/`
Reusable library code (not “run once” scripts):
- `model.py` — `LSTMBaseline`
- `cuda_setup.py` — GPU, seed, DataLoader helpers
- `preprocess.py` — path helpers / light utilities for notebooks

## 3.7 `notebooks/`
Interactive exploration and documentation of decisions. Production path is scripts + `run_pipeline.py`.

## 3.8 Root scripts
Executable entry points for the scientific pipeline (see Part 4).

**Architecture mental model:**  
`raw → processed (tables) → processed (tensors) → models (weights) → reports (evidence)`.

---

# PART 4 — Explain Every File

## 4.1 `run_pipeline.py`

| Aspect | Detail |
|--------|--------|
| **Purpose** | One-command end-to-end orchestration of the locked workflow |
| **When** | Whenever you rebuild from raw or from a mid-step (`--from sequences`) |
| **Input** | Raw Excel (+ existing artifacts if skipping) |
| **Output** | All intermediate + final artifacts for v2 |
| **Key functions** | `step_clean`, `step_features`, `step_audit`, `step_sequences`, `step_scale_y`, `step_train`, `_resolve_python`, `run_script` |
| **Important variables** | `STEPS`, `RAW`, `CLEAN_CSV`, `FEAT_V2_CSV`, force/skip flags |
| **Connections** | Calls cleaning inline; features inline; audit via subprocess; sequences via `generate_sequences_v2.py`; train via `train_lstm_baseline_v2.py` |
| **If removed** | You can still run each script manually, but lose a safe, ordered, skip/force workflow — easy to forget a step or use CPU Python |

Flags: `--from`, `--force`, `--skip-train`, `--skip-audit`.

---

## 4.2 `generate_sequences.py` (v1)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Build 30-day sequences with **7 features** (no past rainfall in X) |
| **When** | Legacy / ablation comparison |
| **Input** | `feature_engineered_v2.csv` |
| **Output** | `X_*.npy`, `y_*.npy`, `minmax_scaler.joblib`, `sequence_metadata.json` |
| **Why still present** | Historical baseline; prove adding past rainfall (v2) helps |
| **If removed** | v1 experiments disappear; locked v2 still works |

---

## 4.3 `generate_sequences_v2.py` (LOCKED)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Build contiguous sequences with **8 features including past rainfall** |
| **When** | After feature engineering; before training |
| **Input** | `feature_engineered_v2.csv` |
| **Output** | `X_*_v2.npy`, `y_*_v2.npy`, `minmax_scaler_v2.joblib`, `minmax_scaler_y_v2.joblib`, `sequence_metadata_v2.json` |
| **Key functions** | `split_name`, `build_sequences`, `stack_split`, `scale_X`, `scale_y`, `main` |
| **Important constants** | `SEQ_LEN=30`, `FEATURE_COLS` (8), chronological split dates |
| **Safety** | 561,092 leakage assertions passed |
| **If removed** | Cannot create/rebuild v2 tensors — training breaks |

---

## 4.4 `audit_temporal_density.py`

| Aspect | Detail |
|--------|--------|
| **Purpose** | Quantify coverage, gaps, feasibility of 31 contiguous days per station |
| **When** | After features; before trusting sequences |
| **Input** | feature CSV (prefers `station_id`) |
| **Output** | stdout report; `temporal_density_by_station.csv`; pipeline saves `reports/temporal_density_audit_v2.txt` |
| **Key ideas** | coverage = observed_days / calendar_span; gap size distribution |
| **If removed** | You lose evidence for “why contiguous-only windows” — mentors will ask |

---

## 4.5 `diagnose_dataset.py`

| Aspect | Detail |
|--------|--------|
| **Purpose** | Quick print diagnostics (shape, missing, stations, rainfall stats) |
| **When** | Debugging data issues |
| **Input** | any CSV/Excel path argument |
| **Output** | console only (no files) |
| **If removed** | Convenience loss only |

---

## 4.6 `train_lstm_baseline.py` (v1)

Trains LSTM on 7-feature tensors; seed 42; writes `lstm_baseline_seed42.pt` + metrics + plots.  
**If removed:** lose v1 comparison numbers (RMSE ~9.93).

---

## 4.7 `train_lstm_baseline_v2.py` (LOCKED single-seed)

| Aspect | Detail |
|--------|--------|
| **Purpose** | Train 8-feature LSTM on CUDA, seed 42 |
| **Input** | `X_*_v2.npy`, `y_*_v2.npy`, `minmax_scaler_y_v2.joblib` |
| **Output** | `lstm_baseline_v2_seed42.pt`, metrics JSON, training curve, pred-vs-actual PNG |
| **Hyperparams** | Adam 1e-3, batch 256, max 100 epochs, patience 15, grad clip 1.0, AMP |
| **If removed** | Primary training entrypoint gone (multiseed still exists) |

---

## 4.8 `train_lstm_baseline_v2_multiseed.py`

Runs seeds **13, 42, 123**; writes per-seed checkpoints/metrics + `lstm_baseline_v2_multiseed_summary.json`.  
**Why needed:** examiners ask “is seed 42 luck?” — you answer with mean±std.

---

## 4.9 `src/model.py`

Defines `LSTMBaseline`:
```text
Input (B, 30, F) → LSTM(2 layers, hidden=64) → last timestep → Linear(64→1) → (B,)
```
Default `input_size=7` for historical reasons; **v2 passes `input_size=8`**.  
**If removed:** both train scripts fail.

---

## 4.10 `src/cuda_setup.py`

| Helpers | Role |
|---------|------|
| `require_cuda()` | Fail fast if no GPU (your train scripts require CUDA) |
| `set_seed()` | Reproducibility |
| `make_loader()` | TensorDataset + DataLoader (batch 256, workers 0) |
| `to_device()` | Move batch to GPU |
| Diagnostics | Print GPU name/memory |

**If removed:** duplicated boilerplate / risk of silent CPU training.

---

## 4.11 `src/preprocess.py`

Path constants and light helpers for notebooks. **Full cleaning logic lives in `run_pipeline.step_clean` / notebook 01**, not only here.  
**If removed:** notebooks that import it may break; pipeline still works.

---

## 4.12 `src/__init__.py`

Marks `src` as a package so `from src.model import LSTMBaseline` works.

---

## 4.13 Notebooks

| Notebook | Purpose | Creates | Consumes |
|----------|---------|---------|----------|
| `01_Data_Preprocessing.ipynb` | Document cleaning decisions | `clean_dataset.csv` | raw Excel |
| `02_EDA.ipynb` | Distributions, monthly/season, correlation | figure PNGs | clean/feature data |
| `04_feature_engineering.ipynb` | doy_sin/cos, drop month/season, station_id | feature CSVs | clean CSV |
| `05_sequence_generation.ipynb` | Wrapper calling v2 sequence script | (via script) NPY | feature_v2 |

Note: **03 is intentionally missing** (numbering history). Say this calmly if asked.

---

## 4.14 `README.md` / `requirements.txt` / `.gitignore`

- README: project contract for humans (metrics, how to run, future work).  
- requirements: library deps.  
- gitignore: large binaries (csv/npy/pt) often not in git — regenerate locally.

---

# PART 5 — Dataset Explanation

## 5.1 Source & size

| Property | Value |
|----------|-------|
| File | `data/raw/india_weather_rainfall_data.xlsx` |
| Approx raw rows | ~**970,339** |
| After cleaning | **712,785** |
| Date range | **2015-01-01 → 2025-02-10** |
| Unique `station_name` | **406** |
| Unique `station_id` | **414** (8 collisions disambiguated) |
| Rainfall unit | **mm/day** |
| Rainfall range (clean) | 0.0 → ~485.9 mm/day; mean ~5.27 |

## 5.2 Columns (raw / cleaned)

| Column | Type | Role | Meaning |
|--------|------|------|---------|
| `date_of_record` | datetime | Time index | Observation day |
| `month` | categorical/int | Dropped later | Calendar month |
| `season` | categorical | Dropped later | Season label |
| `station_name` | categorical | Metadata | Station name (can collide) |
| `state` | categorical | Metadata | Indian state |
| `district` | categorical | Metadata | District |
| `avg_temp` | numeric | **Input feature** | Average temperature |
| `min_temp` | numeric | **Input feature** | Min temperature |
| `max_temp` | numeric | **Input feature** | Max temperature |
| `wind_speed` | numeric | **Input feature** | Wind speed |
| `air_pressure` | numeric | **Input feature** | Air pressure |
| `elevation` | numeric | Metadata / ID | Station elevation |
| `latitude` | numeric | Metadata / ID | Latitude |
| `longitude` | numeric | Metadata / ID | Longitude |
| `rainfall` | numeric | **TARGET** (+ past in X for v2) | Daily rainfall mm |

**Target column:** `rainfall` (next day).  
**Model input columns (v2):**  
`avg_temp, min_temp, max_temp, wind_speed, air_pressure, rainfall, doy_sin, doy_cos`  
(on days 1–30 only).

**Not fed to LSTM as raw features:** state, district, name, lat/lon/elevation (used for `station_id` / future spatial models).

## 5.3 Missing values (raw, approx)

| Column | ~Missing % |
|--------|------------|
| air_pressure | ~31.4% |
| wind_speed | ~28.3% |
| rainfall | ~26.5% |
| max_temp | ~11.4% |
| min_temp | ~4.5% |

## 5.4 Why rainfall-missing rows were removed

**Simple:** You cannot invent the answer key for the exam.  
**Technical:** Target imputation leaks synthetic labels into supervised learning → biased metrics. Dropping NaN rainfall is the correct scientific choice.

## 5.5 Why interpolation for temperatures

Temps change smoothly day-to-day. Linear interpolation within a **station** preserves local climate trajectory better than global mean fill.

## 5.6 Why station-wise (not global) imputation

Delhi in June ≠ Shimla in June. Filling Delhi gaps with India-wide median would inject wrong climate. Station-wise respects local climatology.

## 5.7 Why wind/pressure used median (not always interpolate)

Sparser / noisier; median is robust. Pipeline: station median → global median fallback.

---

# PART 6 — Data Cleaning (deep dive)

## 6.1 Exact cleaning recipe (`run_pipeline.step_clean`)

1. Load Excel → DataFrame.  
2. Write missing-value summary CSV.  
3. `date_of_record = to_datetime(...)`.  
4. Sort by `station_name`, `date_of_record`.  
5. **Drop rows where `rainfall` is NaN.**  
6. For `min_temp`, `max_temp`:  
   - station-wise `interpolate(linear, limit_direction="both")`  
   - then station-wise median fill  
7. For `wind_speed`, `air_pressure`: station-wise median fill.  
8. Global median fallback for any remaining NaNs in those four columns.  
9. Re-sort; save `clean_dataset.csv`.

## 6.2 Decision table (method / why / alt / tradeoff)

| Decision | Why chosen | Alternative | Advantage | Disadvantage |
|----------|------------|-------------|-----------|--------------|
| Drop missing rainfall | Keep labels honest | Impute target | No fake labels | Lose ~26% rows |
| Linear interp temps | Smooth physical variable | Forward fill | Uses both neighbors | Bad across long gaps |
| Station-wise fills | Local climate | Global fill | Less bias | Needs enough station data |
| Median for wind/pressure | Robust to outliers | Mean / KNN | Stable | Ignores temporal structure |
| Sort by station+date | Required for time ops | None | Correct sequences | Must be done every load |

**Duplicate handling:** Later `station_id` construction asserts no duplicate `(station_id, date)` groups. Temporal audit also checks duplicates.

**Date conversion:** Without datetime, day-of-year encoding and contiguous calendar checks fail.

---

# PART 7 — Feature Engineering

## 7.1 Features created

### `doy_sin` and `doy_cos`

**Simple:**  
Seasons are a circle (Dec is close to Jan). Numbers 1…12 make Dec far from Jan. Sine+cosine place days on a circle.

**Math:**
\[
\text{doy\_sin} = \sin\left(2\pi \cdot \frac{\text{day\_of\_year}}{366}\right),\quad
\text{doy\_cos} = \cos\left(2\pi \cdot \frac{\text{day\_of\_year}}{366}\right)
\]

**Graphically in words:**  
Imagine a clock for the year. Day-of-year is the angle. `sin` is the “north-south” coordinate; `cos` is “east-west”. Together they uniquely encode position on the seasonal cycle. LSTM sees continuous seasonal signal without jumps at year boundaries.

**Why 366?** Leap-year-safe denominator used in your code.

### `station_id`

```text
station_id = "{station_name}_{lat:.2f}_{lon:.2f}_{elevation:int}"
```

**Why:** Some `station_name` values collide across locations (406 names → 414 IDs). Without this, sequences from different places get mixed → garbage temporal series.

## 7.2 Why month and season were removed

Once cyclical DOY exists, month/season are **redundant** categorical encodings that:
- create artificial jumps (Dec→Jan),
- waste capacity,
- risk inconsistent encoding.

Code: `df.drop(columns=["month", "season"], errors="ignore")`.

## 7.3 Why cyclical encoding helps LSTM

LSTM learns continuous transitions: monsoon onset is a smooth movement on the (sin, cos) plane, not a hard category switch. This improves generalization across years.

## 7.4 Two output CSVs

| File | Contents |
|------|----------|
| `feature_engineered.csv` | + doy_sin/cos; month/season dropped; **no** station_id |
| `feature_engineered_v2.csv` | same + **station_id** |

Sequences always read **v2**.

---

# PART 8 — Temporal Audit

## 8.1 Why temporal audit is necessary

**Simple:** If a station has missing days, you cannot pretend Jan 1–10 and Jan 20–30 are one continuous 30-day story.

**Technical:** Sequence models assume regularly spaced timesteps. Your data is daily but **coverage is incomplete**. Audit measures real calendar continuity.

Without audit you might fill gaps with invented weather, or sample random 30 rows that skip days — both invalid for a daily LSTM.

## 8.2 What is temporal density?

\[
\text{coverage} = \frac{\text{observed days}}{\text{calendar span days}}
\]

Example: 300 observations over 365 days → coverage ≈ 0.82. Your audit found mean coverage ~**0.816**.

## 8.3 Missing days / gaps

A **gap** is where consecutive records are not consecutive calendar days (`diff(day) != 1`). Audit reports gap counts, size distribution, and worst stations (max gap ~1047 days).

## 8.4 Why contiguous sequences matter

LSTM timestep \(t\) and \(t+1\) must mean “next calendar day,” not “next available row.” Code splits each station into contiguous **segments**, then slides windows only inside segments.

## 8.5 Why random sampling is wrong

Random 30 rows can be non-consecutive, mix seasons wrongly, and destroy the meaning of past rainfall.

## 8.6 Why leakage is dangerous

**Leakage** = seeing information unavailable at prediction time (target day in X, scaling with test stats, random future-in-train splits).

Your v2 script asserts: (1) target date ∉ window, (2) window_end + 1 day = target. **561,092** checks passed.

## 8.7 Reports generated

| Artifact | Meaning |
|----------|---------|
| `reports/temporal_density_audit.txt` | Earlier audit |
| `reports/temporal_density_audit_v2.txt` | Audit on feature_v2 |
| `temporal_density_by_station.csv` | Per-station coverage/gaps |

**Chosen policy:** contiguous-only, **no reindex-and-fill** for model inputs.

---

# PART 9 — Sequence Generation

## 9.1 Why needed

CSV rows are flat. LSTM needs \(X \in \mathbb{R}^{N \times 30 \times 8}\), \(y \in \mathbb{R}^{N}\).

## 9.2 Sequence length 30 → predict day 31

About one month of history. Common meteorological short window; matches paper-style setup; long enough for regimes, short enough for continuity. Need = **31 consecutive calendar days**.

## 9.3 Sliding window example

Contiguous days D1…D40:
```
X=D1..D30 → y=D31
X=D2..D31 → y=D32
...
X=D10..D39 → y=D40
```
A gap ends the segment; no window crosses it.

## 9.4 Chronological split (by TARGET date)

| Split | Target dates | Count |
|-------|--------------|-------|
| Train | ≤ 2022-12-31 | 270,109 |
| Val | 2023-01-01 … 2023-12-31 | 149,720 |
| Test | 2024-01-01 … 2025-02-10 | 141,263 |

Stations used: **414**. Shapes: `X_train_v2` = (270109, 30, 8).

## 9.5 Why chronological not random

Prevents future climate from leaking into training → fake optimistic scores.

## 9.6 Leakage prevention checklist

1. Contiguous windows only  
2. Target day excluded from X (asserted)  
3. Past rainfall in X = days 1–30 only  
4. Scalers fit on train only  
5. Split by target date  

## 9.7 NPY files

`X_{train,val,test}_v2.npy`, `y_{train,val,test}_v2.npy` (scaled), plus `sequence_metadata_v2.json`. v1 equivalents exist without `_v2` (7 features).

---

# PART 10 — Scaling

## 10.1 Why necessary

Features have different units (°C, hPa, mm, sin/cos). Without scaling, large-magnitude features dominate gradients.

## 10.2 Why MinMaxScaler

\[
x' = (x - x_{min}) / (x_{max} - x_{min})
\]
Maps toward [0,1]; good default for bounded weather variables.

## 10.3 Train-only fitting

**fit** learns min/max from train. **transform** applies to val/test. Fitting on all data leaks test extremes into preprocessing.

## 10.4 fit / transform / fit_transform

| Method | Role |
|--------|------|
| fit | Learn parameters |
| transform | Apply parameters |
| fit_transform | fit+transform (train only) |

## 10.5 Inverse transform

Training uses scaled y. Report metrics in mm:
`scaler_y.inverse_transform(pred.reshape(-1,1))`.

## 10.6 Why scale the target

Stabilizes MSE optimization; evaluate after inverse transform. Files: `minmax_scaler_v2.joblib` (X), `minmax_scaler_y_v2.joblib` (y).

---

# PART 11 — Current Model (LSTM Baseline)

## 11.1 Architecture (your code)

```text
x: (batch, 30, 8)
        │
        ▼
   nn.LSTM(input_size=8, hidden_size=64, num_layers=2, batch_first=True, dropout=0.0)
        │
        ▼
   out: (batch, 30, 64)  →  take last timestep out[:, -1, :]  →  (batch, 64)
        │
        ▼
   nn.Linear(64 → 1)  →  prediction (batch,)   # scaled rainfall
```

Class: `LSTMBaseline` in `src/model.py`.

## 11.2 Concepts — simple then technical

| Concept | Simple | Technical |
|---------|--------|-----------|
| Input tensor | 30 days × 8 numbers, many samples | `(B, 30, 8)` FloatTensor |
| Hidden size 64 | Width of LSTM “notebook” | \(h_t \in \mathbb{R}^{64}\) |
| Hidden state \(h_t\) | Summary exposed at time t | Output representation |
| Cell state \(c_t\) | Longer memory highway | Controlled by gates |
| Forget/input/output gates | What to erase / write / show | Standard LSTM equations |
| Output layer | Map memory → one rain number | Linear 64→1 |

## 11.3 Training controls (your settings)

| Item | Value | Meaning |
|------|-------|---------|
| Loss | MSELoss | On scaled y |
| Optimizer | Adam, lr=1e-3 | Adaptive updates |
| Max epochs | 100 | Full passes over train |
| Batch | 256 | Samples per step |
| Early stopping | patience 15, min_delta 1e-5 | Stop if val MSE stalls |
| Grad clip | 1.0 | Stop exploding gradients |
| AMP | GradScaler + autocast | Mixed precision on CUDA |
| Checkpoint | best val weights | Saved in `.pt` |

## 11.4 Forward / backward intuition

1. Forward: X → LSTM → pred → loss vs y  
2. Backward: BPTT gradients  
3. Clip grads → Adam step  

## 11.5 Why this baseline is honest

Simple, reproducible, multi-seeded, and used as the controlled baseline for fair comparison against CNN-LSTM-Attention and GNN-LSTM (see Part 17 and FINAL_AUDIT.md for results).

---

# PART 12 — Training Process (`train_lstm_baseline_v2.py`)

When you run `python train_lstm_baseline_v2.py`:

### A. Setup
1. `set_seed(42)`  
2. `require_cuda()` — GPU required  
3. Load `X_*_v2.npy`, `y_*_v2.npy`  
4. Load `minmax_scaler_y_v2.joblib`  
5. Assert 8 features  

### B. DataLoaders
Train shuffle=True; val/test shuffle=False; batch 256 via `make_loader`.

### C. Model + opt
```python
model = LSTMBaseline(input_size=8, hidden_size=64, num_layers=2).to(device)
optimizer = Adam(lr=1e-3)
criterion = MSELoss()
scaler = GradScaler("cuda")
```

### D. Each epoch (per batch)
1. Move xb, yb to GPU  
2. zero_grad  
3. autocast forward → pred  
4. MSE loss  
5. scaled backward → unscale → clip_grad_norm_(1.0)  
6. optimizer step / scaler update  
7. Accumulate train loss → compute val loss  
8. If val improved: save best state; else patience++; stop at 15  

### E. After loop
1. Reload best weights  
2. Save `.pt` (state_dict, history, seed, feature_cols, …)  
3. Plot training curve PNG  
4. Predict test (scaled) → inverse-transform → metrics in mm  
5. Rebuild test meta → sample station pred-vs-actual PNG  
6. Write `lstm_baseline_v2_seed42_metrics.json`  

**Multiseed script** repeats for seeds 13, 42, 123 → mean±std summary.

---

# PART 13 — Model Evaluation

## 13.1 Metrics

| Metric | Formula idea | Unit |
|--------|--------------|------|
| MSE | mean\((y-\hat y)^2\) | mm² |
| RMSE | √MSE | mm/day |
| MAE | mean\|y−ŷ\| | mm/day |
| R² | 1 − SS_res/SS_tot | dimensionless |

## 13.2 Why these

Standard regression suite; RMSE comparable to persistence; R² shows skill vs predicting the mean; MAE less extreme-sensitive than RMSE.

## 13.3 Your numbers

| Model | RMSE | R² |
|-------|------|-----|
| Persistence | ~11.56 | ~0.05 |
| LSTM v1 (seed 42) | ~9.93 | ~0.30 |
| LSTM v2 (seed 42) | ~9.46 | ~0.37 |
| LSTM v2 (3 seeds) | **9.39 ± 0.06** | **0.375 ± 0.008** |

**Good here:** beat persistence; stable seeds.  
**Not “solved”:** R²~0.37 — expected for daily rainfall (zero-inflated, skewed).  
**Bad:** worse than persistence; R²≤0; huge seed variance.

**Honest viva line:** “Daily rainfall is hard; we beat persistence with stable multi-seed skill, but variance remains — hence advanced models next.”

---

# PART 14 — Current Progress (honest)

| Phase | Status |
|-------|--------|
| Cleaning, EDA, features, audit | Done — research-ready |
| Sequences v1 & v2 | Done — **v2 locked** |
| LSTM multi-seed | Done — research-ready baseline |
| Persistence comparison | Recorded in metrics |
| Climatology baseline script | README says Done — **weak evidence in repo**; admit if asked |
| CNN-LSTM+Attention / GNN | Done — Attention: seeds {13,42,123}, h=1–4; significantly better than CNN-LSTM-Temporal at h=2 and h=4; not shown to outperform plain LSTM. GNN: implemented; does not beat LSTM |
| Extra metrics / paper ablations | Future |

**First-review readiness:** ~**60–70%** of full research vision; **~100%** of Phase-1 baseline pipeline.  
**Production-ready (research):** pipeline + locked LSTM v2.  
**Remaining:** write-up, doc/artifact hygiene; paper-faithful spatial CNN still not applicable to irregular stations.

---

# PART 15 — Files Generated

| File | Why | Creator |
|------|-----|---------|
| `clean_dataset.csv` | Clean tabular truth | pipeline / nb01 |
| `missing_values_summary.csv` | Document raw missingness | clean step |
| `feature_engineered.csv` | DOY features | features step |
| `feature_engineered_v2.csv` | Locked modeling table + station_id | features step |
| `temporal_density_by_station.csv` | Gap/coverage evidence | audit |
| `X_*_v2.npy` / `y_*_v2.npy` | Model tensors | generate_sequences_v2 |
| `sequence_metadata_v2.json` | Reproducibility card | generate_sequences_v2 |
| `minmax_scaler_v2.joblib` | X scale params | generate_sequences_v2 |
| `minmax_scaler_y_v2.joblib` | y scale params | seq / scale_y |
| `lstm_baseline_v2_seed*.pt` | Weights | train scripts |
| `*_metrics.json` | Test scores | train scripts |
| `lstm_baseline_v2_multiseed_summary.json` | mean±std | multiseed |
| training / pred PNGs | Visual evidence | train v2 |
| EDA PNGs | Distribution evidence | nb02 |
| audit txt / train logs | Traceability | pipeline / runs |

---

# PART 16 — Problems Faced

| Problem | Cause | Solution |
|---------|-------|----------|
| Massive missing rainfall | Sensor/reporting gaps | Drop NaN rainfall (never invent target) |
| Missing temps/wind/pressure | Incomplete observations | Station-wise interp / median |
| Station name collisions | Same name, different geo | `station_id` = name+lat+lon+elev |
| Temporal gaps | Real missing days | Contiguous windows only; audit first |
| Leakage risk | Easy to include target/future | Assertions + chrono split + train-only scalers |
| Redundant month/season | Categorical jumps | Replace with doy_sin/cos |
| v1 weaker than v2 | No past rain in X | Add past rainfall (exclude target day) |
| GPU vs CPU confusion | System Python CPU Torch | Project `.venv` with CUDA 12.6 |
| Large data / memory | ~560k windows × 30 × 8 | NumPy pipeline; batch 256; workers 0 |
| Training time | Many sequences | CUDA + AMP; early stopping |
| Zero-rain dominance | Dry days common | MSE regression now; future rain/no-rain metrics |
| Exploding gradients | Deep RNNs | Grad clip 1.0 |
| Seed luck concern | NN stochasticity | Multi-seed 13 / 42 / 123 |

---

# PART 17 — Implemented Extension: CNN–LSTM+Attention

## 17.1 Implemented architecture

```text
Input (B, 30, 8)
    -> CNN along time (local motifs: bursts, fronts, dry spells)
    -> LSTM over CNN features (month-scale memory)
    -> Additive (Bahdanau) attention over LSTM hidden states (which days mattered)
    -> Context -> Dense -> rainfall (mm)
```

Implemented as CNNLSTMAttention, trained across seeds {13,42,123} and horizons h=1-4.

## 17.2 Why CNN first
Extracts short local patterns before long memory - temporal "feature detectors."

## 17.3 Why LSTM second
Composes motifs into a trajectory across the 30-day window.

## 17.4 Why Attention last
Not all days are equal; soft-weights let the model focus on informative days.

## 17.5 Actual result (not speculative)
Significantly better than CNN-LSTM-Temporal at h=2 and h=4 (DM/bootstrap 
significance testing, HAC-corrected). NOT shown to outperform plain LSTM at 
any horizon - Attention-vs-LSTM was formally tested at h=1 (not significant); 
LSTM remains numerically best across all seasons at h=4. See FINAL_AUDIT.md, 
ablation_study.csv, and significance_results.csv for full results.

## 17.6 What stayed fixed across all model comparisons
Cleaning, contiguous sequences, splits, metrics - only the model architecture 
changed between comparisons, enabling a fair, controlled comparison.

---

# PART 18 — 50 Viva Questions + Ideal Answers

### Beginner (1–15)

1. **What is your project?** — Next-day rainfall (mm/day) from 30 contiguous days of Indian station weather using LSTM.  
2. **Target variable?** — `rainfall` on day 31 (mm/day).  
3. **What is mm/day?** — Millimetres of rain in one day.  
4. **Why Deep Learning?** — Nonlinear temporal patterns hard to hand-code.  
5. **What is LSTM?** — RNN with memory gates for sequences.  
6. **Input window?** — 30 contiguous days × 8 features (v2).  
7. **Name 8 features.** — avg/min/max temp, wind_speed, air_pressure, rainfall, doy_sin, doy_cos.  
8. **Splits?** — Train≤2022, Val 2023, Test 2024–Feb 2025.  
9. **Cleaned rows?** — 712,785.  
10. **Stations?** — 414 station_ids.  
11. **Headline metric?** — RMSE (mm/day) and R².  
12. **Best RMSE?** — 9.39 ± 0.06 (3 seeds).  
13. **Persistence?** — Tomorrow = today; RMSE ≈ 11.56.  
14. **Framework?** — PyTorch.  
15. **Model class file?** — `src/model.py` → `LSTMBaseline`.

### Intermediate (16–35)

16. **Why drop missing rainfall?** — Never fabricate labels.  
17. **Why station-wise fill?** — Local climate ≠ global median.  
18. **doy_sin/cos?** — Cyclical day-of-year seasonality.  
19. **Why drop month/season?** — Redundant; avoid category jumps.  
20. **station_id purpose?** — Fix duplicate station names.  
21. **Temporal density?** — observed/span coverage.  
22. **Why contiguous only?** — True daily timesteps for LSTM.  
23. **What is leakage?** — Illegal future/target information.  
24. **How prevent target leak?** — Asserts: target ∉ window; window ends day before.  
25. **Why chrono split?** — No future-in-train.  
26. **Why MinMax?** — Normalize heterogeneous units.  
27. **Fit on train only?** — No test peeking in preprocessing.  
28. **inverse_transform?** — Scaled → mm/day for metrics.  
29. **Hidden size / layers?** — 64, 2 layers.  
30. **Loss / optimizer?** — MSE + Adam(1e-3).  
31. **Early stopping?** — Stop when val loss stalls (patience 15).  
32. **Grad clip?** — Cap norm at 1.0.  
33. **Why multi-seed?** — Prove not lucky init.  
34. **v1 vs v2?** — v2 adds past rainfall; better scores.  
35. **R²=0.37 means?** — ~37% variance explained; useful, incomplete.

### Advanced (36–50)

36. **Why daily rain is hard?** — Zero-inflated, skewed, intermittent, chaotic.  
37. **Is MSE ideal?** — Standard but extreme-sensitive; plan threshold metrics.  
38. **Seq2seq teacher forcing?** — Not used; window→scalar regression.  
39. **batch_first=True?** — `(B,T,F)` matches NumPy pipeline.  
40. **Why last timestep only?** — Many-to-one summary.  
41. **Raw lat/lon now?** — Deferred; better as spatial/GNN stage.  
42. **Why not fill gaps?** — Synthetic weather → learn artifacts.  
43. **Leakage checks count?** — 561,092 passed.  
44. **AMP?** — Mixed precision speed/memory with GradScaler.  
45. **num_workers=0?** — Simplicity / Windows+CUDA stability.  
46. **CNN role later?** — Local temporal filters.  
47. **Attention role?** — Soft-select informative days.  
48. **What stays fixed for fair compare?** — Data, splits, metrics.  
49. **Novelty (careful)?** — Rigorous Indian station pipeline with target-date leakage asserts, train-only scalers, and locked multi-seed baseline for fair advanced-model comparison (covariate imputation limitation documented in FINAL_AUDIT.md) — not “first LSTM ever.”  
50. **Biggest limitation?** — Moderate R²; Attention improves on CNN-LSTM-Temporal at h=2/h=4 but is not shown to outperform plain LSTM; GNN does not beat LSTM; paper spatial CNN not applicable.

---

# PART 19 — Mentor Challenge Questions + Best Answers

**M1. Why trust your RMSE?**  
Chronological split, train-only scalers, target excluded from X with 561k asserts, metrics after inverse-transform, multi-seed std only 0.06.

**M2. Isn’t R²=0.37 weak?**  
Meaningful vs persistence (~0.05). Reported honestly. CNN-LSTM+Attention is implemented (significantly better than CNN-LSTM-Temporal at h=2 and h=4; not shown to outperform plain LSTM).

**M3. Why LSTM not Transformer?**  
Phase-1 baseline aligned with paper Sec 3.3.1; short windows (30); strong and cheap locked baseline first.

**M4. Heavy hyperparameter tuning?**  
Standard locked settings first (64×2, Adam 1e-3). Ablation/tuning is future work.

**M5. Why not rain/no-rain classification?**  
Primary task is mm/day regression; classification metrics planned as extras.

**M6. Does the model use station_id embeddings?**  
No — used only to group sequences correctly. Spatial modeling is future.

**M7. Does scaling y hide bad performance?**  
No — inverse-transform before RMSE/MAE/R².

**M8. Prove no target leakage.**  
Point to asserts in `generate_sequences_v2.py`, metadata splits, train-only scalers, contiguous segments. Separately disclose pre-split covariate imputation leakage (FINAL_AUDIT.md §7.8) — do not claim fully leakage-free preprocessing.

**M9. README says climatology Done but no script?**  
Persistence is recorded. If climatology isn’t in repo, admit it and offer to add before next review.

**M10. What’s novel?**  
Reproducible pipeline with temporal-density-informed contiguous windows, station disambiguation, locked multi-seed baseline preparing fair advanced comparison.

**M11. Overfitting?**  
Early stopping on val; curves saved; test is later years; seeds stable.

**M12. Why 30 not 7/60?**  
History vs continuity tradeoff; paper-style month; ablate later.

**M13. Zero inflation — MSE misleading?**  
Partly; we also watch MAE; plan threshold metrics.

**M14. GPU required — weakness?**  
Training convenience; science used CUDA; inference can be CPU later.

**M15. Remove past rainfall?**  
That is v1: worse RMSE (~9.93) and R² (~0.30).

---