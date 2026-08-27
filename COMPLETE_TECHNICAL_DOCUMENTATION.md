# COMPLETE TECHNICAL DOCUMENTATION
# Rainfall Prediction using Deep Learning

**Document type:** Knowledge-transfer technical specification for AI/human successors  
**Repository root:** `d:\project\Research Project`  
**Primary project path:** `d:\project\Research Project\RainfallPrediction`  
**Git remote:** `https://github.com/Rakshitha2736/Research-project.git`  
**Documentation date:** 2026-07-30  
**Sources consulted:** All Python modules, notebooks, `README.md`, `REPRODUCE.md`, `PROJECT_STATUS.md`, `FINAL_AUDIT.md`, `PROJECT_VERIFICATION_REPORT.md`, `docs/PROJECT_REVIEW_LEARNING_GUIDE.md`, `reports/tables/master_results.csv`, temporal audits, model metrics JSONs  

**Important global facts (read first):**
1. This is a **pure Python deep-learning research pipeline**. There is **no web frontend, no REST API server, no authentication system, and no SQL/NoSQL database**.
2. The base research paper is **referenced by section numbers only** (e.g., Sec 3.3.1, 3.3.3, 3.3.4) inside code comments and docs. **No PDF, BibTeX, title, or authors exist in the repository.** Where paper details are unknown, this document says so explicitly.
3. Implementation phase is marked **complete** by `PROJECT_STATUS.md` / `FINAL_AUDIT.md` (2026-07-29). Next step is **thesis/paper writing**, not new model development (unless scope expands).
4. `docs/PROJECT_REVIEW_LEARNING_GUIDE.md` Part 14 is **outdated** (still claims GNN/CNN-Attention are 0%). Prefer `FINAL_AUDIT.md`, `README.md`, and `PROJECT_STATUS.md` for current status.

---

# Part 1 — Project Overview

## 1.1 Project name

**Rainfall Prediction using Deep Learning**  
Folder / package name: `RainfallPrediction`

## 1.2 Main objective

Build a reproducible deep-learning system with target-date leakage-free sequence construction and train-only scaling (see FINAL_AUDIT.md for a documented, scope-limited covariate imputation leakage finding, empirically negligible in tested cases but not production-corrected) that predicts **daily rainfall in millimetres per day (mm/day)** at Indian meteorological stations from a **contiguous 30-day multivariate weather window**, then rigorously compare:

| Role | Models |
|------|--------|
| Primary | LSTM v2 vs GNN-LSTM across horizons h=1,2,3,4 with multi-seed evaluation |
| Ablations | Temporal CNN-LSTM, Transformer Encoder (h=1, seed 42) |
| Classical baselines | Persistence; rolling ARIMA on a 30-station sample |

## 1.3 Problem statement

**Supervised time-series regression:**

\[
X_{t-29:t} \in \mathbb{R}^{30 \times 8} \;\rightarrow\; y_{t+h} \in \mathbb{R}
\]

where:
- \(X\) = 8 weather/seasonal features over 30 contiguous calendar days at one station
- \(y\) = rainfall (mm/day) on day \(t+h\)
- \(h \in \{1,2,3,4\}\) (1 = next day; 4 = four days ahead)
- Split is chronological by **target date** (never random shuffle across time)
- Target-day rainfall must **not** appear in the input window (no target-day leakage)

## 1.4 Why this project exists

1. **Academic / university research** comparing deep architectures for Indian station rainfall.
2. Daily rainfall is **nonlinear, seasonal, zero-inflated, and station-specific**; classical persistence is weak.
3. The unnamed **base paper** proposes spatial CNN-LSTM(-Attention) on a regular lat/lon grid. Indian stations here are **irregularly spaced** (414 `station_id`s), so spatial CNN is not directly applicable. This project **adapts** the paper by replacing spatial CNN with a **GNN** (paper’s own suggested future direction per `FINAL_AUDIT.md`).
4. Produce thesis-ready evidence: multi-seed metrics, Diebold-Mariano tests, bootstrap CIs.

## 1.5 Target users

| User | Use |
|------|-----|
| Student researcher / author | Run experiments, write thesis |
| Mentors / examiners | Audit methodology & results |
| Future developers / AI agents | Reproduce or extend pipeline |
| Downstream decision-makers | **Not yet** — no deployed product |

## 1.6 Real-world use case

Station-level short-horizon rainfall forecasts supporting agriculture, water resources, and urban planning — **as research prototypes**, not an operational forecasting service. Flood awareness / flood early-warning utility has **NOT been demonstrated** in this project (no flood-specific event validation).

## 1.7 Current development stage

| Stage | Status |
|-------|--------|
| Data pipeline | Complete & verified (integrity ~86–92/100) |
| Model training / evaluation | Complete for planned experiments |
| Statistical significance | Complete (DM, paired-t, bootstrap) |
| Productization (API/UI/deploy) | **Not started / out of scope** |
| Thesis / paper writing | **Next milestone** |

**Claimed implementation completion:** ~100% of planned experimental scope (`PROJECT_STATUS.md`).  
**Claimed full research vision (including write-up):** writing not done; learning guide’s older “60–70%” figure is obsolete for *code*, still relevant for *publication*.

## 1.8 Overall architecture

Batch research pipeline (not client-server):

```
Raw Excel
  → Clean CSV
  → Feature-engineered CSV (+ station_id, doy_sin/cos)
  → Temporal density audit
  → Contiguous 30-day sequences (.npy) + MinMax scalers
  → Optional: station graph + graph-date tensors
  → Train models (PyTorch CUDA + AMP)
  → Inverse-transform metrics (mm/day)
  → Multi-seed / multi-horizon significance tests
  → reports/tables + figures
```

Shared library: `src/model.py`, `src/cuda_setup.py`, `src/preprocess.py`. Orchestration entry: `run_pipeline.py` (Phase-1 path through LSTM v2). Extended experiments via dedicated root scripts.

---

# Part 2 — Technology Stack

## 2.1 Summary table

| Category | Technology | Why used | Where used | Config / notes |
|----------|------------|----------|------------|----------------|
| Language | Python 3.12.x | Research ML ecosystem | Entire project | Tested 3.12 |
| DL framework | PyTorch ≥2.0 (CUDA build) | LSTM/GNN/CNN/Transformer | `src/model.py`, all `train_*.py` | Project uses `torch 2.x+cu126`; AMP autocast + GradScaler |
| GPU | NVIDIA CUDA (RTX 2050 / CUDA 12.6 in docs) | Training speed | `src/cuda_setup.require_cuda()` | Fail-fast if CUDA missing |
| Dataframes | pandas ≥2.0 | Excel/CSV processing | pipeline, audits, seq builders | openpyxl for Excel |
| Arrays | NumPy ≥1.24 | Sequence tensors `.npy` | all sequence/graph scripts | float32 tensors |
| Scaling | scikit-learn MinMaxScaler | Train-only normalization | generate_sequences*, run_pipeline | joblib persist |
| Persistence | joblib ≥1.3 | Save scalers | `models/*.joblib` | |
| Classical ML/stats | statsmodels ≥0.14 | Rolling ARIMA | `arima_and_significance.py` | ARIMA(2,0,2) |
| Scientific stats | scipy ≥1.11 | DM / t-tests / bootstrap | significance scripts | |
| Viz | matplotlib, seaborn | EDA + training curves | notebooks, train scripts | `reports/figures/` |
| Notebooks | Jupyter / ipykernel | Interactive EDA | `notebooks/` | |
| Package mgr | pip + venv | Isolation | `.venv` at parent `Research Project/` | No conda/poetry |
| OS tooling | Windows PowerShell | Reproduction commands | `REPRODUCE.md` | Hardcoded Windows paths in places |
| Version control | Git | Source control | parent + nested `.git` under RainfallPrediction | Large artifacts gitignored |

## 2.2 Explicitly NOT used

| Technology | Status |
|------------|--------|
| Frontend (React/Vue/HTML app) | Absent |
| Backend web framework (Flask/FastAPI/Django) | Absent |
| REST/GraphQL APIs | Absent |
| SQL/NoSQL DB | Absent (CSV/NPY/JSON files only) |
| Auth (JWT/OAuth) | Absent |
| Docker / Kubernetes | Absent |
| Cloud ML platforms | Absent |
| Java / JS / TS | Absent |
| TensorFlow / Keras | Absent |
| Ray / MLflow / W&B | Absent |

## 2.3 Dependency file

`RainfallPrediction/requirements.txt`:

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
torch>=2.0.0
joblib>=1.3.0
openpyxl>=3.1.0
ipykernel>=6.29.0
scipy>=1.11.0
statsmodels>=0.14.0
```

**Install note:** Install CUDA PyTorch from the official wheel index *before* or carefully with `requirements.txt` so CPU-only Torch is not silently used.

## 2.4 Environment path (important)

Preferred interpreter (hardcoded in several places):

`D:\project\Research Project\.venv\Scripts\python.exe`

System path `D:\Programs\Python\python.exe` is documented as **CPU-only** and must not be used for training.

---

# Part 3 — Project Folder Structure

```
d:\project\Research Project\
├── .git/                          # Parent git repository
├── .venv/                         # CUDA PyTorch virtualenv (not project source)
└── RainfallPrediction/            # ★ ALL project code & artifacts
    ├── .git/                      # Nested git (same remote) — hygiene concern
    ├── .gitignore
    ├── README.md
    ├── REPRODUCE.md
    ├── PROJECT_STATUS.md
    ├── PROJECT_VERIFICATION_REPORT.md
    ├── FINAL_AUDIT.md
    ├── requirements.txt
    ├── run_pipeline.py            # End-to-end Phase-1 orchestrator
    ├── diagnose_dataset.py
    ├── audit_temporal_density.py
    ├── generate_sequences.py      # v1 (7 features) — legacy
    ├── generate_sequences_v2.py   # locked h=1 builder
    ├── generate_sequences_multihorizon.py
    ├── build_station_graph.py
    ├── build_graph_batches.py
    ├── build_graph_batches_multihorizon.py
    ├── train_lstm_baseline.py     # v1 trainer
    ├── train_lstm_baseline_v2.py
    ├── train_lstm_baseline_v2_multiseed.py
    ├── train_lstm_multihorizon.py
    ├── train_gnn_lstm.py
    ├── train_gnn_lstm_multihorizon.py
    ├── train_cnn_lstm_temporal_h1.py
    ├── train_transformer_h1.py
    ├── multiseed_gnn_significance.py
    ├── multiseed_multihorizon.py
    ├── arima_and_significance.py
    ├── *.log                      # Root-level run monitors (should live under reports/logs)
    ├── data/
    │   ├── raw/                   # india_weather_rainfall_data.xlsx (~62 MB)
    │   ├── processed/             # CSVs, .npy sequences/graph tensors, JSON meta (~7+ GB)
    │   └── external/              # Empty placeholder
    ├── models/                    # .pt checkpoints, .joblib scalers, adjacency_norm.pt
    ├── notebooks/
    │   ├── 01_Data_Preprocessing.ipynb
    │   ├── 02_EDA.ipynb
    │   ├── 04_feature_engineering.ipynb   # no 03_* (intentional numbering gap)
    │   └── 05_sequence_generation.ipynb
    ├── reports/
    │   ├── figures/               # 13 PNGs
    │   ├── tables/master_results.csv
    │   ├── logs/                  # Reserved (empty)
    │   └── temporal_density_audit*.txt
    ├── results/                   # Empty placeholder (.gitkeep)
    ├── src/
    │   ├── __init__.py
    │   ├── model.py               # All NN architectures
    │   ├── cuda_setup.py
    │   └── preprocess.py          # Path helpers (partial cleaner)
    └── docs/
        └── PROJECT_REVIEW_LEARNING_GUIDE.md
```

### Folder purposes & interactions

| Folder | Purpose | Contains | Interacts with |
|--------|---------|----------|----------------|
| `data/raw` | Immutable source | Excel | `run_pipeline.step_clean`, notebook 01 |
| `data/processed` | All derived tabular/tensor data | CSV, NPY, JSON | Sequence/graph/train scripts |
| `data/external` | Reserved for external datasets | Empty | None currently |
| `models` | Weights + scalers + adjacency | `.pt`, `.joblib` | All train/eval scripts |
| `notebooks` | Interactive exploration | 4 notebooks | Mirrors pipeline steps; writes figures |
| `reports` | Human-readable evidence | figures, tables, audits | Thesis artifacts |
| `results` | Reserved outputs | `.gitkeep` only | Unused |
| `src` | Reusable library | models, CUDA, paths | Imported by train scripts |
| Root `*.py` | Executable pipeline stages | 20 scripts | Chain: clean→…→significance |
| `docs` | Learning / viva guide | 1 large MD | Documentation only (partially stale) |

### `.gitignore` policy

Ignores: `__pycache__`, venvs, raw Excel contents under `data/raw/*` (keeps `.gitkeep`), processed `*.csv/*.npy/*.json`, model `*.pt/*.joblib`, some report txt/logs. **Implication:** cloning GitHub alone is **not** enough to retrain without re-acquiring the Excel and regenerating artifacts (or copying ignored local files).

---

# Part 4 — Explain Every Important File

## 4.1 Library package (`src/`)

### File: `src/__init__.py`
- **Purpose:** Package marker (`"Rainfall prediction project package."`).
- **Classes/Functions:** None.
- **Status:** Complete (minimal).
- **Depends on:** Nothing. Enables `from src.model import ...`.

### File: `src/model.py`
- **Purpose:** Defines all neural network architectures.
- **Main classes:**
  1. **`LSTMBaseline`** — 2-layer LSTM (default `input_size=7`, hidden=64) → last timestep → Linear → scalar. Used with `input_size=8` for v2.
  2. **`CNNLSTMTemporalBaseline`** — Conv1d over **time** (16→32, k=3) + LSTM(64) → FC. Explicitly **not** paper spatial CNN.
  3. **`TransformerEncoderBaseline`** — Linear proj → learnable pos embed → pre-norm TransformerEncoder (d=64, 4 heads, 2 layers, GELU, ff=256) → last step → FC. Adapted from paper Sec 3.3.4 style.
  4. **`GNNLSTM`** — Buffer adjacency; 2-layer GCN (`w1` 8→16, `w2` 16→32) with **per-date masked** \(D^{-1/2}AD^{-1/2}\) → per-station LSTM(64) → FC → `(B,N)` predictions.
- **Key methods (`GNNLSTM`):** `_masked_a_norm(mask)`, `_gcn_encode(x, a_norm)`, `forward(x, mask)`.
- **Consumed by:** All `train_*.py`, significance scripts.
- **Status:** Complete.

### File: `src/cuda_setup.py`
- **Purpose:** CUDA fail-fast, seeding, DataLoader helpers tuned for RTX 2050 4GB.
- **Constants:** `DEFAULT_BATCH_SIZE=256`, `DEFAULT_NUM_WORKERS=0`.
- **Functions:** `require_cuda()`, `set_seed(seed)`, `print_gpu_diagnostics()`, `print_gpu_memory()`, `make_loader()`, `to_device()`.
- **Hardcoded:** Warning against `D:\Programs\Python\python.exe`.
- **Status:** Complete.

### File: `src/preprocess.py`
- **Purpose:** Path constants + lightweight notebook helpers. **Not** the production cleaner.
- **Constants:** `RAW_DATASET_PATH`, `CLEAN_DATASET_PATH`, `FEATURE_ENGINEERED_PATH`, aliases.
- **Functions:** `load_dataset()`, `inspect_dataset()`, `clean_dataset()` (date parse+sort only), `save_clean_dataset()`.
- **Status:** **Partial by design** — full cleaning is in `run_pipeline.step_clean` / notebook 01.

## 4.2 Orchestration & diagnostics

### File: `run_pipeline.py`
- **Purpose:** End-to-end Phase-1: clean → features → audit → sequences v2 → optional y-scale → train LSTM v2 seed 42.
- **Functions:** `log`, `_resolve_python`, `run_script`, `step_clean`, `step_features`, `step_audit`, `step_sequences`, `step_scale_y`, `step_train`, `parse_args`, `main`.
- **CLI:** `--from {clean,features,audit,sequences,scale_y,train}`, `--force`, `--skip-train`, `--skip-audit`.
- **Outputs:** `clean_dataset.csv`, `missing_values_summary.csv`, `feature_engineered.csv`, `feature_engineered_v2.csv`, audit txt, triggers NPY/scalers/checkpoint.
- **Does NOT run:** multi-horizon, GNN, CNN, Transformer, significance (separate scripts).
- **Status:** Complete.

### File: `diagnose_dataset.py`
- **Purpose:** Ad-hoc CSV/Excel diagnostic printer (shape, missingness, stations, rainfall zeros).
- **CLI:** positional path.
- **Outputs:** stdout only.
- **Status:** Complete. Not on critical path.

### File: `audit_temporal_density.py`
- **Purpose:** Per-station coverage, gap stats, contiguous 31-day feasibility → justifies no gap-fill policy.
- **Key functions:** `station_key_column`, `per_station_stats`, `all_gap_sizes`, `main`.
- **Outputs:** `temporal_density_by_station.csv` (+ stdout captured to `reports/temporal_density_audit_v2.txt`).
- **Key finding (v2):** 414 stations, mean coverage ~0.816, 22,604 gaps, 0 duplicate (station_id, date).
- **Status:** Complete.

## 4.3 Sequence generation

### File: `generate_sequences.py` (v1 / legacy)
- **Purpose:** Contiguous 30→day-31 sequences with **7 features** (no past rainfall in X). Scales X only; **does not scale y**.
- **Outputs:** `X_*.npy`, `y_*.npy` (raw mm), `minmax_scaler.joblib`, `sequence_metadata.json`.
- **Consumed by:** `train_lstm_baseline.py`.
- **Status:** Complete as v1 builder; **integration risk:** trainer expects `minmax_scaler_y.joblib` which this script may not create (scaler may exist from earlier manual run). Prefer v2 path.

### File: `generate_sequences_v2.py` (locked h=1)
- **Purpose:** 8-feature sequences including past rainfall; leakage asserts; MinMax X+y train-only.
- **Feature cols:** `avg_temp, min_temp, max_temp, wind_speed, air_pressure, rainfall, doy_sin, doy_cos`.
- **Outputs:** `X_*_v2.npy`, `y_*_v2.npy`, `minmax_scaler_v2.joblib`, `minmax_scaler_y_v2.joblib`, `sequence_metadata_v2.json`.
- **Sample counts h=1:** train 270,109 / val 149,720 / test 141,263; shape `(N,30,8)`.
- **Status:** Complete (canonical).

### File: `generate_sequences_multihorizon.py`
- **Purpose:** Horizons h=2,3,4; contiguous 30-day X; target = real obs at `window_end+h` (intermediates may be missing); reuses X-scaler; new y-scaler per horizon; never mutates `*_v2`.
- **Outputs:** `X_*_h{h}.npy`, `y_*_h{h}.npy`, `minmax_scaler_y_h{h}.joblib`, `sequence_metadata_h{h}.json`.
- **Status:** Complete.

## 4.4 Graph construction

### File: `build_station_graph.py`
- **Purpose:** Symmetrized kNN (k=8) via haversine; assert 414 stations; remote-station diagnostics (300 km).
- **Outputs:** `station_graph_edges.csv` (3,856 edges), `station_id_to_index.json`.
- **Status:** Complete.

### File: `build_graph_batches.py`
- **Purpose:** Align v2 sequences into dense per-date tensors `(D, 414, 30, 8)` + y + mask; export date-index JSON.
- **Outputs:** `{X,y,mask}_{train,val,test}_graph.npy`, `graph_date_index_{split}.json`.
- **Exports:** `replay_meta` reused by significance.
- **Status:** Complete.

### File: `build_graph_batches_multihorizon.py`
- **Purpose:** Same for h=2,3,4 without overwriting h=1 graph files. Notes train coverage ≥20% filter used later.
- **Status:** Complete.

## 4.5 Training scripts

### File: `train_lstm_baseline.py`
- **Purpose:** Train v1 LSTM (7 feats, seed 42, AMP, early stop patience 15).
- **Outputs:** `lstm_baseline_seed42.pt`, metrics JSON, training/pred PNGs.
- **Status:** Complete (legacy).

### File: `train_lstm_baseline_v2.py`
- **Purpose:** Locked h=1 LSTM v2 seed 42; records persistence RMSE 11.5559 in metrics.
- **Hyperparams:** Adam 1e-3, batch 256, max 100 epochs, patience 15, grad clip 1.0, AMP.
- **Outputs:** `lstm_baseline_v2_seed42.pt`, metrics JSON, figures.
- **Called by:** `run_pipeline`.
- **Status:** Complete.

### File: `train_lstm_baseline_v2_multiseed.py`
- **Purpose:** Seeds {13,42,123}; per-seed ckpt/metrics + summary JSON.
- **Note:** Summary JSON can go **stale** if seed-42 later retrained alone — prefer canonical `mh_multiseed_monitor.log` / master_results for headlines.
- **Status:** Complete.

### File: `train_lstm_multihorizon.py`
- **Purpose:** LSTM for h=2,3,4 (historically seed 42; multi-seed also via `multiseed_multihorizon.py`). Batch 64, patience 30.
- **Outputs:** `lstm_h{h}_seed42.pt` (console metrics; limited JSON persistence).
- **Status:** Complete.

### File: `train_gnn_lstm.py`
- **Purpose:** GNN-LSTM h=1; build `adjacency_norm.pt`; filter train dates with ≥20% valid stations; masked MSE; batch_size=1; compare vs LSTM.
- **Outputs:** `gnn_lstm_seed42.pt`, adjacency, training curve.
- **Status:** Complete.

### File: `train_gnn_lstm_multihorizon.py`
- **Purpose:** GNN-LSTM h=2,3,4 seed 42; patience 30.
- **Outputs:** `gnn_lstm_h{h}_seed42.pt`.
- **Status:** Complete.

### File: `train_cnn_lstm_temporal_h1.py`
- **Purpose:** Temporal CNN-LSTM ablation h=1 seed 42.
- **Outputs:** `cnn_lstm_temporal_h1_seed42.pt` + curve; metrics printed (no JSON).
- **Status:** Complete (single-seed ablation).

### File: `train_transformer_h1.py`
- **Purpose:** Transformer encoder ablation h=1 seed 42.
- **Outputs:** `transformer_h1_seed42.pt` + curve; metrics printed.
- **Status:** Complete (single-seed ablation).

## 4.6 Evaluation / significance

### File: `arima_and_significance.py`
- **Purpose:** (1) Rolling ARIMA(2,0,2) on random 30 stations; (2) DM + paired-t LSTM vs persistence; **exports** `diebold_mariano()`.
- **Outputs:** stdout only (ARIMA aggregates not saved).
- **Status:** Complete.

### File: `multiseed_gnn_significance.py`
- **Purpose:** Multi-seed GNN h=1 + DM vs LSTM v2 on masked test.
- **Status:** Complete.

### File: `multiseed_multihorizon.py`
- **Purpose:** Canonical multi-seed LSTM+GNN across h=1..4; DM, paired-t, bootstrap 95% CI (1000 resamples, seed 42). Writes monitor log with **headline table**.
- **Status:** Complete. **Primary source of truth for multi-horizon numbers.**

## 4.7 Notebooks

| Notebook | Purpose | Status |
|----------|---------|--------|
| `01_Data_Preprocessing.ipynb` | Load Excel, clean, save CSV | Complete (mirrors pipeline) |
| `02_EDA.ipynb` | Distributions, monthly/season, correlation heatmaps | Complete |
| `04_feature_engineering.ipynb` | doy_sin/cos, station_id | Complete |
| `05_sequence_generation.ipynb` | Thin wrapper calling `generate_sequences_v2.py` | Complete |
| `03_*` | **Missing by design** (numbering history) | N/A |

## 4.8 Documentation / config files

| File | Purpose | Status |
|------|---------|--------|
| `README.md` | Project overview + results | Complete (prefer verification table for multi-horizon) |
| `REPRODUCE.md` | Step-by-step reproduction | Complete |
| `PROJECT_STATUS.md` | Completion checklist | Complete (2026-07-29) |
| `PROJECT_VERIFICATION_REPORT.md` | Independent pipeline audit 92/100 scorecard | Complete |
| `FINAL_AUDIT.md` | Frozen-model final audit + paper framing | Complete |
| `docs/PROJECT_REVIEW_LEARNING_GUIDE.md` | Viva/learning guide | **Partially stale** on advanced models |
| `requirements.txt` | pip deps | Complete |
| `.gitignore` | Exclude large artifacts | Complete |
| `reports/tables/master_results.csv` | Consolidated metrics table | Complete with documented N/As |

## 4.9 Data / model artifacts (non-source but critical)

| Artifact pattern | Role |
|------------------|------|
| `india_weather_rainfall_data.xlsx` | Raw input (~970k rows) |
| `clean_dataset.csv` | 712,785 cleaned rows |
| `feature_engineered_v2.csv` | Locked modeling table |
| `X_*_v2.npy` / `y_*_v2.npy` | h=1 tensors |
| `X_*_h{2,3,4}.npy` | Multi-horizon tensors |
| `*_graph*.npy` | GNN date×station tensors (very large) |
| `adjacency_norm.pt` | Normalized adjacency with self-loops |
| `lstm_*.pt` / `gnn_lstm_*.pt` / ablation `.pt` | Checkpoints |
| `minmax_scaler*.joblib` | Scalers |
| `mh_multiseed_monitor.log` | Canonical multi-horizon eval log |

---

# Part 5 — System Architecture

## 5.1 Overall architecture (ASCII)

```
┌──────────────────────────────────────────────────────────────────┐
│                     RESEARCH PIPELINE (offline)                   │
│                                                                  │
│  ┌─────────┐   ┌──────────┐   ┌──────────┐   ┌────────────────┐ │
│  │ Raw XLSX │→ │ Clean/FE │→ │ Sequences│→ │ Scalers (.joblib)│ │
│  └─────────┘   └──────────┘   └────┬─────┘   └────────┬───────┘ │
│                                    │                   │         │
│                    ┌───────────────┼───────────────────┘         │
│                    ▼               ▼                             │
│            ┌─────────────┐  ┌─────────────┐                      │
│            │ Flat LSTM / │  │ Station     │                      │
│            │ CNN / Trans │  │ Graph + GNN │                      │
│            └──────┬──────┘  └──────┬──────┘                      │
│                   │                │                             │
│                   └───────┬────────┘                             │
│                           ▼                                      │
│              Inverse-transform metrics (mm/day)                  │
│                           ▼                                      │
│              Multi-seed + DM / bootstrap significance            │
│                           ▼                                      │
│              reports/tables + figures + logs                     │
└──────────────────────────────────────────────────────────────────┘
```

## 5.2 Data flow

1. Excel → drop NaN rainfall → station-wise fills → `clean_dataset.csv`
2. Add `doy_sin/cos`, `station_id` → `feature_engineered_v2.csv`
3. Audit temporal density (coverage/gaps)
4. Per-station contiguous sliding windows → NumPy X/y
5. Chronological split by **target date**
6. MinMax fit on **train only**, transform val/test
7. Optional reshape to graph batches
8. Train → predict scaled y → inverse_transform → RMSE/MAE/MSE/R²

## 5.3 User flow

There is **no end-user application**. Operator flow:

1. Place Excel in `data/raw/`
2. Create CUDA venv; `pip install`
3. `python run_pipeline.py` (Phase-1)
4. Run multihorizon / graph / GNN / significance scripts as in `REPRODUCE.md`
5. Read `master_results.csv` / monitor logs for thesis tables

## 5.4 Backend / Frontend / API / Auth flows

**Not applicable** — no server, UI, API, or authentication.

## 5.5 AI / ML training flow

```
require_cuda() → set_seed → DataLoader
→ for epoch:
     AMP autocast forward → MSE loss → GradScaler backward → clip_grad → step
     validate → early stop on val loss
→ save best checkpoint
→ test predict → inverse y → metrics_mm
```

GNN variant: batch = one (or few) dates; mask invalid stations; masked MSE.

## 5.6 “Database” flow

File-based store only:

```
Excel/CSV  ↔  pandas
NPY        ↔  numpy
joblib/pt  ↔  sklearn/torch
JSON meta  ↔  dict dumps
```

## 5.7 Request / Response lifecycle

N/A (batch scripts). Closest analogue: CLI invocation → stdout/logs + files on disk.

## 5.8 Error handling flow

| Mechanism | Behavior |
|-----------|----------|
| `require_cuda()` | Raises if no GPU |
| File existence checks | Raise `FileNotFoundError` with hints |
| Leakage asserts in seq builders | Fail build if target in window |
| GNN unstable detection | Retry / adjust batch in `train_gnn_lstm` |
| Subprocess `check=True` | Pipeline aborts on child failure |
| No centralized exception framework | Script-local try/raise/print |

---

# Part 6 — Features

| Feature | Purpose | Files | Flow | Status | Limitations |
|---------|---------|-------|------|--------|-------------|
| Raw data ingest | Load Excel | `run_pipeline`, nb01, `preprocess.load_dataset` | Excel→DF | Complete | Dataset not in git |
| Missing-value analysis | Document NaNs | `missing_values_summary.csv` | raw→summary | Complete | |
| Data cleaning | Usable panel | `step_clean`, nb01 | drop rain NaN; fills | Complete | Drops ~257k rows |
| Feature engineering | Model inputs | `step_features`, nb04 | doy + station_id | Complete | |
| EDA visualization | Understand data | nb02, figures | plots | Complete | |
| Temporal density audit | Justify contiguous windows | `audit_temporal_density.py` | coverage/gaps | Complete | |
| Sequence gen v1 | Legacy 7-feat windows | `generate_sequences.py` | windows→npy | Complete (legacy) | Weaker; y-scale mismatch risk |
| Sequence gen v2 | Locked h=1 | `generate_sequences_v2.py` | 8 feats + asserts | Complete | |
| Multi-horizon sequences | h=2,3,4 | `generate_sequences_multihorizon.py` | Option A targets | Complete | |
| Train-only MinMax | Prevent leakage | seq scripts, `step_scale_y` | fit train | Complete | |
| Chronological split | Time-safe eval | all seq builders | by target date | Complete | Windows may cross year boundary |
| LSTM v2 training | Primary baseline | `train_lstm_baseline_v2*.py` | CUDA AMP | Complete | |
| LSTM multi-horizon | Longer forecasts | `train_lstm_multihorizon.py`, `multiseed_multihorizon.py` | | Complete | Aggregate MAE/R² not always logged |
| Station graph | Spatial topology | `build_station_graph.py` | kNN k=8 | Complete | Distance≠meteorology |
| Graph batches | GNN inputs | `build_graph_batches*.py` | dense tensors | Complete | Huge disk (~GB) |
| GNN-LSTM training | Spatial+temporal model | `train_gnn_lstm*.py` | masked GCN+LSTM | Complete | Underperforms LSTM |
| Temporal CNN-LSTM ablation | Local temporal filters | `train_cnn_lstm_temporal_h1.py` | | Complete | h=1 seed42 only; not paper spatial CNN |
| Transformer ablation | Attention encoder | `train_transformer_h1.py` | | Complete | h=1 seed42 only |
| Persistence baseline | Naive comparator | recorded in LSTM v2 metrics | last rain→pred | Complete | RMSE only |
| ARIMA baseline | Classical comparator | `arima_and_significance.py` | rolling 1-step | Complete | 30/414 stations; metrics not saved |
| Diebold-Mariano / bootstrap | Significance | `arima_and_significance`, `multiseed_*` | | Complete | |
| Pipeline automation | One-command Phase-1 | `run_pipeline.py` | | Complete | Doesn’t cover full experiment matrix |
| Reproduction docs | Re-run guide | `REPRODUCE.md` | | Complete | Windows-centric |
| Master results table | Paper tables | `master_results.csv` | | Complete | Some N/As documented |
| Web app / API / auth | Productization | — | — | **Not started** | Out of current scope |
| Deployment | Cloud/ops | — | — | **Not started** | |

---

# Part 7 — APIs

## Verdict

**There are no HTTP/REST/gRPC APIs in this repository.**

| Item | Detail |
|------|--------|
| Methods / Endpoints | None |
| Input / Output contracts | CLI scripts + filesystem artifacts |
| Authentication | None |
| Status | N/A — research batch jobs only |

### CLI “interfaces” (closest equivalent)

| Invocation | Input | Output |
|------------|-------|--------|
| `python run_pipeline.py [--from STEP] [--force] [--skip-train] [--skip-audit]` | Excel / prior artifacts | CSV, NPY, LSTM ckpt |
| `python audit_temporal_density.py <csv>` | Feature CSV | stdout + per-station CSV |
| `python diagnose_dataset.py <path>` | CSV/XLSX | stdout |
| `python generate_sequences_v2.py` | feature CSV | `*_v2.npy` + scalers |
| `python generate_sequences_multihorizon.py` | feature CSV + X-scaler | `*_h{2,3,4}.npy` |
| `python build_station_graph.py` | feature CSV | edges + index JSON |
| `python build_graph_batches.py` | v2 npy + index | graph npy |
| `python train_*.py` / `multiseed_*.py` / `arima_*.py` | tensors/scalers | checkpoints, logs, figures |

---

# Part 8 — Database

## Verdict

**No relational or document database.** Persistence is filesystem artifacts.

### Logical “tables” (file analogues)

| Logical entity | Storage | Key fields / shape | Relationships |
|----------------|---------|--------------------|---------------|
| Raw observations | Excel | ~970k×15 weather cols | — |
| Clean observations | `clean_dataset.csv` | 712,785 rows | 1 row per station_name×date (pre-disambiguation) |
| Feature table | `feature_engineered_v2.csv` | + doy_sin/cos, station_id | 414 station_ids |
| Sequences | `X_*.npy`, `y_*.npy` | `(N,30,8)`, `(N,)` | Split by target date |
| Graph edges | `station_graph_edges.csv` | source, target, distance_km | 3856 edges on 414 nodes |
| Graph batches | `X_*_graph*.npy` | `(D,414,30,8)` | Aligned to sequence meta |
| Masks | `mask_*_graph*.npy` | `(D,414)` bool | Valid station-days |
| Scalers | `*.joblib` | MinMax params | Fit on train |
| Models | `*.pt` | state_dict + meta | — |
| Metrics | `*_metrics.json`, CSV, logs | RMSE etc. | — |

### Indexes / FKs / migrations

None (no DBMS). Integrity enforced by script asserts and verification report.

### Migration status

N/A. Schema evolution is versioned by filenames (`_v2`, `_h2`, etc.).

---

# Part 9 — AI / ML Components

## 9.1 Models used

| Model | Architecture summary | Horizons | Seeds |
|-------|----------------------|----------|-------|
| LSTM v2 | 2×LSTM(64)→FC | 1–4 | 13,42,123 |
| GNN-LSTM | Masked 2-layer GCN (8→16→32) + LSTM(64)→FC | 1–4 | 13,42,123 |
| CNN-LSTM-Temporal | Conv1d 16→32 + LSTM→FC | 1 | 42 |
| Transformer | Pre-norm encoder d=64, 4 heads, 2 layers | 1 | 42 |
| Persistence | ŷ = last observed rainfall in window | 1 | n/a |
| ARIMA | Rolling ARIMA(2,0,2) | 1 | 30-station sample |

## 9.2 Dataset

- Source: `india_weather_rainfall_data.xlsx`
- Cleaned: 712,785 rows; dates 2015-01-01 → 2025-02-10
- Stations: 406 names → **414** `station_id`s
- Features (8): temps (avg/min/max), wind_speed, air_pressure, rainfall, doy_sin, doy_cos

## 9.3 Feature engineering

- Cyclical DOY: \(\sin/\cos(2\pi \cdot \mathrm{doy}/366)\)
- Drop categorical `month`, `season`
- `station_id = name_lat_lon_elev` to fix collisions
- **No** station embedding in models; ID used for grouping only

## 9.4 Preprocessing

1. Drop missing rainfall (never invent labels)
2. Station-wise linear interpolate min/max temp; median fill wind/pressure; global median fallback
3. Contiguous windows only (no gap-fill) after temporal audit
4. Chronological split: train ≤2022, val 2023, test 2024–2025-02-10
5. MinMax X and y on **train only**

## 9.5 Training process

- Loss: MSE on scaled targets
- Optimizer: Adam lr=1e-3
- Grad clip: 1.0
- Early stopping: patience 15 (h=1 LSTM) or 30 (multihorizon / GNN)
- AMP: `torch.amp.autocast` + GradScaler
- GNN train filter: dates with <20% valid stations excluded from **train** only

## 9.6 Inference / evaluation

- Forward pass → scaled predictions → `inverse_transform` with y-scaler → mm/day
- Metrics: RMSE, MAE, MSE, R²
- GNN metrics: masked stations only
- Canonical numbers: **CUDA + autocast** path (`multiseed_multihorizon.py` / monitor log)
- AMP sensitivity: ~0.0005 RMSE vs pure FP32

## 9.7 Headline results (canonical)

| Horizon | LSTM RMSE mean±std | GNN RMSE mean±std | DM p | Bootstrap 95% CI (GNN−LSTM) |
|--------:|--------------------|-------------------|------|-----------------------------|
| 1 | 9.3745 ± 0.0408 | 9.7476 ± 0.0441 | 7.18e-08 | (0.1820, 0.3775) |
| 2 | 10.2295 ± 0.0184 | 10.4880 ± 0.0880 | 1.279e-25 | (0.2825, 0.4019) |
| 3 | 10.4892 ± 0.0187 | 10.7174 ± 0.0628 | 6.900e-32 | (0.2250, 0.3081) |
| 4 | 10.5841 ± 0.0178 | 10.9702 ± 0.0326 | 6.975e-40 | (0.3107, 0.4082) |

Ablations (h=1, seed 42, CPU FP32 re-eval in master table): Transformer 9.4355; CNN-LSTM-Temporal 9.4835; Persistence 11.5559.

**Finding:** LSTM significantly outperforms GNN-LSTM at every horizon.

## 9.8 Current ML limitations

1. Irregular coverage (mean ~0.816); many gaps
2. Distance kNN graph may be suboptimal
3. No paper-faithful spatial CNN (base paper has no attention mechanism, so this is not applicable to attention). The project's own CNN-LSTM+Attention (Bahdanau attention over LSTM hidden states) is an original extension beyond the base paper, not a reproduction of it - see Part 13 for its implementation status.
4. Ablations single-seed / h=1 only
5. ARIMA only 30 stations; metrics not persisted
6. R² ~0.37 — meaningful vs persistence but far from “solved”
7. Zero-inflated rainfall; MSE-only primary objective
8. Single geographic dataset

## 9.9 Future ML improvements (documented)

- Paper write-up + richer tables
- Attention visualization
- Correlation / learned adjacency graphs
- Rain/no-rain threshold metrics
- Ensembles
- Multi-seed ablations; full ARIMA coverage
- Other regions

---

# Part 10 — Current Progress

## Completed (~100% of planned implementation)

- Cleaning, EDA, features, temporal audit
- Sequences h=1..4; graph construction & batches
- LSTM & GNN multi-seed multi-horizon
- CNN-LSTM-Temporal & Transformer ablations (h=1)
- Persistence + sampled ARIMA
- DM / paired-t / bootstrap
- Pipeline automation, verification (92/100 stage scorecard), final audit docs

## Partially completed

| Item | % | Notes |
|------|--:|-------|
| Metrics persistence | ~70 | Some MAE/MSE/R² aggregates N/A; ablation metrics not JSON |
| Documentation consistency | ~85 | Learning guide stale; stale multiseed summary JSON; naming inconsistency h=1 |
| ARIMA baseline | ~40 | Works for sample, not full coverage / not saved |
| Paper / thesis writing | ~10 | Artifacts ready; prose not in repo |
| Learning guide accuracy | ~60 | Parts claim advanced models missing |

## In progress

None claimed in status docs (implementation frozen 2026-07-29).

## Planned / Not started

| Item | Approx % |
|------|---------:|
| Thesis/paper manuscript | 0–10 |
| Spatial CNN-LSTM-Attention (literal paper) | 0 |
| Web UI / API / Auth / Deploy | 0 |
| Climatology baseline script | 0 (persistence exists; climatology weakly evidenced) |
| Attention visualizations | 0 |
| Docker packaging | 0 |

### Module completion percentages (implementation scope)

| Module | % |
|--------|--:|
| Data pipeline | 100 |
| Feature engineering | 100 |
| Sequence generation | 100 |
| Graph pipeline | 100 |
| LSTM experiments | 100 |
| GNN experiments | 100 |
| Ablation models | 90 (single-seed) |
| Statistical testing | 100 |
| Classical baselines | 75 |
| Documentation (code-facing) | 95 |
| Documentation (learning guide sync) | 60 |
| Thesis writing | 10 |
| Productization | 0 |
| **Overall planned ML implementation** | **~100** |
| **Overall research delivery including paper** | **~75–80** |

---

# Part 11 — TODO List

## High Priority

1. Write thesis/paper with correct framing: **adaptation/extension**, not literal spatial-CNN reproduction
2. Cite canonical multi-horizon table (CUDA+autocast), not stale JSON/README variants
3. Refresh or delete `lstm_baseline_v2_multiseed_summary.json`; align README headlines
4. Obtain/record base paper bibliographic citation (title/authors) — currently missing from repo
5. Ensure raw Excel availability for any external reproduction

## Medium Priority

1. Persist full MAE/MSE/R² aggregates for multi-horizon multi-seed runs
2. Save ablation metrics JSON (CNN/Transformer)
3. Move root `*.log` into `reports/logs/`
4. Unify h=1 checkpoint naming (`lstm_h1_*`, `gnn_lstm_h1_*`)
5. Update learning guide Parts 14/17 to reflect completed GNN/Transformer/CNN-temporal work
6. Resolve nested `.git` under `RainfallPrediction/` (repo hygiene)

## Low Priority

1. Add/renumber notebook 03 or document gap in README only
2. Multi-seed ablations for CNN/Transformer
3. Full-station ARIMA (expensive)
4. Climatology baseline script
5. Fill `results/` or remove unused folder

## Technical Debt

1. Duplicated `split_name` / metrics helpers across many scripts
2. `src/preprocess.clean_dataset` vs `run_pipeline.step_clean` divergence
3. Hardcoded Windows absolute paths in `cuda_setup` / `run_pipeline`
4. v1 path retained alongside locked v2 (confusing for newcomers)
5. Large graph NPY disk footprint

## Bug Fixes

1. No open critical training bugs reported in FINAL_AUDIT
2. Historical: stray `5` prefix in `generate_sequences_v2.py` — **already fixed**
3. Watch for y-scaler missing if someone runs v1 `generate_sequences.py` + `train_lstm_baseline.py` alone

## Performance

1. GNN batch_size=1 is slow; investigate larger safe batches
2. Graph tensor generation memory/disk heavy
3. Consider half-precision storage for archived NPY

## Security

1. No secrets found; low risk (offline research)
2. Do not commit raw proprietary datasets if restricted by license (currently gitignored pattern)

## UI Improvements

N/A unless productization starts.

## Testing

1. No automated unit/integration test suite — add smoke tests for leakage asserts, scaler fit scope, shapes
2. CI not present

## Deployment

1. Not in scope; if needed: containerize CUDA env + artifact volumes

## Documentation

1. Sync learning guide with FINAL_AUDIT
2. Add paper BibTeX when known
3. This complete technical doc (knowledge transfer)

---

# Part 12 — Code Quality Review

| Dimension | Assessment | Suggestions |
|-----------|------------|-------------|
| Structure | Clear stage scripts + small `src/` | Extract shared `splits.py`, `metrics.py`, `paths.py` |
| Naming | Mostly clear; h=1 naming inconsistent | Rename/symlink checkpoints |
| Duplication | High across train scripts | Shared train loop utilities |
| Performance | Good for LSTM; GNN slow | Profile GNN loader; mixed precision storage |
| Scalability | Single-GPU research scale | Fine; not multi-node ready |
| Maintainability | Good docs; some stale guides | Single source of truth for results |
| Security | Minimal attack surface | Keep ignoring large data; no web surface |
| Patterns | Script-oriented procedural | Acceptable for research; optional light package |
| SOLID / Clean Arch | Weak (monolithic scripts) | OK for thesis code; refactor only if extending heavily |
| Verification | Strong independent audit | Keep canonical eval path pinned |

---

# Part 13 — Missing Components

| Issue | Detail |
|-------|--------|
| No REST API / frontend / DB / auth | By design for current scope |
| No Docker | Missing for portable repro |
| No automated tests | Missing |
| Empty `data/external`, `results/`, `reports/logs/` | Placeholders |
| Notebook 03 | Intentionally absent |
| Spatial CNN-LSTM-Attention | Missing vs paper primary architecture |
| Attention module (`CNNLSTMAttention`) | Implemented (additive Bahdanau over LSTM hidden states); seeds {13,42,123}, h=1–4; significantly better than CNN-LSTM-Temporal at h=2 and h=4; not shown to outperform plain LSTM |
| Ablation metrics JSON | Missing |
| ARIMA saved aggregates | Missing |
| Base paper PDF / citation | **Missing from repository** |
| Learning guide stale claims | Says GNN/Attention 0% — false now |
| Stale multiseed summary JSON | Seed-42 RMSE mismatch |
| Nested git repo | Potential confusion |
| Hardcoded machine paths | Portability issue |
| v1 incomplete y-scaling path | Dead-end risk if used |
| Dead/legacy code | `generate_sequences.py` + `train_lstm_baseline.py` still present but superseded |
| Climatology baseline | Mentioned in guide as weakly evidenced |

No systematic `TODO`/`FIXME`/`NotImplemented` stubs found in current Python sources.

---

# Part 14 — Compare with the Research / Base Paper

## 14.1 What is known about the paper (from repo only)

**Cannot determine title, authors, year, or venue** — not present in repository.

From `FINAL_AUDIT.md`, `src/model.py` comments, and learning guide:

### Core idea (as documented in-repo)

A deep learning rainfall forecasting approach using a **30-day window → scalar rainfall** formulation, with architectures including:
- LSTM baseline (Sec **3.3.1**)
- Spatial **CNN-LSTM** over a **2D lat/lon grid** (Sec **3.3.3**)
- Transformer-style encoder (Sec **3.3.4**)
- Likely **Attention** as part of CNN-LSTM-Attention upgrade path (guide)

### Architecture proposed (paper)

Spatial convolution on a regular geographic grid → temporal LSTM → (attention) → rainfall prediction.

### Algorithms (paper)

Spatial CNN; LSTM; Attention; Transformer encoder variant.

### Workflow (paper)

Grid-based spatial-temporal deep learning for rainfall regression (details beyond section numbers **not available in repo**).

### Components required by paper (inferred)

1. Regular spatial grid of stations/cells  
2. Spatial CNN  
3. LSTM temporal model  
4. Attention (per learning guide upgrade path)  
5. Possibly Transformer comparator  
6. Standard train/eval protocol for rainfall  

## 14.2 Comparison table

| Paper Requirement | Current Implementation | Status | Notes |
| ----------------- | ---------------------- | ------ | ----- |
| LSTM scalar baseline (Sec 3.3.1) | `LSTMBaseline` + locked v2 multi-seed multi-horizon | Implemented exactly (adapted features/data) | Primary strong baseline |
| 30-day input window | Contiguous 30-day windows | Implemented exactly | Contiguity constraint is project-specific rigor |
| Per-station / scalar rainfall target | mm/day regression | Implemented exactly | |
| Spatial CNN on lat/lon grid (Sec 3.3.3) | **Not implemented** | Missing | Irregular 414 stations; no valid 2D grid |
| Replace spatial CNN with GNN (paper future work per FINAL_AUDIT) | `GNNLSTM` + kNN graph | Implemented (extension) | Core novelty of this project |
| CNN-LSTM+Attention (project temporal extension) | `CNNLSTMAttention` — additive Bahdanau over LSTM hidden states; seeds {13,42,123}, h=1–4 | Implemented (adaptation) | Significantly better than CNN-LSTM-Temporal at h=2 and h=4; not shown to outperform plain LSTM (see FINAL_AUDIT.md / `ablation_study.csv` / `significance_results.csv`). Paper spatial CNN-LSTM-Attention stack remains unreproduced. |
| Transformer (Sec 3.3.4) | `TransformerEncoderBaseline` | Modified / ablation | Per-station non-spatial; h=1 seed42 |
| Multi-horizon evaluation | h=1..4 with significance | Additional beyond paper (unknown if paper did this) | Cannot confirm paper’s horizons without PDF |
| Diebold-Mariano / bootstrap | Implemented | Additional | Strengthens claims |
| Indian station Excel dataset | Used | Modified data domain | Paper dataset unknown |
| Target-date leakage-free contiguous protocol (covariate imputation: see FINAL_AUDIT.md) | Strong target-day asserts + audits; covariate imputation not train-only | Additional engineering rigor | |
| Production API | None | N/A | |

## 14.3 Summary vs paper

| Category | Items |
|----------|-------|
| Implemented exactly (spirit) | LSTM baseline; window→scalar; Transformer-as-comparator (partial) |
| Modified | Data domain (India irregular stations); temporal CNN instead of spatial CNN; GNN instead of spatial CNN |
| Missing (vs. base paper) | Spatial CNN; paper dataset reproduction; paper citation artifacts. (Note: the base paper itself has no attention mechanism, so "missing attention vs. paper" does not apply - the project's CNN-LSTM+Attention is an original extension, implemented and evaluated, not a paper-comparison item.) |
| Partially | CNN-LSTM (temporal only); Transformer (single-seed h=1) |
| Beyond paper | Multi-horizon+multi-seed GNN vs LSTM with DM/bootstrap; temporal density audit; station_id disambiguation; verification reports |

### Overall implementation percentage vs paper

**Cannot compute a precise % without the PDF.**  
Using in-repo framing as an **adaptation**:

| Lens | Estimate |
|------|---------:|
| Literal paper reproduction (paper's actual architecture: spatial CNN-LSTM, plus separate Transformer comparator - the paper has no attention mechanism) | **~25-35%** (LSTM + partial Transformer implemented; spatial CNN not reproducible on irregular station data) |
| Stated project goal (adapt paper via GNN for irregular stations + rigorous LSTM baseline) | **~95–100%** of planned experimental scope |

---

# Part 15 — Current Project Status Summary

**Maturity:** Research-implementation complete; publication stage beginning. Independent verification scorecard 92/100 (integrity ~86/100 after doc hygiene penalties).

**Completed work:** Target-date leakage-free and train-only-scaled data pipeline (see FINAL_AUDIT.md for a documented, scope-limited covariate imputation leakage finding, empirically negligible in tested cases but not production-corrected); multi-horizon LSTM & GNN multi-seed experiments; ablations; classical baselines; significance tests; reproduction docs; master results.

**Remaining work:** Thesis/paper writing; doc/artifact hygiene; optional enrichments (ablation multi-seed, full ARIMA, Attention, better graphs).

**Biggest challenges:** Irregular sparse stations blocking spatial CNN; GNN not beating LSTM; daily rainfall inherent difficulty (R²~0.37); AMP/eval-path sensitivity; stale secondary docs/JSON.

**Next milestones:**
1. Thesis narrative with adaptation framing  
2. Finalize tables/figures from canonical logs  
3. Bibliographic citation of base paper  
4. Optional hygiene PR (naming, logs, guide sync)

**Estimated completion:** ML experiments **done**. Thesis depends on writing timeline (typically weeks, not blocked by code).

**Top priorities:** Write-up; cite canonical metrics; fix stale artifacts; record paper citation.

---

# Part 16 — Knowledge Transfer Summary

## For a new developer / AI agent taking over

### Architecture in one paragraph

Offline Python pipeline: clean Indian station Excel → engineer 8 features → build contiguous 30-day sequences with chronological splits and train-only MinMax → train PyTorch models on CUDA → evaluate in mm/day → statistically compare LSTM vs GNN across horizons. Shared nets live in `src/model.py`. Phase-1 orchestration is `run_pipeline.py`; full experiments need the extra scripts in `REPRODUCE.md`.

### Critical files

| File | Why critical |
|------|----------------|
| `generate_sequences_v2.py` | Locked data protocol + leakage asserts |
| `src/model.py` | All architectures |
| `src/cuda_setup.py` | Prevents silent CPU training |
| `multiseed_multihorizon.py` | Canonical results + significance |
| `FINAL_AUDIT.md` / `PROJECT_VERIFICATION_REPORT.md` | Truth about status & caveats |
| `reports/tables/master_results.csv` | Consolidated metrics |
| `mh_multiseed_monitor.log` | Canonical multi-horizon numbers |

### Configuration essentials

- Python 3.12 + CUDA Torch in `Research Project/.venv`
- Raw Excel at `data/raw/india_weather_rainfall_data.xlsx`
- Seeds: 13, 42, 123
- Splits: ≤2022 / 2023 / 2024+
- Features: 8 listed above; SEQ_LEN=30
- Eval: CUDA+autocast for paper numbers

### Current blockers

- None for planned coding (frozen complete)
- External: need base paper citation; disk space for ~7GB processed data; GPU for training

### Development workflow

1. Activate CUDA venv  
2. Prefer regenerating via scripts rather than hand-editing NPY  
3. Never shuffle across time; never fit scalers on test  
4. After any retrain, refresh summary JSON / master table  
5. Do not treat learning guide Part 14 as current status  

### Common mistakes

1. Using system CPU Python → wrong/slow training  
2. Citing stale `lstm_baseline_v2_multiseed_summary.json` (9.4584) instead of canonical ~9.37–9.42  
3. Evaluating FP32-only and expecting bitwise match to autocast logs  
4. Claiming spatial CNN-LSTM-Attention is implemented  
5. Claiming GNN beats LSTM (opposite is true)  
6. Gap-filling missing days (violates project methodology)  
7. Assuming GitHub clone includes Excel/NPY/checkpoints (often gitignored)

### Future roadmap

1. Thesis writing (primary)  
2. Doc/artifact hygiene  
3. Optional: Attention model, better graphs, threshold metrics, multi-seed ablations, packaging  

### What another AI must not invent

- Do not invent paper title/authors  
- Do not invent REST APIs or databases  
- Do not mark spatial CNN as done  
- Do not ignore AMP canonical-path caveat  

---

# Appendix A — Quick command cheat sheet

```powershell
cd "D:\project\Research Project\RainfallPrediction"
& "D:\project\Research Project\.venv\Scripts\python.exe" run_pipeline.py
& "D:\project\Research Project\.venv\Scripts\python.exe" generate_sequences_multihorizon.py
& "D:\project\Research Project\.venv\Scripts\python.exe" build_station_graph.py
& "D:\project\Research Project\.venv\Scripts\python.exe" build_graph_batches.py
& "D:\project\Research Project\.venv\Scripts\python.exe" build_graph_batches_multihorizon.py
& "D:\project\Research Project\.venv\Scripts\python.exe" train_lstm_baseline_v2_multiseed.py
& "D:\project\Research Project\.venv\Scripts\python.exe" train_lstm_multihorizon.py
& "D:\project\Research Project\.venv\Scripts\python.exe" train_gnn_lstm.py
& "D:\project\Research Project\.venv\Scripts\python.exe" multiseed_multihorizon.py
& "D:\project\Research Project\.venv\Scripts\python.exe" train_cnn_lstm_temporal_h1.py
& "D:\project\Research Project\.venv\Scripts\python.exe" train_transformer_h1.py
```

---

# Appendix B — Document confidence notes

| Claim area | Confidence |
|------------|------------|
| File inventory & ML pipeline | High — read from repo |
| Metrics tables | High — match verification report / master CSV |
| Base paper identity | **Low / unknown** — not in repo |
| Learning guide advanced-model status | Outdated — superseded by FINAL_AUDIT |
| APIs/DB/frontend | High confidence they do not exist |

---

*End of Complete Technical Documentation.*
