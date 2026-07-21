# Rainfall Prediction using Deep Learning

**University research project** — current stage: **data pipeline + LSTM baseline**

Daily rainfall forecasting from Indian meteorological station observations using a PyTorch LSTM baseline (adapted from the base paper’s Section 3.3.1 for per-station scalar targets).

---

## Project overview

| Item | Detail |
|------|--------|
| Task | Next-day rainfall regression (mm/day) |
| Input window | 30 contiguous calendar days → predict day 31 |
| Model | 2-layer LSTM (64 units) → FC → 1 output |
| Split | Train ≤2022 · Val 2023 · Test 2024–Feb 2025 |
| Locked baseline | **v2** (8 features including past rainfall) |

**Headline test results (v2, 3 seeds):**  
RMSE **9.39 ± 0.06** mm/day · R² **0.375 ± 0.008**  
(vs persistence RMSE 11.56 / R² 0.05)

---

## Dataset

- Source: India weather / rainfall Excel (`data/raw/india_weather_rainfall_data.xlsx`)
- After cleaning: **712,785** rows, **414** disambiguated stations (`station_id`)
- Features used by the LSTM (v2):  
  `avg_temp`, `min_temp`, `max_temp`, `wind_speed`, `air_pressure`, `rainfall` (past only), `doy_sin`, `doy_cos`

---

## Folder structure

```
RainfallPrediction/
├── data/
│   ├── raw/                 # place Excel here
│   └── processed/           # clean + feature CSVs, .npy arrays
├── models/                  # .pt checkpoints, MinMax scalers
├── notebooks/
│   ├── 01_Data_Preprocessing.ipynb
│   ├── 02_EDA.ipynb
│   ├── 04_feature_engineering.ipynb
│   └── 05_sequence_generation.ipynb
├── reports/figures/         # EDA + training plots
├── src/
│   ├── model.py             # LSTMBaseline
│   ├── cuda_setup.py        # GPU helpers
│   └── preprocess.py        # path helpers
├── run_pipeline.py          # end-to-end orchestration
├── generate_sequences_v2.py # preferred sequence builder
├── train_lstm_baseline_v2.py
└── requirements.txt
```

> Note: notebook `03` was reserved / skipped; numbering jumps from 02 → 04 by design history.

---

## Workflow

```
Raw Excel
  → Cleaning (drop missing rainfall; station-wise fills)
  → Feature engineering (doy_sin/cos + station_id)
  → Temporal density audit
  → Sequence generation (contiguous 30-day windows, no gap-fill)
  → MinMax scaling (fit on train only)
  → LSTM training (CUDA)
  → Metrics in mm/day (inverse-transform y)
```

---

## How to run

### 1. Environment (use the CUDA venv)

```powershell
cd "D:\project\Research Project\RainfallPrediction"

# Always use the project venv — system Python is often CPU-only
& "D:\project\Research Project\.venv\Scripts\python.exe" -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.13.0+cu126 True` and GPU name `NVIDIA GeForce RTX 2050`.

Install deps if needed:

```powershell
& "D:\project\Research Project\.venv\Scripts\python.exe" -m pip install -r requirements.txt
```

### 2. Full pipeline (skips steps that already have outputs)

```powershell
& "D:\project\Research Project\.venv\Scripts\python.exe" run_pipeline.py
```

Useful flags:

```powershell
python run_pipeline.py --skip-train
python run_pipeline.py --from sequences
python run_pipeline.py --from train --force
```

### 3. Train LSTM v2 only

```powershell
& "D:\project\Research Project\.venv\Scripts\python.exe" train_lstm_baseline_v2.py
```

Multi-seed (13, 42, 123):

```powershell
& "D:\project\Research Project\.venv\Scripts\python.exe" train_lstm_baseline_v2_multiseed.py
```

---

## Requirements

See `requirements.txt` (pandas, numpy, matplotlib, scikit-learn, **PyTorch**, joblib, openpyxl).

For GPU training install a **CUDA** build of PyTorch matching your driver (this project uses `torch 2.13.0+cu126`).

---

## Current progress

| Phase | Status |
|-------|--------|
| Data cleaning | Done |
| EDA + temporal audit | Done |
| Feature engineering | Done |
| Sequence generation (v1 & v2) | Done |
| LSTM baseline (v2, multi-seed) | Done |
| Classical baselines (persistence, climatology) | Done |
| Advanced models (CNN-LSTM-Attention / GNN) | Future |

---

## Future work (second review+)

- CNN-LSTM-Attention (base paper architecture)
- Spatial / graph models using lat/lon/elevation metadata
- Additional metrics (tolerance accuracy, rain/no-rain)
- Paper-style tables and ablation study write-up

---

## License / academic use

Prepared for university project review. Cite the base paper and dataset source in your report as required by your institution.
