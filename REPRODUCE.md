# Reproduction Guide

Complete instructions to reproduce all results from a clean environment.

---

## 1. Prerequisites

- **Python:** 3.12.x (tested with 3.12)
- **GPU:** NVIDIA GPU with CUDA support (tested on RTX 2050, CUDA 12.6)
- **OS:** Windows 10/11 (PowerShell commands below; adapt for Linux)
- **Disk:** ~2 GB for data + models + venv

---

## 2. Environment Setup

```powershell
cd "D:\project\Research Project"

# Create virtual environment
python -m venv .venv

# Activate
& ".venv\Scripts\Activate.ps1"

# Install CUDA PyTorch (adjust cu126 to match your CUDA driver)
pip install torch --index-url https://download.pytorch.org/whl/cu126

# Install remaining dependencies
cd RainfallPrediction
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Expected: 2.x.x+cu126 True
```

---

## 3. Dataset Placement

Place the raw dataset at:

```
RainfallPrediction/data/raw/india_weather_rainfall_data.xlsx
```

This is an Excel file with ~970k rows of Indian meteorological station observations (2015–2025).

---

## 4. Data Pipeline (run_pipeline.py)

The pipeline handles cleaning, feature engineering, sequence generation, and scaling:

```powershell
python run_pipeline.py
```

This produces:
- `data/processed/clean_dataset.csv` — cleaned data (712,785 rows)
- `data/processed/feature_engineered_v2.csv` — with doy_sin/cos + station_id
- `data/processed/X_train_v2.npy`, `y_train_v2.npy`, etc. — h=1 sequences
- `models/minmax_scaler_y_v2.joblib` — target scaler

To skip already-completed steps: `python run_pipeline.py --skip-train`

---

## 5. Multi-Horizon Sequence Generation

For horizons h=2,3,4:

```powershell
python generate_sequences_multihorizon.py
```

This produces `X_train_h{2,3,4}.npy`, `y_train_h{2,3,4}.npy`, etc.

---

## 6. Graph Data Preparation (for GNN-LSTM)

```powershell
python build_station_graph.py
python build_graph_batches.py
python build_graph_batches_multihorizon.py
```

This produces:
- `data/processed/station_graph_edges.csv` — 3,856 edges
- `data/processed/station_id_to_index.json` — 414 nodes
- `models/adjacency_norm.pt` — normalized adjacency matrix
- `data/processed/X_train_graph.npy`, etc. — graph-format tensors

---

## 7. Model Training Order

### 7a. LSTM Baseline (h=1, multi-seed)

```powershell
python train_lstm_baseline_v2_multiseed.py
```

Outputs: `models/lstm_baseline_v2_seed{13,42,123}.pt` + metrics JSON

### 7b. LSTM Multi-Horizon (h=2,3,4)

```powershell
python train_lstm_multihorizon.py
```

Outputs: `models/lstm_h{2,3,4}_seed{13,42,123}.pt`

### 7c. GNN-LSTM (h=1)

```powershell
python train_gnn_lstm.py
```

Outputs: `models/gnn_lstm_seed{13,42,123}.pt`

### 7d. GNN-LSTM Multi-Horizon + Full Evaluation

```powershell
python multiseed_multihorizon.py
```

Trains GNN h=2,3,4 (if missing), evaluates all horizons, runs statistical significance tests.

### 7e. Ablation Models (h=1 only, seed 42)

```powershell
python train_cnn_lstm_temporal_h1.py
python train_transformer_h1.py
```

Outputs: `models/cnn_lstm_temporal_h1_seed42.pt`, `models/transformer_h1_seed42.pt`

---

## 8. Evaluation

Test metrics are printed to stdout during training and saved to:
- `models/lstm_baseline_v2_seed*_metrics.json` (h=1 per-seed)
- `models/lstm_baseline_v2_multiseed_summary.json` (h=1 summary)
- Monitor logs: `mh_multiseed_monitor.log` (canonical multi-horizon results)

The multi-horizon comparison table is written to stderr by `multiseed_multihorizon.py`.

---

## 9. Output Locations

| Artifact | Location |
|----------|----------|
| Cleaned data | `data/processed/clean_dataset.csv` |
| Feature-engineered data | `data/processed/feature_engineered_v2.csv` |
| Sequences (h=1) | `data/processed/X_*_v2.npy`, `y_*_v2.npy` |
| Sequences (h=2,3,4) | `data/processed/X_*_h{2,3,4}.npy` |
| Graph tensors | `data/processed/X_*_graph*.npy` |
| Model checkpoints | `models/*.pt` |
| Scalers | `models/*.joblib` |
| Training curves | `reports/figures/*.png` |
| Results table | `reports/tables/master_results.csv` |
| Verification report | `PROJECT_VERIFICATION_REPORT.md` |

---

## 10. Notes on Reproducibility

- **AMP sensitivity:** RMSE values may differ by ~0.0005 between CUDA+autocast and pure FP32 evaluation paths. Always use the same eval settings as the training scripts for consistent numbers.
- **Checkpoint naming:** h=1 LSTM checkpoints are named `lstm_baseline_v2_seed*.pt` (not `lstm_h1_*`); h=1 GNN checkpoints are named `gnn_lstm_seed*.pt` (not `gnn_lstm_h1_*`).
- **GNN train filter:** dates with <20% valid stations are excluded from GNN training only (documented design choice).
- **CUDA required:** All training scripts require CUDA. CPU-only evaluation is possible but may produce slightly different metric values.
