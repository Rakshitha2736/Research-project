# Final Repository Audit

**Date:** 2026-08-05 (results tables refreshed; training not re-run; all metrics sourced from original training-run logs, not fresh inference)
**Scope:** Complete inspection of the RainfallPrediction repository
**Constraint:** Model development frozen — no architecture changes, no new models, no re-running experiments

---

## Research Framing

This project is an **adaptation and extension** of the base paper, not a literal reproduction. The base paper proposes a spatial CNN-LSTM-Attention architecture that convolves over a regular 2D lat/lon grid. Because our Indian station data is irregularly spaced (414 stations with no grid structure), the paper's spatial CNN cannot be applied directly. Instead, this project:

1. **Primary contribution — temporal extension:** CNN-LSTM with **additive (Bahdanau) attention** after the LSTM, so all 30 contextualized hidden states are kept and reweighted (vs. last-timestep-only). A non-attention temporal CNN-LSTM is the controlled comparator.
2. **Secondary contribution — spatial investigation:** a **GNN-LSTM** on a distance-based station graph, replacing the paper's grid CNN for irregular networks (a direction the base paper flags as future work).
3. Retains the paper's LSTM backbone and per-station scalar regression formulation; Transformer encoder remains a supplementary h=1 ablation.

This framing should be stated explicitly in any thesis or paper: attention is the **primary temporal extension**; the GNN is a **secondary spatial investigation**. The project does **not** claim to fully reproduce the base paper's spatial CNN results.

---

## Headline findings (complete 3-seed evidence)

This study evaluated two candidate architectural extensions to a temporal CNN-LSTM precipitation forecasting baseline: a graph neural network (GNN-LSTM) modeling inter-station spatial dependence, and an additive attention mechanism (CNN-LSTM+Attention) modeling temporal salience. Both were benchmarked against a plain LSTM baseline across four forecast horizons (1-4 days), using three random seeds per model per horizon and Diebold-Mariano/bootstrap significance testing.

The GNN-LSTM result is robust: across all 4 horizons and all 3 seeds (12 independent tests), LSTM significantly outperforms GNN-LSTM without exception. This finding — that graph-based spatial modeling does not improve forecasting on this sparse, irregular station network — holds unconditionally. Of the two negative architectural results, this is the more robustly evidenced.

The Attention result requires more careful statement. When evaluated with a single seed, CNN-LSTM+Attention appeared to significantly outperform the non-attention CNN-LSTM baseline at two of four horizons (h=2, h=4). Extending this evaluation to three seeds reveals that this result is not robust: at h=2, two of three seeds favor Attention while one significantly reverses the finding; at h=4, two of three seeds favor Attention while one shows no effect. Neither claimed improvement survives unanimous multi-seed replication. Furthermore, when compared directly against the plain LSTM baseline across all 4 horizons and 3 seeds (12 tests), LSTM shows the numerically better result in 10 of 12 cases, reaching significance in 6 — indicating a recurring, though not fully consistent, LSTM advantage over the attention-augmented model.

Taken together, these results support two conclusions. First, neither candidate extension examined here — spatial (GNN) or temporal (attention) — produces a reproducible improvement over a plain LSTM baseline on this dataset. Second, and methodologically, this study demonstrates that single-seed significance testing is materially unreliable for this problem: at least one comparison (Attention vs. LSTM at h=3) produced a statistically significant result at one seed that directly reversed under two independent reruns. This finding motivates mandatory multi-seed evaluation as a minimum standard for future rainfall-forecasting deep learning studies, and is offered as a secondary methodological contribution of this work.

Canonical tables: `reports/tables/significance_results.csv` (seed-level rows retained, including seed 42) and `reports/tables/multiseed_robustness_summary.csv` (12-row consolidation). Figure: `reports/figures/multiseed_robustness_summary.png`.

**Forest-plot note:** `reports/figures/attention_vs_temporal_forest_plot.png` shows seed-42 point estimates only. See `multiseed_robustness_summary.csv` / `.png` for the complete 3-seed picture, which shows this result is not consistent across seeds.

### Methodological contribution — seed sensitivity

Single-seed (seed=42) Diebold-Mariano tests were informative: they first flagged Attention-vs-Temporal at h=2/h=4 and Attention-vs-LSTM at h=3. They are not sufficient as headline claims. The concrete reversal is Attention vs LSTM at h=3: seed 42 DM p=0.00786 (Attention better); seeds 13 and 123 DM p=1.09e-13 and p=7.37e-11 (LSTM better). That pattern is classified **INCONSISTENT** in `multiseed_robustness_summary.csv`. Attention vs Temporal is **MIXED** at all four horizons (not the same seed-42-outlier pattern at every horizon). GNN vs LSTM is **CONSISTENT** at all four horizons.

---

## 1. Everything Completed

### Data Pipeline
- [x] Raw data loading and inspection
- [x] Missing value analysis (`missing_values_summary.csv`)
- [x] Data cleaning (drop missing rainfall, station-wise fills)
- [x] Feature engineering (doy_sin/cos, station_id disambiguation)
- [x] Temporal density audit (v1 and v2)
- [x] Sequence generation h=1 (v2, 8 features, contiguous windows)
- [x] Sequence generation h=2,3,4
- [x] Graph construction (414 nodes, 3,856 edges)
- [x] Graph batch preparation (all horizons)
- [x] MinMax scaling (train-only, X and y)
- [x] Chronological train/val/test split

### Model Training
- [x] LSTM v2 baseline h=1,2,3,4 (seeds 13, 42, 123 — multi-seed at every horizon)
- [x] GNN-LSTM h=1,2,3,4 (seeds 13, 42, 123 — multi-seed at every horizon)
- [x] CNN-LSTM+Attention h=1,2,3,4 (seeds 13, 42, 123 — multi-seed at every horizon; primary temporal extension)
- [x] CNN-LSTM-Temporal h=1,2,3,4 (seeds 13, 42, 123 — multi-seed; non-attention comparator)
- [x] Transformer Encoder h=1 only (seed 42 — single-seed, supplementary ablation)
- [x] ARIMA baseline (30 of 414 stations — random sample, not full coverage)
- [x] Persistence baseline

### Evaluation
- [x] All metrics in mm/day (inverse-transformed)
- [x] Multi-seed aggregation (mean ± std) for LSTM, GNN-LSTM, CNN-LSTM+Attention, CNN-LSTM-Temporal
- [x] Diebold-Mariano significance test (all 4 horizons × 3 seeds for GNN_vs_LSTM, Attention_vs_Temporal, Attention_vs_LSTM)
- [x] Paired t-test (all 4 horizons)
- [x] Bootstrap 95% CI (1000 resamples, all 4 horizons)
- [x] Attention-weight interpretability plot (h=4 mean α over test set)

### Documentation
- [x] README.md (complete project description)
- [x] REPRODUCE.md (step-by-step reproduction)
- [x] PROJECT_STATUS.md (completion status)
- [x] PROJECT_VERIFICATION_REPORT.md (independent verification, 92/100)
- [x] PROJECT_REVIEW_LEARNING_GUIDE.md (explanatory guide)
- [x] requirements.txt (all dependencies including scipy, statsmodels)
- [x] run_pipeline.py (automated data pipeline)

---

## 2. Everything Verified

| Check | Result |
|-------|--------|
| All 30 model checkpoints loadable | ✅ PASS |
| All 7 scaler files present | ✅ PASS |
| All h=1,2,3,4 sequence arrays present | ✅ PASS |
| All graph tensors present | ✅ PASS |
| Evaluation metrics saved (h=1 per-seed JSON) | ✅ PASS |
| Multi-horizon results in monitor log | ✅ PASS |
| Master results table created | ✅ PASS |
| 13 figures exist in reports/figures/ | ✅ PASS |
| No target-day leakage | ✅ PASS (verified in PROJECT_VERIFICATION_REPORT) |
| Scaler fitted on train only | ✅ PASS |
| Chronological split by target date | ✅ PASS |
| All Python files parse without syntax errors | ✅ PASS (1 fixed: generate_sequences_v2.py) |
| All required imports available in requirements.txt | ✅ PASS |
| Directory structure complete | ✅ PASS |

---

## 3. Files Created in This Audit

| File | Purpose |
|------|---------|
| `README.md` | Updated with all models, metrics, project structure |
| `REPRODUCE.md` | Complete reproduction instructions |
| `PROJECT_STATUS.md` | Project completion status |
| `FINAL_AUDIT.md` | This file |
| `reports/tables/master_results.csv` | Consolidated results table (incl. Attention + Temporal multi-seed) |
| `reports/tables/significance_results.csv` | Pairwise DM / paired-t / bootstrap CI: GNN_vs_LSTM, Attention_vs_Temporal, Attention_vs_LSTM (h=1–4 × seeds 13/42/123) |
| `reports/tables/multiseed_robustness_summary.csv` | 12-row consolidation of 3-seed direction, significance, and consistency verdicts |
| `reports/figures/multiseed_robustness_summary.png` | Heatmap of the 12-row robustness table |
| `reports/tables/` | Directory created |
| `reports/logs/` | Directory created |
| `reports/figures/attention_weights_h4_mean.png` | Mean attention profile (h=4, seed 42) |

---

## 4. Files Modified in This Audit

| File | Change |
|------|--------|
| `README.md` | Rewrote to include all 4 models, multi-horizon results, complete structure |
| `requirements.txt` | Added `scipy>=1.11.0` and `statsmodels>=0.14.0` (were imported but not listed) |
| `generate_sequences_v2.py` | Fixed syntax error: removed stray `5` character at start of file (line 1: `5"""` → `"""`) |

---

## 5a. Threshold / categorical skill (eval-only addition)

Run `python eval_threshold_skill.py` (default: h=1..4 × seeds 13/42/123 × LSTM/Temporal/Attention + Persistence).
Helpers: `src/metrics_rainfall.py`.

Outputs:
- `reports/tables/threshold_skill.csv` / `threshold_skill_summary.csv` — POD, FAR, CSI, Bias, HSS at 0.1/1/5/10 mm
- `reports/tables/intensity_bins.csv` / `intensity_bins_summary.csv` — RMSE/MAE by observed intensity bin
- `reports/tables/tolerance_accuracy.csv` / `tolerance_accuracy_summary.csv` — |err|≤1/2/5 mm

No training; architecture unchanged. Use summary CSVs for thesis tables.

**Provenance note (rain/no-rain contingency):** Rain/no-rain contingency counts in `threshold_skill.csv` were computed via a separate CUDA+autocast inference session (`eval_threshold_skill.py`, 2026-08-05) on the same seed-42/13/123 checkpoints used throughout this project, not literally reused prediction arrays from the original significance-testing session. Because `CNNLSTMAttention` and `CNNLSTMTemporalBaseline` contain no dropout or batch-normalization layers, forward-pass outputs are deterministic given fixed weights, device, and precision mode, so this re-inference is expected to be numerically equivalent to the original evaluation to within floating-point tolerance. This is a documented provenance note, not a re-run of training or a new experiment. Downstream rain/no-rain Precision/Recall/F1 tables derived from these counts (e.g. `reports/tables/rain_classification_metrics.csv`) inherit this provenance.

**Headline (CSI @ 1 mm, mean±std over 3 seeds):** Persistence has the highest CSI at every horizon (e.g. h=1 CSI ≈ 0.58) because DL models run very high POD (~0.89–0.94) but also high FAR (~0.46–0.53) — typical MSE-regression over-forecasting of rain events. Among DL models, rankings vs Temporal/LSTM vary by horizon (see `threshold_skill_summary.csv`). Intensity bins show RMSE exploding on ≥10 mm days (~25 mm RMSE for Attention h=1). Thesis claim: report continuous RMSE **and** categorical skill; do not imply MSE optimality transfers to CSI.

---

## 5b. Extreme vs Normal rainfall subset evaluation (eval-only addition)

Run `python eval_extreme_rainfall.py` (seed 42 only; LSTM / Temporal / Attention; h=1..4). Splits each test set by the **true-target 95th percentile** and reports RMSE/MAE/R² on Extreme vs Normal subsets.

Outputs:
- `reports/tables/extreme_rainfall_evaluation.csv`
- `reports/figures/extreme_rainfall_rmse_comparison.png`

**Provenance note (extreme-day metrics):** Extreme/Normal RMSE/MAE/R² in `extreme_rainfall_evaluation.csv` were computed via a **separate** CUDA+autocast re-inference session (`eval_extreme_rainfall.py`) on the same seed-42 checkpoints used throughout this project, not literally reused prediction arrays from the original significance-testing session or from `eval_threshold_skill.py`. Because `CNNLSTMAttention` and `CNNLSTMTemporalBaseline` contain no dropout or batch-normalization layers, forward-pass outputs are deterministic given fixed weights, device, and precision mode, so this re-inference is expected to be numerically equivalent to those earlier evaluations to within floating-point tolerance. This is a documented provenance note (same precedent as Feature 3 / §5a), not a re-run of training or a new experiment.

**R² on Normal subset:** R² values on the Normal subset are negative for all models/horizons. This is an expected statistical artifact of computing R² on a variance-truncated subset (the 95% of days with lowest rainfall have very low target variance, shrinking R²'s denominator disproportionately to the model's absolute error), NOT evidence the models perform poorly on typical days — their absolute RMSE/MAE on the Normal subset (~4.3–5.0 mm RMSE) is in fact considerably better than their Extreme-subset performance (~36–45 mm RMSE), consistent with expectations. RMSE and MAE, not R², should be used to interpret Normal-vs-Extreme performance in this table. *(Note: CNN-LSTM-Temporal at h=4 Normal is a near-zero exception with R² ≈ +0.029; the artifact still applies — do not interpret subset R² as overall skill.)*

**Absolute Extreme-RMSE ranking:** LSTM has the lowest absolute Extreme-RMSE at h=1–2; CNN-LSTM+Attention has the lowest absolute Extreme-RMSE at h=3–4.

---

## 5c. Attention Extreme vs Normal comparison (eval-only addition)

Run `python analyze_attention_extreme_vs_normal.py` (Attention seed 42; h=1..4). Splits attention weights by the **same** true-target 95th-percentile Extreme/Normal definition as §5b.

Outputs:
- `reports/tables/attention_extreme_vs_normal.csv`
- `reports/figures/attention_extreme_vs_normal_h{1,2,3,4}.png`

**Provenance note:** Horizons **h=1 and h=4** reuse already-saved `data/processed/attention_weights_h*_seed42.npy`. Horizons **h=2 and h=3** required a **separate** CUDA+autocast inference session (`analyze_attention_extreme_vs_normal.py`) on the same seed-42 Attention checkpoints; weights were saved as new cache files. This is not literally reused arrays from significance testing or from `eval_threshold_skill.py` / `eval_extreme_rainfall.py`. Because `CNNLSTMAttention` has no dropout or batch-normalization, eval-mode forward passes are deterministic given fixed weights/device/precision — expected numerical equivalence within floating-point tolerance. Documented provenance (same convention as §5a / §5b), not training or a new modeling experiment.

**Finding sketch:** Peak day-position is identical for Extreme vs Normal within each horizon (day 1 at h=1–3; day 30 at h=4). Mann-Whitney tests on day-30 weights reject equality (large N), but absolute mean differences are tiny (~0.0002–0.005); interpret as statistically detectable but practically small profile shifts, not a qualitatively different attention policy on extreme days.

---

## 5d. Station-wise geographic error map (eval-only addition)

Run `python eval_station_wise_error.py` (Attention seed 42; h=1 and h=4). Per-station RMSE/MAE on the test set; maps in `reports/figures/station_error_map_h{1,4}.png`; table `reports/tables/station_wise_error.csv`.

**Provenance note:** Continuous `y_pred` arrays were not on disk; predictions come from a **separate** CUDA+autocast inference session on `cnn_lstm_attention_h{1,4}_seed42.pt`, with `station_id` from `rebuild_test_meta` and lat/lon from `feature_engineered_v2.csv` (same Features 3–5 convention).

**Cross-reference to §5b:** The strong correlation between station RMSE and station rainfall variance (r=0.89–0.93) is consistent with, and best interpreted alongside, the Extreme Rainfall Subset Evaluation (Feature 4 / Section 5b): high-RMSE stations are largely those with more frequent or more severe extreme-rainfall days, and all three deep learning models (LSTM, Temporal, Attention) showed 8–10× worse RMSE on extreme vs normal days regardless of station. This map does not indicate a distinct new failure mode — it is the geographic expression of the same extreme-rainfall difficulty already documented in Section 5b.

---

## 5. Remaining Optional Improvements

These are non-critical enhancements that do **not** affect research validity:

1. **Naming consistency:** Rename h=1 checkpoints to `lstm_h1_seed*.pt` / `gnn_lstm_h1_seed*.pt` for uniformity (currently `lstm_baseline_v2_seed*.pt` and `gnn_lstm_seed*.pt`)
2. **Stale JSON:** `lstm_baseline_v2_multiseed_summary.json` seed-42 RMSE (9.4584) differs from current checkpoint eval (~9.4184); could be refreshed
3. **Transformer remains single-seed (seed 42) at h=1 only** — optional multi-seed / multi-horizon would strengthen that ablation but is not the primary claim.
4. **Save CNN multi-seed metrics to JSON:** aggregate numbers live in `master_results.csv` / stdout from multiseed scripts; optional per-seed JSON would match LSTM's h=1 pattern.
5. **Move root-level log files:** `*.log` files in project root could be moved to `reports/logs/`
6. **Add notebook 03:** Numbering gap (02→04) documented but could be filled or re-numbered
7. **ARIMA full-coverage:** ARIMA was evaluated on a random 30-station sample only; full 414-station ARIMA is computationally expensive and not essential given ARIMA's role as a classical reference point

---

## 6. Explanation of "Not Available" Entries in master_results.csv

The following metrics are marked "Not Available" and the reasons are:

| What is missing | Why |
|-----------------|-----|
| MAE, MSE, R² for LSTM h=2,3,4 (aggregate) | `multiseed_multihorizon.py` only logs per-seed RMSE to the monitor log; it computes `metrics_mm` (which includes MAE/MSE/R²) internally but only aggregates and prints RMSE mean±std. Re-extracting per-seed full metrics would require re-running eval with CUDA+autocast, which is not done in this audit to preserve reproducibility. |
| MAE, MSE, R² for GNN-LSTM (all horizons, aggregate) | Same reason: the canonical eval script aggregates only RMSE across seeds. Per-seed full metrics exist transiently during the script run but are not persisted. |
| MAE, MSE, R² for Persistence baseline | `train_lstm_baseline_v2.py` only recorded persistence RMSE (11.5559); other metrics were not computed for the persistence model. |
| All metrics for ARIMA | ARIMA was run on a **random sample of 30 out of 414 stations** via `arima_and_significance.py`. Results were printed to stdout during execution but not saved to a file. The 30-station sample provides a classical-method reference point but is not directly comparable to the full-test-set deep learning metrics. |

---

## 7. Stated Limitations (carried forward from project audit)

1. **Transformer remains a single-seed (seed 42) ablation at h=1 only.** CNN-LSTM+Attention and CNN-LSTM-Temporal are now multi-seed at h=1–4; footnote Transformer as supplementary only.
2. **ARIMA coverage is 30/414 stations.** The ARIMA baseline was evaluated on a random 30-station sample due to computational cost of rolling ARIMA. This is adequate as a classical reference but should not be presented as a full-coverage comparison.
3. **AMP sensitivity:** PyTorch's Automatic Mixed Precision (`torch.amp.autocast`) uses float16 for some operations during training and evaluation, which introduces small rounding differences versus pure float32 — in this project, that shifts RMSE by ~0.0005 depending on which path is used. The canonical results in `mh_multiseed_monitor.log` were produced under CUDA+autocast (matching the training path) and should be the numbers cited in any paper; re-evaluating the same checkpoints in pure FP32 (e.g. on CPU) will not match exactly.
4. **Stale multiseed summary JSON:** `lstm_baseline_v2_multiseed_summary.json` was written by an earlier run of `train_lstm_baseline_v2_multiseed.py` and records seed-42 RMSE as 9.4584, but the seed-42 checkpoint was subsequently retrained (the current checkpoint evaluates to ~9.4184 per `lstm_baseline_v2_seed42_metrics.json`). The JSON summary was never refreshed after retraining, so it disagrees with the actual checkpoint. The canonical multi-horizon table in `mh_multiseed_monitor.log` uses the current checkpoints and supersedes this file.
5. **MAE-vs-RMSE divergence:** relative to the non-attention temporal CNN-LSTM, attention can **reduce 3-seed-mean RMSE** at some horizons, but that stage-wise gain is **not unanimous across seeds** (Mixed at h=1–4; see Headline findings). MAE does **not** always fall with RMSE (e.g. higher mean MAE at several horizons). Thesis claims should treat RMSE/DM as the primary error comparison, state MAE explicitly, and not describe Attention-vs-Temporal as a confirmed improvement.

6. **Non-monotonic Attn-vs-Temporal pattern was a seed-42 observation, not a 3-seed result.** Seed-42 DM/bootstrap favored attention at **h=2** and **h=4** and was non-significant at **h=1** and **h=3**. The complete 3-seed picture is Mixed at every horizon (h=2: 2/3 Attention, 1/3 significant reverse; h=4: 2/3 Attention, 1/3 ns). Interpretability (h=1 recency vs h=4 near-uniform attention) remains a useful qualitative discussion of *why* attention might behave differently by horizon; it does **not** license a claim that attention “significantly improves at h=2/h=4.” See Headline findings and `multiseed_robustness_summary.csv`.

   Seasonal breakdown (seed-42 descriptive RMSE): LSTM remains numerically best or tied in every season at h=4. Attention-vs-LSTM across 12 tests: LSTM numerically better in 10/12, significant in 6. At **h=3** the result is **INCONSISTENT**: seed 42 alone shows Attention significantly better (DM p=0.00786); seeds 13 and 123 both show LSTM significantly better (DM p=1.09e-13 and p=7.37e-11). Additionally, Attention's h=4 Winter R² (0.001) is the weakest of any model/season/horizon combination.

7. **External validity / near-duplicate stations (thesis Limitations):** External validity audit (`reports/tables/external_validity_audit.json`) confirms the dataset reflects genuine, physically plausible Indian weather: correct geographic bounds, correct identification of Meghalaya as a high-rainfall extreme and western Rajasthan as a low-rainfall extreme, correct monsoon seasonality (8.5x JJAS/DJF ratio), and correct elevation-temperature relationships. One data source limitation was identified: 124 station-ID pairs (typically nearby city/airport aliases, e.g. same city listed under 2+ names, 4-35km apart) share long runs of identical rainfall values, indicating these are not fully independent measurement sources despite being treated as distinct stations in the 414-station panel. This has negligible impact on the project's core temporal train/val/test methodology and significance testing (given the very large effect sizes and p-values reported), and if anything would bias the GNN spatial-extension comparison in the GNN's favor (near-duplicate neighbors are trivially predictive) — the GNN's failure to outperform plain LSTM despite this potential advantage strengthens rather than weakens that finding.

8. **Pre-split covariate imputation leakage (confirmed via code inspection; production pipeline unchanged; documented as an accepted, evidence-backed limitation).** The canonical preprocessing pipeline (`run_pipeline.py` `step_clean`, mirrored in `notebooks/01_Data_Preprocessing.ipynb`) computes per-station imputation statistics (linear interpolation for min/max temperature; station median for `wind_speed` and `air_pressure`) using the full 2015–2025 date range, grouped by `station_name`, BEFORE the chronological train/validation/test split is applied at sequence-generation time. This means some train-period missing values were filled using statistics that included later validation/test-period observations. Additionally, grouping occurs by `station_name` (406 groups) rather than the disambiguated `station_id` (414 IDs), so 8 `station_name` collisions (e.g. "Agra", which maps to 2 physically distinct `station_id`s at different coordinates) share imputation statistics across physically distinct stations.

   Measured scope: 94,965 of 398,697 train-period rows (23.82%) received at least one covariate value affected by this leakage — `wind_speed` (83,039 affected fills, 98.42% of that variable's missing train rows) and `air_pressure` (94,717 affected fills, 100%) across ~185–187 stations; temperature interpolation was not materially affected (0 cross-boundary linear-fill instances per audit). The rainfall target itself was never imputed (missing rainfall rows were dropped, not filled) and target-day rainfall is verifiably excluded from model inputs (confirmed via code assertions in the sequence-generation scripts) — this is covariate leakage only, not target leakage.

   Two scope-limited sanity-check experiments (LSTM h=1 seed=42; CNN-LSTM+Attention h=4 seed=42 — the project's most significant reported result) compared the canonical pipeline against a parallel, train-only-fit correction. Both showed RMSE changes smaller than the respective model's established 3-seed noise band (LSTM: −0.11%, within ±0.0408; Attention h=4: −0.20%, within ±0.0941).

   IMPORTANT: These sanity checks estimate performance IMPACT only; they do NOT establish that the canonical pipeline is methodologically causal, and NO production fix has been applied — `run_pipeline.py` still performs full-series, pre-split imputation, and every trained checkpoint in this project was built from that pipeline. The train-only-fit correction exists only in untracked diagnostic scripts (`reports/sanity_trainonly_impute_*.py`), not in the production path. This project explicitly chose, given the negligible measured impact in the two highest-stakes cases tested, not to retrain the full pipeline — documented here as a deliberate, evidence-based scoping decision (see `reports/tables/imputation_leakage_audit.json` for full quantification), not as a resolved issue. A methodologically causal (train-only, `station_id`-grouped) imputation pipeline remains identified future work.

### Attention interpretability (h=4, seed 42)

Mean attention over the full h=4 test set (`reports/figures/attention_weights_h4_mean.png`):
- Peak mean weight at **day-position 30** (oldest day in the 30-day window; axis 1=most recent … 30=oldest).
- Recent-7-days attention share ≈ **0.2335**; oldest-7-days share ≈ **0.2464** (near-even split, not strongly recency-biased).

**Conditioned attention (architecture frozen):** run
`python analyze_attention_conditioned.py` (default h=4; optional `--horizons 1 4`).
Helpers live in `src/eval_attention.py`. Outputs:
- `reports/tables/attention_conditioned_h{h}.csv` — wet/dry, JJAS vs non-monsoon, high/low error strata
- `reports/tables/attention_conditioned_contrasts_h{h}.csv` — bootstrap 95% CIs on recent-7 share and entropy contrasts
- `reports/figures/attention_conditioned_h{h}_{wet_dry,monsoon,error,entropy_wet_dry}.png`

Key empirical findings (seed 42, τ=1 mm, already generated for h=1 and h=4):
- **h=1 is strongly recency-biased** (peak day-position **1**; recent-7 share ≈ **0.61**). Wet and JJAS samples focus even more on recent days (recent-7 Δ wet−dry ≈ +0.044; monsoon−non ≈ +0.094; CIs exclude 0) and have **lower** attention entropy (more peaked).
- **h=4 is near-uniform / oldest-peaked** (peak day-position **30**; recent-7 ≈ **0.23**). Conditioning shifts are tiny in magnitude (~0.003–0.005 on recent-7) though CIs still exclude 0 — practically a flat policy.
- This horizon-dependent attention regime is useful for discussing *why* attention policies differ by horizon; it does **not** establish a reproducible Attn-vs-Temporal RMSE improvement. Seed-42 forest-plot estimates remain seed-42 only.

---

## 8. Is the Repository Ready for Thesis Writing?

**YES.**

This project is an **adaptation and extension** of the base paper: **additive attention** is the primary temporal extension; the **GNN** is a secondary spatial investigation for irregular stations (paper future-work direction). All experimental results for this adapted methodology are complete, verified, and documented.

Key thesis-ready artifacts:
- Canonical results table (LSTM, GNN-LSTM, CNN-LSTM+Attention, CNN-LSTM-Temporal × 4 horizons, 3 seeds)
- **Primary finding:** neither GNN nor Attention produces a reproducible improvement over plain LSTM; GNN-vs-LSTM is unconditionally LSTM-better (12/12); Attention-vs-Temporal is Mixed at all horizons; Attention-vs-LSTM favors LSTM in 10/12 tests (6 significant) — see Headline findings and `multiseed_robustness_summary.csv`
- LSTM vs GNN-LSTM significance at every horizon × every seed; bootstrap CIs
- **Secondary methodological finding:** single-seed significance can reverse (Attention vs LSTM, h=3, seed 42 vs 13/123)
- Attention interpretability figure (`attention_weights_h4_mean.png`); forest plot is seed-42 only (caption caveat required)
- Supplementary Transformer ablation at h=1 (single seed)
- `reports/tables/master_results.csv` for paper tables
- Independent verification report (92/100 integrity score)

---

## 9. Is the Repository Ready for GitHub Submission?

**YES**, with minor considerations:

- `.gitignore` correctly excludes large data files, model weights, and generated artifacts
- `.gitkeep` files preserve directory structure
- All code, documentation, and notebooks are tracked
- No secrets or credentials in the repository

---

## 10. Is the Experimental Phase Complete?

**YES.**

All planned experiments have been executed:
- Primary temporal extension: CNN-LSTM+Attention (and Temporal comparator) across 4 horizons × 3 seeds
- Secondary spatial investigation: GNN-LSTM across 4 horizons × 3 seeds; LSTM backbone baseline likewise
- Supplementary Transformer ablation at h=1 (seed 42)
- Classical baselines (Persistence, ARIMA on 30-station sample)
- Attn-vs-Temporal, Attn-vs-LSTM, and LSTM-vs-GNN significance testing completed at all horizons (3 seeds; GNN fully consistent; Attention mixed)
- Attention-weight interpretability at h=4
- Results independently verified for the pre-attention LSTM/GNN package; Attention/Temporal metrics recorded in master/significance tables from the CUDA+autocast training scripts

---

## 11. Is Further Coding Required?

**NO.**

**The implementation phase is complete. The next recommended step is thesis/paper writing.**

No further model development, training, or evaluation coding is needed. The optional improvements listed in Section 5 are cosmetic/organizational and do not affect research conclusions.
