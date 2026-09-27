# Phase 0 — Locked FR/ML benchmarks (canonical for VLM)

**Official locked benchmarks for Phase 0 and all later VLM comparisons:**

| Model | Buffered LOOCV AUC (r = 1 km) |
|---|---|
| Random Forest | **0.8337** |
| Logistic Regression | **0.7707** |

These match regenerated `outputs_real/` and curated `results/aizawl_17fr/`. Use them everywhere Stage A / Stage B report against FR/ML. Do not cite the older kickoff draft figure RF **0.8275**.

Generated on this machine via:

```text
python run_pipeline.py ^
  --input data/landslide_training_data.csv ^
  --grid data/aizawl_grid.csv ^
  --output outputs_real/ ^
  --skip-xgboost ^
  --feature-selection none
```

XGBoost / PU bagging / GA feature selection were **off**. Buffered LOOCV radius = **1000 m** (from `configs/aizawl.yaml`). Study = `aizawl`, 17 FR features, 44 samples (22/22).

## Benchmark table (detail)

| Model | Buffered LOOCV AUC (r=1 km) | 5-fold CV AUC (mean ± std) |
|---|---|---|
| Random Forest | **0.8337** | **0.8625 ± 0.1179** |
| Logistic Regression | **0.7707** | **0.7200 ± 0.1483** |

Per-fold 5-fold AUCs:

- RF: 0.850, 0.700, 0.950, 1.000, 0.812
- LR: 0.700, 0.650, 0.550, 0.950, 0.750

Full-dataset (training-fit) metrics are in `outputs_real/model_metrics_report.txt` — **not for citation**.

## Comparison to prior quoted numbers

| Source | RF buffered LOOCV | LR buffered LOOCV |
|---|---|---|
| **Locked Phase 0 / VLM canonical** | **0.8337** | **0.7707** |
| This regeneration (`outputs_real/`) | 0.8337 | 0.7707 |
| Curated `results/aizawl_17fr/` | 0.8337 | 0.7707 |
| Older kickoff draft quote | 0.8275 | 0.7707 |

- **LR** matches the kickoff draft and curated results exactly (0.7707).
- **RF** matches curated `results/aizawl_17fr/` and this regeneration (0.8337). That is **+0.0062** vs the older kickoff draft quote (0.8275). The 0.8275 figure is treated as an outdated run quote, not the current reproducible pipeline.
- Work continues with **RF 0.8337 / LR 0.7707**. Metrics under `results/aizawl_17fr/` are left as-is; **this file is the canonical reference for VLM comparisons**.

## Stage A constraint (class bands in the stack)

Of the 21 bands in `outputs_vlm/aizawl_stack.tif`, only **6 are continuous** (`rainfall_mm`, `gravity_mgal`, `slope_final_deg`, `elevation_m`, `hillshade`, `slope_dem_deg`). The other **15 are integer class codes** (not continuous values, not one-hot). Phase 0 already lists the missing continuous rebuilds. **Until TWI, SPI, STI, TRI, distance layers, and NDVI are rebuilt as continuous, Stage A prompts must use class names/labels for those 15 bands** (not raw code integers as if they were physical units). See `outputs_vlm/stack_notes.md` and `outputs_vlm/audit.md`.

## Top RF feature importances (full-fit model)

From `outputs_real/rf_model.pkl` (Gini importance; full-dataset fit):

| Rank | Feature | Importance |
|---|---|---|
| 1 | `lulc_fr` | 0.2791 |
| 2 | `distance_faults_fr` | 0.1220 |
| 3 | `ndvi_fr` | 0.1210 |
| 4 | `gravity_anomaly_fr` | 0.1155 |
| 5 | `rainfall_fr` | 0.0997 |
| 6 | `distance_drainage_fr` | 0.0490 |
| 7 | `sti_fr` | 0.0459 |
| 8 | `slope_fr` | 0.0438 |
| 9 | `geomorphology_fr` | 0.0424 |
| 10 | `twi_fr` | 0.0327 |

Near-zero importance (constant FR across the 44 points): `elevation_fr`, `tri_fr`, `geology_fr`, `distance_roads_fr` — consistent with the Phase 0 audit.

## Artefact paths

| Artefact | Path |
|---|---|
| Metrics report | `outputs_real/model_metrics_report.txt` |
| RF / LR models | `outputs_real/rf_model.pkl`, `outputs_real/lr_model.pkl` |
| ROC plot | `outputs_real/roc_comparison.png` |
| Feature importance plot | `outputs_real/feature_importance.png` |
| Run metadata | `outputs_real/run_metadata.txt` |
| Curated twin (leave numbers as-is) | `results/aizawl_17fr/` |

## Checkpoint note

Stage A was **not** started. Aizawl remains the **test set only** for VLM work.
