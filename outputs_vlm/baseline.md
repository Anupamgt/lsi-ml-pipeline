# Phase 0 — Regenerated FR/ML baseline (BENCHMARKS)

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

## Benchmark table

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
| **This regeneration (`outputs_real/`)** | **0.8337** | **0.7707** |
| Curated `results/aizawl_17fr/` | 0.8337 | 0.7707 |
| `VLM_KICKOFF.md` quote | 0.8275 | 0.7707 |

- **LR matches** the kickoff quote and curated results exactly (0.7707).
- **RF matches** curated `results/aizawl_17fr/` (0.8337) and is **+0.0062** vs the older kickoff quote (0.8275). Not catastrophic — pipeline is healthy. Prefer **0.8337 / 0.7707** as the Phase 0 benchmark for all later VLM comparisons.
- Divergence vs 0.8275 is likely an older run / rounding / code-path difference, not a broken pipeline.

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

## Checkpoint note

Stage A was **not** started. Aizawl remains the **test set only** for VLM work.
