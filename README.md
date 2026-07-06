# LSI ML Pipeline — Landslide Susceptibility Index
### IIT Patna B.Tech Research Project | Aizawl, Mizoram

A modular **Machine Learning pipeline** for predicting Landslide Susceptibility Index (LSI) from Frequency Ratio (FR) weighted raster features produced in QGIS.

---

## Overview

The baseline LSI model (Frequency Ratio bivariate statistical method, QGIS) achieves **AUC ≈ 0.77** using 3 parameters. This pipeline extends it to **17 FR-weighted parameters** using scikit-learn, applying spatially-honest **Buffered Leave-One-Out Cross-Validation** (1 km exclusion radius) to prevent data leakage from spatial autocorrelation.

### Study Area
- **Location**: Aizawl, Mizoram, India
- **CRS**: WGS 84 / UTM Zone 46N (EPSG:32646)
- **Landslide inventory**: 22 confirmed events (NASA Global Landslide Catalog)

---

## Results (3 Features: Slope, Aspect, Elevation FR)

| Model | LOOCV AUC | 5-Fold CV AUC | vs FR Baseline |
|-------|:---------:|:-------------:|:--------------:|
| Logistic Regression | **0.808** | 0.870 ± 0.089 | +3.8% |
| Random Forest | 0.727 | 0.750 ± 0.183 | — |
| *QGIS FR Baseline* | *~0.770* | — | — |

### Feature Importance (RF Gini)
| Rank | Feature | Importance |
|:----:|---------|:----------:|
| 1 | `elevation_fr` | 0.488 |
| 2 | `aspect_fr` | 0.361 |
| 3 | `slope_fr` | 0.151 |

---

## Pipeline Architecture

```
lsi_ml_pipeline/
├── run_pipeline.py              ← CLI entry point (argparse)
├── extract_training_data.py     ← Extract FR values from QGIS rasters
├── generate_sample_data.py      ← Synthetic data generator for testing
├── pyproject.toml               ← Dependencies
│
├── lsi_pipeline/
│   ├── config.py                ← ONLY file to edit when adding features
│   ├── data_ingestion.py        ← Steps 1 & 2: load + pseudo-absence generation
│   ├── preprocessing.py         ← Buffered LOOCV + Stratified K-Fold + scaler
│   ├── model_training.py        ← RF + LR training with joblib persistence
│   ├── evaluation.py            ← ROC plot + metrics report
│   ├── feature_importance.py    ← Gini bar chart + ranked table
│   └── susceptibility_map.py   ← Grid prediction + 5-zone classification
│
└── tests/
    ├── test_data_ingestion.py   (8 tests)
    └── test_preprocessing.py    (5 tests)
```

---

## Validation Strategy

**Primary**: Buffered Spatial LOOCV (1 km exclusion radius)
- Each of 44 samples (22 landslide + 22 pseudo-absence) is held out in turn
- All training samples within 1 km of the test point are excluded from the training fold
- Guards against spatial autocorrelation leakage — the correct metric for spatial data

**Secondary**: Stratified 5-Fold CV
- Maintains class balance across folds
- Provides stability estimate (mean ± std)

---

## Quick Start

### 1. Install dependencies
```bash
pip install scikit-learn pandas numpy matplotlib geopandas rasterio pyproj
```

### 2. Generate synthetic test data and run pipeline
```bash
python generate_sample_data.py
python run_pipeline.py --input data/landslide_training_data.csv --output outputs/
```

### 3. Extract from your real QGIS rasters
Edit `extract_training_data.py` to point to your FR raster paths, then:
```bash
python extract_training_data.py
python run_pipeline.py --input data/landslide_training_data.csv --output outputs_real/
```

### 4. Add susceptibility map grid
```bash
python run_pipeline.py \
  --input  data/landslide_training_data.csv \
  --grid   data/aizawl_grid.csv \
  --output outputs_real/
```

### 5. Run tests
```bash
python -m pytest tests/ -v
```

---

## Input CSV Schema

`landslide_training_data.csv` must contain:

| Column | Type | Description |
|--------|------|-------------|
| `x` | float | UTM Easting (m, EPSG:32646) |
| `y` | float | UTM Northing (m, EPSG:32646) |
| `slope_fr` | float | Slope Frequency Ratio |
| `aspect_fr` | float | Aspect Frequency Ratio |
| `elevation_fr` | float | Elevation Frequency Ratio |
| `target` | int | 1 = landslide, 0 = non-landslide |

> Column names are defined in `lsi_pipeline/config.py` → `FEATURE_COLUMNS`. Adding a new FR feature requires only uncommenting it in `config.py` — zero refactoring downstream.

---

## Adding All 17 Features

In `lsi_pipeline/config.py`, uncomment features as their QGIS FR rasters are ready:

```python
FEATURE_COLUMNS: list[str] = [
    "slope_fr",
    "aspect_fr",
    "elevation_fr",
    # "rainfall_fr",            # ← uncomment when raster is ready
    # "earthquake_fr",
    # "distance_drainage_fr",
    # "tri_fr",
    # ... 11 more
]
```

---

## Technical Notes

- **Pseudo-absence generation**: Spatially constrained random sampling from valid raster pixels (≥ 500 m buffer from known landslide points). FR values are sampled directly from the actual QGIS rasters — no synthetic placeholders.
- **Scaler**: `StandardScaler` fitted per-fold inside LOOCV (no leakage), and on the full dataset for final model training.
- **Random seed**: 42 throughout for reproducibility.
- **RF hyperparameters**: `n_estimators=200`, `class_weight='balanced'`, `n_jobs=-1`.

---

## References

- Mandal, B. & Mandal, S. (2018). Spatial modeling and vulnerability assessment of landslide hazards using ML in Darjeeling Himalaya. *Advances in Space Research*.
- Lee, S. & Talib, J.A. (2005). Probabilistic landslide susceptibility and factor effect analysis. *Environmental Geology*.
- NASA Global Landslide Catalog: https://catalog.data.gov/dataset/global-landslide-catalog

---

## License

MIT License — see [LICENSE](LICENSE)
