"""
config.py — Central configuration for the LSI ML Pipeline.

Edit FEATURE_COLUMNS to add new FR features as they become available
from QGIS without touching any other module.
"""

# ─── FEATURE COLUMNS ────────────────────────────────────────────────────────
# Single source of truth. Currently 3 active; extend to 17 by appending.
FEATURE_COLUMNS: list[str] = [
    # --- Currently Available ---
    "slope_fr",
    "aspect_fr",
    "elevation_fr",
    # --- To be added after QGIS FR analysis is complete ---
    # "rainfall_fr",
    # "earthquake_fr",
    # "distance_drainage_fr",
    # "tri_fr",
    # "gravity_anomaly_fr",
    # "distance_faults_fr",
    # "sti_fr",
    # "twi_fr",
    # "spi_fr",
    # "distance_roads_fr",
    # "ndvi_fr",
    # "geomorphology_fr",
    # "geology_fr",
    # "soil_fr",
    # "lulc_fr",
]

# ─── TARGET COLUMN ──────────────────────────────────────────────────────────
TARGET_COLUMN: str = "target"
COORD_COLUMNS: list[str] = ["x", "y"]

# ─── SPATIAL / SAMPLING CONSTANTS ───────────────────────────────────────────
# Aizawl, Mizoram study area bounding box (UTM Zone 46N, EPSG:32646, metres)
STUDY_BBOX: dict[str, float] = {
    "x_min": 520_000.0,
    "x_max": 545_000.0,
    "y_min": 2_580_000.0,
    "y_max": 2_610_000.0,
}

# Minimum Euclidean distance (metres) between a non-landslide sample
# and any known landslide point during random generation.
MIN_SAMPLE_BUFFER_M: float = 500.0

# ─── SPATIAL LOOCV BUFFER ───────────────────────────────────────────────────
# Points within this radius (metres) of the held-out point are excluded
# from the training fold during buffered Leave-One-Out CV.
BUFFER_RADIUS: float = 1_000.0  # 1 km

# ─── LOW-RISK FR PLACEHOLDER RANGE ──────────────────────────────────────────
# Fallback: when no raster is available to sample real pseudo-absence FR values,
# draw from the FULL study-area FR range so classes overlap realistically.
# NOTE: The preferred approach is to use extract_training_data.py which samples
# real raster values at random spatial locations (not synthetic placeholder values).
FR_LOW_MIN: float = 0.0
FR_LOW_MAX: float = 1.5  # widened to cover real Aizawl raster value range

# ─── MODEL HYPERPARAMETERS ──────────────────────────────────────────────────
RANDOM_STATE: int = 42

RF_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": None,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LR_PARAMS: dict = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# ─── CROSS-VALIDATION ───────────────────────────────────────────────────────
N_FOLDS: int = 5

# ─── SUSCEPTIBILITY ZONE LABELS ─────────────────────────────────────────────
ZONE_LABELS: list[str] = ["Very Low", "Low", "Moderate", "High", "Very High"]
N_ZONES: int = len(ZONE_LABELS)

# ─── OUTPUT FILE NAMES ──────────────────────────────────────────────────────
OUT_FULL_DATASET: str = "landslide_full_dataset.csv"
OUT_RF_MODEL: str = "rf_model.pkl"
OUT_LR_MODEL: str = "lr_model.pkl"
OUT_SCALER: str = "scaler.pkl"
OUT_ROC_PLOT: str = "roc_comparison.png"
OUT_FI_PLOT: str = "feature_importance.png"
OUT_SUSC_SCORES: str = "susceptibility_scores.csv"
OUT_SUSC_CLASS: str = "susceptibility_classified.csv"
OUT_METRICS: str = "model_metrics_report.txt"
