"""
config.py — Central configuration for the LSI ML Pipeline.

Edit FEATURE_COLUMNS / feature_registry to change the active FR feature set
without touching training, evaluation, or map modules.
"""

from lsi_pipeline.feature_registry import (  # noqa: F401 — re-exported for callers
    FEATURE_REGISTRY,
    FEATURE_SPECS,
    FR_RESULTS_PATH,
    FR_TABLE_PATH,
    LANDSLIDE_POINTS_PATH,
    SONKER_17_DIR,
    TRAINING_CSV_PATH,
)

# ─── FEATURE COLUMNS ────────────────────────────────────────────────────────
# Single source of truth: 17 Sonker Frequency Ratio factors (no aspect_fr).
FEATURE_COLUMNS: list[str] = [spec.column for spec in FEATURE_SPECS]

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
