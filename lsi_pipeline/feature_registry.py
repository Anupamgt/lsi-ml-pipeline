"""
feature_registry.py — Map each FEATURE_COLUMNS entry to FR table key + class raster.

Class integers are sampled from the GeoTIFF, then mapped to raw FR via
``data/fr_class_table_17factor.csv`` (column ``FR``, not ``FRn``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Repo root = parent of lsi_pipeline/
REPO_ROOT = Path(__file__).resolve().parent.parent

FR_TABLE_PATH: Path = REPO_ROOT / "data" / "fr_class_table_17factor.csv"
FR_RESULTS_PATH: Path = REPO_ROOT / "data" / "fr_results_17factor.json"
SONKER_17_DIR: Path = REPO_ROOT / "final_maps" / "sonker_17"
LANDSLIDE_POINTS_PATH: Path = REPO_ROOT / "final_maps" / "Aizawl_Points_UTM.gpkg"
TRAINING_CSV_PATH: Path = REPO_ROOT / "data" / "landslide_training_data.csv"

# CRS used by all Sonker 17 class rasters and for training CSV coordinates
TARGET_CRS: str = "EPSG:32646"


@dataclass(frozen=True)
class FeatureSpec:
    """One training feature ↔ FR analysis factor ↔ class GeoTIFF."""

    column: str
    fr_table_key: str
    raster_name: str

    @property
    def raster_path(self) -> Path:
        return SONKER_17_DIR / self.raster_name


# Order matches FEATURE_COLUMNS in config.py (17 factors, no aspect_fr).
FEATURE_SPECS: tuple[FeatureSpec, ...] = (
    FeatureSpec("rainfall_fr", "rainfall", "P2_03_Rainfall_quantile5.tif"),
    FeatureSpec("earthquake_fr", "earthquake", "S17_02_Earthquake_classes.tif"),
    FeatureSpec("slope_fr", "slope", "S17_03_Slope_classes.tif"),
    FeatureSpec("elevation_fr", "altitude", "S17_04_Altitude_classes.tif"),
    FeatureSpec("distance_drainage_fr", "drainage_dist", "S17_05_Distance_drainages_classes.tif"),
    FeatureSpec("tri_fr", "tri", "S17_06_TRI_classes.tif"),
    FeatureSpec("geomorphology_fr", "geomorphology", "S17_07_Geomorphology_classes.tif"),
    FeatureSpec("geology_fr", "geology", "S17_08_Geology_classes.tif"),
    FeatureSpec("soil_fr", "soil", "S17_09_Soil_India_classes.tif"),
    FeatureSpec("gravity_anomaly_fr", "gravity", "P2_04_Gravity_quantile5.tif"),
    FeatureSpec("distance_faults_fr", "faults_dist", "S17_11_Distance_faults_classes.tif"),
    FeatureSpec("sti_fr", "sti", "S17_12_STI_classes.tif"),
    FeatureSpec("twi_fr", "twi", "S17_13_TWI_classes.tif"),
    FeatureSpec("spi_fr", "spi", "S17_14_SPI_classes.tif"),
    FeatureSpec("distance_roads_fr", "roads_dist", "S17_15_Distance_roads_classes.tif"),
    FeatureSpec("lulc_fr", "lulc", "S17_16_LULC_classes.tif"),
    FeatureSpec("ndvi_fr", "ndvi", "S17_17_NDVI_classes.tif"),
)

FEATURE_REGISTRY: dict[str, FeatureSpec] = {spec.column: spec for spec in FEATURE_SPECS}


def raster_paths_by_column() -> dict[str, Path]:
    """Return ``{feature_column: absolute GeoTIFF path}`` for all registered features."""
    return {spec.column: spec.raster_path for spec in FEATURE_SPECS}


def missing_rasters() -> list[Path]:
    """Return registered GeoTIFF paths that do not exist on disk."""
    return [p for p in raster_paths_by_column().values() if not p.exists()]
