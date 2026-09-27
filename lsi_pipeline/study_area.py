"""
study_area.py — Load and apply study-area YAML configs.

Makes the pipeline transferable beyond Aizawl: paths, CRS, bbox, buffers,
and optional FEATURE_SPECS overrides live in ``configs/<area>.yaml``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from lsi_pipeline.feature_registry import FeatureSpec, REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_STUDY_CONFIG = REPO_ROOT / "configs" / "aizawl.yaml"


@dataclass
class PUBaggingConfig:
    n_bags: int = 10
    sample_ratio: float = 1.0
    inner_buffer_m: float = 500.0
    outer_buffer_m: float | None = None


@dataclass
class FeatureSelectionConfig:
    corr_threshold: float = 0.85
    ga_population: int = 20
    ga_generations: int = 15
    ga_mutation_rate: float = 0.15
    ga_crossover_rate: float = 0.7
    ga_evaluator: str = "rf"


@dataclass
class StudyAreaConfig:
    """Resolved study-area settings (paths absolute relative to repo root)."""

    name: str
    crs: str
    study_bbox: dict[str, float]
    buffer_radius: float
    min_sample_buffer_m: float
    fr_table: Path
    fr_results: Path
    rasters_dir: Path
    landslide_points: Path
    training_csv: Path
    grid_csv: Path
    feature_specs: tuple[FeatureSpec, ...] | None = None
    pu_bagging: PUBaggingConfig = field(default_factory=PUBaggingConfig)
    feature_selection: FeatureSelectionConfig = field(
        default_factory=FeatureSelectionConfig
    )
    config_path: Path | None = None


def _resolve(repo_root: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _parse_feature_specs(
    raw: list[dict[str, str]] | None,
    rasters_dir: Path,
) -> tuple[FeatureSpec, ...] | None:
    if not raw:
        return None
    specs = []
    for item in raw:
        specs.append(
            FeatureSpec(
                column=item["column"],
                fr_table_key=item["fr_table_key"],
                raster_name=item["raster_name"],
                rasters_dir=rasters_dir,
            )
        )
    return tuple(specs)


def load_study_config(
    path: str | Path | None = None,
    repo_root: Path | None = None,
) -> StudyAreaConfig:
    """Load a study-area YAML file and return a resolved ``StudyAreaConfig``.

    Parameters
    ----------
    path:
        Path to YAML. Defaults to ``configs/aizawl.yaml``.
    repo_root:
        Repository root for resolving relative paths. Defaults to package parent.
    """
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    cfg_path = Path(path) if path is not None else DEFAULT_STUDY_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Study-area config not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    paths = raw.get("paths") or {}
    required_path_keys = (
        "fr_table",
        "rasters_dir",
        "landslide_points",
        "training_csv",
        "grid_csv",
    )
    missing = [k for k in required_path_keys if k not in paths]
    if missing:
        raise ValueError(f"Study config missing paths.{missing}: {cfg_path}")

    bbox = raw.get("study_bbox") or {}
    for key in ("x_min", "x_max", "y_min", "y_max"):
        if key not in bbox:
            raise ValueError(f"Study config missing study_bbox.{key}: {cfg_path}")

    rasters_dir = _resolve(repo_root, paths["rasters_dir"])
    pu_raw = raw.get("pu_bagging") or {}
    fs_raw = raw.get("feature_selection") or {}

    cfg = StudyAreaConfig(
        name=str(raw.get("name", cfg_path.stem)),
        crs=str(raw.get("crs", "EPSG:32646")),
        study_bbox={k: float(bbox[k]) for k in ("x_min", "x_max", "y_min", "y_max")},
        buffer_radius=float(raw.get("buffer_radius", 1000.0)),
        min_sample_buffer_m=float(raw.get("min_sample_buffer_m", 500.0)),
        fr_table=_resolve(repo_root, paths["fr_table"]),
        fr_results=_resolve(
            repo_root, paths.get("fr_results", "data/fr_results_17factor.json")
        ),
        rasters_dir=rasters_dir,
        landslide_points=_resolve(repo_root, paths["landslide_points"]),
        training_csv=_resolve(repo_root, paths["training_csv"]),
        grid_csv=_resolve(repo_root, paths["grid_csv"]),
        feature_specs=_parse_feature_specs(raw.get("feature_specs"), rasters_dir),
        pu_bagging=PUBaggingConfig(
            n_bags=int(pu_raw.get("n_bags", 10)),
            sample_ratio=float(pu_raw.get("sample_ratio", 1.0)),
            inner_buffer_m=float(pu_raw.get("inner_buffer_m", 500.0)),
            outer_buffer_m=(
                None
                if pu_raw.get("outer_buffer_m") is None
                else float(pu_raw["outer_buffer_m"])
            ),
        ),
        feature_selection=FeatureSelectionConfig(
            corr_threshold=float(fs_raw.get("corr_threshold", 0.85)),
            ga_population=int(fs_raw.get("ga_population", 20)),
            ga_generations=int(fs_raw.get("ga_generations", 15)),
            ga_mutation_rate=float(fs_raw.get("ga_mutation_rate", 0.15)),
            ga_crossover_rate=float(fs_raw.get("ga_crossover_rate", 0.7)),
            ga_evaluator=str(fs_raw.get("ga_evaluator", "rf")),
        ),
        config_path=cfg_path,
    )
    logger.info("Loaded study-area config '%s' from %s", cfg.name, cfg_path)
    return cfg


def apply_study_config(cfg: StudyAreaConfig) -> StudyAreaConfig:
    """Apply ``cfg`` to mutable module-level settings in config / feature_registry.

    Call once at process start (CLI) before importing FEATURE_COLUMNS consumers
    re-read values — modules that already bound names at import time should
    re-import from ``lsi_pipeline.config`` after this call, or read via
    ``lsi_pipeline.config.FEATURE_COLUMNS`` attribute access.
    """
    import lsi_pipeline.config as config_mod
    import lsi_pipeline.feature_registry as reg

    reg.FR_TABLE_PATH = cfg.fr_table
    reg.FR_RESULTS_PATH = cfg.fr_results
    reg.SONKER_17_DIR = cfg.rasters_dir
    reg.LANDSLIDE_POINTS_PATH = cfg.landslide_points
    reg.TRAINING_CSV_PATH = cfg.training_csv
    reg.TARGET_CRS = cfg.crs

    if cfg.feature_specs is not None:
        reg.FEATURE_SPECS = cfg.feature_specs
    else:
        # Rebind rasters_dir on default specs so raster_path resolves correctly
        reg.FEATURE_SPECS = tuple(
            FeatureSpec(
                column=s.column,
                fr_table_key=s.fr_table_key,
                raster_name=s.raster_name,
                rasters_dir=cfg.rasters_dir,
            )
            for s in reg.FEATURE_SPECS
        )
    reg.FEATURE_REGISTRY = {s.column: s for s in reg.FEATURE_SPECS}

    config_mod.FEATURE_COLUMNS = [s.column for s in reg.FEATURE_SPECS]
    config_mod.STUDY_BBOX = dict(cfg.study_bbox)
    config_mod.BUFFER_RADIUS = cfg.buffer_radius
    config_mod.MIN_SAMPLE_BUFFER_M = cfg.min_sample_buffer_m
    config_mod.FR_TABLE_PATH = cfg.fr_table
    config_mod.FR_RESULTS_PATH = cfg.fr_results
    config_mod.SONKER_17_DIR = cfg.rasters_dir
    config_mod.LANDSLIDE_POINTS_PATH = cfg.landslide_points
    config_mod.TRAINING_CSV_PATH = cfg.training_csv
    config_mod.FEATURE_SPECS = reg.FEATURE_SPECS
    config_mod.FEATURE_REGISTRY = reg.FEATURE_REGISTRY
    config_mod.ACTIVE_STUDY_CONFIG = cfg

    logger.info(
        "Applied study config '%s' - %d features, CRS=%s, buffer=%.0fm",
        cfg.name,
        len(config_mod.FEATURE_COLUMNS),
        cfg.crs,
        cfg.buffer_radius,
    )
    return cfg
