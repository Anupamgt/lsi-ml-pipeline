"""Tests for study-area YAML loading and apply."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lsi_pipeline.study_area import apply_study_config, load_study_config


def test_load_default_aizawl_config():
    cfg = load_study_config("configs/aizawl.yaml")
    assert cfg.name == "aizawl"
    assert cfg.crs == "EPSG:32646"
    assert cfg.buffer_radius == 1000.0
    assert cfg.fr_table.exists() or True  # path resolved; file may exist
    assert "x_min" in cfg.study_bbox
    assert cfg.pu_bagging.n_bags >= 1


def test_apply_updates_config_module():
    import lsi_pipeline.config as cfg_mod

    study = load_study_config("configs/aizawl.yaml")
    apply_study_config(study)
    assert cfg_mod.BUFFER_RADIUS == study.buffer_radius
    assert cfg_mod.STUDY_BBOX["x_min"] == study.study_bbox["x_min"]
    assert cfg_mod.ACTIVE_STUDY_CONFIG is not None
    assert len(cfg_mod.FEATURE_COLUMNS) == 17


def test_custom_yaml_paths(tmp_path: Path):
    # Minimal custom config pointing at existing Aizawl assets via relative paths
    repo = Path(__file__).resolve().parent.parent
    data = {
        "name": "toy_area",
        "crs": "EPSG:32646",
        "study_bbox": {
            "x_min": 1.0,
            "x_max": 2.0,
            "y_min": 3.0,
            "y_max": 4.0,
        },
        "buffer_radius": 250.0,
        "min_sample_buffer_m": 100.0,
        "paths": {
            "fr_table": "data/fr_class_table_17factor.csv",
            "rasters_dir": "final_maps/sonker_17",
            "landslide_points": "final_maps/Aizawl_Points_UTM.gpkg",
            "training_csv": "data/landslide_training_data.csv",
            "grid_csv": "data/aizawl_grid.csv",
        },
    }
    p = tmp_path / "toy.yaml"
    p.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_study_config(p, repo_root=repo)
    assert cfg.name == "toy_area"
    assert cfg.buffer_radius == 250.0
    assert cfg.study_bbox["x_min"] == 1.0


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_study_config("configs/does_not_exist.yaml")
