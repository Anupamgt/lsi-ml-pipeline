"""Unit tests for lsi_vlm Phase 0 audit helpers and stack band naming."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lsi_vlm.audit import (
    CATEGORICAL_UNIQUE_MAX,
    audit_training_csv,
    classify_kind,
)
from lsi_vlm.stack import default_band_specs, slope_degrees_from_dem


def test_classify_kind_by_filename():
    assert classify_kind(5, "uint8", "S17_03_Slope_classes.tif") == "categorical"
    assert classify_kind(1000, "float32", "S17_01_Rainfall_CHIRPS_mm.tif") == "continuous"
    assert classify_kind(1000, "float32", "DEM_UTF_FINAL.tif") == "continuous"
    assert classify_kind(10, "float32", "mystery.tif") == "categorical"
    assert classify_kind(CATEGORICAL_UNIQUE_MAX + 1, "float32", "mystery.tif") == "continuous"


def test_slope_degrees_from_dem_flat_and_ramp():
    flat = np.ones((5, 5), dtype=np.float32) * 100.0
    s_flat = slope_degrees_from_dem(flat, 30.0, 30.0)
    assert float(np.nanmax(np.abs(s_flat))) < 1e-6

    # Rising 30 m per cell in x → slope = arctan(1) = 45°
    ramp = np.tile(np.arange(5, dtype=np.float32) * 30.0, (5, 1))
    s_ramp = slope_degrees_from_dem(ramp, 30.0, 30.0)
    # Interior should be near 45 degrees
    assert 40.0 < float(s_ramp[2, 2]) < 50.0


def test_default_band_specs_unique_names():
    specs = default_band_specs(Path("."))
    names = [s.name for s in specs]
    assert len(names) == len(set(names))
    assert "rainfall_mm" in names
    assert "slope_dem_deg" in names
    assert any(s.kind == "categorical" for s in specs)
    assert any(s.kind == "derived" for s in specs)


def test_audit_training_csv_near_constant(tmp_path: Path):
    csv_path = Path("data/landslide_training_data.csv")
    if not csv_path.is_file():
        pytest.skip("training CSV not present")
    info = audit_training_csv(csv_path)
    assert info["n_rows"] == 44
    assert info["n_pos"] == 22
    assert "elevation_fr" in info["near_constant_fr"]
    assert "ndvi_fr" in info["extreme_fr"]
    assert info["pseudo_absence_north_extent_m"] > 50_000
