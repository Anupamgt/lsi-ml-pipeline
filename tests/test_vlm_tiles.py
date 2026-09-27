"""Tests for Stage A tile cutting / composite rendering (no network required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lsi_vlm.tiles import (
    COLOUR_LEGEND_MD,
    LSI_ZONE_COLOURS,
    LSI_ZONE_NAMES,
    TilesConfig,
    build_point_tiles,
    build_study_area_grid,
    iter_study_area_grid,
    load_tiles_config,
    make_tile_window,
    read_window_array,
    render_composite,
    render_satellite_placeholder,
    study_area_bbox_from_points,
    write_colour_legend,
)


REPO = Path(__file__).resolve().parents[1]
TRAINING_CSV = REPO / "data" / "landslide_training_data.csv"
HILLSHADE = REPO / "final_maps" / "HILLSHADE_2.tif"
SLOPE = REPO / "final_maps" / "Slope_FINAL.tif"
CONFIG = REPO / "configs" / "vlm_stage_a.yaml"


def test_colour_legend_documents_lsi_and_slope():
    assert "very_high" in COLOUR_LEGEND_MD.lower() or "very_high" in LSI_ZONE_NAMES[5]
    assert set(LSI_ZONE_COLOURS) == {1, 2, 3, 4, 5}
    assert LSI_ZONE_NAMES[1] == "very_low"
    assert LSI_ZONE_NAMES[5] == "very_high"
    assert "YlOrRd" in COLOUR_LEGEND_MD or "slope" in COLOUR_LEGEND_MD.lower()


def test_make_tile_window_geometry():
    w = make_tile_window(500_000.0, 2_600_000.0, size_px=64, resolution_m=30.0)
    assert w.size_px == 64
    assert w.width_m == pytest.approx(1920.0)
    left, bottom, right, top = w.bounds
    assert right - left == pytest.approx(1920.0)
    assert top - bottom == pytest.approx(1920.0)
    assert (left + right) / 2 == pytest.approx(500_000.0)
    assert (bottom + top) / 2 == pytest.approx(2_600_000.0)
    assert w.transform.a == pytest.approx(30.0)
    assert abs(w.transform.e) == pytest.approx(30.0)


def test_render_composite_shapes_and_lsi_flag():
    rng = np.random.default_rng(0)
    hs = rng.integers(0, 255, size=(32, 32)).astype(np.float32)
    slope = rng.uniform(0, 45, size=(32, 32)).astype(np.float32)
    zones = rng.integers(1, 6, size=(32, 32)).astype(np.float32)

    img = render_composite(hs, slope, include_lsi_overlay=False)
    assert img.size == (32, 32)
    assert img.mode == "RGB"

    img_lsi = render_composite(
        hs, slope, lsi_zones=zones, include_lsi_overlay=True, lsi_alpha=0.5
    )
    assert img_lsi.size == (32, 32)
    # With a strong LSI overlay the images should differ.
    assert not np.array_equal(np.asarray(img), np.asarray(img_lsi))


def test_satellite_placeholder():
    img = render_satellite_placeholder(64, "SATELLITE UNAVAILABLE")
    assert img.size == (64, 64)
    assert img.mode == "RGB"


def test_iter_study_area_grid_counts():
    # 1920 m tiles, no overlap, bbox exactly 2×2 tiles
    bbox = (0.0, 0.0, 3840.0, 3840.0)
    windows = list(
        iter_study_area_grid(bbox, tile_size_px=64, resolution_m=30.0, overlap_px=0)
    )
    assert len(windows) == 4
    assert {(w.row, w.col) for w in windows} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    # Centres of corner tiles
    assert windows[0].center_x == pytest.approx(960.0)
    assert windows[0].center_y == pytest.approx(960.0)


def test_study_area_bbox_from_points():
    if not TRAINING_CSV.is_file():
        pytest.skip("training CSV missing")
    bbox = study_area_bbox_from_points(TRAINING_CSV, buffer_m=1000.0)
    assert bbox[2] > bbox[0] and bbox[3] > bbox[1]


def test_load_tiles_config_yaml():
    if not CONFIG.is_file():
        pytest.skip("vlm_stage_a.yaml missing")
    cfg = load_tiles_config(CONFIG)
    assert cfg.window_size_px == 64
    assert cfg.resolution_m == 30.0
    assert cfg.crs == "EPSG:32646"
    assert "months" in cfg.satellite
    assert set(cfg.satellite["months"]) == {11, 12, 1, 2}
    assert cfg.include_lsi_overlay is False


def test_write_colour_legend(tmp_path: Path):
    path = write_colour_legend(tmp_path)
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "LSI" in text
    assert "slope" in text.lower()


@pytest.mark.skipif(not HILLSHADE.is_file() or not SLOPE.is_file(), reason="rasters missing")
def test_read_window_and_build_two_points(tmp_path: Path):
    cfg = TilesConfig(
        window_size_px=64,
        resolution_m=30.0,
        output_dir=tmp_path / "tiles",
        training_csv=TRAINING_CSV,
        hillshade_path=HILLSHADE,
        slope_path=SLOPE,
        include_lsi_overlay=False,
        satellite={"enabled": False},
    )
    # Smoke-read first training point
    import pandas as pd

    df = pd.read_csv(TRAINING_CSV)
    w = make_tile_window(float(df.iloc[0]["x"]), float(df.iloc[0]["y"]))
    hs = read_window_array(HILLSHADE, w)
    assert hs.shape == (64, 64)
    assert np.isfinite(hs).any()

    results = build_point_tiles(cfg, skip_satellite=True, limit=2)
    assert len(results) == 2
    for r in results:
        assert r.composite.is_file()
        assert r.satellite.is_file()
        with Image.open(r.composite) as im:
            assert im.size == (64, 64)
        with Image.open(r.satellite) as im:
            assert im.size == (64, 64)
    assert (cfg.output_dir / "manifest.json").is_file()
    assert (cfg.output_dir / "colour_legend.md").is_file()


@pytest.mark.skipif(not TRAINING_CSV.is_file(), reason="training CSV missing")
def test_build_study_area_grid_api():
    cfg = load_tiles_config(CONFIG) if CONFIG.is_file() else TilesConfig()
    # Tiny bbox → few tiles; do not render.
    windows = build_study_area_grid(
        cfg, bbox=(500_000.0, 2_600_000.0, 502_000.0, 2_602_000.0)
    )
    assert len(windows) >= 1
    assert all(w.size_px == cfg.grid.get("tile_size_px", cfg.window_size_px) for w in windows)
