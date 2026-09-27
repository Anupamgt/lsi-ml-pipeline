"""Tests for Stage A VLM prompt building and JSON response parsing."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from lsi_vlm.prompts import (
    CATEGORICAL_STACK_BANDS,
    CONTINUOUS_CENTRE_BANDS,
    REASON_MAX_WORDS,
    SUSCEPTIBILITY_CLASSES,
    build_prompt,
    format_centre_values,
    label_for_class_band,
    load_class_label_maps,
    parse_vlm_response,
    validate_prediction,
)


@pytest.fixture(scope="module")
def label_maps() -> dict:
    path = Path("data/fr_class_table_17factor.csv")
    if not path.is_file():
        pytest.skip("FR class table not present")
    maps = load_class_label_maps(path)
    assert "lulc_class" in maps
    return maps


def test_load_class_label_maps_covers_15_bands(label_maps):
    for band in CATEGORICAL_STACK_BANDS:
        assert band in label_maps, f"missing labels for {band}"
        assert label_maps[band], f"empty label map for {band}"


def test_label_for_class_band_uses_names_not_raw_units(label_maps):
    # LULC code 6 → Forest (not "6 m" or similar)
    assert "Forest" in label_for_class_band("lulc_class", 6, label_maps)
    # Slope class 1 has a degree-range *label* from the FR table (class name).
    label = label_for_class_band("slope_class", 1, label_maps)
    assert "0-15" in label or "Sonker" in label
    # Unknown code falls back without inventing physical units as the primary form.
    unknown = label_for_class_band("lulc_class", 99, label_maps)
    assert "class code 99" in unknown


def test_format_centre_values_separates_continuous_and_categorical(label_maps):
    centre = {
        "elevation_m": 812.5,
        "slope_final_deg": 28.1,
        "rainfall_mm": 2450.0,
        "gravity_mgal": -118.0,
        "hillshade": 180.0,
        "lulc_class": 6,
        "ndvi_class": 5,
        "drainage_dist_class": 3,
    }
    text, summary = format_centre_values(centre, label_maps=label_maps)
    assert "elevation_m: 812.5 m" in text
    assert "rainfall_mm" in text
    assert "Categorical class labels" in text
    assert "Forest" in text
    assert "not physical units" in text
    # Summary stores categorical as code+label, not a bare float pretending to be metres.
    assert summary["lulc_class"]["label"]
    assert summary["lulc_class"]["code"] == 6


def test_build_prompt_includes_legend_scale_context_and_json_schema(label_maps):
    centre = {
        "elevation_m": 700.0,
        "slope_dem_deg": 22.0,
        "rainfall_mm": 2400.0,
        "lulc_class": 3,
        "geology_class": 1,
    }
    result = build_prompt(
        scale_bar_m=1920.0,
        centre_values=centre,
        label_maps=label_maps,
        include_numbers=True,
        include_lsi_legend=False,
    )
    text = result.text
    assert "colour legend" in text.lower() or "color legend" in text.lower() or "Composite tile" in text
    assert "1920 m" in text
    assert "Mizoram" in text
    assert "JSON only" in text
    assert "very_low|low|moderate|high|very_high" in text
    assert "Build-up" in text or "Build" in text
    assert "does **not** include the LSI-zone overlay" in text
    assert result.scale_bar_m == 1920.0


def test_build_prompt_ablation_without_numbers(label_maps):
    result = build_prompt(
        scale_bar_m=1000.0,
        centre_values={"elevation_m": 500.0, "lulc_class": 6},
        label_maps=label_maps,
        include_numbers=False,
    )
    assert "omitted for this ablation" in result.text
    assert "elevation_m" not in result.text
    assert result.include_numbers is False


def test_validate_and_parse_happy_path():
    raw = json.dumps(
        {
            "class": "high",
            "score": 72,
            "key_factors": ["steep slope", "built-up LULC"],
            "reason": "Steep cut slope next to road with monsoon saturation risk.",
        }
    )
    pred = parse_vlm_response(raw)
    assert pred is not None
    assert pred.class_ == "high"
    assert pred.score == 72.0
    assert "steep slope" in pred.key_factors


def test_parse_accepts_markdown_fence_and_aliases():
    raw = """```json
{"class": "Very High", "score": 90.5, "key_factors": ["rainfall"], "reason": "Intense monsoon on steep slope."}
```"""
    pred = parse_vlm_response(raw)
    assert pred is not None
    assert pred.class_ == "very_high"
    assert pred.score == 90.5


def test_parse_logs_and_returns_none_on_bad_json(caplog):
    with caplog.at_level(logging.ERROR):
        pred = parse_vlm_response("sorry, I cannot help with that")
    assert pred is None
    assert any("parse failure" in r.message.lower() for r in caplog.records)


def test_validate_rejects_bad_class_and_score():
    with pytest.raises(ValueError, match="invalid class"):
        validate_prediction(
            {"class": "extreme", "score": 10, "key_factors": [], "reason": "x"}
        )
    with pytest.raises(ValueError, match="score out of range"):
        validate_prediction(
            {"class": "low", "score": 101, "key_factors": [], "reason": "x"}
        )


def test_continuous_and_categorical_band_counts():
    assert len(CONTINUOUS_CENTRE_BANDS) == 6
    assert len(CATEGORICAL_STACK_BANDS) == 15
    assert len(SUSCEPTIBILITY_CLASSES) == 5
    assert REASON_MAX_WORDS == 60
