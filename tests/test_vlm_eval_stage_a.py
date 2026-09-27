"""Tests for Stage A evaluation harness (A4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from lsi_vlm.eval_stage_a import (
    LOCKED_LR_AUC,
    LOCKED_RF_AUC,
    MockVLMClient,
    PointPrediction,
    flag_post_event,
    load_config,
    run_stage_a,
    score_predictions,
    summarize_ablations,
    write_results_md,
)
from lsi_vlm.prompts import build_prompt, parse_vlm_response
from lsi_vlm.tiles import load_tiles_config


REPO = Path(__file__).resolve().parents[1]
TRAINING_CSV = REPO / "data" / "landslide_training_data.csv"
CONFIG = REPO / "configs" / "vlm_stage_a.yaml"
HILLSHADE = REPO / "final_maps" / "HILLSHADE_2.tif"
SLOPE = REPO / "final_maps" / "Slope_FINAL.tif"


@pytest.fixture
def tiny_config(tmp_path: Path) -> Path:
    base = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    base["tiles"]["training_csv"] = str(TRAINING_CSV)
    base["tiles"]["output_dir"] = str(tmp_path / "tiles")
    base["tiles"]["hillshade_path"] = str(HILLSHADE)
    base["tiles"]["slope_path"] = str(SLOPE)
    base["tiles"]["satellite"] = {"enabled": False, "months": [11, 12, 1, 2]}
    base["model"]["cache_dir"] = str(tmp_path / "cache")
    base["eval"]["cache_dir"] = str(tmp_path / "cache")
    base["eval"]["output_results"] = str(tmp_path / "stage_a_results.md")
    base["eval"]["output_examples"] = str(tmp_path / "stage_a_examples.md")
    base["eval"]["n_repeats"] = 2
    base["eval"]["n_examples"] = 4
    base["eval"]["skip_satellite"] = True
    path = tmp_path / "vlm_stage_a.yaml"
    path.write_text(yaml.safe_dump(base), encoding="utf-8")
    return path


def test_locked_baselines_constants():
    assert LOCKED_RF_AUC == 0.8337
    assert LOCKED_LR_AUC == 0.7707


def test_config_eval_and_tiles_sections():
    cfg = load_config(CONFIG)
    eval_cfg = cfg["eval"]
    assert eval_cfg["locked_baselines"]["rf_auc"] == 0.8337
    assert eval_cfg["locked_baselines"]["lr_auc"] == 0.7707
    names = {a["name"] for a in eval_cfg["ablations"]}
    assert "no_lsi_numbers" in names
    assert "with_lsi_numbers" in names
    assert any(a.get("include_numbers") is False for a in eval_cfg["ablations"])
    assert any(a.get("honest_baseline") for a in eval_cfg["ablations"])
    # Tiles agent expectations
    tcfg = load_tiles_config(CONFIG)
    assert tcfg.window_size_px == 64
    assert set(tcfg.satellite["months"]) == {11, 12, 1, 2}
    assert tcfg.include_lsi_overlay is False
    # Prompt / model sections preserved for sibling agents
    assert cfg["model"]["backend"] == "ollama"
    assert "prompt" in cfg


def test_parse_vlm_response_accepts_fenced_json():
    text = """```json
{"class": "high", "score": 77, "key_factors": ["slope"], "reason": "steep"}
```"""
    parsed = parse_vlm_response(text)
    assert parsed is not None
    assert parsed.class_ == "high"
    assert parsed.score == 77.0


def test_build_prompt_image_only_omits_numbers():
    with_nums = build_prompt(
        scale_bar_m=1920,
        centre_values={"elevation_m": 700.0, "lulc_class": 6},
        include_numbers=True,
        include_lsi_legend=False,
    )
    without = build_prompt(
        scale_bar_m=1920,
        centre_values={"elevation_m": 700.0},
        include_numbers=False,
        include_lsi_legend=False,
    )
    assert "Site measurements:" in with_nums.text
    assert "omitted for this ablation" in without.text


def test_score_predictions_auc_perfect_and_none():
    perfect = [
        PointPrediction(0, 0, 10.0, "very_low", 0.0, "{}"),
        PointPrediction(1, 0, 20.0, "low", 1.0, "{}"),
        PointPrediction(2, 1, 80.0, "high", 3.0, "{}"),
        PointPrediction(3, 1, 90.0, "very_high", 4.0, "{}"),
    ]
    auc_s, auc_c = score_predictions(perfect)
    assert auc_s == pytest.approx(1.0)
    assert auc_c == pytest.approx(1.0)

    only_pos = [
        PointPrediction(0, 1, 80.0, "high", 3.0, "{}"),
        PointPrediction(1, 1, 90.0, "very_high", 4.0, "{}"),
    ]
    assert score_predictions(only_pos) == (None, None)


def test_flag_post_event():
    assert flag_post_event("2020-06-01", "2019-11-01", "2024-02-28") is True
    assert flag_post_event("2025-01-01", "2019-11-01", "2024-02-28") is False
    assert flag_post_event(None, "2019-11-01", "2024-02-28") is None


def test_mock_client_deterministic(tmp_path: Path):
    client = MockVLMClient(
        target_by_index={0: 1, 1: 0},
        cache_dir=tmp_path / "cache",
        use_cache=False,
    )
    # Need a real tiny PNG path for hash_tile_payload inside complete().
    from lsi_vlm.tiles import render_satellite_placeholder

    img = tmp_path / "pt000_deadbeef_composite.png"
    render_satellite_placeholder(8, "t").save(img)
    r1 = client.complete([img], "prompt-a LSI-zone overlay is present\nSite measurements:\nx")
    r2 = client.complete([img], "prompt-a LSI-zone overlay is present\nSite measurements:\nx")
    assert r1.raw_text == r2.raw_text
    parsed = parse_vlm_response(r1.raw_text)
    assert parsed is not None
    assert 0.0 <= parsed.score <= 100.0


@pytest.mark.skipif(
    not TRAINING_CSV.is_file() or not HILLSHADE.is_file() or not SLOPE.is_file(),
    reason="training CSV or rasters missing",
)
def test_run_stage_a_dry_run_writes_reports(tiny_config: Path):
    # Training CSV is grouped (22 absences then 22 positives) — need both classes for AUC.
    out = run_stage_a(
        tiny_config,
        backend="ollama",
        dry_run=True,
        max_points=24,
        use_cache=True,
    )
    assert out["used_mock"] is True
    assert out["n_points"] == 24
    assert out["locked_baselines"]["rf_auc"] == 0.8337
    assert out["locked_baselines"]["lr_auc"] == 0.7707

    results = Path(out["results_path"])
    examples = Path(out["examples_path"])
    sidecar = Path(out["sidecar_path"])
    assert results.is_file()
    assert examples.is_file()
    assert sidecar.is_file()

    text = results.read_text(encoding="utf-8")
    assert "0.8337" in text
    assert "0.7707" in text
    assert "no_lsi_numbers" in text
    assert "honest baseline" in text.lower()
    assert "mock" in text.lower()

    ex = examples.read_text(encoding="utf-8")
    assert ex.count("## Example") == 4

    summary = out["summary"]
    assert set(summary) >= {
        "no_lsi_numbers",
        "with_lsi_numbers",
        "no_lsi_image_only",
        "with_lsi_image_only",
    }
    assert summary["no_lsi_numbers"]["auc_score_mean"] is not None

    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["locked_baselines"]["rf_auc"] == 0.8337


def test_summarize_and_write_results_md(tmp_path: Path):
    from lsi_vlm.eval_stage_a import AblationRunResult

    runs = [
        AblationRunResult(
            ablation="no_lsi_numbers",
            repeat=1,
            backend="mock",
            model="mock-vlm",
            n_points=4,
            n_scored=4,
            n_errors=0,
            auc_score=0.75,
            auc_class=0.70,
            predictions=[
                PointPrediction(
                    0,
                    1,
                    80,
                    "high",
                    3,
                    '{"class":"high","score":80}',
                    imagery_date_start="2019-11-01",
                    imagery_date_end="2024-02-28",
                    post_event=None,
                )
            ],
            used_mock=True,
        ),
        AblationRunResult(
            ablation="no_lsi_numbers",
            repeat=2,
            backend="mock",
            model="mock-vlm",
            n_points=4,
            n_scored=4,
            n_errors=0,
            auc_score=0.80,
            auc_class=0.72,
            predictions=[],
            used_mock=True,
        ),
    ]
    summary = summarize_ablations(runs)
    assert summary["no_lsi_numbers"]["auc_score_mean"] == pytest.approx(0.775)
    out = tmp_path / "r.md"
    write_results_md(
        out,
        summary=summary,
        results=runs,
        rf_auc=0.8337,
        lr_auc=0.7707,
        backend="mock",
        model="mock-vlm",
        honest_ablation="no_lsi_numbers",
        config_path="configs/vlm_stage_a.yaml",
        dry_run=True,
    )
    body = out.read_text(encoding="utf-8")
    assert "**0.8337**" in body
    assert "**0.7707**" in body
