"""Stage A VLM prompt builder and JSON response parsing.

Prompts include colour legend, scale bar (metres), continuous centre values,
and **class names/labels** for the 15 categorical stack bands (see
``outputs_vlm/stack_notes.md``). Answers must be JSON only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SUSCEPTIBILITY_CLASSES: tuple[str, ...] = (
    "very_low",
    "low",
    "moderate",
    "high",
    "very_high",
)

# Continuous / derived stack bands used as numeric centre values when present.
CONTINUOUS_CENTRE_BANDS: tuple[str, ...] = (
    "elevation_m",
    "slope_final_deg",
    "slope_dem_deg",
    "rainfall_mm",
    "gravity_mgal",
    "hillshade",
)

# The 15 integer class-code bands — never treat codes as physical units.
CATEGORICAL_STACK_BANDS: tuple[str, ...] = (
    "earthquake_class",
    "slope_class",
    "altitude_class",
    "drainage_dist_class",
    "tri_class",
    "geomorphology_class",
    "geology_class",
    "soil_class",
    "faults_dist_class",
    "sti_class",
    "twi_class",
    "spi_class",
    "roads_dist_class",
    "lulc_class",
    "ndvi_class",
)

# stack band name → ``factor`` key in data/fr_class_table_17factor.csv
BAND_TO_FR_FACTOR: dict[str, str] = {
    "earthquake_class": "earthquake",
    "slope_class": "slope",
    "altitude_class": "altitude",
    "drainage_dist_class": "drainage_dist",
    "tri_class": "tri",
    "geomorphology_class": "geomorphology",
    "geology_class": "geology",
    "soil_class": "soil",
    "faults_dist_class": "faults_dist",
    "sti_class": "sti",
    "twi_class": "twi",
    "spi_class": "spi",
    "roads_dist_class": "roads_dist",
    "lulc_class": "lulc",
    "ndvi_class": "ndvi",
}

# Fixed composite-tile colour legend (must match Stage A tile renderer).
DEFAULT_COLOUR_LEGEND: str = """\
Composite tile colour legend:
- Base: hillshade greyscale (dark = shadow, light = illuminated).
- Slope overlay (semi-transparent): blue (gentle) → cyan → yellow → orange → red (steep).
- Optional LSI-zone overlay (when present): Very Low=green, Low=lime, Moderate=yellow, High=orange, Very High=red.
Satellite tile: Sentinel-2 true-colour RGB (dry-season median when available)."""

MIZORAM_CONTEXT: str = (
    "Regional context: Aizawl sits in the Mizoram fold belt (Indo-Burman ranges) — "
    "steep Tertiary sandstone/shale slopes with intense summer monsoon rainfall that "
    "commonly triggers shallow debris slides and cut-slope failures."
)

REASON_MAX_WORDS = 60


@dataclass(frozen=True)
class VLMPrediction:
    """Validated Stage A JSON answer."""

    class_: str  # susceptibility class (``class`` in JSON)
    score: float
    key_factors: list[str]
    reason: str

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "class": self.class_,
            "score": self.score,
            "key_factors": list(self.key_factors),
            "reason": self.reason,
        }


@dataclass
class PromptBuildResult:
    """Prompt text plus hashes useful for cache keys."""

    text: str
    include_numbers: bool = True
    include_lsi_legend: bool = False
    scale_bar_m: float | None = None
    centre_summary: dict[str, Any] = field(default_factory=dict)


def load_class_label_maps(
    fr_table_path: str | Path | None = None,
) -> dict[str, dict[int, str]]:
    """Load factor → {class_code: label} from the 17-factor FR class table.

    Returns maps keyed by **stack band name** (e.g. ``lulc_class``), not FR factor.
    """
    path = Path(fr_table_path) if fr_table_path else Path("data/fr_class_table_17factor.csv")
    if not path.is_file():
        logger.warning("FR class table missing at %s; categorical labels will be generic", path)
        return {}

    # Avoid a hard pandas dependency in the prompt module — CSV is small.
    import csv

    factor_maps: dict[str, dict[int, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            factor = (row.get("factor") or "").strip()
            label = (row.get("label") or "").strip()
            try:
                code = int(float(row["class"]))
            except (KeyError, TypeError, ValueError):
                continue
            if not factor or not label:
                continue
            factor_maps.setdefault(factor, {})[code] = label

    band_maps: dict[str, dict[int, str]] = {}
    for band, factor in BAND_TO_FR_FACTOR.items():
        if factor in factor_maps:
            band_maps[band] = dict(factor_maps[factor])
    return band_maps


def _as_int_code(value: Any) -> int | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return int(round(f))


def label_for_class_band(
    band: str,
    code: Any,
    label_maps: Mapping[str, Mapping[int, str]] | None = None,
) -> str:
    """Return a human-readable class label; never invent physical units from the code."""
    maps = label_maps or {}
    icode = _as_int_code(code)
    if icode is None:
        return "unknown / nodata"
    band_map = maps.get(band) or {}
    if icode in band_map:
        return band_map[icode]
    # Fallback: explicit class-code wording (not a physical measurement).
    return f"class code {icode} (label unavailable)"


def format_centre_values(
    centre_values: Mapping[str, Any] | None,
    *,
    label_maps: Mapping[str, Mapping[int, str]] | None = None,
    include_continuous: bool = True,
    include_categorical: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Format tile-centre stack samples for the prompt.

    Continuous bands are shown with units. Categorical bands are shown as
    **class labels**, never as raw integers treated like metres/indices.
    """
    centre = dict(centre_values or {})
    summary: dict[str, Any] = {}
    lines: list[str] = []

    if include_continuous:
        cont_lines: list[str] = []
        for band in CONTINUOUS_CENTRE_BANDS:
            if band not in centre:
                continue
            raw = centre[band]
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if val != val:  # NaN
                continue
            unit = {
                "elevation_m": "m",
                "slope_final_deg": "deg",
                "slope_dem_deg": "deg",
                "rainfall_mm": "mm",
                "gravity_mgal": "mGal",
                "hillshade": "",
            }.get(band, "")
            unit_s = f" {unit}" if unit else ""
            cont_lines.append(f"- {band}: {val:.4g}{unit_s}")
            summary[band] = val
        if cont_lines:
            lines.append("Continuous values at tile centre:")
            lines.extend(cont_lines)

    if include_categorical:
        cat_lines: list[str] = []
        for band in CATEGORICAL_STACK_BANDS:
            if band not in centre:
                continue
            code = centre[band]
            label = label_for_class_band(band, code, label_maps)
            icode = _as_int_code(code)
            cat_lines.append(f"- {band}: {label}" + (f" [code={icode}]" if icode is not None else ""))
            summary[band] = {"code": icode, "label": label}
        if cat_lines:
            lines.append(
                "Categorical class labels at tile centre "
                "(integer codes are class IDs, not physical units):"
            )
            lines.extend(cat_lines)

    if not lines:
        return "(no centre stack values supplied)", summary
    return "\n".join(lines), summary


def build_colour_legend(*, include_lsi_overlay: bool = False) -> str:
    """Return the fixed colour legend text for Stage A tiles."""
    text = DEFAULT_COLOUR_LEGEND
    if not include_lsi_overlay:
        # Honest baseline: clarify LSI colours are absent.
        text += "\nNote: this tile does **not** include the LSI-zone overlay."
    else:
        text += "\nNote: LSI-zone overlay is present (treat as prior FR map, not ground truth)."
    return text


def build_prompt(
    *,
    scale_bar_m: float,
    centre_values: Mapping[str, Any] | None = None,
    label_maps: Mapping[str, Mapping[int, str]] | None = None,
    include_numbers: bool = True,
    include_lsi_legend: bool = False,
    colour_legend: str | None = None,
    regional_context: str | None = None,
    fr_table_path: str | Path | None = None,
    extra_instructions: str | None = None,
) -> PromptBuildResult:
    """Build the Stage A zero-shot susceptibility prompt.

    Parameters
    ----------
    scale_bar_m:
        Ground length represented by the tile scale bar / tile width (metres).
    centre_values:
        Mapping of stack band name → value at the tile centre pixel.
    include_numbers:
        Ablation flag: when False, omit centre stack values (image-only).
    include_lsi_legend:
        Whether the composite tile includes the LSI-zone overlay.
    """
    maps = label_maps
    if maps is None and include_numbers:
        maps = load_class_label_maps(fr_table_path)

    legend = colour_legend or build_colour_legend(include_lsi_overlay=include_lsi_legend)
    context = regional_context or MIZORAM_CONTEXT

    parts: list[str] = [
        "You are assessing landslide susceptibility from map tiles for a site in "
        "Aizawl, Mizoram, India.",
        "",
        legend.strip(),
        "",
        f"Scale: the tile spans approximately {scale_bar_m:.0f} m across "
        f"(scale bar / window width in metres).",
        "",
        context.strip(),
    ]

    centre_summary: dict[str, Any] = {}
    if include_numbers:
        centre_block, centre_summary = format_centre_values(
            centre_values, label_maps=maps
        )
        parts.extend(["", "Site measurements:", centre_block])
    else:
        parts.extend(
            [
                "",
                "Site measurements: omitted for this ablation (judge from the imagery only).",
            ]
        )

    parts.extend(
        [
            "",
            "Task: estimate landslide susceptibility for the tile centre.",
            "Reply with **JSON only** (no markdown fences, no extra text) using exactly:",
            '{"class": "very_low|low|moderate|high|very_high", '
            '"score": 0-100, "key_factors": ["..."], "reason": "<=60 words"}',
            "- class: ordinal susceptibility class.",
            "- score: integer or float from 0 (safe) to 100 (most susceptible).",
            "- key_factors: short list of the main drivers you used.",
            f"- reason: plain English, at most {REASON_MAX_WORDS} words.",
        ]
    )
    if extra_instructions:
        parts.extend(["", extra_instructions.strip()])

    text = "\n".join(parts).strip() + "\n"
    return PromptBuildResult(
        text=text,
        include_numbers=include_numbers,
        include_lsi_legend=include_lsi_legend,
        scale_bar_m=float(scale_bar_m),
        centre_summary=centre_summary,
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort JSON object extraction from a model reply."""
    if not text or not text.strip():
        return None
    stripped = text.strip()
    # Strip common markdown fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        stripped = fence.group(1)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(stripped[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def validate_prediction(obj: Mapping[str, Any]) -> VLMPrediction:
    """Validate a parsed JSON object; raise ValueError on failure."""
    if "class" not in obj:
        raise ValueError("missing required field 'class'")
    class_raw = str(obj["class"]).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "verylow": "very_low",
        "veryhigh": "very_high",
    }
    class_ = aliases.get(class_raw, class_raw)
    if class_ not in SUSCEPTIBILITY_CLASSES:
        raise ValueError(
            f"invalid class {obj['class']!r}; expected one of {SUSCEPTIBILITY_CLASSES}"
        )

    if "score" not in obj:
        raise ValueError("missing required field 'score'")
    try:
        score = float(obj["score"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"score must be numeric, got {obj['score']!r}") from exc
    if score < 0 or score > 100:
        raise ValueError(f"score out of range [0, 100]: {score}")

    factors_raw = obj.get("key_factors", [])
    if factors_raw is None:
        factors_raw = []
    if isinstance(factors_raw, str):
        key_factors = [factors_raw.strip()] if factors_raw.strip() else []
    elif isinstance(factors_raw, (list, tuple)):
        key_factors = [str(x).strip() for x in factors_raw if str(x).strip()]
    else:
        raise ValueError("key_factors must be a list of strings")

    reason = str(obj.get("reason", "") or "").strip()
    if _word_count(reason) > REASON_MAX_WORDS:
        # Soft-trim for validation success, but log — callers may still reject.
        logger.warning(
            "reason exceeds %d words (%d); truncating for validation",
            REASON_MAX_WORDS,
            _word_count(reason),
        )
        words = re.findall(r"\S+", reason)
        reason = " ".join(words[:REASON_MAX_WORDS])

    return VLMPrediction(
        class_=class_,
        score=score,
        key_factors=key_factors,
        reason=reason,
    )


def parse_vlm_response(text: str) -> VLMPrediction | None:
    """Parse and validate a VLM reply. Returns None and logs on failure."""
    try:
        obj = _extract_json_object(text)
        if obj is None:
            logger.error("VLM response parse failure: no JSON object in reply: %r", text[:500])
            return None
        return validate_prediction(obj)
    except (ValueError, TypeError, KeyError) as exc:
        logger.error("VLM response validation failure: %s | raw=%r", exc, text[:500])
        return None


def prediction_to_json(pred: VLMPrediction) -> str:
    return json.dumps(pred.to_json_dict(), ensure_ascii=False)


def prompt_build_result_meta(result: PromptBuildResult) -> dict[str, Any]:
    """Serializable metadata for cache / eval logs."""
    return {
        "include_numbers": result.include_numbers,
        "include_lsi_legend": result.include_lsi_legend,
        "scale_bar_m": result.scale_bar_m,
        "centre_summary": result.centre_summary,
    }
