"""Stage A evaluation harness (A4).

Score VLM ``score`` and ordinal ``class`` vs ``target`` on the 44 Aizawl points,
with ablations (LSI overlay / numbers / imagery flags), repeat-run variance,
and markdown reports next to locked RF **0.8337** / LR **0.7707**.

CLI (default free path)::

    python -m lsi_vlm.eval_stage_a --config configs/vlm_stage_a.yaml --backend ollama

If Ollama / API keys are unavailable, the harness falls back to a dry-run
``mock`` backend and still writes a results skeleton. Live AUCs need a real
backend — see ``outputs_vlm/stage_a_results.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from lsi_vlm.clients import (
    DEFAULT_CACHE_DIR,
    VLMClient,
    VLMCallResult,
    hash_tile_payload,
    make_client,
)
from lsi_vlm.prompts import (
    SUSCEPTIBILITY_CLASSES,
    VLMPrediction,
    build_prompt,
    parse_vlm_response,
)
from lsi_vlm.tiles import TilePaths, TilesConfig, build_point_tiles, load_tiles_config

logger = logging.getLogger(__name__)

# Locked Phase 0 benchmarks (outputs_vlm/baseline.md). Do not use RF 0.8275.
LOCKED_RF_AUC = 0.8337
LOCKED_LR_AUC = 0.7707

CLASS_TO_ORDINAL = {c: float(i) for i, c in enumerate(SUSCEPTIBILITY_CLASSES)}


def _ollama_available(base_url: str = "http://127.0.0.1:11434", timeout_s: float = 2.0) -> bool:
    try:
        import httpx
    except ImportError:
        import urllib.request

        try:
            req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.get(f"{base_url.rstrip('/')}/api/tags")
            return resp.is_success
    except Exception:
        return False


@dataclass
class PointPrediction:
    point_index: int
    target: int
    score: float
    class_name: str
    ordinal: float
    raw_text: str
    error: str | None = None
    post_event: bool | None = None
    imagery_date_start: str | None = None
    imagery_date_end: str | None = None
    event_date: str | None = None
    satellite_ok: bool | None = None
    satellite_note: str = ""
    tile_hash: str = ""
    prompt_hash: str = ""
    cache_hit: bool = False


@dataclass
class AblationRunResult:
    ablation: str
    repeat: int
    backend: str
    model: str
    n_points: int
    n_scored: int
    n_errors: int
    auc_score: float | None
    auc_class: float | None
    predictions: list[PointPrediction] = field(default_factory=list)
    used_mock: bool = False


class MockVLMClient(VLMClient):
    """Deterministic offline backend so the harness always produces a skeleton.

    Lives in the eval module so we do not alter the owned ``clients.py`` surface.
    Live AUCs require ollama / cloud backends.
    """

    backend = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-vlm",
        target_by_index: Mapping[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, **kwargs)
        self.target_by_index = dict(target_by_index or {})

    def _complete_once(
        self,
        images: Sequence[Any],
        prompt: str,
    ) -> str:
        import hashlib

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        point_index = 0
        for img in images:
            name = Path(img).name if not isinstance(img, (bytes, bytearray)) else ""
            m = re.match(r"(?:point_|pt)(\d+)_", name)
            if m:
                point_index = int(m.group(1))
                break

        target = self.target_by_index.get(point_index)
        noise = (int(digest[:8], 16) % 21) - 10
        if target is None:
            score = 50.0 + noise
        elif int(target) == 1:
            score = 72.0 + noise * 0.5
        else:
            score = 28.0 + noise * 0.5
        score = float(max(0.0, min(100.0, score)))

        if score < 20:
            cls = "very_low"
        elif score < 40:
            cls = "low"
        elif score < 60:
            cls = "moderate"
        elif score < 80:
            cls = "high"
        else:
            cls = "very_high"

        lsi = "with LSI" if "LSI-zone overlay is present" in prompt else "no LSI"
        nums = "numbers" if "Site measurements:" in prompt and "omitted" not in prompt else "image-only"
        return json.dumps(
            {
                "class": cls,
                "score": round(score, 2),
                "key_factors": [lsi, nums, "mock_backend"],
                "reason": (
                    f"Mock VLM for point {point_index} ({lsi}, {nums}). "
                    "Replace with a live backend for paper AUCs."
                ),
            }
        )


def load_config(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return cfg


def _locked_baselines(cfg: Mapping[str, Any]) -> tuple[float, float]:
    eval_cfg = cfg.get("eval") or {}
    locked = eval_cfg.get("locked_baselines") or {}
    return float(locked.get("rf_auc", LOCKED_RF_AUC)), float(
        locked.get("lr_auc", LOCKED_LR_AUC)
    )


def _safe_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    y = np.asarray(list(y_true), dtype=int)
    s = np.asarray(list(y_score), dtype=float)
    if y.size < 2 or len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def score_predictions(
    preds: Sequence[PointPrediction],
) -> tuple[float | None, float | None]:
    ok = [p for p in preds if p.error is None]
    if not ok:
        return None, None
    y = [p.target for p in ok]
    return (
        _safe_auc(y, [p.score for p in ok]),
        _safe_auc(y, [p.ordinal for p in ok]),
    )


def sample_centre_values(
    stack_tif: Path,
    x: float,
    y: float,
) -> dict[str, Any]:
    """Sample all stack bands at a point (band descriptions as names)."""
    if not stack_tif.is_file():
        return {}
    try:
        import rasterio
    except ImportError:
        return {}

    values: dict[str, Any] = {}
    with rasterio.open(stack_tif) as ds:
        row, col = ds.index(x, y)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            return {}
        window = ((row, row + 1), (col, col + 1))
        for i in range(1, ds.count + 1):
            tag_name = ds.tags(i).get("name") or ds.tags(i).get("band_name")
            desc = ds.descriptions[i - 1] if ds.descriptions else None
            name = tag_name or desc or f"band_{i}"
            try:
                val = float(ds.read(i, window=window)[0, 0])
            except Exception:
                continue
            if ds.nodata is not None and val == ds.nodata:
                continue
            if val != val:  # NaN
                continue
            values[str(name)] = val
    return values


def _parse_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text[:10], fmt).date()
        except ValueError:
            continue
    return None


def load_event_dates(
    points_gpkg: Path,
    training: pd.DataFrame,
    *,
    max_dist_m: float = 150.0,
) -> dict[int, str]:
    """Nearest inventory ``event_date`` per training row index (when within radius)."""
    if not points_gpkg.is_file():
        return {}
    try:
        import geopandas as gpd
    except ImportError:
        return {}

    gdf = gpd.read_file(points_gpkg)
    if gdf.empty or "event_date" not in gdf.columns:
        return {}
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:32646")
    else:
        gdf = gdf.to_crs("EPSG:32646")

    out: dict[int, str] = {}
    for i, row in training.iterrows():
        x, y = float(row["x"]), float(row["y"])
        # Approximate nearest in projected metres.
        dx = gdf.geometry.x - x
        dy = gdf.geometry.y - y
        dist = np.hypot(dx.to_numpy(), dy.to_numpy())
        j = int(np.argmin(dist))
        if dist[j] > max_dist_m:
            continue
        d = _parse_date(gdf.iloc[j]["event_date"])
        if d is not None:
            out[int(i)] = d.isoformat()
    return out


def imagery_window_from_cfg(tiles_cfg: TilesConfig) -> tuple[str, str]:
    sat = tiles_cfg.satellite or {}
    y0 = int(sat.get("year_start", 2019))
    y1 = int(sat.get("year_end", 2024))
    months = list(sat.get("months", [11, 12, 1, 2]))
    # Dry-season span Nov (y0) → Feb (y1) as documented for Stage A.
    start = f"{y0}-11-01" if 11 in months else f"{y0}-01-01"
    end = f"{y1}-02-28" if 2 in months else f"{y1}-12-31"
    return start, end


def flag_post_event(
    event_date_s: str | None,
    imagery_start: str | None,
    imagery_end: str | None,
) -> bool | None:
    ev = _parse_date(event_date_s)
    end = _parse_date(imagery_end)
    start = _parse_date(imagery_start)
    if ev is None or end is None:
        return None
    # Post-event if the composite window ends after the event (scar may be visible).
    if end >= ev:
        return True
    if start is not None and end < ev:
        return False
    return None


def resolve_client(
    backend: str,
    *,
    cfg: Mapping[str, Any],
    dry_run: bool,
    target_by_index: Mapping[int, int],
    cache_dir: Path,
    use_cache: bool,
) -> tuple[VLMClient, bool]:
    model_cfg = dict(cfg.get("model") or {})
    name = backend.strip().lower()
    model_name = model_cfg.get("name") or model_cfg.get("model")
    base_url = str(model_cfg.get("ollama_base_url", "http://127.0.0.1:11434"))
    temperature = float(model_cfg.get("temperature", 0.0))
    max_retries = int(model_cfg.get("max_retries", 3))

    force_mock = dry_run or name == "mock"
    if name == "ollama" and not force_mock and not _ollama_available(base_url):
        logger.warning("Ollama unavailable at %s; falling back to mock", base_url)
        force_mock = True

    if force_mock:
        return (
            MockVLMClient(
                model=str(model_name or "mock-vlm"),
                target_by_index=target_by_index,
                temperature=temperature,
                max_retries=1,
                cache_dir=cache_dir,
                use_cache=use_cache,
            ),
            True,
        )

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_retries": max_retries,
        "cache_dir": cache_dir,
        "use_cache": use_cache,
    }
    if model_name:
        kwargs["model"] = model_name
    if name == "ollama":
        kwargs["base_url"] = base_url
        if model_cfg.get("timeout_s") is not None:
            kwargs["timeout_s"] = float(model_cfg["timeout_s"])
    try:
        client = make_client(name, **kwargs)
    except Exception as exc:
        logger.warning("Backend %s failed (%s); falling back to mock", name, exc)
        return (
            MockVLMClient(
                model=str(model_name or "mock-vlm"),
                target_by_index=target_by_index,
                temperature=temperature,
                max_retries=1,
                cache_dir=cache_dir,
                use_cache=use_cache,
            ),
            True,
        )
    return client, getattr(client, "backend", name) == "mock"


def predict_point(
    client: VLMClient,
    tile: TilePaths,
    *,
    include_numbers: bool,
    prompt_cfg: Mapping[str, Any],
    centre_values: Mapping[str, Any] | None,
    scale_bar_m: float,
    imagery_start: str | None,
    imagery_end: str | None,
    event_date: str | None,
) -> PointPrediction:
    include_lsi = bool(tile.include_lsi_overlay)
    built = build_prompt(
        scale_bar_m=scale_bar_m,
        centre_values=centre_values,
        include_numbers=include_numbers,
        include_lsi_legend=include_lsi,
        regional_context=prompt_cfg.get("regional_context"),
        fr_table_path=prompt_cfg.get("fr_class_table"),
    )
    images = [tile.composite, tile.satellite]
    tile_hash = hash_tile_payload(
        tile.composite,
        tile.satellite,
        f"lsi={include_lsi}",
    )
    result: VLMCallResult = client.complete(
        images,
        built.text,
        tile_hash=tile_hash,
        parse=True,
    )

    post = flag_post_event(event_date, imagery_start, imagery_end)
    base_kwargs = dict(
        point_index=tile.point_index,
        target=int(tile.target if tile.target is not None else -1),
        post_event=post,
        imagery_date_start=imagery_start,
        imagery_date_end=imagery_end,
        event_date=event_date,
        satellite_ok=tile.satellite_ok,
        satellite_note=tile.satellite_note,
        tile_hash=tile_hash,
        prompt_hash=result.cache_key.prompt_hash,
        cache_hit=result.cached,
    )

    if result.error and not result.raw_text:
        return PointPrediction(
            score=float("nan"),
            class_name="",
            ordinal=float("nan"),
            raw_text=result.raw_text,
            error=result.error,
            **base_kwargs,
        )

    pred: VLMPrediction | None = result.prediction
    if pred is None and result.raw_text:
        pred = parse_vlm_response(result.raw_text)
    if pred is None:
        return PointPrediction(
            score=float("nan"),
            class_name="",
            ordinal=float("nan"),
            raw_text=result.raw_text,
            error=result.error or "parse failure",
            **base_kwargs,
        )

    return PointPrediction(
        score=float(pred.score),
        class_name=pred.class_,
        ordinal=CLASS_TO_ORDINAL[pred.class_],
        raw_text=result.raw_text or json.dumps(pred.to_json_dict()),
        error=None,
        **base_kwargs,
    )


def run_ablation(
    *,
    name: str,
    repeat: int,
    tiles: Sequence[TilePaths],
    client: VLMClient,
    include_numbers: bool,
    prompt_cfg: Mapping[str, Any],
    centres: Mapping[int, Mapping[str, Any]],
    scale_bar_m: float,
    imagery_start: str | None,
    imagery_end: str | None,
    event_dates: Mapping[int, str],
    used_mock: bool,
) -> AblationRunResult:
    preds: list[PointPrediction] = []
    for tile in tiles:
        preds.append(
            predict_point(
                client,
                tile,
                include_numbers=include_numbers,
                prompt_cfg=prompt_cfg,
                centre_values=centres.get(tile.point_index),
                scale_bar_m=scale_bar_m,
                imagery_start=imagery_start,
                imagery_end=imagery_end,
                event_date=event_dates.get(tile.point_index),
            )
        )
    auc_score, auc_class = score_predictions(preds)
    n_errors = sum(1 for p in preds if p.error is not None)
    return AblationRunResult(
        ablation=name,
        repeat=repeat,
        backend=getattr(client, "backend", "unknown"),
        model=getattr(client, "model", "unknown"),
        n_points=len(preds),
        n_scored=len(preds) - n_errors,
        n_errors=n_errors,
        auc_score=auc_score,
        auc_class=auc_class,
        predictions=preds,
        used_mock=used_mock,
    )


def _fmt_auc(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4f}"


def _mean_std(vals: Sequence[float]) -> str:
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{statistics.mean(vals):.4f} ± {statistics.stdev(vals):.4f}"


def summarize_ablations(
    results: Sequence[AblationRunResult],
) -> dict[str, dict[str, Any]]:
    by_name: dict[str, list[AblationRunResult]] = {}
    for r in results:
        by_name.setdefault(r.ablation, []).append(r)
    summary: dict[str, dict[str, Any]] = {}
    for name, runs in by_name.items():
        score_vals = [r.auc_score for r in runs if r.auc_score is not None]
        class_vals = [r.auc_class for r in runs if r.auc_class is not None]
        summary[name] = {
            "n_repeats": len(runs),
            "auc_score_mean": statistics.mean(score_vals) if score_vals else None,
            "auc_score_std": statistics.stdev(score_vals) if len(score_vals) > 1 else 0.0,
            "auc_class_mean": statistics.mean(class_vals) if class_vals else None,
            "auc_class_std": statistics.stdev(class_vals) if len(class_vals) > 1 else 0.0,
            "auc_score_runs": score_vals,
            "auc_class_runs": class_vals,
            "used_mock": any(r.used_mock for r in runs),
            "n_errors_total": sum(r.n_errors for r in runs),
        }
    return summary


def write_results_md(
    path: Path,
    *,
    summary: Mapping[str, Mapping[str, Any]],
    results: Sequence[AblationRunResult],
    rf_auc: float,
    lr_auc: float,
    backend: str,
    model: str,
    honest_ablation: str,
    config_path: str,
    dry_run: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        "# Stage A — Zero-shot VLM results",
        "",
        f"Config: `{config_path}`  ",
        f"Backend: `{backend}` / model `{model}`  ",
        f"Dry-run / mock path: **{'yes' if dry_run else 'no'}**",
        "",
        "## Locked FR/ML baselines",
        "",
        "Canonical numbers from `outputs_vlm/baseline.md` (do **not** cite RF 0.8275):",
        "",
        "| Model | Buffered LOOCV AUC (r = 1 km) |",
        "|---|---|",
        f"| Random Forest | **{rf_auc:.4f}** |",
        f"| Logistic Regression | **{lr_auc:.4f}** |",
        "",
        "## VLM ROC-AUC vs locked baselines",
        "",
        "Primary reportable VLM row is the **no-LSI** ablation (honest baseline; "
        "with-LSI can read FR zones back from the image).",
        "",
        "| System | Ablation | AUC (`score`) | AUC (ordinal `class`) | Notes |",
        "|---|---|---|---|---|",
        f"| RF (locked) | — | **{rf_auc:.4f}** | — | buffered LOOCV |",
        f"| LR (locked) | — | **{lr_auc:.4f}** | — | buffered LOOCV |",
    ]

    for name, s in summary.items():
        score_s = _mean_std(s["auc_score_runs"])
        class_s = _mean_std(s["auc_class_runs"])
        note_bits = []
        if name == honest_ablation:
            note_bits.append("**honest baseline (report this)**")
        if s.get("used_mock"):
            note_bits.append("mock/dry-run — not for paper citation")
        if s.get("n_errors_total"):
            note_bits.append(f"parse/request errors={s['n_errors_total']}")
        notes = "; ".join(note_bits) if note_bits else ""
        lines.append(f"| VLM | `{name}` | {score_s} | {class_s} | {notes} |")

    lines.extend(
        [
            "",
            "## Ablation detail (required)",
            "",
            "1. **With vs without LSI-zone overlay** — no-LSI is the honest baseline.",
            "2. **Numbers in prompt vs image-only** — tests text leakage vs pixels.",
            "3. **Imagery date / post-event flags** — recorded per point when known "
            "(see examples + prediction JSON).",
            "",
            "### Per-repeat AUCs",
            "",
            "| Ablation | Repeat | AUC score | AUC class | n_scored | n_errors | mock |",
            "|---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for r in results:
        lines.append(
            f"| `{r.ablation}` | {r.repeat} | {_fmt_auc(r.auc_score)} | "
            f"{_fmt_auc(r.auc_class)} | {r.n_scored} | {r.n_errors} | "
            f"{'Y' if r.used_mock else 'N'} |"
        )

    honest_runs = [r for r in results if r.ablation == honest_ablation]
    if honest_runs:
        preds = honest_runs[0].predictions
        n_post = sum(1 for p in preds if p.post_event is True)
        n_pre = sum(1 for p in preds if p.post_event is False)
        n_unk = sum(1 for p in preds if p.post_event is None)
        dates = sorted(
            {
                f"{p.imagery_date_start or '?'}→{p.imagery_date_end or '?'}"
                for p in preds
            }
        )
        n_with_event = sum(1 for p in preds if p.event_date)
        lines.extend(
            [
                "",
                "## Imagery date / post-event flags",
                "",
                f"- Distinct imagery windows: {', '.join(f'`{d}`' for d in dates) or 'n/a'}",
                f"- Training points with matched inventory `event_date`: **{n_with_event}**",
                f"- post_event=True: **{n_post}**, False: **{n_pre}**, unknown: **{n_unk}**",
                "- `post_event=True` when the satellite composite window ends on/after "
                "the matched event date (scar-detection risk).",
            ]
        )

    lines.extend(
        [
            "",
            "## Stability",
            "",
            "Configured repeats are aggregated as mean ± sample stdev above. "
            "With temperature 0 and response caching, variance is often ~0.",
            "",
            "## How to reproduce",
            "",
            "```text",
            "python -m lsi_vlm.eval_stage_a --config configs/vlm_stage_a.yaml --backend ollama",
            "```",
            "",
            "If Ollama is down and no API keys are set, add `--dry-run` (or let the "
            "harness auto-fallback to `mock`). **Mock AUCs are scaffolding only** — "
            "cite live backend numbers for the paper, always next to RF "
            f"**{rf_auc:.4f}** / LR **{lr_auc:.4f}**.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_examples_md(
    path: Path,
    *,
    run: AblationRunResult,
    tiles_by_index: Mapping[int, TilePaths],
    n_examples: int = 5,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = list(run.predictions)
    pos = [p for p in preds if p.target == 1]
    neg = [p for p in preds if p.target == 0]
    chosen: list[PointPrediction] = []
    for a, b in zip(pos, neg):
        if len(chosen) >= n_examples:
            break
        chosen.append(a)
        if len(chosen) >= n_examples:
            break
        chosen.append(b)
    for p in preds:
        if len(chosen) >= n_examples:
            break
        if p not in chosen:
            chosen.append(p)

    lines = [
        "# Stage A — Example tiles and VLM responses",
        "",
        f"Ablation: `{run.ablation}` (repeat {run.repeat})  ",
        f"Backend: `{run.backend}` / `{run.model}`",
        "",
    ]
    for i, p in enumerate(chosen, start=1):
        tile = tiles_by_index.get(p.point_index)
        lines.append(f"## Example {i} — point {p.point_index}")
        lines.append("")
        lines.append(f"- **target**: `{p.target}`")
        lines.append(f"- **VLM class / score**: `{p.class_name}` / `{p.score}`")
        if p.error:
            lines.append(f"- **error**: `{p.error}`")
        lines.append(
            f"- **imagery**: `{p.imagery_date_start}` → `{p.imagery_date_end}` "
            f"(event_date={p.event_date}, post_event={p.post_event})"
        )
        if tile is not None:
            lines.append(f"- **composite**: `{tile.composite.as_posix()}`")
            lines.append(f"- **satellite**: `{tile.satellite.as_posix()}`")
            lines.append(
                f"- **include_lsi**: `{tile.include_lsi_overlay}` "
                f"(sat_ok={tile.satellite_ok})"
            )
        lines.append("")
        lines.append("```json")
        try:
            lines.append(json.dumps(json.loads(p.raw_text), indent=2))
        except (json.JSONDecodeError, TypeError):
            lines.append(p.raw_text or "{}")
        lines.append("```")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _default_ablations() -> list[dict[str, Any]]:
    return [
        {
            "name": "no_lsi_numbers",
            "include_lsi": False,
            "include_numbers": True,
            "honest_baseline": True,
        },
        {"name": "with_lsi_numbers", "include_lsi": True, "include_numbers": True},
        {"name": "no_lsi_image_only", "include_lsi": False, "include_numbers": False},
        {"name": "with_lsi_image_only", "include_lsi": True, "include_numbers": False},
    ]


def run_stage_a(
    config_path: Path | str,
    *,
    backend: str | None = None,
    dry_run: bool = False,
    max_points: int | None = None,
    use_cache: bool | None = None,
    skip_satellite: bool | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    eval_cfg = cfg.get("eval") or {}
    prompt_cfg = dict(cfg.get("prompt") or {})
    model_cfg = dict(cfg.get("model") or {})

    tiles_cfg = load_tiles_config(config_path)
    if max_points is not None:
        # build_point_tiles uses limit=; also shrink training via temporary override
        pass

    be = (backend or model_cfg.get("backend") or "ollama").strip().lower()
    cache_dir = Path(model_cfg.get("cache_dir") or eval_cfg.get("cache_dir") or DEFAULT_CACHE_DIR)
    cache_on = model_cfg.get("use_cache", True) if use_cache is None else use_cache
    n_repeats = int(eval_cfg.get("n_repeats", 3))
    n_examples = int(eval_cfg.get("n_examples", 5))
    results_path = Path(eval_cfg.get("output_results", "outputs_vlm/stage_a_results.md"))
    examples_path = Path(eval_cfg.get("output_examples", "outputs_vlm/stage_a_examples.md"))
    skip_sat = (
        bool(eval_cfg.get("skip_satellite", True))
        if skip_satellite is None
        else skip_satellite
    )
    if dry_run:
        skip_sat = True

    ablations = list(eval_cfg.get("ablations") or _default_ablations())
    honest = next(
        (a["name"] for a in ablations if a.get("honest_baseline")),
        "no_lsi_numbers",
    )
    rf_auc, lr_auc = _locked_baselines(cfg)

    training = pd.read_csv(tiles_cfg.training_csv)
    if max_points is not None:
        training = training.iloc[: int(max_points)].copy()
    target_by_index = {int(i): int(t) for i, t in training["target"].items()}

    scale_bar_m = float(
        prompt_cfg.get("scale_bar_m")
        or (tiles_cfg.window_size_px * tiles_cfg.resolution_m)
    )
    stack_tif = Path(prompt_cfg.get("stack_tif") or "outputs_vlm/aizawl_stack.tif")
    centres: dict[int, dict[str, Any]] = {}
    for i, row in training.iterrows():
        centres[int(i)] = sample_centre_values(stack_tif, float(row["x"]), float(row["y"]))

    event_dates = load_event_dates(
        Path("final_maps/Aizawl_Points_UTM.gpkg"),
        training,
    )
    imagery_start, imagery_end = imagery_window_from_cfg(tiles_cfg)

    # Per-ablation cache dirs when stability_bust_cache is on.
    client, used_mock = resolve_client(
        be,
        cfg=cfg,
        dry_run=dry_run,
        target_by_index=target_by_index,
        cache_dir=cache_dir,
        use_cache=bool(cache_on),
    )

    tile_cache: dict[bool, list[TilePaths]] = {}
    all_results: list[AblationRunResult] = []
    limit = int(max_points) if max_points is not None else None

    for abl in ablations:
        include_lsi = bool(abl.get("include_lsi", False))
        include_numbers = bool(abl.get("include_numbers", True))
        name = str(abl["name"])
        if include_lsi not in tile_cache:
            # Separate output subdirs so with/without LSI do not overwrite.
            sub = TilesConfig(
                window_size_px=tiles_cfg.window_size_px,
                resolution_m=tiles_cfg.resolution_m,
                crs=tiles_cfg.crs,
                output_dir=tiles_cfg.output_dir
                / ("lsi" if include_lsi else "no_lsi"),
                training_csv=tiles_cfg.training_csv,
                hillshade_path=tiles_cfg.hillshade_path,
                slope_path=tiles_cfg.slope_path,
                lsi_zones_path=tiles_cfg.lsi_zones_path,
                include_lsi_overlay=include_lsi,
                slope_alpha=tiles_cfg.slope_alpha,
                lsi_alpha=tiles_cfg.lsi_alpha,
                slope_vmax_deg=tiles_cfg.slope_vmax_deg,
                satellite=dict(tiles_cfg.satellite or {}),
                grid=dict(tiles_cfg.grid or {}),
                repo_root=tiles_cfg.repo_root,
            )
            if dry_run:
                sat = dict(sub.satellite)
                sat["enabled"] = False
                sub.satellite = sat
            tile_cache[include_lsi] = build_point_tiles(
                sub,
                skip_satellite=skip_sat or dry_run,
                limit=limit,
                include_lsi_overlay=include_lsi,
            )
            # Align targets if training was truncated (build_point_tiles reads full CSV + limit).
            for t in tile_cache[include_lsi]:
                if t.point_index in target_by_index:
                    t.target = target_by_index[t.point_index]

        tiles = tile_cache[include_lsi]
        for rep in range(1, n_repeats + 1):
            bust = bool(eval_cfg.get("stability_bust_cache", False))
            if bust:
                # Remake client with isolated cache dir per repeat.
                client_rep, used_mock = resolve_client(
                    be,
                    cfg=cfg,
                    dry_run=dry_run or used_mock,
                    target_by_index=target_by_index,
                    cache_dir=cache_dir / name / f"rep{rep}",
                    use_cache=bool(cache_on),
                )
            else:
                client_rep = client
            run = run_ablation(
                name=name,
                repeat=rep,
                tiles=tiles,
                client=client_rep,
                include_numbers=include_numbers,
                prompt_cfg=prompt_cfg,
                centres=centres,
                scale_bar_m=scale_bar_m,
                imagery_start=imagery_start,
                imagery_end=imagery_end,
                event_dates=event_dates,
                used_mock=used_mock,
            )
            all_results.append(run)

    summary = summarize_ablations(all_results)
    write_results_md(
        results_path,
        summary=summary,
        results=all_results,
        rf_auc=rf_auc,
        lr_auc=lr_auc,
        backend=getattr(client, "backend", be),
        model=getattr(client, "model", ""),
        honest_ablation=honest,
        config_path=str(config_path),
        dry_run=used_mock,
    )

    example_run = next(
        (r for r in all_results if r.ablation == honest and r.repeat == 1),
        all_results[0],
    )
    honest_lsi = bool(
        next(a for a in ablations if a["name"] == example_run.ablation).get(
            "include_lsi", False
        )
    )
    tiles_for_examples = {t.point_index: t for t in tile_cache.get(honest_lsi, [])}
    write_examples_md(
        examples_path,
        run=example_run,
        tiles_by_index=tiles_for_examples,
        n_examples=n_examples,
    )

    sidecar = results_path.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "locked_baselines": {"rf_auc": rf_auc, "lr_auc": lr_auc},
                "backend": getattr(client, "backend", be),
                "model": getattr(client, "model", ""),
                "used_mock": used_mock,
                "honest_ablation": honest,
                "summary": summary,
                "n_points": int(len(training)),
                "imagery_window": {"start": imagery_start, "end": imagery_end},
                "n_event_dates_matched": len(event_dates),
            },
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    return {
        "results_path": results_path,
        "examples_path": examples_path,
        "sidecar_path": sidecar,
        "summary": summary,
        "used_mock": used_mock,
        "n_points": len(training),
        "locked_baselines": {"rf_auc": rf_auc, "lr_auc": lr_auc},
    }


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage A VLM evaluation harness (A4)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/vlm_stage_a.yaml"),
        help="Stage A YAML config",
    )
    p.add_argument(
        "--backend",
        default=None,
        help="vlm backend: ollama (default) | mock | anthropic | gemini | openai",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Force mock backend + skip satellite STAC (no API keys / Ollama needed)",
    )
    p.add_argument("--max-points", type=int, default=None, help="Cap points (smoke tests)")
    p.add_argument("--no-cache", action="store_true", help="Disable response cache")
    p.add_argument(
        "--with-satellite",
        action="store_true",
        help="Fetch Sentinel-2 tiles (overrides eval.skip_satellite)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_arg_parser().parse_args(argv)
    t0 = time.time()
    backend = args.backend or ("mock" if args.dry_run else "ollama")
    try:
        out = run_stage_a(
            args.config,
            backend=backend,
            dry_run=args.dry_run or backend == "mock",
            max_points=args.max_points,
            use_cache=not args.no_cache,
            skip_satellite=False if args.with_satellite else None,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 — CLI boundary
        logger.exception("Stage A eval failed")
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    summary = out["summary"]
    rf = out["locked_baselines"]["rf_auc"]
    lr = out["locked_baselines"]["lr_auc"]
    print(
        f"Stage A eval done in {time.time() - t0:.1f}s | "
        f"n={out['n_points']} | mock={out['used_mock']}"
    )
    print(f"Locked baselines: RF {rf:.4f} / LR {lr:.4f}")
    for name, s in summary.items():
        print(
            f"  {name}: score_auc={_mean_std(s['auc_score_runs'])} "
            f"class_auc={_mean_std(s['auc_class_runs'])}"
        )
    print(f"Wrote {out['results_path']}")
    print(f"Wrote {out['examples_path']}")
    if out["used_mock"]:
        print(
            "NOTE: mock/dry-run AUCs are a harness skeleton only. "
            "Live AUCs need Ollama or a cloud backend.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
