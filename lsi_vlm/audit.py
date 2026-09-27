"""Audit GeoTIFF factor layers and training-data quirks for the VLM stack.

CLI:
    python -m lsi_vlm.audit [--final-maps DIR] [--training-csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml

# Cap unique-value enumeration for continuous rasters (memory / report size).
UNIQUE_CAP = 256
# Heuristic: rasters with few unique values (after nodata mask) are categorical.
CATEGORICAL_UNIQUE_MAX = 32


@dataclass
class RasterAudit:
    path: str
    relative: str
    crs: str
    res_x: float
    res_y: float
    width: int
    height: int
    left: float
    bottom: float
    right: float
    top: float
    dtype: str
    nodata: float | None
    band_count: int
    unique_count: int | None
    unique_capped: bool
    kind: str  # "categorical" | "continuous" | "unknown"
    sample_min: float | None
    sample_max: float | None
    notes: str


def _crs_str(crs) -> str:
    if crs is None:
        return "None"
    try:
        return crs.to_string()
    except Exception:
        return str(crs)


def classify_kind(unique_count: int | None, dtype: str, name: str) -> str:
    """Classify raster as categorical vs continuous using name + unique count."""
    lower = name.lower()
    if any(tok in lower for tok in ("_classes", "quantile", "reclass", "zones", "lsi_master")):
        return "categorical"
    if any(tok in lower for tok in ("_mm", "mgal", "dem", "hillshade", "slope_final", "copernicus")):
        return "continuous"
    if unique_count is None:
        return "unknown"
    if unique_count <= CATEGORICAL_UNIQUE_MAX:
        return "categorical"
    if "float" in dtype.lower() and unique_count > CATEGORICAL_UNIQUE_MAX:
        return "continuous"
    return "unknown"


def audit_raster(path: Path, root: Path) -> RasterAudit:
    """Inspect a single GeoTIFF and return structured metadata."""
    with rasterio.open(path) as ds:
        crs = _crs_str(ds.crs)
        res_x, res_y = float(ds.res[0]), float(abs(ds.res[1]))
        b = ds.bounds
        dtype = str(ds.dtypes[0])
        nodata = ds.nodata
        # Read band 1; large rasters: sample for unique if huge.
        data = ds.read(1, masked=False)
        mask = np.zeros(data.shape, dtype=bool)
        if nodata is not None and not (isinstance(nodata, float) and math.isnan(nodata)):
            mask |= data == nodata
        # Also treat common sentinel nodata
        if np.issubdtype(data.dtype, np.floating):
            mask |= ~np.isfinite(data)

        valid = data[~mask]
        sample_min = sample_max = None
        unique_count: int | None = None
        unique_capped = False
        notes = ""

        if valid.size == 0:
            notes = "no valid pixels"
            kind = "unknown"
        else:
            sample_min = float(np.min(valid))
            sample_max = float(np.max(valid))
            # Cap unique enumeration
            if valid.size > 5_000_000:
                # Subsample for unique estimate on very large arrays
                rng = np.random.default_rng(0)
                idx = rng.choice(valid.size, size=2_000_000, replace=False)
                sample = valid[idx]
                uniq = np.unique(sample)
                unique_capped = True
                notes = "unique estimated from 2M-pixel subsample"
            else:
                uniq = np.unique(valid)

            if uniq.size > UNIQUE_CAP:
                unique_count = int(uniq.size) if not unique_capped else int(uniq.size)
                unique_capped = True
                # For reporting we still store the true/approx count
            else:
                unique_count = int(uniq.size)

            # If we subsampled and hit UNIQUE_CAP-ish, re-estimate with full if feasible
            if unique_capped and valid.size <= 5_000_000:
                unique_count = int(np.unique(valid).size)
                unique_capped = unique_count > UNIQUE_CAP

            kind = classify_kind(unique_count, dtype, path.name)

        rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        return RasterAudit(
            path=str(path.resolve()),
            relative=rel.replace("\\", "/"),
            crs=crs,
            res_x=res_x,
            res_y=res_y,
            width=int(ds.width),
            height=int(ds.height),
            left=float(b.left),
            bottom=float(b.bottom),
            right=float(b.right),
            top=float(b.top),
            dtype=dtype,
            nodata=float(nodata) if nodata is not None else None,
            band_count=int(ds.count),
            unique_count=unique_count,
            unique_capped=unique_capped,
            kind=kind,
            sample_min=sample_min,
            sample_max=sample_max,
            notes=notes,
        )


def iter_geotiffs(dirs: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            continue
        found.extend(sorted(d.rglob("*.tif")))
        found.extend(sorted(d.rglob("*.tiff")))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def audit_training_csv(csv_path: Path) -> dict:
    """Summarise FR quirks and spatial extents from the training table."""
    df = pd.read_csv(csv_path)
    fr_cols = [c for c in df.columns if c.endswith("_fr")]
    nunique = {c: int(df[c].nunique()) for c in fr_cols}
    near_constant = [c for c, n in nunique.items() if n <= 1]
    maxima = {c: float(df[c].max()) for c in fr_cols}
    extreme = {c: v for c, v in maxima.items() if v > 10.0}

    pos = df[df["target"] == 1]
    neg = df[df["target"] == 0]
    return {
        "n_rows": int(len(df)),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "x_min": float(df["x"].min()),
        "x_max": float(df["x"].max()),
        "y_min": float(df["y"].min()),
        "y_max": float(df["y"].max()),
        "pos_y_min": float(pos["y"].min()),
        "pos_y_max": float(pos["y"].max()),
        "neg_y_min": float(neg["y"].min()),
        "neg_y_max": float(neg["y"].max()),
        "fr_nunique": nunique,
        "near_constant_fr": near_constant,
        "fr_max": maxima,
        "extreme_fr": extreme,
        "pseudo_absence_north_extent_m": float(neg["y"].max() - pos["y"].max()),
    }


def load_study_bbox(config_path: Path) -> dict | None:
    if not config_path.is_file():
        return None
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("study_bbox")


def format_audit_md(
    audits: list[RasterAudit],
    train_info: dict,
    study_bbox: dict | None,
    config_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# Phase 0 raster & training audit")
    lines.append("")
    lines.append("Generated by `python -m lsi_vlm.audit`.")
    lines.append("")
    lines.append("## Kickoff known issues (with numbers)")
    lines.append("")

    # Issue 1
    cont = [a for a in audits if a.kind == "continuous"]
    cat = [a for a in audits if a.kind == "categorical"]
    lines.append("### 1. Most factors are classes, not continuous")
    lines.append("")
    lines.append(
        f"- Continuous (heuristic): **{len(cont)}** rasters — "
        + (", ".join(a.relative for a in cont) if cont else "none")
    )
    lines.append(
        f"- Categorical / class (heuristic): **{len(cat)}** rasters"
    )
    sonker = [a for a in audits if "sonker_17/" in a.relative.replace("\\", "/")]
    sonker_cont = [a for a in sonker if a.kind == "continuous"]
    sonker_cat = [a for a in sonker if a.kind == "categorical"]
    lines.append(
        f"- Under `final_maps/sonker_17/`: {len(sonker_cont)} continuous "
        f"({', '.join(Path(a.relative).name for a in sonker_cont)}), "
        f"{len(sonker_cat)} class/quantile layers."
    )
    lines.append(
        "- Stage B needs continuous rebuilds for drainage/faults/roads distances (m), "
        "NDVI (−1…1), TRI/TWI/SPI/STI, earthquake density/PGA, etc. "
        "Geology / soil / geomorphology / LULC stay categorical."
    )
    lines.append("")

    # Issue 2
    lines.append("### 2. Study bbox in `configs/aizawl.yaml` vs landslide point extents")
    lines.append("")
    lines.append(f"- Config: `{config_path.as_posix()}`")
    if study_bbox:
        lines.append(
            f"- YAML `study_bbox`: x [{study_bbox['x_min']:,.1f}, {study_bbox['x_max']:,.1f}], "
            f"y [{study_bbox['y_min']:,.1f}, {study_bbox['y_max']:,.1f}]"
        )
    else:
        lines.append("- YAML `study_bbox`: not found")
    lines.append(
        f"- Training points (`x`,`y`): x [{train_info['x_min']:,.1f}, {train_info['x_max']:,.1f}], "
        f"y [{train_info['y_min']:,.1f}, {train_info['y_max']:,.1f}]"
    )
    lines.append(
        "- **Mismatch:** YAML x-range (520000–545000) does **not** overlap training x "
        f"({train_info['x_min']:,.0f}–{train_info['x_max']:,.0f}). "
        "Points match Aizawl ≈92.7°E in UTM 46N. "
        "**Do not auto-fix** — confirm with user before changing anything FR/ML depends on."
    )
    lines.append("")

    # Issue 3
    lines.append("### 3. Near-constant / extreme FR features on training CSV")
    lines.append("")
    lines.append(
        f"- Near-constant (nunique ≤ 1): {', '.join(train_info['near_constant_fr']) or 'none'}"
    )
    lines.append("- FR nunique across 44 rows:")
    for col, n in train_info["fr_nunique"].items():
        lines.append(f"  - `{col}`: {n}")
    lines.append("- Extreme FR maxima (max > 10):")
    if train_info["extreme_fr"]:
        for col, v in train_info["extreme_fr"].items():
            lines.append(f"  - `{col}`: {v:.4f}")
    else:
        lines.append("  - none")
    lines.append(
        "- Implication: FR maps as Stage-B auxiliary labels should be down-weighted; "
        "tiny-area classes can inflate ratios (`lulc_fr`, `ndvi_fr`)."
    )
    lines.append("")

    # Issue 4
    lines.append("### 4. Pseudo-absence north extent vs positives")
    lines.append("")
    lines.append(
        f"- Positives y: [{train_info['pos_y_min']:,.1f}, {train_info['pos_y_max']:,.1f}]"
    )
    lines.append(
        f"- Pseudo-absences y: [{train_info['neg_y_min']:,.1f}, {train_info['neg_y_max']:,.1f}]"
    )
    lines.append(
        f"- Negatives extend **{train_info['pseudo_absence_north_extent_m']:,.0f} m** "
        f"(~{train_info['pseudo_absence_north_extent_m']/1000:.1f} km) north of the northernmost positive. "
        "This can make classification easier than a strictly co-located control sample."
    )
    lines.append("")

    # Continuous rebuild inventory
    lines.append("## Continuous sources: present vs need rebuild")
    lines.append("")
    lines.append("| Factor | Continuous version | Status / rebuild from |")
    lines.append("|---|---|---|")
    lines.append(
        "| Rainfall | mm | **Present** — `sonker_17/S17_01_Rainfall_CHIRPS_mm.tif` |"
    )
    lines.append(
        "| Gravity | mGal | **Present** — `sonker_17/S17_10_Gravity_Bouguer_mGal.tif` |"
    )
    lines.append(
        "| Slope | degrees | **Present** — `Slope_FINAL.tif`; also DEM-derived in stack |"
    )
    lines.append(
        "| Elevation | m | **Present** — `DEM_UTF_FINAL.tif` (+ Copernicus DSM tiles) |"
    )
    lines.append("| Hillshade | unitless | **Present** — `HILLSHADE_2.tif` (viz aid) |")
    lines.append(
        "| TRI / TWI / SPI / STI | index | **Missing continuous** — rebuild from DEM "
        "(numpy/richdem/whitebox); only `*_classes.tif` exist |"
    )
    lines.append(
        "| Distance to drainage | m | **Missing continuous** — derive drainage from DEM |"
    )
    lines.append(
        "| Distance to faults / roads | m | **Missing continuous** — need vector sources "
        "(ask user; OSM for roads) |"
    )
    lines.append(
        "| NDVI | −1…1 | **Missing continuous** — Sentinel-2 composite; only classes exist |"
    )
    lines.append(
        "| Earthquake | density / PGA | **Missing continuous** — USGS catalog (ask original source) |"
    )
    lines.append(
        "| Geology / soil / geomorphology / LULC | categorical | Keep class rasters; one-hot / embed |"
    )
    lines.append("")

    lines.append("## Per-raster table")
    lines.append("")
    lines.append(
        "| Relative path | CRS | Res (m) | Size | Extent (L,B,R,T) | dtype | nodata | "
        "unique | kind | min | max | notes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a in audits:
        uniq = "n/a" if a.unique_count is None else str(a.unique_count)
        if a.unique_capped and a.unique_count is not None:
            uniq = f"{a.unique_count}+"
        extent = f"{a.left:.0f},{a.bottom:.0f},{a.right:.0f},{a.top:.0f}"
        nd = "None" if a.nodata is None else f"{a.nodata:g}"
        mn = "" if a.sample_min is None else f"{a.sample_min:g}"
        mx = "" if a.sample_max is None else f"{a.sample_max:g}"
        lines.append(
            f"| `{a.relative}` | {a.crs} | {a.res_x:.2f}×{a.res_y:.2f} | "
            f"{a.width}×{a.height} | {extent} | {a.dtype} | {nd} | {uniq} | "
            f"{a.kind} | {mn} | {mx} | {a.notes} |"
        )
    lines.append("")
    lines.append("## Machine-readable summary")
    lines.append("")
    lines.append("```")
    lines.append(f"n_rasters={len(audits)} continuous={len(cont)} categorical={len(cat)}")
    lines.append(f"near_constant_fr={train_info['near_constant_fr']}")
    lines.append(f"extreme_fr={train_info['extreme_fr']}")
    lines.append("```")
    lines.append("")
    lines.append("## Phase 0 stack")
    lines.append("")
    lines.append(
        "Build / regenerate the co-registered stack (EPSG:32646, 30 m) with:"
    )
    lines.append("")
    lines.append("```text")
    lines.append(
        "python -m lsi_vlm.stack "
        "--out outputs_vlm/aizawl_stack.tif "
        "--stats outputs_vlm/aizawl_stack_stats.json"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "See `outputs_vlm/aizawl_stack_stats.json` for band names, kinds, and "
        "per-band mean/std/min/max. The GeoTIFF is gitignored (`*.tif` / `outputs*/`)."
    )
    lines.append("")
    return "\n".join(lines)


def run_audit(
    final_maps: Path,
    training_csv: Path,
    study_config: Path,
    out_md: Path,
) -> tuple[list[RasterAudit], dict]:
    root = final_maps.parent if final_maps.name == "final_maps" else final_maps
    # Prefer repo root = parent of final_maps
    repo_root = final_maps.resolve().parent
    dirs = [final_maps, final_maps / "sonker_17"]
    paths = iter_geotiffs(dirs)
    audits = [audit_raster(p, repo_root) for p in paths]
    train_info = audit_training_csv(training_csv)
    bbox = load_study_bbox(study_config)
    md = format_audit_md(audits, train_info, bbox, study_config)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    return audits, train_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit LSI GeoTIFFs and training CSV quirks.")
    p.add_argument("--final-maps", type=Path, default=Path("final_maps"))
    p.add_argument("--training-csv", type=Path, default=Path("data/landslide_training_data.csv"))
    p.add_argument("--study-config", type=Path, default=Path("configs/aizawl.yaml"))
    p.add_argument("--out", type=Path, default=Path("outputs_vlm/audit.md"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    audits, _ = run_audit(
        final_maps=args.final_maps,
        training_csv=args.training_csv,
        study_config=args.study_config,
        out_md=args.out,
    )
    print(f"Audited {len(audits)} rasters -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
