#!/usr/bin/env python3
"""
extract_training_data.py — Sample 17 Sonker FR class rasters → training CSV
===========================================================================
Loads class→FR maps from ``data/fr_class_table_17factor.csv`` (raw ``FR`` column),
samples all registered class GeoTIFFs under ``final_maps/sonker_17/`` at
landslide points and buffered pseudo-absences (EPSG:32646), and writes
``data/landslide_training_data.csv`` with columns::

    x, y, <17 *_fr features>, target

Repo-relative paths only — no machine-specific absolute paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy as raster_xy

from lsi_pipeline.config import FEATURE_COLUMNS, MIN_SAMPLE_BUFFER_M, RANDOM_STATE
from lsi_pipeline.feature_registry import (
    FEATURE_SPECS,
    FR_TABLE_PATH,
    LANDSLIDE_POINTS_PATH,
    TARGET_CRS,
    TRAINING_CSV_PATH,
    missing_rasters,
)

PIXEL_STRIDE = 10  # stride for building the valid-pixel candidate pool
DISTANCE_BATCH = 50_000


# ─── FR TABLE ────────────────────────────────────────────────────────────────


def load_class_to_fr_maps(fr_table_path: Path) -> dict[str, dict[int, float]]:
    """Build ``{fr_table_key: {class_int: raw_FR}}`` from the FR class CSV.

    Prefers the raw ``FR`` column (not ``FRn``).
    """
    if not fr_table_path.exists():
        raise FileNotFoundError(f"FR class table not found: {fr_table_path}")

    df = pd.read_csv(fr_table_path)
    required = {"factor", "class", "FR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FR table missing columns {sorted(missing)}: {fr_table_path}")

    maps: dict[str, dict[int, float]] = {}
    for factor, group in df.groupby("factor", sort=False):
        class_map: dict[int, float] = {}
        for _, row in group.iterrows():
            cls = int(row["class"])
            class_map[cls] = float(row["FR"])
        maps[str(factor)] = class_map
    return maps


def class_to_fr(cls_arr: np.ndarray, fr_map: dict[int, float]) -> np.ndarray:
    """Map integer class values → raw FR. Unknown / NaN classes stay NaN."""
    out = np.full(len(cls_arr), np.nan, dtype=np.float64)
    finite = np.isfinite(cls_arr)
    if not finite.any():
        return out
    # Round to nearest int for float class codes from GeoTIFF
    rounded = np.rint(cls_arr[finite]).astype(int)
    mapped = np.full(rounded.shape, np.nan, dtype=np.float64)
    for cls, fr in fr_map.items():
        mapped[rounded == cls] = fr
    out[finite] = mapped
    return out


# ─── RASTER HELPERS ──────────────────────────────────────────────────────────


def sample_raster(path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Nearest-pixel sampling in the raster CRS. Returns NaN for nodata / OOB."""
    with rasterio.open(path) as src:
        nd = src.nodata
        band = src.read(1).astype(np.float64)
        rows, cols = rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        nr, nc = band.shape
        out = np.full(len(xs), np.nan, dtype=np.float64)
        for i, (r, c) in enumerate(zip(rows, cols)):
            if 0 <= r < nr and 0 <= c < nc:
                v = band[r, c]
                is_nd = nd is not None and abs(float(v) - float(nd)) < 1e-3
                if not is_nd and np.isfinite(v):
                    out[i] = v
    return out


def build_valid_pixel_pool(
    raster_path: Path,
    stride: int = PIXEL_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return UTM (x, y) centroids of valid (non-nodata, positive class) pixels."""
    with rasterio.open(raster_path) as src:
        nd = src.nodata
        arr = src.read(1)
        nr, nc = arr.shape
        ri = np.arange(0, nr, stride)
        ci = np.arange(0, nc, stride)
        rr, cc = np.meshgrid(ri, ci, indexing="ij")
        rr, cc = rr.ravel(), cc.ravel()
        vals = arr[rr, cc].astype(np.float64)
        if nd is not None:
            valid = (vals != nd) & np.isfinite(vals) & (vals > 0)
        else:
            valid = np.isfinite(vals) & (vals > 0)
        rr_v, cc_v = rr[valid], cc[valid]
        xs, ys = raster_xy(src.transform, rr_v, cc_v)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def sample_all_features(
    xs: np.ndarray,
    ys: np.ndarray,
    fr_maps: dict[str, dict[int, float]],
) -> dict[str, np.ndarray]:
    """Sample every registered class raster and map classes → raw FR."""
    out: dict[str, np.ndarray] = {}
    for spec in FEATURE_SPECS:
        if spec.fr_table_key not in fr_maps:
            raise KeyError(
                f"FR table has no factor '{spec.fr_table_key}' "
                f"(needed for feature '{spec.column}')"
            )
        cls = sample_raster(spec.raster_path, xs, ys)
        out[spec.column] = class_to_fr(cls, fr_maps[spec.fr_table_key])
    return out


def all_features_finite(feat: dict[str, np.ndarray]) -> np.ndarray:
    """Boolean mask: True where every feature value is finite."""
    mask = np.ones(len(next(iter(feat.values()))), dtype=bool)
    for col in FEATURE_COLUMNS:
        mask &= np.isfinite(feat[col])
    return mask


# ─── MAIN ────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "═" * 68)
    print("  DATA EXTRACTION — 17 Sonker FR class rasters → training CSV")
    print("═" * 68)

    missing = missing_rasters()
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} registered GeoTIFF(s). "
            f"Expected under final_maps/sonker_17/:\n{lines}"
        )

    if not LANDSLIDE_POINTS_PATH.exists():
        raise FileNotFoundError(
            f"Landslide points not found: {LANDSLIDE_POINTS_PATH}"
        )

    rng = np.random.default_rng(RANDOM_STATE)
    fr_maps = load_class_to_fr_maps(FR_TABLE_PATH)
    print(f"\n[0] FR table: {FR_TABLE_PATH.name}  "
          f"({len(fr_maps)} factors, raw FR lookups)")

    # ── 1. Landslide points in EPSG:32646 ────────────────────────────────
    print(f"\n[1] Loading landslide points from {LANDSLIDE_POINTS_PATH.name} …")
    gdf = gpd.read_file(LANDSLIDE_POINTS_PATH)
    if gdf.crs is None:
        raise ValueError(f"Landslide points have no CRS: {LANDSLIDE_POINTS_PATH}")
    if str(gdf.crs) != TARGET_CRS:
        gdf = gdf.to_crs(TARGET_CRS)
    ls_x = gdf.geometry.x.to_numpy(dtype=np.float64)
    ls_y = gdf.geometry.y.to_numpy(dtype=np.float64)
    n_pos = len(ls_x)
    print(f"    {n_pos} points  |  x [{ls_x.min():.1f}, {ls_x.max():.1f}]  "
          f"y [{ls_y.min():.1f}, {ls_y.max():.1f}]  CRS={TARGET_CRS}")

    # ── 2. Sample 17 FR features at landslide points ─────────────────────
    print("\n[2] Sampling 17 class rasters at landslide points …")
    ls_feat = sample_all_features(ls_x, ls_y, fr_maps)
    ls_ok = all_features_finite(ls_feat)
    print(f"    Valid (all 17 finite): {ls_ok.sum()}/{n_pos}")
    for col in FEATURE_COLUMNS:
        arr = ls_feat[col]
        print(f"    {col:<22} valid={np.isfinite(arr).sum()}/{n_pos}  "
              f"range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
    if not np.all(ls_ok):
        bad = np.where(~ls_ok)[0].tolist()
        raise RuntimeError(
            f"Landslide points missing one or more FR values at indices {bad}. "
            "Check CRS alignment and class→FR coverage."
        )

    # ── 3. Valid-pixel pool from slope class raster (EPSG:32646) ─────────
    pool_raster = next(s for s in FEATURE_SPECS if s.column == "slope_fr").raster_path
    print(f"\n[3] Building valid-pixel pool (stride={PIXEL_STRIDE}) "
          f"from {pool_raster.name} …")
    pool_x, pool_y = build_valid_pixel_pool(pool_raster, stride=PIXEL_STRIDE)
    print(f"    Pool size: {len(pool_x):,} candidate pixel centroids")

    # ── 4. Buffer filter (≥ MIN_SAMPLE_BUFFER_M from every landslide) ────
    print(f"\n[4] Filtering pool: ≥{MIN_SAMPLE_BUFFER_M:.0f} m from every "
          "landslide point …")
    ls_coords = np.stack([ls_x, ls_y], axis=1)
    keep = np.zeros(len(pool_x), dtype=bool)
    for start in range(0, len(pool_x), DISTANCE_BATCH):
        end = min(start + DISTANCE_BATCH, len(pool_x))
        batch = np.stack([pool_x[start:end], pool_y[start:end]], axis=1)
        dists = np.linalg.norm(
            batch[:, None, :] - ls_coords[None, :, :], axis=2
        ).min(axis=1)
        keep[start:end] = dists >= MIN_SAMPLE_BUFFER_M

    cand_x, cand_y = pool_x[keep], pool_y[keep]
    print(f"    Candidates after buffer: {len(cand_x):,}")
    if len(cand_x) == 0:
        raise RuntimeError("No pseudo-absence candidates remain after buffer filter.")

    # ── 5. Select & validate pseudo-absences (all 17 finite) ─────────────
    print(f"\n[5] Selecting {n_pos} pseudo-absence locations "
          "(require finite values on all 17 features) …")
    # Oversample then filter; reshuffle if needed
    n_needed = n_pos
    max_attempts = 5
    neg_x = neg_y = None
    neg_feat: dict[str, np.ndarray] | None = None
    remaining_idx = np.arange(len(cand_x))
    rng.shuffle(remaining_idx)
    collected_x: list[float] = []
    collected_y: list[float] = []
    collected_feat: dict[str, list[float]] = {c: [] for c in FEATURE_COLUMNS}

    attempt = 0
    cursor = 0
    while len(collected_x) < n_needed and attempt < max_attempts:
        attempt += 1
        take = min(max(n_needed * 4, n_needed), len(remaining_idx) - cursor)
        if take <= 0:
            break
        batch_idx = remaining_idx[cursor: cursor + take]
        cursor += take
        bx = cand_x[batch_idx]
        by = cand_y[batch_idx]
        bf = sample_all_features(bx, by, fr_maps)
        ok = all_features_finite(bf)
        for i in np.where(ok)[0]:
            if len(collected_x) >= n_needed:
                break
            collected_x.append(float(bx[i]))
            collected_y.append(float(by[i]))
            for col in FEATURE_COLUMNS:
                collected_feat[col].append(float(bf[col][i]))
        print(f"    attempt {attempt}: +{int(ok.sum())} valid "
              f"(have {len(collected_x)}/{n_needed})")

    if len(collected_x) < n_needed:
        raise RuntimeError(
            f"Could only find {len(collected_x)}/{n_needed} pseudo-absences "
            "with finite values on all 17 features."
        )

    neg_x = np.asarray(collected_x[:n_needed], dtype=np.float64)
    neg_y = np.asarray(collected_y[:n_needed], dtype=np.float64)
    neg_feat = {
        col: np.asarray(collected_feat[col][:n_needed], dtype=np.float64)
        for col in FEATURE_COLUMNS
    }
    print(f"    Selected {n_needed} pseudo-absences with all 17 features finite")

    # ── 6. Assemble training CSV ─────────────────────────────────────────
    print("\n[6] Assembling landslide_training_data.csv …")
    pos_data: dict[str, np.ndarray] = {"x": ls_x, "y": ls_y}
    pos_data.update(ls_feat)
    pos_data["target"] = np.ones(n_pos, dtype=int)
    df_pos = pd.DataFrame(pos_data)

    neg_data: dict[str, np.ndarray] = {"x": neg_x, "y": neg_y}
    neg_data.update(neg_feat)
    neg_data["target"] = np.zeros(n_needed, dtype=int)
    df_neg = pd.DataFrame(neg_data)

    col_order = ["x", "y", *FEATURE_COLUMNS, "target"]
    df = pd.concat([df_pos, df_neg], ignore_index=True)[col_order]
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    if df[FEATURE_COLUMNS].isnull().any().any():
        raise RuntimeError("Output CSV still contains null feature values.")

    TRAINING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TRAINING_CSV_PATH, index=False)
    print(f"\n  ✓ Saved → {TRAINING_CSV_PATH}  (shape: {df.shape})")
    print(f"  columns: {list(df.columns)}")
    print("\n  Per-class feature means:")
    print(df.groupby("target")[FEATURE_COLUMNS].mean().round(4).to_string())
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
