#!/usr/bin/env python3
"""
extract_training_data.py  (v4 — definitive QGIS FR rasters)
============================================================
Uses the final, authoritative FR rasters as they appear in BTP_1.qgz:

  Slope FR    → ~/Slope_fr_Final.tif          (EPSG:4326, already FR-weighted)
  Aspect FR   → ~/Aspect_FR_final.tif         (EPSG:4326, already FR-weighted)
  Elevation   → ~/Elevation_reclass_final.tif (EPSG:32646, class 1–4)
                mapped to FR via cross-tab:
                  class 1 → 0.0000
                  class 2 → 0.3553
                  class 3 → 0.5401
                  class 4 → 3.2359

Coordinate handling
-------------------
  Slope/Aspect rasters are EPSG:4326 → sample landslide points directly in lon/lat.
  Elevation raster is EPSG:32646     → reproject landslide points to UTM first.
  Pseudo-absence pool is built from the EPSG:4326 slope raster valid pixels.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy as raster_xy

# ─── PATHS ────────────────────────────────────────────────────────────────────
HOME_DIR     = Path("/Users/rakeshkumar")
AIZWAL_DIR   = Path("/Users/rakeshkumar/Desktop/AIZWAL")
PIPELINE_DIR = Path("/Users/rakeshkumar/.gemini/antigravity/scratch/lsi_ml_pipeline")
OUTPUT_CSV   = PIPELINE_DIR / "data" / "landslide_training_data.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

RANDOM_SEED  = 42
MIN_BUFFER_M = 500.0     # min distance from any landslide point (metres, UTM)
PIXEL_STRIDE = 10        # stride for building the valid-pixel candidate pool

# ─── DEFINITIVE RASTERS ───────────────────────────────────────────────────────
# Both in EPSG:4326 — sample lon/lat directly (no reprojection)
SLOPE_FR_4326  = HOME_DIR / "Slope_fr_Final.tif"        # FR already applied
ASPECT_FR_4326 = HOME_DIR / "Aspect_FR_final.tif"       # FR already applied

# EPSG:32646 — need to reproject points to UTM for sampling
ELEV_CLS_32646 = HOME_DIR / "Elevation_reclass_final.tif"  # class integers 1–4

# Class → FR lookup (derived from cross-tabulation with Elevation_FR_final.tif)
ELEV_FR_MAP: dict[int, float] = {
    1: 0.0000,
    2: 0.35529,
    3: 0.54013,
    4: 3.23588,
}


# ─── HELPERS ──────────────────────────────────────────────────────────────────


def sample_raster(path: Path, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Nearest-pixel sampling. Returns NaN for nodata / out-of-bounds."""
    with rasterio.open(path) as src:
        nd   = src.nodata
        band = src.read(1).astype(np.float64)
        rows, cols = rowcol(src.transform, xs, ys)
        rows, cols = np.asarray(rows), np.asarray(cols)
        nr, nc = band.shape
        out = np.full(len(xs), np.nan)
        for i, (r, c) in enumerate(zip(rows, cols)):
            if 0 <= r < nr and 0 <= c < nc:
                v = band[r, c]
                is_nd = nd is not None and abs(float(v) - float(nd)) < 1e-3
                if not is_nd and np.isfinite(v):
                    out[i] = v
    return out


def elev_class_to_fr(cls_arr: np.ndarray) -> np.ndarray:
    """Map integer elevation class (1–4) → FR value."""
    out = np.full(len(cls_arr), np.nan)
    for c, fr in ELEV_FR_MAP.items():
        out[cls_arr == c] = fr
    return out


def build_valid_pixel_pool_4326(
    raster_path: Path,
    stride: int = PIXEL_STRIDE,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lon/lat centroids of valid (non-zero, non-nodata) pixels at stride."""
    with rasterio.open(raster_path) as src:
        nd  = src.nodata
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
        lons, lats = raster_xy(src.transform, rr_v, cc_v)
    return np.asarray(lons), np.asarray(lats)


# ─── MAIN ─────────────────────────────────────────────────────────────────────


def main() -> None:
    print("\n" + "═" * 68)
    print("  DATA EXTRACTION v4 — Definitive QGIS FR rasters (BTP_1.qgz)")
    print("═" * 68)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Load landslide points in both CRS ─────────────────────────────
    print("\n[1] Loading 22 Aizawl landslide points …")
    gdf_4326 = gpd.read_file(AIZWAL_DIR / "Aizawl_Points_UTM.gpkg")
    # natively EPSG:4326 despite the name
    if str(gdf_4326.crs) != "EPSG:4326":
        gdf_4326 = gdf_4326.to_crs("EPSG:4326")
    ls_lon = gdf_4326.geometry.x.values
    ls_lat = gdf_4326.geometry.y.values

    gdf_utm = gdf_4326.to_crs("EPSG:32646")
    ls_utmx = gdf_utm.geometry.x.values
    ls_utmy = gdf_utm.geometry.y.values
    print(f"    {len(ls_lon)} points  |  lon [{ls_lon.min():.4f}, {ls_lon.max():.4f}]  "
          f"lat [{ls_lat.min():.4f}, {ls_lat.max():.4f}]")

    # ── 2. Sample FR rasters at landslide points ──────────────────────────
    print("\n[2] Sampling definitive QGIS FR rasters at landslide points …")
    ls_slope_fr  = sample_raster(SLOPE_FR_4326,  ls_lon, ls_lat)
    ls_aspect_fr = sample_raster(ASPECT_FR_4326, ls_lon, ls_lat)
    ls_elev_cls  = sample_raster(ELEV_CLS_32646, ls_utmx, ls_utmy)
    ls_elev_fr   = elev_class_to_fr(ls_elev_cls)

    print(f"    slope_fr     : valid={np.isfinite(ls_slope_fr).sum()}/22  "
          f"range=[{np.nanmin(ls_slope_fr):.4f}, {np.nanmax(ls_slope_fr):.4f}]")
    print(f"    aspect_fr    : valid={np.isfinite(ls_aspect_fr).sum()}/22  "
          f"range=[{np.nanmin(ls_aspect_fr):.4f}, {np.nanmax(ls_aspect_fr):.4f}]")
    print(f"    elevation_fr : valid={np.isfinite(ls_elev_fr).sum()}/22  "
          f"range=[{np.nanmin(ls_elev_fr):.4f}, {np.nanmax(ls_elev_fr):.4f}]")
    print(f"    elev classes : {np.unique(ls_elev_cls[np.isfinite(ls_elev_cls)]).astype(int).tolist()}")

    # ── 3. Build valid-pixel pool in EPSG:4326 ────────────────────────────
    print(f"\n[3] Building valid-pixel pool (stride={PIXEL_STRIDE}) from Slope_fr_Final.tif …")
    pool_lon, pool_lat = build_valid_pixel_pool_4326(SLOPE_FR_4326, stride=PIXEL_STRIDE)
    print(f"    Pool size: {len(pool_lon):,} candidate pixel centroids")

    # ── 4. Buffer filter in UTM (metric distances) ────────────────────────
    print(f"\n[4] Filtering pool: ≥{MIN_BUFFER_M:.0f} m from every landslide point …")
    # Convert pool lon/lat → UTM for distance check
    pool_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(pool_lon, pool_lat),
        crs="EPSG:4326"
    ).to_crs("EPSG:32646")
    pool_utmx = pool_gdf.geometry.x.values
    pool_utmy = pool_gdf.geometry.y.values

    ls_coords = np.stack([ls_utmx, ls_utmy], axis=1)
    BATCH = 50_000
    keep = np.zeros(len(pool_utmx), dtype=bool)
    for start in range(0, len(pool_utmx), BATCH):
        end = min(start + BATCH, len(pool_utmx))
        batch = np.stack([pool_utmx[start:end], pool_utmy[start:end]], axis=1)
        dists = np.linalg.norm(batch[:, None, :] - ls_coords[None, :, :], axis=2).min(axis=1)
        keep[start:end] = dists >= MIN_BUFFER_M

    cand_lon = pool_lon[keep]
    cand_lat = pool_lat[keep]
    print(f"    Candidates after filter: {len(cand_lon):,}")

    # ── 5. Randomly select 22 pseudo-absence locations ────────────────────
    print("\n[5] Selecting 22 pseudo-absence locations …")
    idx = rng.choice(len(cand_lon), size=min(22, len(cand_lon)), replace=False)
    neg_lon = cand_lon[idx]
    neg_lat = cand_lat[idx]

    # Convert to UTM for elevation raster sampling
    neg_gdf = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(neg_lon, neg_lat), crs="EPSG:4326"
    ).to_crs("EPSG:32646")
    neg_utmx = neg_gdf.geometry.x.values
    neg_utmy = neg_gdf.geometry.y.values

    # ── 6. Sample FR rasters at pseudo-absence locations ──────────────────
    print("[6] Sampling FR rasters at pseudo-absence locations …")
    neg_slope_fr  = sample_raster(SLOPE_FR_4326,  neg_lon, neg_lat)
    neg_aspect_fr = sample_raster(ASPECT_FR_4326, neg_lon, neg_lat)
    neg_elev_cls  = sample_raster(ELEV_CLS_32646, neg_utmx, neg_utmy)
    neg_elev_fr   = elev_class_to_fr(neg_elev_cls)

    valid_neg = np.isfinite(neg_slope_fr) & np.isfinite(neg_aspect_fr) & np.isfinite(neg_elev_fr)
    print(f"    Valid (all 3 non-null): {valid_neg.sum()}/{len(neg_lon)}")
    print(f"    slope_fr     : [{np.nanmin(neg_slope_fr):.4f}, {np.nanmax(neg_slope_fr):.4f}]")
    print(f"    aspect_fr    : [{np.nanmin(neg_aspect_fr):.4f}, {np.nanmax(neg_aspect_fr):.4f}]")
    print(f"    elevation_fr : [{np.nanmin(neg_elev_fr):.4f}, {np.nanmax(neg_elev_fr):.4f}]")

    # Fill residual NaNs with column means (rare edge pixels)
    for arr in [neg_slope_fr, neg_aspect_fr, neg_elev_fr]:
        if not np.all(np.isfinite(arr)):
            arr[~np.isfinite(arr)] = np.nanmean(arr)

    # ── 7. Assemble training CSV ───────────────────────────────────────────
    print("\n[7] Assembling landslide_training_data.csv …")
    df_pos = pd.DataFrame({
        "x": ls_utmx, "y": ls_utmy,
        "slope_fr": ls_slope_fr, "aspect_fr": ls_aspect_fr,
        "elevation_fr": ls_elev_fr, "target": 1,
    })
    df_neg = pd.DataFrame({
        "x": neg_utmx, "y": neg_utmy,
        "slope_fr": neg_slope_fr, "aspect_fr": neg_aspect_fr,
        "elevation_fr": neg_elev_fr, "target": 0,
    })
    df = pd.concat([df_pos, df_neg], ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Fill any remaining NaNs
    for col in ["slope_fr", "aspect_fr", "elevation_fr"]:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)

    # ── 8. Class overlap report ───────────────────────────────────────────
    pos = df[df.target == 1]
    neg = df[df.target == 0]
    print("\n  ─── CLASS OVERLAP (higher = better challenge for model) ───")
    print(f"  {'Feature':<16}  {'Positive':<22}  {'Negative':<22}  Overlap")
    print("  " + "─" * 76)
    for col in ["slope_fr", "aspect_fr", "elevation_fr"]:
        pm, px = pos[col].min(), pos[col].max()
        nm, nx = neg[col].min(), neg[col].max()
        ov = max(0.0, min(px, nx) - max(pm, nm))
        print(f"  {col:<16}  [{pm:.4f}, {px:.4f}]        "
              f"[{nm:.4f}, {nx:.4f}]        {ov:.4f}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  ✓ Saved → {OUTPUT_CSV}  (shape: {df.shape})")

    print("\n  Per-class statistics:")
    print(df.groupby("target")[["slope_fr","aspect_fr","elevation_fr"]]
            .agg(["min","mean","max"]).round(4).to_string())
    print()


if __name__ == "__main__":
    main()
