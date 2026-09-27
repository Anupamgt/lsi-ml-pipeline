#!/usr/bin/env python3
"""
extract_training_data.py — Sample FR class rasters → training CSV
================================================================
Loads class→FR maps from the study-area FR table, samples registered
class GeoTIFFs at landslide points and buffered pseudo-absences, and
writes the training CSV configured in the study-area YAML.

Also supports a coarse prediction grid via ``--grid`` / ``--grid-only``.

Examples::

    python extract_training_data.py --study-config configs/aizawl.yaml
    python extract_training_data.py --grid
    python extract_training_data.py --grid-only --grid-output data/aizawl_grid.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy as raster_xy

from lsi_pipeline.study_area import apply_study_config, load_study_config

PIXEL_STRIDE = 10  # stride for building the valid-pixel candidate pool
GRID_STRIDE = 40  # coarser stride for prediction-grid CSV (keeps file small)
DISTANCE_BATCH = 50_000


# ─── FR TABLE ────────────────────────────────────────────────────────────────


def load_class_to_fr_maps(fr_table_path: Path) -> dict[str, dict[int, float]]:
    """Build ``{fr_table_key: {class_int: raw_FR}}`` from the FR class CSV."""
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
    """Return (x, y) centroids of valid (non-nodata, positive class) pixels."""
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
    feature_specs,
    feature_columns: list[str],
) -> dict[str, np.ndarray]:
    """Sample every registered class raster and map classes → raw FR."""
    out: dict[str, np.ndarray] = {}
    for spec in feature_specs:
        if spec.column not in feature_columns:
            continue
        if spec.fr_table_key not in fr_maps:
            raise KeyError(
                f"FR table has no factor '{spec.fr_table_key}' "
                f"(needed for feature '{spec.column}')"
            )
        cls = sample_raster(spec.raster_path, xs, ys)
        out[spec.column] = class_to_fr(cls, fr_maps[spec.fr_table_key])
    return out


def all_features_finite(
    feat: dict[str, np.ndarray],
    feature_columns: list[str],
) -> np.ndarray:
    """Boolean mask: True where every feature value is finite."""
    mask = np.ones(len(next(iter(feat.values()))), dtype=bool)
    for col in feature_columns:
        mask &= np.isfinite(feat[col])
    return mask


# ─── GRID EXPORT ─────────────────────────────────────────────────────────────


def generate_prediction_grid(
    output_path: Path,
    fr_maps: dict[str, dict[int, float]],
    feature_specs,
    feature_columns: list[str],
    stride: int = GRID_STRIDE,
) -> pd.DataFrame:
    """Build a coarse prediction grid: x, y + all feature columns (no target)."""
    pool_spec = next(
        (s for s in feature_specs if s.column == "slope_fr"),
        feature_specs[0],
    )
    pool_raster = pool_spec.raster_path
    print(
        f"\n[G1] Building strided pixel lattice (stride={stride}) "
        f"from {pool_raster.name} …"
    )
    xs, ys = build_valid_pixel_pool(pool_raster, stride=stride)
    print(f"    Lattice size: {len(xs):,} candidate centroids")

    n_feat = len(feature_columns)
    print(f"\n[G2] Sampling {n_feat} class rasters → FR …")
    feat = sample_all_features(xs, ys, fr_maps, feature_specs, feature_columns)
    ok = all_features_finite(feat, feature_columns)
    print(f"    Valid (all {n_feat} finite): {ok.sum():,}/{len(xs):,}")

    data: dict[str, np.ndarray] = {"x": xs[ok], "y": ys[ok]}
    for col in feature_columns:
        data[col] = feat[col][ok]

    col_order = ["x", "y", *feature_columns]
    df = pd.DataFrame(data)[col_order]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n  ✓ Grid saved → {output_path}  (shape: {df.shape})")
    print(f"  columns: {list(df.columns)}")
    return df


# ─── MAIN ────────────────────────────────────────────────────────────────────


def extract_training(
    fr_maps: dict[str, dict[int, float]],
    study_cfg,
) -> None:
    """Sample landslides + buffered pseudo-absences → training CSV."""
    import lsi_pipeline.config as cfg
    import lsi_pipeline.feature_registry as reg

    feature_columns = list(cfg.FEATURE_COLUMNS)
    feature_specs = reg.FEATURE_SPECS
    n_feat = len(feature_columns)
    min_buffer = cfg.MIN_SAMPLE_BUFFER_M
    training_csv = study_cfg.training_csv
    points_path = study_cfg.landslide_points
    target_crs = study_cfg.crs

    print("\n" + "═" * 68)
    print(f"  DATA EXTRACTION — {study_cfg.name} ({n_feat} FR features → training CSV)")
    print("═" * 68)

    if not points_path.exists():
        raise FileNotFoundError(f"Landslide points not found: {points_path}")

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    print(f"\n[0] FR table: {study_cfg.fr_table.name}  "
          f"({len(fr_maps)} factors, raw FR lookups)")

    print(f"\n[1] Loading landslide points from {points_path.name} …")
    gdf = gpd.read_file(points_path)
    if gdf.crs is None:
        raise ValueError(f"Landslide points have no CRS: {points_path}")
    if str(gdf.crs) != target_crs:
        gdf = gdf.to_crs(target_crs)
    ls_x = gdf.geometry.x.to_numpy(dtype=np.float64)
    ls_y = gdf.geometry.y.to_numpy(dtype=np.float64)
    n_pos = len(ls_x)
    print(f"    {n_pos} points  |  x [{ls_x.min():.1f}, {ls_x.max():.1f}]  "
          f"y [{ls_y.min():.1f}, {ls_y.max():.1f}]  CRS={target_crs}")

    print(f"\n[2] Sampling {n_feat} class rasters at landslide points …")
    ls_feat = sample_all_features(ls_x, ls_y, fr_maps, feature_specs, feature_columns)
    ls_ok = all_features_finite(ls_feat, feature_columns)
    print(f"    Valid (all {n_feat} finite): {ls_ok.sum()}/{n_pos}")
    for col in feature_columns:
        arr = ls_feat[col]
        print(f"    {col:<22} valid={np.isfinite(arr).sum()}/{n_pos}  "
              f"range=[{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
    if not np.all(ls_ok):
        bad = np.where(~ls_ok)[0].tolist()
        raise RuntimeError(
            f"Landslide points missing one or more FR values at indices {bad}. "
            "Check CRS alignment and class→FR coverage."
        )

    pool_spec = next(
        (s for s in feature_specs if s.column == "slope_fr"),
        feature_specs[0],
    )
    pool_raster = pool_spec.raster_path
    print(f"\n[3] Building valid-pixel pool (stride={PIXEL_STRIDE}) "
          f"from {pool_raster.name} …")
    pool_x, pool_y = build_valid_pixel_pool(pool_raster, stride=PIXEL_STRIDE)
    print(f"    Pool size: {len(pool_x):,} candidate pixel centroids")

    print(f"\n[4] Filtering pool: ≥{min_buffer:.0f} m from every "
          "landslide point …")
    ls_coords = np.stack([ls_x, ls_y], axis=1)
    keep = np.zeros(len(pool_x), dtype=bool)
    for start in range(0, len(pool_x), DISTANCE_BATCH):
        end = min(start + DISTANCE_BATCH, len(pool_x))
        batch = np.stack([pool_x[start:end], pool_y[start:end]], axis=1)
        dists = np.linalg.norm(
            batch[:, None, :] - ls_coords[None, :, :], axis=2
        ).min(axis=1)
        keep[start:end] = dists >= min_buffer

    cand_x, cand_y = pool_x[keep], pool_y[keep]
    print(f"    Candidates after buffer: {len(cand_x):,}")
    if len(cand_x) == 0:
        raise RuntimeError("No pseudo-absence candidates remain after buffer filter.")

    print(f"\n[5] Selecting {n_pos} pseudo-absence locations "
          f"(require finite values on all {n_feat} features) …")
    n_needed = n_pos
    max_attempts = 5
    remaining_idx = np.arange(len(cand_x))
    rng.shuffle(remaining_idx)
    collected_x: list[float] = []
    collected_y: list[float] = []
    collected_feat: dict[str, list[float]] = {c: [] for c in feature_columns}

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
        bf = sample_all_features(bx, by, fr_maps, feature_specs, feature_columns)
        ok = all_features_finite(bf, feature_columns)
        for i in np.where(ok)[0]:
            if len(collected_x) >= n_needed:
                break
            collected_x.append(float(bx[i]))
            collected_y.append(float(by[i]))
            for col in feature_columns:
                collected_feat[col].append(float(bf[col][i]))
        print(f"    attempt {attempt}: +{int(ok.sum())} valid "
              f"(have {len(collected_x)}/{n_needed})")

    if len(collected_x) < n_needed:
        raise RuntimeError(
            f"Could only find {len(collected_x)}/{n_needed} pseudo-absences "
            f"with finite values on all {n_feat} features."
        )

    neg_x = np.asarray(collected_x[:n_needed], dtype=np.float64)
    neg_y = np.asarray(collected_y[:n_needed], dtype=np.float64)
    neg_feat = {
        col: np.asarray(collected_feat[col][:n_needed], dtype=np.float64)
        for col in feature_columns
    }
    print(f"    Selected {n_needed} pseudo-absences with all {n_feat} features finite")

    print("\n[6] Assembling training CSV …")
    pos_data: dict[str, np.ndarray] = {"x": ls_x, "y": ls_y}
    pos_data.update(ls_feat)
    pos_data["target"] = np.ones(n_pos, dtype=int)
    df_pos = pd.DataFrame(pos_data)

    neg_data: dict[str, np.ndarray] = {"x": neg_x, "y": neg_y}
    neg_data.update(neg_feat)
    neg_data["target"] = np.zeros(n_needed, dtype=int)
    df_neg = pd.DataFrame(neg_data)

    col_order = ["x", "y", *feature_columns, "target"]
    df = pd.concat([df_pos, df_neg], ignore_index=True)[col_order]
    df = df.sample(frac=1, random_state=cfg.RANDOM_STATE).reset_index(drop=True)

    if df[feature_columns].isnull().any().any():
        raise RuntimeError("Output CSV still contains null feature values.")

    training_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(training_csv, index=False)
    print(f"\n  ✓ Saved → {training_csv}  (shape: {df.shape})")
    print(f"  columns: {list(df.columns)}")
    print("\n  Per-class feature means:")
    print(df.groupby("target")[feature_columns].mean().round(4).to_string())
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract FR training data and/or a coarse prediction grid "
        "from class GeoTIFFs (study-area configurable)."
    )
    parser.add_argument(
        "--study-config",
        default="configs/aizawl.yaml",
        help="Study-area YAML (default: configs/aizawl.yaml)",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Also write the prediction grid CSV from study config paths.grid_csv",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only regenerate the prediction grid (skip training extraction)",
    )
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=GRID_STRIDE,
        help=f"Pixel stride for grid lattice (default: {GRID_STRIDE})",
    )
    parser.add_argument(
        "--grid-output",
        type=Path,
        default=None,
        help="Override grid CSV path (default: study config paths.grid_csv)",
    )
    args = parser.parse_args()

    study_cfg = apply_study_config(load_study_config(args.study_config))

    import lsi_pipeline.config as cfg
    import lsi_pipeline.feature_registry as reg
    from lsi_pipeline.feature_registry import missing_rasters

    missing = missing_rasters()
    if missing:
        lines = "\n".join(f"  - {p}" for p in missing)
        raise FileNotFoundError(
            f"Missing {len(missing)} registered GeoTIFF(s). "
            f"Expected under {study_cfg.rasters_dir}:\n{lines}"
        )

    fr_maps = load_class_to_fr_maps(study_cfg.fr_table)
    grid_output = args.grid_output or study_cfg.grid_csv

    do_grid = args.grid or args.grid_only
    do_train = not args.grid_only

    if do_train:
        extract_training(fr_maps, study_cfg)

    if do_grid:
        print("\n" + "═" * 68)
        print(f"  GRID EXPORT — {study_cfg.name} → {grid_output.name}")
        print("═" * 68)
        generate_prediction_grid(
            grid_output,
            fr_maps,
            reg.FEATURE_SPECS,
            list(cfg.FEATURE_COLUMNS),
            stride=args.grid_stride,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 — CLI entrypoint
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
