"""Build a co-registered multi-band continuous factor stack for Aizawl (VLM Stage B prep).

CLI:
    python -m lsi_vlm.stack [--out PATH] [--stats PATH]

Uses rasterio + numpy only (no richdem/whitebox). Target CRS EPSG:32646, 30 m.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

TARGET_CRS = "EPSG:32646"
TARGET_RES = 30.0
STACK_NODATA = -9999.0


@dataclass(frozen=True)
class BandSpec:
    name: str
    source: Path | None  # None → derived
    kind: str  # "continuous" | "categorical" | "derived"
    description: str


def default_band_specs(repo: Path) -> list[BandSpec]:
    fm = repo / "final_maps"
    s17 = fm / "sonker_17"
    return [
        BandSpec(
            "rainfall_mm",
            s17 / "S17_01_Rainfall_CHIRPS_mm.tif",
            "continuous",
            "CHIRPS rainfall (mm)",
        ),
        BandSpec(
            "gravity_mgal",
            s17 / "S17_10_Gravity_Bouguer_mGal.tif",
            "continuous",
            "Bouguer gravity anomaly (mGal)",
        ),
        BandSpec(
            "slope_final_deg",
            fm / "Slope_FINAL.tif",
            "continuous",
            "Existing Slope_FINAL raster (degrees assumed)",
        ),
        BandSpec(
            "elevation_m",
            fm / "DEM_UTF_FINAL.tif",
            "continuous",
            "DEM elevation (m)",
        ),
        BandSpec(
            "hillshade",
            fm / "HILLSHADE_2.tif",
            "continuous",
            "Hillshade (visual / illumination aid)",
        ),
        BandSpec(
            "slope_dem_deg",
            None,
            "derived",
            "Slope (degrees) from DEM via numpy gradient",
        ),
        # Categorical Sonker class layers (integer-coded, clearly tagged)
        BandSpec(
            "earthquake_class",
            s17 / "S17_02_Earthquake_classes.tif",
            "categorical",
            "Earthquake class codes (not continuous)",
        ),
        BandSpec(
            "slope_class",
            s17 / "S17_03_Slope_classes.tif",
            "categorical",
            "Slope class codes",
        ),
        BandSpec(
            "altitude_class",
            s17 / "S17_04_Altitude_classes.tif",
            "categorical",
            "Altitude class codes",
        ),
        BandSpec(
            "drainage_dist_class",
            s17 / "S17_05_Distance_drainages_classes.tif",
            "categorical",
            "Distance-to-drainage class codes (continuous m missing)",
        ),
        BandSpec(
            "tri_class",
            s17 / "S17_06_TRI_classes.tif",
            "categorical",
            "TRI class codes",
        ),
        BandSpec(
            "geomorphology_class",
            s17 / "S17_07_Geomorphology_classes.tif",
            "categorical",
            "Geomorphology class codes",
        ),
        BandSpec(
            "geology_class",
            s17 / "S17_08_Geology_classes.tif",
            "categorical",
            "Geology class codes",
        ),
        BandSpec(
            "soil_class",
            s17 / "S17_09_Soil_India_classes.tif",
            "categorical",
            "Soil class codes",
        ),
        BandSpec(
            "faults_dist_class",
            s17 / "S17_11_Distance_faults_classes.tif",
            "categorical",
            "Distance-to-faults class codes (continuous m missing)",
        ),
        BandSpec(
            "sti_class",
            s17 / "S17_12_STI_classes.tif",
            "categorical",
            "STI class codes",
        ),
        BandSpec(
            "twi_class",
            s17 / "S17_13_TWI_classes.tif",
            "categorical",
            "TWI class codes",
        ),
        BandSpec(
            "spi_class",
            s17 / "S17_14_SPI_classes.tif",
            "categorical",
            "SPI class codes",
        ),
        BandSpec(
            "roads_dist_class",
            s17 / "S17_15_Distance_roads_classes.tif",
            "categorical",
            "Distance-to-roads class codes (continuous m missing)",
        ),
        BandSpec(
            "lulc_class",
            s17 / "S17_16_LULC_classes.tif",
            "categorical",
            "LULC class codes",
        ),
        BandSpec(
            "ndvi_class",
            s17 / "S17_17_NDVI_classes.tif",
            "categorical",
            "NDVI class codes (continuous NDVI missing)",
        ),
    ]


def reference_grid_from_rainfall(rainfall_path: Path) -> dict:
    """Use the CHIRPS rainfall grid as the 30 m EPSG:32646 template."""
    with rasterio.open(rainfall_path) as ds:
        if ds.crs is None or ds.crs.to_string() not in ("EPSG:32646", "epsg:32646"):
            # Still accept if equivalent
            pass
        transform = ds.transform
        # Force exact 30 m if close
        res_x, res_y = abs(transform.a), abs(transform.e)
        if abs(res_x - TARGET_RES) > 0.5 or abs(res_y - TARGET_RES) > 0.5:
            raise ValueError(
                f"Rainfall grid resolution {res_x}×{res_y} is not ~{TARGET_RES} m"
            )
        return {
            "crs": TARGET_CRS,
            "transform": transform,
            "width": ds.width,
            "height": ds.height,
            "bounds": ds.bounds,
        }


def slope_degrees_from_dem(dem: np.ndarray, res_x: float, res_y: float) -> np.ndarray:
    """Horn-style slope in degrees from elevation array (cell sizes in metres)."""
    # np.gradient: first axis = rows (y), second = cols (x)
    gy, gx = np.gradient(dem.astype(np.float64), res_y, res_x)
    slope = np.degrees(np.arctan(np.sqrt(gx * gx + gy * gy)))
    return slope.astype(np.float32)


def warp_to_grid(
    src_path: Path,
    grid: dict,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject/resample a source raster onto the reference grid (float32, nodata filled)."""
    dst = np.full((grid["height"], grid["width"]), STACK_NODATA, dtype=np.float32)
    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=grid["transform"],
            dst_crs=grid["crs"],
            dst_nodata=STACK_NODATA,
            resampling=resampling,
        )
    return dst


def band_stats(arr: np.ndarray, nodata: float = STACK_NODATA) -> dict:
    valid = arr[np.isfinite(arr) & (arr != nodata)]
    if valid.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "valid_pixels": 0}
    return {
        "mean": float(np.mean(valid)),
        "std": float(np.std(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "valid_pixels": int(valid.size),
    }


def build_stack(
    repo: Path,
    out_tif: Path,
    stats_json: Path,
    include_categorical: bool = True,
) -> dict:
    """Build multi-band stack GeoTIFF + per-band stats JSON."""
    specs = default_band_specs(repo)
    if not include_categorical:
        specs = [s for s in specs if s.kind != "categorical"]

    rainfall = repo / "final_maps" / "sonker_17" / "S17_01_Rainfall_CHIRPS_mm.tif"
    if not rainfall.is_file():
        raise FileNotFoundError(f"Reference rainfall raster missing: {rainfall}")
    grid = reference_grid_from_rainfall(rainfall)

    dem_path = repo / "final_maps" / "DEM_UTF_FINAL.tif"
    dem_arr = warp_to_grid(dem_path, grid, Resampling.bilinear)
    dem_valid = dem_arr.copy()
    dem_valid[dem_valid == STACK_NODATA] = np.nan
    slope_dem = slope_degrees_from_dem(
        np.nan_to_num(dem_valid, nan=0.0), TARGET_RES, TARGET_RES
    )
    slope_dem[~np.isfinite(dem_valid)] = STACK_NODATA

    bands: list[np.ndarray] = []
    names: list[str] = []
    kinds: list[str] = []
    descriptions: list[str] = []
    sources: list[str] = []
    missing: list[str] = []

    for spec in specs:
        if spec.kind == "derived" and spec.name == "slope_dem_deg":
            arr = slope_dem
            src_label = "derived:numpy_gradient(DEM_UTF_FINAL)"
        elif spec.source is None:
            missing.append(spec.name)
            continue
        elif not spec.source.is_file():
            missing.append(f"{spec.name} ({spec.source})")
            continue
        else:
            resampling = (
                Resampling.nearest if spec.kind == "categorical" else Resampling.bilinear
            )
            arr = warp_to_grid(spec.source, grid, resampling)
            src_label = str(spec.source.relative_to(repo)).replace("\\", "/")

        bands.append(arr)
        names.append(spec.name)
        kinds.append(spec.kind)
        descriptions.append(spec.description)
        sources.append(src_label)

    if not bands:
        raise RuntimeError("No bands written — check source rasters")

    out_tif.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": len(bands),
        "width": grid["width"],
        "height": grid["height"],
        "crs": grid["crs"],
        "transform": grid["transform"],
        "nodata": STACK_NODATA,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(out_tif, "w", **profile) as dst:
        for i, arr in enumerate(bands, start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, names[i - 1])
        # GDAL band names via descriptions; also store tags
        dst.update_tags(
            BAND_NAMES=",".join(names),
            BAND_KINDS=",".join(kinds),
            PURPOSE="lsi_vlm Phase 0 continuous/categorical stack",
            AIZAWL_ROLE="test_set_only",
        )
        for i, (n, k, d, s) in enumerate(
            zip(names, kinds, descriptions, sources, strict=True), start=1
        ):
            dst.update_tags(i, name=n, kind=k, description=d, source=s)

    stats = {
        "crs": TARGET_CRS,
        "resolution_m": TARGET_RES,
        "width": grid["width"],
        "height": grid["height"],
        "bounds": {
            "left": float(grid["bounds"].left),
            "bottom": float(grid["bounds"].bottom),
            "right": float(grid["bounds"].right),
            "top": float(grid["bounds"].top),
        },
        "nodata": STACK_NODATA,
        "bands": [],
        "missing_sources": missing,
        "missing_continuous_rebuild": [
            "drainage_distance_m",
            "faults_distance_m",
            "roads_distance_m",
            "ndvi_continuous",
            "tri_continuous",
            "twi_continuous",
            "spi_continuous",
            "sti_continuous",
            "earthquake_density_or_pga",
        ],
        "regenerate": (
            "python -m lsi_vlm.stack "
            f"--out {out_tif.as_posix()} --stats {stats_json.as_posix()}"
        ),
    }
    for n, k, d, s, arr in zip(
        names, kinds, descriptions, sources, bands, strict=True
    ):
        st = band_stats(arr)
        st.update({"name": n, "kind": k, "description": d, "source": s})
        stats["bands"].append(st)

    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Aizawl multi-band VLM factor stack.")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--out", type=Path, default=Path("outputs_vlm/aizawl_stack.tif"))
    p.add_argument("--stats", type=Path, default=Path("outputs_vlm/aizawl_stack_stats.json"))
    p.add_argument(
        "--continuous-only",
        action="store_true",
        help="Omit integer categorical class bands.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = build_stack(
        repo=args.repo.resolve(),
        out_tif=args.out,
        stats_json=args.stats,
        include_categorical=not args.continuous_only,
    )
    print(f"Wrote {args.out} ({len(stats['bands'])} bands)")
    print(f"Stats -> {args.stats}")
    print("Bands:", ", ".join(b["name"] for b in stats["bands"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
