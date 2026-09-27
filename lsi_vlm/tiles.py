"""Stage A tile rendering: composite terrain PNGs + Sentinel-2 RGB windows.

For each training point, cut a window centred on (x, y). Default is 64×64 px at
30 m ≈ 1.92 km (configurable via ``configs/vlm_stage_a.yaml`` → ``tiles``).

Composite layers
----------------
1. **Hillshade** — greyscale base (``HILLSHADE_2.tif``).
2. **Slope** — continuous degrees colour-ramped (YlOrRd) with alpha blend.
3. **LSI zones** (optional flag) — discrete 1–5 overlay; leave off for the
   honest zero-shot baseline (with-LSI is an ablation / leak check).

Colour legend (fixed; also written to ``outputs_vlm/tiles/colour_legend.md``)
----------------------------------------------------------------------------
**Slope ramp** (degrees, YlOrRd, transparent below ~0° / nodata):

| Slope (°) | Colour (approx RGB) |
|-----------|---------------------|
| 0         | #ffffcc (pale yellow) |
| 15        | #fed976 |
| 30        | #fd8d3c |
| 45        | #e31a1c |
| ≥60       | #800026 (dark red) |

**LSI zones** (``LSI_Final_Zones.tif`` quantile classes):

| Code | Class      | RGB           |
|------|------------|---------------|
| 1    | very_low   | (50, 160, 50) |
| 2    | low        | (150, 200, 50)|
| 3    | moderate   | (250, 220, 50)|
| 4    | high       | (240, 140, 40)|
| 5    | very_high  | (200, 40, 40) |

Satellite RGB
-------------
Sentinel-2 L2A true colour via Microsoft Planetary Computer STAC
(``pystac-client`` + ``planetary-computer``). Dry-season (Nov–Feb) median
composite. On network/STAC failure a grey placeholder PNG is written and a
warning is logged — generation never aborts the batch.

CLI
---
::

    python -m lsi_vlm.tiles [--config PATH] [--skip-satellite] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import rasterio
import yaml
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, transform_bounds

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "vlm_stage_a.yaml"

# Documented colour legend (RGB 0–255).
LSI_ZONE_COLOURS: dict[int, tuple[int, int, int]] = {
    1: (50, 160, 50),  # very_low
    2: (150, 200, 50),  # low
    3: (250, 220, 50),  # moderate
    4: (240, 140, 40),  # high
    5: (200, 40, 40),  # very_high
}

LSI_ZONE_NAMES: dict[int, str] = {
    1: "very_low",
    2: "low",
    3: "moderate",
    4: "high",
    5: "very_high",
}

COLOUR_LEGEND_MD = """# Stage A tile colour legend

## Slope ramp (degrees, YlOrRd, alpha-blended over hillshade)

| Slope (°) | Colour (approx) |
|-----------|-----------------|
| 0 | `#ffffcc` pale yellow |
| 15 | `#fed976` |
| 30 | `#fd8d3c` orange |
| 45 | `#e31a1c` red |
| ≥60 | `#800026` dark red |

Nodata / invalid slope pixels stay transparent (hillshade only).

## LSI zones (`LSI_Final_Zones.tif`, optional overlay)

| Code | Class | RGB |
|------|-------|-----|
| 1 | very_low | (50, 160, 50) |
| 2 | low | (150, 200, 50) |
| 3 | moderate | (250, 220, 50) |
| 4 | high | (240, 140, 40) |
| 5 | very_high | (200, 40, 40) |

Leave `include_lsi_overlay: false` for the honest Stage A baseline (LSI in the
image lets a VLM read the FR answer back).
"""


@dataclass(frozen=True)
class TileWindow:
    """Geographic window for one tile (centre + pixel grid)."""

    center_x: float
    center_y: float
    size_px: int
    resolution_m: float
    crs: str
    row: int | None = None  # grid row index (study-area grid only)
    col: int | None = None  # grid col index
    point_index: int | None = None  # training-point index

    @property
    def half_m(self) -> float:
        return 0.5 * self.size_px * self.resolution_m

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(left, bottom, right, top) in ``crs``."""
        h = self.half_m
        return (
            self.center_x - h,
            self.center_y - h,
            self.center_x + h,
            self.center_y + h,
        )

    @property
    def transform(self) -> Affine:
        left, bottom, right, top = self.bounds
        return from_bounds(left, bottom, right, top, self.size_px, self.size_px)

    @property
    def width_m(self) -> float:
        return self.size_px * self.resolution_m


@dataclass
class TilePaths:
    composite: Path
    satellite: Path
    point_index: int
    x: float
    y: float
    target: int | None = None
    satellite_ok: bool = False
    satellite_note: str = ""
    include_lsi_overlay: bool = False


@dataclass
class TilesConfig:
    window_size_px: int = 64
    resolution_m: float = 30.0
    crs: str = "EPSG:32646"
    output_dir: Path = field(default_factory=lambda: REPO_ROOT / "outputs_vlm" / "tiles")
    training_csv: Path = field(
        default_factory=lambda: REPO_ROOT / "data" / "landslide_training_data.csv"
    )
    hillshade_path: Path = field(
        default_factory=lambda: REPO_ROOT / "final_maps" / "HILLSHADE_2.tif"
    )
    slope_path: Path = field(
        default_factory=lambda: REPO_ROOT / "final_maps" / "Slope_FINAL.tif"
    )
    lsi_zones_path: Path = field(
        default_factory=lambda: REPO_ROOT / "final_maps" / "LSI_Final_Zones.tif"
    )
    include_lsi_overlay: bool = False
    slope_alpha: float = 0.55
    lsi_alpha: float = 0.35
    slope_vmax_deg: float = 60.0
    satellite: dict[str, Any] = field(default_factory=dict)
    grid: dict[str, Any] = field(default_factory=dict)
    repo_root: Path = field(default_factory=lambda: REPO_ROOT)

    @staticmethod
    def from_dict(raw: dict[str, Any], *, repo_root: Path | None = None) -> TilesConfig:
        root = Path(repo_root) if repo_root is not None else REPO_ROOT
        t = dict(raw.get("tiles") or raw)

        def _p(key: str, default: str | Path) -> Path:
            val = t.get(key, default)
            p = Path(val)
            return p if p.is_absolute() else root / p

        sat = dict(t.get("satellite") or {})
        grid = dict(t.get("grid") or {})
        return TilesConfig(
            window_size_px=int(t.get("window_size_px", 64)),
            resolution_m=float(t.get("resolution_m", 30.0)),
            crs=str(t.get("crs", "EPSG:32646")),
            output_dir=_p("output_dir", "outputs_vlm/tiles"),
            training_csv=_p("training_csv", "data/landslide_training_data.csv"),
            hillshade_path=_p("hillshade_path", "final_maps/HILLSHADE_2.tif"),
            slope_path=_p("slope_path", "final_maps/Slope_FINAL.tif"),
            lsi_zones_path=_p("lsi_zones_path", "final_maps/LSI_Final_Zones.tif"),
            include_lsi_overlay=bool(t.get("include_lsi_overlay", False)),
            slope_alpha=float(t.get("slope_alpha", 0.55)),
            lsi_alpha=float(t.get("lsi_alpha", 0.35)),
            slope_vmax_deg=float(t.get("slope_vmax_deg", 60.0)),
            satellite=sat,
            grid=grid,
            repo_root=root,
        )


def load_tiles_config(path: Path | str | None = None) -> TilesConfig:
    """Load ``tiles`` section from YAML (defaults if file missing)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    if not cfg_path.is_file():
        logger.warning("Config not found at %s; using defaults", cfg_path)
        return TilesConfig()
    with cfg_path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return TilesConfig.from_dict(raw, repo_root=REPO_ROOT)


def make_tile_window(
    center_x: float,
    center_y: float,
    *,
    size_px: int = 64,
    resolution_m: float = 30.0,
    crs: str = "EPSG:32646",
    point_index: int | None = None,
    row: int | None = None,
    col: int | None = None,
) -> TileWindow:
    return TileWindow(
        center_x=float(center_x),
        center_y=float(center_y),
        size_px=int(size_px),
        resolution_m=float(resolution_m),
        crs=str(crs),
        point_index=point_index,
        row=row,
        col=col,
    )


def read_window_array(
    raster_path: Path | str,
    window: TileWindow,
    *,
    resampling: Resampling = Resampling.bilinear,
    band: int = 1,
) -> np.ndarray:
    """Resample one band into the tile window grid (float32)."""
    path = Path(raster_path)
    dst = np.full((window.size_px, window.size_px), np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=window.transform,
            dst_crs=window.crs,
            resampling=resampling,
            src_nodata=src_nodata,
            dst_nodata=np.nan,
        )
    # Extra guard for float rasters that use extreme nodata sentinels.
    if src_nodata is not None and np.isfinite(src_nodata):
        dst = np.where(np.isclose(dst, float(src_nodata), atol=0.0), np.nan, dst)
    dst = np.where(np.abs(dst) > 1e30, np.nan, dst)
    return dst


def _slope_rgba(slope_deg: np.ndarray, vmax: float, alpha: float) -> np.ndarray:
    """Map slope degrees → RGBA uint8 using a YlOrRd-like ramp."""
    from matplotlib import colormaps

    cmap = colormaps["YlOrRd"]
    valid = np.isfinite(slope_deg)
    norm = np.zeros(slope_deg.shape, dtype=np.float64)
    norm[valid] = np.clip(slope_deg[valid] / max(vmax, 1e-6), 0.0, 1.0)
    rgba = (cmap(norm) * 255.0).astype(np.uint8)
    rgba[~valid, 3] = 0
    rgba[valid, 3] = int(np.clip(alpha, 0.0, 1.0) * 255)
    return rgba


def _lsi_rgba(zones: np.ndarray, alpha: float) -> np.ndarray:
    h, w = zones.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    a = int(np.clip(alpha, 0.0, 1.0) * 255)
    for code, rgb in LSI_ZONE_COLOURS.items():
        mask = np.isfinite(zones) & (np.rint(zones) == code)
        rgba[mask, 0] = rgb[0]
        rgba[mask, 1] = rgb[1]
        rgba[mask, 2] = rgb[2]
        rgba[mask, 3] = a
    return rgba


def _blend_over(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Porter-Duff 'over' blend; base and result are HxWx3 uint8."""
    base = base_rgb.astype(np.float32)
    over = overlay_rgba.astype(np.float32)
    a = over[..., 3:4] / 255.0
    out = base * (1.0 - a) + over[..., :3] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def render_composite(
    hillshade: np.ndarray,
    slope_deg: np.ndarray,
    *,
    lsi_zones: np.ndarray | None = None,
    include_lsi_overlay: bool = False,
    slope_alpha: float = 0.55,
    lsi_alpha: float = 0.35,
    slope_vmax_deg: float = 60.0,
) -> Image.Image:
    """Hillshade greyscale + slope colour ramp (+ optional LSI overlay)."""
    hs = np.asarray(hillshade, dtype=np.float32)
    valid_hs = np.isfinite(hs)
    if valid_hs.any():
        lo, hi = float(np.nanmin(hs)), float(np.nanmax(hs))
        if hi > lo:
            hs_n = (hs - lo) / (hi - lo)
        else:
            hs_n = np.zeros_like(hs)
    else:
        hs_n = np.zeros_like(hs)
    hs_n = np.where(valid_hs, hs_n, 0.0)
    base = np.stack([(hs_n * 255).astype(np.uint8)] * 3, axis=-1)

    base = _blend_over(base, _slope_rgba(np.asarray(slope_deg, dtype=np.float32), slope_vmax_deg, slope_alpha))

    if include_lsi_overlay and lsi_zones is not None:
        base = _blend_over(base, _lsi_rgba(np.asarray(lsi_zones, dtype=np.float32), lsi_alpha))

    return Image.fromarray(base, mode="RGB")


def render_satellite_placeholder(
    size_px: int,
    message: str = "SATELLITE UNAVAILABLE",
) -> Image.Image:
    """Grey placeholder when STAC/network fails."""
    img = Image.new("RGB", (size_px, size_px), color=(96, 96, 96))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None
    text = message[:48]
    # Rough centre; default bitmap font has no getbbox on very old Pillow.
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = 8 * len(text), 11
    draw.text(((size_px - tw) / 2, (size_px - th) / 2), text, fill=(220, 220, 220), font=font)
    return img


def _month_in_dry_season(month: int, months: Sequence[int]) -> bool:
    return int(month) in {int(m) for m in months}


def fetch_sentinel2_rgb(
    window: TileWindow,
    *,
    stac_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1",
    collection: str = "sentinel-2-l2a",
    months: Sequence[int] = (11, 12, 1, 2),
    year_start: int = 2019,
    year_end: int = 2024,
    max_cloud: float = 30.0,
    max_items: int = 12,
    assets: Sequence[str] = ("B04", "B03", "B02"),
) -> tuple[np.ndarray | None, str]:
    """Dry-season median Sentinel-2 L2A true-colour for ``window``.

    Returns ``(HxWx3 uint8 array or None, note)``. Never raises for network
    errors — callers should fall back to a placeholder.

    Uses one STAC search over ``[year_start, year_end]`` then filters to the
    configured dry-season months and cloud cover client-side (PC STAC does not
    advertise the QUERY conformance class for ``eo:cloud_cover``).
    """
    try:
        import planetary_computer as pc
        from pystac_client import Client
    except ImportError as exc:
        return None, f"missing optional deps: {exc}"

    left, bottom, right, top = window.bounds
    try:
        west, south, east, north = transform_bounds(
            window.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21
        )
    except Exception as exc:  # pragma: no cover
        return None, f"bbox transform failed: {exc}"

    bbox_wgs84 = [west, south, east, north]
    # Pad search window so Nov of year_start-1 / Feb of year_end+1 are included.
    dt_start = f"{year_start - 1}-11-01"
    dt_end = f"{year_end + 1}-02-28"
    # Fetch a modest surplus; we keep the clearest dry-season scenes only.
    search_limit = max(max_items * 6, 24)

    try:
        catalog = Client.open(stac_url, modifier=pc.sign_inplace)
        search = catalog.search(
            collections=[collection],
            bbox=bbox_wgs84,
            datetime=f"{dt_start}/{dt_end}",
            max_items=search_limit,
        )
        items = list(search.items())
    except Exception as exc:
        return None, f"STAC search failed: {exc}"

    # Rank by cloud cover; keep dry-season months only.
    ranked: list[tuple[float, Any]] = []
    for item in items:
        item_dt = item.datetime
        if item_dt is None:
            raw = item.properties.get("datetime")
            if raw is None:
                continue
            try:
                month = int(str(raw)[5:7])
            except Exception:
                continue
        else:
            month = item_dt.month
        if not _month_in_dry_season(month, months):
            continue
        cloud = float(item.properties.get("eo:cloud_cover", 100.0))
        if cloud >= max_cloud:
            continue
        ranked.append((cloud, item))
    ranked.sort(key=lambda t: t[0])
    ranked = ranked[:max_items]

    def _read_visual(item: Any) -> np.ndarray:
        href = pc.sign(item.assets["visual"].href)
        rgb = np.full((window.size_px, window.size_px, 3), np.nan, dtype=np.float32)
        with rasterio.open(href) as src:
            for b in range(1, min(4, src.count + 1)):
                band_dst = np.full((window.size_px, window.size_px), np.nan, dtype=np.float32)
                reproject(
                    source=rasterio.band(src, b),
                    destination=band_dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=window.transform,
                    dst_crs=window.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
                rgb[..., b - 1] = band_dst
        return rgb

    def _read_spectral(item: Any) -> np.ndarray:
        band_arrs: list[np.ndarray] = []
        for asset_key in assets:
            if asset_key not in item.assets:
                raise KeyError(f"asset {asset_key} missing")
            href = pc.sign(item.assets[asset_key].href)
            arr = np.full((window.size_px, window.size_px), np.nan, dtype=np.float32)
            with rasterio.open(href) as src:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=arr,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=window.transform,
                    dst_crs=window.crs,
                    resampling=Resampling.bilinear,
                    src_nodata=src.nodata,
                    dst_nodata=np.nan,
                )
            band_arrs.append(arr)
        return np.stack(band_arrs, axis=-1)

    def _collect(reader, label: str) -> tuple[list[np.ndarray], str]:
        out: list[np.ndarray] = []
        for _cloud, item in ranked:
            try:
                if label == "visual" and "visual" not in item.assets:
                    continue
                rgb = reader(item)
                if float(np.isfinite(rgb).mean()) < 0.2:
                    continue
                out.append(rgb)
            except Exception as exc:
                logger.debug("Skip item %s (%s): %s", getattr(item, "id", "?"), label, exc)
        return out, label

    stacks, src_kind = _collect(_read_visual, "visual")
    if not stacks:
        stacks, src_kind = _collect(_read_spectral, "B04/B03/B02")

    if not stacks:
        return None, "no usable Sentinel-2 scenes in dry-season window"

    cube = np.stack(stacks, axis=0)  # N,H,W,3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(cube, axis=0)

    finite = med[np.isfinite(med)]
    if finite.size == 0:
        return None, "median composite all-nodata"

    # ``visual`` is already 0–255; spectral bands are often 0–10000.
    vmax = float(np.nanpercentile(finite, 99))
    if src_kind == "visual" or vmax <= 255.5:
        stretched = np.clip(med, 0.0, 255.0) / 255.0
    else:
        scale = 10000.0 if vmax > 1.5 else 1.0
        stretched = np.clip(med / scale, 0.0, 1.0)
        p2 = float(np.nanpercentile(stretched, 2))
        p98 = float(np.nanpercentile(stretched, 98))
        if p98 <= p2:
            p2, p98 = 0.0, 1.0
        stretched = np.clip((stretched - p2) / (p98 - p2), 0.0, 1.0)

    stretched = np.nan_to_num(stretched, nan=0.0)
    rgb8 = (stretched * 255.0).astype(np.uint8)
    note = (
        f"dry-season median of {len(stacks)} scene(s) via {src_kind}; "
        f"years {year_start}-{year_end}"
    )
    return rgb8, note


def write_colour_legend(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "colour_legend.md"
    path.write_text(COLOUR_LEGEND_MD, encoding="utf-8")
    return path


def point_tile_stem(index: int, x: float, y: float) -> str:
    return f"point_{index:03d}_x{int(round(x))}_y{int(round(y))}"


def build_point_tiles(
    cfg: TilesConfig | None = None,
    *,
    skip_satellite: bool = False,
    limit: int | None = None,
    include_lsi_overlay: bool | None = None,
) -> list[TilePaths]:
    """Render composite (+ satellite) PNGs for each training point."""
    cfg = cfg or load_tiles_config()
    use_lsi = cfg.include_lsi_overlay if include_lsi_overlay is None else include_lsi_overlay
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_colour_legend(out_dir)

    df = pd.read_csv(cfg.training_csv)
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError(f"{cfg.training_csv} must have x,y columns")

    sat_cfg = cfg.satellite or {}
    sat_enabled = bool(sat_cfg.get("enabled", True)) and not skip_satellite

    results: list[TilePaths] = []
    n = len(df) if limit is None else min(limit, len(df))

    for i in range(n):
        row = df.iloc[i]
        x, y = float(row["x"]), float(row["y"])
        target = int(row["target"]) if "target" in df.columns and pd.notna(row["target"]) else None
        window = make_tile_window(
            x,
            y,
            size_px=cfg.window_size_px,
            resolution_m=cfg.resolution_m,
            crs=cfg.crs,
            point_index=i,
        )
        stem = point_tile_stem(i, x, y)
        # Ablation-friendly suffix when LSI overlay is on.
        comp_name = f"{stem}_composite_lsi.png" if use_lsi else f"{stem}_composite.png"
        sat_name = f"{stem}_satellite.png"
        comp_path = out_dir / comp_name
        sat_path = out_dir / sat_name

        hs = read_window_array(cfg.hillshade_path, window, resampling=Resampling.bilinear)
        slope = read_window_array(cfg.slope_path, window, resampling=Resampling.bilinear)
        lsi = None
        if use_lsi:
            lsi = read_window_array(
                cfg.lsi_zones_path, window, resampling=Resampling.nearest
            )

        img = render_composite(
            hs,
            slope,
            lsi_zones=lsi,
            include_lsi_overlay=use_lsi,
            slope_alpha=cfg.slope_alpha,
            lsi_alpha=cfg.lsi_alpha,
            slope_vmax_deg=cfg.slope_vmax_deg,
        )
        img.save(comp_path)

        sat_ok = False
        sat_note = "skipped"
        if sat_enabled:
            rgb, sat_note = fetch_sentinel2_rgb(
                window,
                stac_url=str(sat_cfg.get("stac_url", "https://planetarycomputer.microsoft.com/api/stac/v1")),
                collection=str(sat_cfg.get("collection", "sentinel-2-l2a")),
                months=tuple(sat_cfg.get("months", [11, 12, 1, 2])),
                year_start=int(sat_cfg.get("year_start", 2019)),
                year_end=int(sat_cfg.get("year_end", 2024)),
                max_cloud=float(sat_cfg.get("max_cloud", 30)),
                max_items=int(sat_cfg.get("max_items", 12)),
                assets=tuple(sat_cfg.get("assets", ["B04", "B03", "B02"])),
            )
            if rgb is not None:
                Image.fromarray(rgb, mode="RGB").save(sat_path)
                sat_ok = True
            else:
                logger.warning("Satellite fallback for point %s: %s", i, sat_note)
                render_satellite_placeholder(cfg.window_size_px, "SATELLITE UNAVAILABLE").save(
                    sat_path
                )
        else:
            sat_note = "satellite disabled"
            render_satellite_placeholder(cfg.window_size_px, "SATELLITE SKIPPED").save(sat_path)

        results.append(
            TilePaths(
                composite=comp_path,
                satellite=sat_path,
                point_index=i,
                x=x,
                y=y,
                target=target,
                satellite_ok=sat_ok,
                satellite_note=sat_note,
                include_lsi_overlay=use_lsi,
            )
        )
        logger.info(
            "Tile %s/%s → %s (sat_ok=%s)",
            i + 1,
            n,
            comp_path.name,
            sat_ok,
        )

    manifest = {
        "n_points": len(results),
        "window_size_px": cfg.window_size_px,
        "resolution_m": cfg.resolution_m,
        "width_m": cfg.window_size_px * cfg.resolution_m,
        "crs": cfg.crs,
        "include_lsi_overlay": use_lsi,
        "colour_legend": str(out_dir / "colour_legend.md"),
        "tiles": [
            {
                "point_index": r.point_index,
                "x": r.x,
                "y": r.y,
                "target": r.target,
                "composite": str(r.composite.as_posix()),
                "satellite": str(r.satellite.as_posix()),
                "satellite_ok": r.satellite_ok,
                "satellite_note": r.satellite_note,
                "include_lsi_overlay": r.include_lsi_overlay,
            }
            for r in results
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return results


def iter_study_area_grid(
    bbox: tuple[float, float, float, float],
    *,
    tile_size_px: int = 64,
    resolution_m: float = 30.0,
    overlap_px: int = 0,
    crs: str = "EPSG:32646",
) -> Iterator[TileWindow]:
    """Yield coarse tile windows covering ``bbox`` (left, bottom, right, top).

    API for a later map-style VLM run — do **not** invoke full-grid inference
    until the 44-point Stage A results are reviewed.
    """
    left, bottom, right, top = map(float, bbox)
    if right <= left or top <= bottom:
        raise ValueError(f"invalid bbox: {bbox}")
    step_px = max(1, tile_size_px - int(overlap_px))
    step_m = step_px * resolution_m
    half = 0.5 * tile_size_px * resolution_m

    # Centres start so the first tile's left/bottom edges sit on the bbox edge.
    x0 = left + half
    y0 = bottom + half
    cols = int(np.ceil((right - left) / step_m))
    rows = int(np.ceil((top - bottom) / step_m))

    for r in range(rows):
        cy = y0 + r * step_m
        for c in range(cols):
            cx = x0 + c * step_m
            yield make_tile_window(
                cx,
                cy,
                size_px=tile_size_px,
                resolution_m=resolution_m,
                crs=crs,
                row=r,
                col=c,
            )


def study_area_bbox_from_points(
    training_csv: Path | str,
    *,
    buffer_m: float = 2000.0,
) -> tuple[float, float, float, float]:
    """Axis-aligned bbox of training points plus buffer (left, bottom, right, top)."""
    df = pd.read_csv(training_csv)
    return (
        float(df["x"].min()) - buffer_m,
        float(df["y"].min()) - buffer_m,
        float(df["x"].max()) + buffer_m,
        float(df["y"].max()) + buffer_m,
    )


def build_study_area_grid(
    cfg: TilesConfig | None = None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> list[TileWindow]:
    """Materialise the coarse study-area grid (no PNG rendering).

    Returns the list of :class:`TileWindow` only — rendering / VLM over the
    full grid is deferred until after Stage A point review.
    """
    cfg = cfg or load_tiles_config()
    g = cfg.grid or {}
    tile_size = int(g.get("tile_size_px", cfg.window_size_px))
    res = float(g.get("resolution_m", cfg.resolution_m))
    overlap = int(g.get("overlap_px", 0))
    if bbox is None:
        raw_bbox = g.get("bbox")
        if raw_bbox is not None:
            bbox = tuple(float(v) for v in raw_bbox)  # type: ignore[assignment]
        else:
            buffer_m = float(g.get("buffer_m", 2000.0))
            bbox = study_area_bbox_from_points(cfg.training_csv, buffer_m=buffer_m)
    assert bbox is not None
    return list(
        iter_study_area_grid(
            bbox,
            tile_size_px=tile_size,
            resolution_m=res,
            overlap_px=overlap,
            crs=cfg.crs,
        )
    )


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # Keep third-party (rasterio/GDAL) quiet even with -v.
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    if verbose:
        logger.setLevel(logging.DEBUG)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage A VLM tile renderer")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to vlm_stage_a.yaml",
    )
    parser.add_argument(
        "--skip-satellite",
        action="store_true",
        help="Write satellite placeholders only (no STAC calls)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only first N points")
    parser.add_argument(
        "--with-lsi",
        action="store_true",
        help="Force LSI-zone overlay on (ablation)",
    )
    parser.add_argument(
        "--grid-count-only",
        action="store_true",
        help="Print study-area grid tile count and exit (no rendering)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(args.verbose)

    cfg = load_tiles_config(args.config)
    if args.grid_count_only:
        windows = build_study_area_grid(cfg)
        print(json.dumps({"n_grid_tiles": len(windows), "crs": cfg.crs}, indent=2))
        return 0

    results = build_point_tiles(
        cfg,
        skip_satellite=args.skip_satellite,
        limit=args.limit,
        include_lsi_overlay=True if args.with_lsi else None,
    )
    n_sat = sum(1 for r in results if r.satellite_ok)
    print(
        json.dumps(
            {
                "n_points": len(results),
                "n_composite": len(results),
                "n_satellite_ok": n_sat,
                "n_satellite_fallback": len(results) - n_sat,
                "output_dir": str(cfg.output_dir.as_posix()),
                "include_lsi_overlay": results[0].include_lsi_overlay if results else False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
