# Build the supervisor share pack: compressed GeoTIFFs + JPEG previews.
# Run inside QGIS (osgeo.gdal) or any Python with GDAL.

from __future__ import annotations

from pathlib import Path

from osgeo import gdal, gdalconst

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_PAM_ENABLED", "NO")

ROOT = Path(r"C:\Users\sharm\lsi-ml-pipeline")
SRC_FINAL = ROOT / "final_maps"
SRC_PROC = Path(r"C:\Users\sharm\LSI_btp_sikkim_replication\01_processed")
OUT = ROOT / "supervisor"
GTIFF = OUT / "maps" / "geotiff"
PREV = OUT / "maps" / "preview"
VEC = OUT / "maps" / "vectors"

COPTS = ["COMPRESS=LZW", "TILED=YES", "PREDICTOR=2", "BIGTIFF=IF_SAFER"]


def translate_tif(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    gdal.Translate(
        str(dst),
        str(src),
        format="GTiff",
        creationOptions=COPTS,
    )
    print(f"TIF  {dst.name:48s} {dst.stat().st_size / 1e6:6.2f} MB")


def preview_jpeg(src: Path, dst: Path, width: int = 1400) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    ds = gdal.Open(str(src), gdalconst.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    stats = band.GetStatistics(True, True)
    vmin, vmax = stats[0], stats[1]
    nd = band.GetNoDataValue()
    if nd is not None and vmin == nd:
        vmin = stats[0]
    if vmax <= vmin:
        vmin, vmax = 0, 1
    gdal.Translate(
        str(dst),
        str(src),
        format="JPEG",
        width=width,
        outputType=gdalconst.GDT_Byte,
        scaleParams=[[vmin, vmax, 1, 255]],
        noData=None,
        creationOptions=["QUALITY=88"],
    )
    ds = None
    print(f"JPG  {dst.name:48s} {dst.stat().st_size / 1e6:6.2f} MB")


COPIES = [
    # Phase 1 — completed 3-factor FR / LSI
    (SRC_FINAL / "LSI_Final_Zones.tif", GTIFF / "P1_01_LSI_hazard_zones.tif", True),
    (SRC_FINAL / "LSI_Master.tif", GTIFF / "P1_02_LSI_continuous.tif", True),
    (SRC_FINAL / "Slope_fr_Final.tif", GTIFF / "P1_03_Slope_FR.tif", True),
    (SRC_FINAL / "Aspect_FR_final.tif", GTIFF / "P1_04_Aspect_FR.tif", True),
    (SRC_FINAL / "Elevation_FR_final.tif", GTIFF / "P1_05_Elevation_FR.tif", True),
    (SRC_FINAL / "SLOPE_RECLASS.tif", GTIFF / "P1_06_Slope_classes.tif", True),
    (SRC_FINAL / "ASPECT_RECLASS.tif", GTIFF / "P1_07_Aspect_classes.tif", True),
    (SRC_FINAL / "Elevation_reclass_final.tif", GTIFF / "P1_08_Elevation_classes.tif", True),
    (SRC_FINAL / "DEM_UTF_FINAL.tif", GTIFF / "P1_09_DEM_Aizawl.tif", True),
    (SRC_FINAL / "Slope_FINAL.tif", GTIFF / "P1_10_Slope_degrees.tif", True),
    (SRC_FINAL / "HILLSHADE_2.tif", GTIFF / "P1_11_Hillshade.tif", True),
    # Sonker 17-factor rasters (inputs; 17-factor LSI not yet computed)
    (SRC_PROC / "chirps_mean_annual_30m.tif", GTIFF / "S17_01_Rainfall_CHIRPS_mm.tif", True),
    (SRC_PROC / "earthquake_reclass_sonker.tif", GTIFF / "S17_02_Earthquake_classes.tif", True),
    (SRC_PROC / "slope_reclass_sonker.tif", GTIFF / "S17_03_Slope_classes.tif", True),
    (SRC_PROC / "altitude_reclass_sonker.tif", GTIFF / "S17_04_Altitude_classes.tif", True),
    (SRC_PROC / "drainage_dist_reclass_sonker.tif", GTIFF / "S17_05_Distance_drainages_classes.tif", True),
    (SRC_PROC / "tri_reclass_sonker.tif", GTIFF / "S17_06_TRI_classes.tif", True),
    (SRC_PROC / "geomorph_reclass_sonker.tif", GTIFF / "S17_07_Geomorphology_classes.tif", True),
    (SRC_PROC / "geology_reclass_sonker_fr.tif", GTIFF / "S17_08_Geology_classes.tif", True),
    (SRC_PROC / "soil_india_reclass_sonker.tif", GTIFF / "S17_09_Soil_India_classes.tif", True),
    (SRC_PROC / "gravity_bouguer_simple_30m_clip.tif", GTIFF / "S17_10_Gravity_Bouguer_mGal.tif", True),
    (SRC_PROC / "faults_dist_reclass_sonker.tif", GTIFF / "S17_11_Distance_faults_classes.tif", True),
    (SRC_PROC / "sti_reclass_sonker.tif", GTIFF / "S17_12_STI_classes.tif", True),
    (SRC_PROC / "twi_reclass_sonker.tif", GTIFF / "S17_13_TWI_classes.tif", True),
    (SRC_PROC / "spi_reclass_sonker.tif", GTIFF / "S17_14_SPI_classes.tif", True),
    (SRC_PROC / "roads_dist_reclass_sonker.tif", GTIFF / "S17_15_Distance_roads_classes.tif", True),
    (SRC_PROC / "lulc_reclass_sonker.tif", GTIFF / "S17_16_LULC_classes.tif", True),
    (SRC_PROC / "ndvi_reclass_sonker.tif", GTIFF / "S17_17_NDVI_classes.tif", True),
]

JPEG_EXTRAS = [
    (SRC_PROC / "geomorph_sonker_aizawl.jpg", PREV / "S17_07_Geomorphology.jpg"),
    (SRC_PROC / "geology_glim_sonker_aizawl.jpg", PREV / "S17_08_Geology_GLiM.jpg"),
    (SRC_PROC / "soil_wrb_sonker_aizawl.jpg", PREV / "S17_09_Soil.jpg"),
    (SRC_PROC / "gravity_bouguer_simple_aizawl.jpg", PREV / "S17_10_Gravity_Bouguer.jpg"),
]

VECTORS = [
    (SRC_FINAL / "Aizawl_Points_UTM.gpkg", VEC / "Aizawl_landslide_points.gpkg"),
    (SRC_FINAL / "AIZWAL.shp", VEC / "AIZWAL.shp"),
    (SRC_FINAL / "AIZWAL.shx", VEC / "AIZWAL.shx"),
    (SRC_FINAL / "AIZWAL.dbf", VEC / "AIZWAL.dbf"),
    (SRC_FINAL / "AIZWAL.prj", VEC / "AIZWAL.prj"),
    (SRC_FINAL / "sonker_roads_aizawl_network.gpkg", VEC / "Aizawl_OSM_roads.gpkg"),
    (SRC_FINAL / "sonker_earthquakes_aizawl.gpkg", VEC / "Aizawl_USGS_earthquakes.gpkg"),
]


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    print(f"COPY {dst.name:48s} {dst.stat().st_size / 1e6:6.2f} MB")


def main() -> None:
    for folder in (GTIFF, PREV, VEC, OUT / "analysis"):
        folder.mkdir(parents=True, exist_ok=True)

    missing = []
    for src, dst, make_jpg in COPIES:
        if not src.exists():
            missing.append(str(src))
            print("MISSING", src)
            continue
        translate_tif(src, dst)
        if make_jpg:
            preview_jpeg(src, PREV / (dst.stem + ".jpg"))

    for src, dst in JPEG_EXTRAS:
        if src.exists():
            copy_file(src, dst)
        else:
            missing.append(str(src))

    for src, dst in VECTORS:
        if src.exists():
            copy_file(src, dst)
        else:
            missing.append(str(src))

    print("DONE missing=", len(missing))
    for m in missing:
        print("  ", m)


if __name__ == "__main__":
    main()
