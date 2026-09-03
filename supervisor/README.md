# Aizawl Landslide Susceptibility — supervisor pack

**Start here.** This folder is the review set for the IIT Patna B.Tech project on landslide susceptibility in **Aizawl, Mizoram**.

| | |
|---|---|
| **Study area** | Aizawl district, Mizoram (WGS84 ~92.62–93.21°E, 23.31–24.41°N) |
| **CRS for all GeoTIFFs** | EPSG:32646 (WGS 84 / UTM zone 46N) |
| **Grid** | 30 m (Sonker stack) / ~28 m (Phase 1 DEM derivatives) |
| **Inventory** | 22 landslide points (`maps/vectors/Aizawl_landslide_points.gpkg`) |
| **Method** | Frequency Ratio (FR) → Landslide Susceptibility Index (LSI) → 5 hazard zones → ROC/AUC |

---

## Open in 30 seconds

| If you want to… | Do this |
|---|---|
| **View maps in the browser** (no GIS) | Open [`maps/preview/`](maps/preview/) and click any `.jpg` |
| **Read the analysis** | Open [`analysis/FREQUENCY_RATIO_REPORT.md`](analysis/FREQUENCY_RATIO_REPORT.md) |
| **Open everything in QGIS** | Open [`Aizawl_LSI_supervisor.qgz`](Aizawl_LSI_supervisor.qgz) |
| **Use rasters in ArcGIS / QGIS / Python** | Use GeoTIFFs in [`maps/geotiff/`](maps/geotiff/) |
| **See the file list** | [`FILE_INDEX.csv`](FILE_INDEX.csv) |

GeoTIFF is the analysis format (georeferenced, lossless). JPEG is only for quick viewing on GitHub.

---

## What is finished vs still in progress

### Phase 1 — finished (this is the current LSI result)

Three terrain factors (**slope, aspect, elevation**) were classed, weighted by Frequency Ratio, summed to LSI, and split into **five hazard zones**.

| Result | Value |
|---|---|
| Validation AUC (ROC) | **0.77** |
| Strongest class | Elevation > 1100 m (**FR = 3.24**; 17 of 22 landslides) |
| Hazard map | [`maps/geotiff/P1_01_LSI_hazard_zones.tif`](maps/geotiff/P1_01_LSI_hazard_zones.tif) |
| FR table | [`analysis/fr_class_table_phase1.csv`](analysis/fr_class_table_phase1.csv) |

Full write-up: [`analysis/FREQUENCY_RATIO_REPORT.md`](analysis/FREQUENCY_RATIO_REPORT.md)

### Phase 2 — 17 Sonker et al. (2022) factors prepared; 17-factor LSI **not** computed yet

Integer class rasters and substitutes (CHIRPS rainfall, India soil, GLiM geology, OSM roads, etc.) are in `maps/geotiff/S17_*.tif`. FR tables, FRn, a new 17-factor LSI, and a 70/30 ROC for that model have **not** been run. Treat `S17_*` as **input maps**, not as a finished susceptibility product.

Factor-by-factor notes: [`analysis/SONKER_17_FACTOR_STATUS.md`](analysis/SONKER_17_FACTOR_STATUS.md)

---

## Folder map

```
supervisor/
│
├── README.md                          ← you are here
├── FILE_INDEX.csv                     ← every map with description + CRS
├── Aizawl_LSI_supervisor.qgz          ← QGIS project (relative paths)
│
├── analysis/
│   ├── FREQUENCY_RATIO_REPORT.md      ← method, tables, AUC, limitations
│   ├── fr_class_table_phase1.csv      ← Phase 1 FR numbers
│   └── SONKER_17_FACTOR_STATUS.md     ← 17-factor catalogue and caveats
│
└── maps/
    ├── preview/                       ← JPEG (click on GitHub)
    ├── geotiff/                       ← GeoTIFF for GIS
    │     P1_*.tif                     Phase 1 results
    │     S17_*.tif                    Sonker 17-factor inputs
    └── vectors/                       ← points, boundary, roads, quakes
```

---

## Map catalogue

### A. Phase 1 results — start with these

| # | Preview (browser) | GeoTIFF (GIS) | What it is |
|---|---|---|---|
| 1 | [P1_01_LSI_hazard_zones.jpg](maps/preview/P1_01_LSI_hazard_zones.jpg) | [P1_01_LSI_hazard_zones.tif](maps/geotiff/P1_01_LSI_hazard_zones.tif) | **Final 5-zone hazard map** (Very Low → Very High) |
| 2 | [P1_02_LSI_continuous.jpg](maps/preview/P1_02_LSI_continuous.jpg) | [P1_02_LSI_continuous.tif](maps/geotiff/P1_02_LSI_continuous.tif) | Continuous LSI = Σ FR (slope + aspect + elevation) |
| 3 | [P1_03_Slope_FR.jpg](maps/preview/P1_03_Slope_FR.jpg) | [P1_03_Slope_FR.tif](maps/geotiff/P1_03_Slope_FR.tif) | Slope weighted by Frequency Ratio |
| 4 | [P1_04_Aspect_FR.jpg](maps/preview/P1_04_Aspect_FR.jpg) | [P1_04_Aspect_FR.tif](maps/geotiff/P1_04_Aspect_FR.tif) | Aspect weighted by Frequency Ratio |
| 5 | [P1_05_Elevation_FR.jpg](maps/preview/P1_05_Elevation_FR.jpg) | [P1_05_Elevation_FR.tif](maps/geotiff/P1_05_Elevation_FR.tif) | Elevation weighted by Frequency Ratio |
| 6 | [P1_09_DEM_Aizawl.jpg](maps/preview/P1_09_DEM_Aizawl.jpg) | [P1_09_DEM_Aizawl.tif](maps/geotiff/P1_09_DEM_Aizawl.tif) | Clipped DEM (Copernicus DSM) |
| 7 | [P1_10_Slope_degrees.jpg](maps/preview/P1_10_Slope_degrees.jpg) | [P1_10_Slope_degrees.tif](maps/geotiff/P1_10_Slope_degrees.tif) | Slope in degrees |
| 8 | [P1_11_Hillshade.jpg](maps/preview/P1_11_Hillshade.jpg) | [P1_11_Hillshade.tif](maps/geotiff/P1_11_Hillshade.tif) | Hillshade for context |

Class rasters used to compute FR: `P1_06` slope, `P1_07` aspect, `P1_08` elevation.

### B. Sonker 17-factor inputs (not yet an LSI map)

| # | Factor | Preview | GeoTIFF |
|---|---|---|---|
| 1 | Rainfall (CHIRPS mean annual, mm) | [jpg](maps/preview/S17_01_Rainfall_CHIRPS_mm.jpg) | [tif](maps/geotiff/S17_01_Rainfall_CHIRPS_mm.tif) |
| 2 | Earthquake magnitude classes | [jpg](maps/preview/S17_02_Earthquake_classes.jpg) | [tif](maps/geotiff/S17_02_Earthquake_classes.tif) |
| 3 | Slope classes (paper breaks) | [jpg](maps/preview/S17_03_Slope_classes.jpg) | [tif](maps/geotiff/S17_03_Slope_classes.tif) |
| 4 | Altitude classes | [jpg](maps/preview/S17_04_Altitude_classes.jpg) | [tif](maps/geotiff/S17_04_Altitude_classes.tif) |
| 5 | Distance to drainages | [jpg](maps/preview/S17_05_Distance_drainages_classes.jpg) | [tif](maps/geotiff/S17_05_Distance_drainages_classes.tif) |
| 6 | TRI | [jpg](maps/preview/S17_06_TRI_classes.jpg) | [tif](maps/geotiff/S17_06_TRI_classes.tif) |
| 7 | Geomorphology | [jpg](maps/preview/S17_07_Geomorphology.jpg) | [tif](maps/geotiff/S17_07_Geomorphology_classes.tif) |
| 8 | Geology (GLiM stand-in) | [jpg](maps/preview/S17_08_Geology_GLiM.jpg) | [tif](maps/geotiff/S17_08_Geology_classes.tif) |
| 9 | Soil (NBSS India) | [jpg](maps/preview/S17_09_Soil.jpg) | [tif](maps/geotiff/S17_09_Soil_India_classes.tif) |
| 10 | Gravity (simple Bouguer) | [jpg](maps/preview/S17_10_Gravity_Bouguer.jpg) | [tif](maps/geotiff/S17_10_Gravity_Bouguer_mGal.tif) |
| 11 | Distance to faults | [jpg](maps/preview/S17_11_Distance_faults_classes.jpg) | [tif](maps/geotiff/S17_11_Distance_faults_classes.tif) |
| 12 | STI | [jpg](maps/preview/S17_12_STI_classes.jpg) | [tif](maps/geotiff/S17_12_STI_classes.tif) |
| 13 | TWI | [jpg](maps/preview/S17_13_TWI_classes.jpg) | [tif](maps/geotiff/S17_13_TWI_classes.tif) |
| 14 | SPI | [jpg](maps/preview/S17_14_SPI_classes.jpg) | [tif](maps/geotiff/S17_14_SPI_classes.tif) |
| 15 | Distance to roads | [jpg](maps/preview/S17_15_Distance_roads_classes.jpg) | [tif](maps/geotiff/S17_15_Distance_roads_classes.tif) |
| 16 | LULC | [jpg](maps/preview/S17_16_LULC_classes.jpg) | [tif](maps/geotiff/S17_16_LULC_classes.tif) |
| 17 | NDVI | [jpg](maps/preview/S17_17_NDVI_classes.jpg) | [tif](maps/geotiff/S17_17_NDVI_classes.tif) |

**Weak discriminators (almost one class across Aizawl):** TRI, geology (GLiM = all siliciclastic sedimentary), distance to roads (OSM is dense), NDVI (very high canopy). See the status note before using these in FR.

### C. Vectors

| File | Contents |
|---|---|
| [`maps/vectors/Aizawl_landslide_points.gpkg`](maps/vectors/Aizawl_landslide_points.gpkg) | 22 landslide locations (EPSG:32646) |
| [`maps/vectors/AIZWAL.shp`](maps/vectors/AIZWAL.shp) | Study-area boundary (source CRS EPSG:4326 — reproject on the fly in QGIS) |
| [`maps/vectors/Aizawl_OSM_roads.gpkg`](maps/vectors/Aizawl_OSM_roads.gpkg) | Vehicle roads used for distance-to-roads |
| [`maps/vectors/Aizawl_USGS_earthquakes.gpkg`](maps/vectors/Aizawl_USGS_earthquakes.gpkg) | Regional USGS events used for the magnitude surface |

---

## How to open the GeoTIFFs in QGIS

1. Install [QGIS](https://qgis.org/) 3.x.
2. Either open `Aizawl_LSI_supervisor.qgz`, **or** Layer → Add Layer → Add Raster Layer and select files in `maps/geotiff/`.
3. Project CRS should be **EPSG:32646**.
4. For the 5-zone map, use a discrete palette (Unique Values), not a continuous stretch.

Python (optional):

```python
import rasterio
with rasterio.open("maps/geotiff/P1_01_LSI_hazard_zones.tif") as src:
    print(src.crs, src.res, src.read(1).shape)
```

---

## Recommended reading order

1. This README (status + catalogue)
2. [`analysis/FREQUENCY_RATIO_REPORT.md`](analysis/FREQUENCY_RATIO_REPORT.md)
3. Preview JPEGs of **P1_01** (hazard zones) and **P1_05** (elevation FR)
4. [`analysis/SONKER_17_FACTOR_STATUS.md`](analysis/SONKER_17_FACTOR_STATUS.md) if reviewing the 17-factor stack
5. GeoTIFFs in QGIS only if you need to inspect values or overlay the 22 points

---

## Parent repository

The rest of the code (ML pipeline, GEE scripts, unit tests) lives one level up:

- Repository: https://github.com/Anupamgt/lsi-ml-pipeline
- Pipeline README (Phase 1 narrative + ML baseline): [`../README.md`](../README.md)
- GEE rainfall / LULC scripts: [`../docs/gee/`](../docs/gee/)

**Author:** Anupamgt · IIT Patna B.Tech project · maps in GeoTIFF (LZW, tiled) unless noted.
