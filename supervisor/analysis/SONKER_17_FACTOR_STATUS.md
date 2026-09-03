# Sonker 17-factor stack — status for Aizawl

These rasters follow the factor list in Sonker et al. (2022) and sit on a common **30 m** grid (EPSG:32646, extent 461520, 2578020, 521610, 2700000).

**FR / FRn / LSI / 5 zones / ROC have been computed** from these inputs. Read [`FR_17_FACTOR_REPORT.md`](FR_17_FACTOR_REPORT.md) for the numbers. Treat that product as **exploratory**: 22 NASA GLC points, several near-constant factors, and explosive FR on urban/NDVI classes.

Rainfall and gravity were 5-classed by equal-area quantiles for that run (`P2_03` / `P2_04` in `maps/geotiff/`).

Legend for **Status**: prepared = raster exists and is in this pack; weak = raster exists but is a poor discriminator in Aizawl as currently classed.

| # | Factor | Status | GeoTIFF in this pack | How it was built | Caveat |
|---|---|---|---|---|---|
| 1 | Rainfall | Prepared + 5-class for FR | `S17_01_Rainfall_CHIRPS_mm.tif` + `P2_03_Rainfall_quantile5.tif` | CHIRPS mean annual mm, complete years 2010–2018 (GEE), warped to 30 m | Paper used IMD 1988–2018. FR used equal-area quantiles, not Sikkim mm bins |
| 2 | Earthquake | Prepared | `S17_02_Earthquake_classes.tif` | USGS events → IDW magnitude (3.60–5.40) → paper magnitude breaks | Outer paper classes are empty |
| 3 | Slope | Prepared | `S17_03_Slope_classes.tif` | DEM slope; breaks 0–17 / 17–33 / 33–50 / 50–67 / >67° | Paper classes; denser than Phase 1’s 3 bins |
| 4 | Altitude | Prepared | `S17_04_Altitude_classes.tif` | DEM; paper breaks at 1500 / 3000 / 4500 / 6000 m | Aizawl max ~1906 m → only classes 1–2 populate |
| 5 | Distance to drainages | Prepared | `S17_05_Distance_drainages_classes.tif` | D8 streams + proximity; 300 m rings | — |
| 6 | TRI | Weak | `S17_06_TRI_classes.tif` | Riley TRI from DEM | Paper breaks are ~0–1; Riley values go to hundreds → almost all pixels in one class |
| 7 | Geomorphology | Prepared | `S17_07_Geomorphology_classes.tif` | GRASS geomorphons remapped to paper legend | No glacial class in Aizawl |
| 8 | Geology | Weak | `S17_08_Geology_classes.tif` | GLiM lithology | **100% siliciclastic sedimentary** in the AOI; GSI/NGDR needed to discriminate |
| 9 | Soil | Prepared | `S17_09_Soil_India_classes.tif` | NBSS-style India soil map (texture) | Use this, not SoilGrids, for FR. Classes: loamy, clayey, skeletal |
| 10 | Gravity | Prepared + 5-class for FR | `S17_10_Gravity_Bouguer_mGal.tif` + `P2_04_Gravity_quantile5.tif` | GGMPlus gravity minus 0.1119 × elevation (simple Bouguer) | Optional in the paper (digitised Sikkim map). FR used equal-area quantiles |
| 11 | Distance to faults | Prepared | `S17_11_Distance_faults_classes.tif` | GEM Global Active Faults proximity | Few traces; large area in the farthest class |
| 12 | STI | Prepared | `S17_12_STI_classes.tif` | Sediment transport index; paper breaks | — |
| 13 | TWI | Prepared | `S17_13_TWI_classes.tif` | Topographic wetness; paper breaks | Closest match to paper class ranges |
| 14 | SPI | Weak | `S17_14_SPI_classes.tif` | Stream power index | Paper classes look like ln(SPI); linear SPI piles into one class |
| 15 | Distance to roads | Weak | `S17_15_Distance_roads_classes.tif` | OSM vehicle roads, 30 m proximity | Max distance ~284 m → almost all pixels in the <300 m class |
| 16 | LULC | Prepared | `S17_16_LULC_classes.tif` | ESRI 10 m annual 2017–2018 → paper 6 classes | Forest dominates (~94%) |
| 17 | NDVI | Weak | `S17_17_NDVI_classes.tif` | Landsat 8 dry-season median | Mean NDVI ~0.82 → almost all pixels in the highest class |

Styled overview JPEGs (geomorphology, geology, soil, gravity) are in `maps/preview/S17_07`–`S17_10`.

## What would still change a 17-factor FR

- **Geology:** replace GLiM with GSI/NGDR lithology.
- **Roads:** restrict OSM to trunk/primary/secondary/tertiary, then rebuild distance classes.
- **TRI / SPI / NDVI:** re-bin with Aizawl quantiles or Natural Breaks instead of Sikkim paper breaks.
- **Inventory:** more than 22 points (ideally scar polygons) before a 70/30 ROC is meaningful.
