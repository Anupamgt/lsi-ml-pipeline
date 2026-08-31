# LSI_btp — Work Achieved

**Project:** Landslide Susceptibility Index (LSI) for Aizawl, Mizoram  
**QGIS project:** `C:\Users\sharm\LSI_btp.qgz`  
**CRS:** EPSG:32646 (WGS 84 / UTM zone 46N)  
**Last saved:** 19 August 2026  
**QGIS:** 3.44.12-Solothurn (connected via MCP)

This note records what has been set up and processed so far. It is a progress log, not a finished susceptibility map.

---

## 1. Goal

Replicate the Frequency Ratio (FR) landslide-susceptibility method from:

> Sonker, I., Tripathi, J.N., Swarnim (2022). Remote sensing and GIS-based landslide susceptibility mapping using frequency ratio method in Sikkim Himalaya. *Quaternary Science Advances* 8, 100067.

**Applied to Aizawl**, not remapped to Sikkim.

Paper recipe:

1. Build **17 factor rasters**
2. Reclassify each factor
3. Compute **FR** and normalized **FRn** from training landslides (70%)
4. **LSI = Σ FRn**, Natural Breaks into 5 zones
5. Validate with held-out 30% using **ROC/AUC** (paper: ~87.9%)

Existing Phase-1 baseline (3 factors only: slope, aspect, elevation FR) lives in the GitHub pipeline and is already loaded in QGIS.

---

## 2. QGIS project status

| Item | Value |
|---|---|
| File | `C:\Users\sharm\LSI_btp.qgz` |
| Layers | 30 |
| CRS | EPSG:32646 |
| Study area | `AIZWAL` polygon (1 feature) |
| Inventory in use | `Aizawl_Points_UTM` (22 landslide points) |

### Layer groups

| Group | Contents | Currently visible |
|---|---|---|
| **Points & Boundaries** | Landslide points, Aizawl boundary, analysis/reprojected inventories, 500-pt grid, NASA global catalog | Aizawl points + boundary |
| **LSI Results** | `LSI_Final_Zones`, `LSI_Master` (plus leftover roads/earthquake source layers) | off |
| **Frequency Ratio** | Slope / Aspect / Elevation FR rasters | off |
| **Reclassified** | Slope, aspect, elevation class rasters | off |
| **Terrain Derivatives** | Slope, aspect, DEM, hillshade | off |
| **DEM Sources** | Four Copernicus DSM tiles (N23/N24 × E092/E093) | off |
| **Sonker 2022 factors** | Roads, distance-to-roads, earthquake points, magnitude surface | roads + distance + quake points |

---

## 3. What has been done

### 3.1 Loaded the existing Aizawl LSI pipeline

Cloned [Anupamgt/lsi-ml-pipeline](https://github.com/Anupamgt/lsi-ml-pipeline) and added all maps and points from `final_maps/` into `LSI_btp`.

**Vectors**

| Layer | Features | Role |
|---|---|---|
| `Aizawl_Points_UTM` | 22 | Standardized landslide inventory (EPSG:32646) |
| `AIZWAL` | 1 | Study-area boundary |
| `Analysis_layer` | 22 | Duplicate inventory |
| `Reprojected_LANDSLIDE_layers_UTM` | 22 | Duplicate inventory |
| `aizawl_grid` | 500 | Sample point grid from `data/aizawl_grid.csv` |
| `Global_Landslide_Catalog` | 11,033 | NASA GLC worldwide points (hidden) |

**Rasters already in the project (Phase-1 FR, 3 factors)**

- LSI: `LSI_Final_Zones`, `LSI_Master`
- FR-weighted: `Slope_fr_Final`, `Aspect_FR_final`, `Elevation_FR_final`
- Reclass: `SLOPE_RECLASS`, `ASPECT_RECLASS`, `Elevation_reclass_final`, `Elevation_Classes`
- Terrain: `Slope_FINAL`, `ASPECT_FINAL`, `ASPECT_DEM1`, `DEM_UTF_FINAL`, `HILLSHADE_2`
- DEM tiles: Copernicus DSM N23E092, N23E093, N24E092, N24E093

`DEM_UTF_FINAL` is the clipped Aizawl DEM (~28 m, UTM 46N, elevation 20–1906 m). ALOS PALSAR 12.5 m (as in the paper) was **skipped** for now; Copernicus is the working DEM.

### 3.2 Set up the Sonker 2022 replication workspace

Folder:

`C:\Users\sharm\LSI_btp_sikkim_replication\`

```
00_drop_raw/
  01_landslide_inventory/
  02_dem_alos_palsar/
  03_rainfall_imd/
  04_earthquakes_usgs/     ← query.csv (in)
  05_geology_bhukosh/
  06_soil/
  07_geomorphology_landsat/
  08_lulc_landsat/
  09_ndvi_landsat/
  10_faults/
  11_roads/                ← OSM roads shapefile (in)
  12_gravity_anomaly/
  13_study_area_boundary/
01_processed/              ← derived GeoPackages and GeoTIFFs
02_fr_tables/              ← empty (FR not run yet)
03_lsi_outputs/            ← empty (new 17-factor LSI not built yet)
```

### 3.3 Processed roads (OSM)

**Source:** `C:\Users\sharm\Downloads\north-eastern-zone-260817-free.shp.zip`  
(Geofabrik North-Eastern Zone OSM extract)

**Steps**

1. Extracted `gis_osm_roads_free_1.*`
2. Clipped to Aizawl bounding box, then to the `AIZWAL` polygon
3. Reprojected to EPSG:32646
4. Dropped footways, paths, pedestrian, steps, and tracks
5. Kept vehicle roads: trunk, primary, secondary, tertiary, unclassified, residential, living_street, service
6. Rasterized at 30 m and computed **GDAL proximity** (distance in metres)
7. Clipped the distance raster to Aizawl

**Outputs**

| File | What it is |
|---|---|
| `01_processed\roads_aizawl_network.gpkg` | 4,584 road segments |
| `01_processed\distance_to_roads_clip.tif` | Distance to roads, 30 m |

**Stats (distance to roads):** min 0 m, max 284 m, mean ~42 m. OSM is dense inside Aizawl, so most of the district is close to a mapped road. If FR later looks weak for this factor, restrict the network to trunk/primary/secondary/tertiary only.

QGIS layers: `Roads_Aizawl`, `Distance_to_roads`.

### 3.4 Processed earthquakes (USGS)

**Source:** `C:\Users\sharm\Downloads\query.csv`

**Steps**

1. Loaded 2,341 USGS events as points
2. Kept events with magnitude, `type = earthquake`, and within a buffer around Aizawl (lon 91.5–94.5, lat 22.0–25.5)
3. Reprojected to EPSG:32646 → **733 points** (mag 1.8–6.7)
4. IDW interpolation of **magnitude** (power = 2, 50 m cells)
5. Clipped to Aizawl

First IDW pass accidentally used **depth** (GeoPackage inserted an `fid` column). That was discarded and re-run on field `mag`.

**Outputs**

| File | What it is |
|---|---|
| `01_processed\earthquakes_aizawl_utm.gpkg` | 733 regional earthquake points |
| `01_processed\earthquake_magnitude_clip.tif` | IDW magnitude surface |

**Stats (magnitude raster):** min 3.60, max 5.40, mean 4.51

QGIS layers: `Earthquakes_Aizawl_region` (visible), `Earthquake_magnitude` (in the group, currently off).

---

## 4. 17 paper factors — status

| # | Factor | Status | Notes |
|---|---|---|---|
| 1 | Rainfall | **Not started** | IMD 0.25° NetCDF; download skipped for now |
| 2 | Earthquake magnitude | **Done** | USGS CSV → IDW surface |
| 3 | Slope angle | **Have raster** | `Slope_FINAL` from existing DEM; not yet reclassed to paper breaks |
| 4 | Altitude | **Have raster** | `DEM_UTF_FINAL` |
| 5 | Distance to drainages | **Not started** | Can be derived from DEM (no extra download) |
| 6 | TRI | **Not started** | Can be derived from DEM |
| 7 | Geomorphology | **Not started** | Landsat or DEM-derived classes |
| 8 | Geology / lithology | **Not started** | Bhukosh skipped; see alternatives below |
| 9 | Soil | **Not started** | Hardest layer |
| 10 | Gravity anomaly | **Optional / skip** | Paper digitised a published map |
| 11 | Distance to faults | **Not started** | GEM Global Active Faults is the easy download |
| 12 | STI | **Not started** | DEM-derived |
| 13 | TWI | **Not started** | DEM-derived |
| 14 | SPI | **Not started** | DEM-derived |
| 15 | Distance to roads | **Done** | OSM → 30 m proximity |
| 16 | LULC | **Not started** | Landsat 8 or WorldCover |
| 17 | NDVI | **Not started** | Landsat 8 bands 4 + 5 |

**FR tables, 5-zone LSI, and ROC/AUC for the 17-factor model have not been run yet.**

The 3-factor Phase-1 LSI (`LSI_Final_Zones`, AUC ~0.77 in the pipeline README) is in the project as a baseline only.

---

## 5. Data still needed (and Bhukosh alternatives)

Bhukosh was the paper’s source for inventory, geology, and faults. It is not required.

### Landslide inventory (need more than 22 points)

- [Bhusanket](https://bhusanket.gsi.gov.in/) — GSI field-validated inventory download
- [Bhuvan landslide](https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/landslide.php?uname=empty) — ISRO/NRSC, Mizoram listed
- [NASA COOLR](https://gpm.nasa.gov/landslides/) — global points, sparse locally
- [Zenodo Mizoram set](https://doi.org/10.5281/zenodo.20783995) — 19 Aizawl-area events (2016–2025)
- Already in QGIS: 22 Aizawl points + 11,033 NASA GLC points (global)

Drop files in `00_drop_raw\01_landslide_inventory`.

### Geology

- [NGDR](https://geodataindia.gov.in/login) — official GSI replacement for Bhukosh downloads
- [GLiM](https://www.geo.uni-hamburg.de/en/geologie/forschung/aquatische-geochemie/glim.html) — global lithology, clip to Aizawl

### Faults (easiest remaining download)

- [GEM Global Active Faults](https://github.com/GEMScienceTools/gem-global-active-faults) — shapefile, no login
- [AFDI](https://github.com/iitrseismo/AFDI) — Active Fault Database of India

### Rainfall / Landsat (when ready)

- IMD 0.25° rainfall: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html (years 1988–2018 to match the paper)
- Landsat 8-9 C2 L2 from [EarthExplorer](https://earthexplorer.usgs.gov/) — Path 136, Row 44, dry season, &lt;10% cloud

### Can be built from data we already have (no download)

Slope, altitude, drainage distance, TRI, STI, TWI, SPI, and a first-pass geomorphology from `DEM_UTF_FINAL`.

---

## 6. Aizawl extent used for processing

WGS84: west 92.62, south 23.31, east 93.21, north 24.41  

UTM 46N (approx.): 461539, 2578028 → 521581, 2699993 (~60 km × 122 km)

---

## 7. Key paths

| What | Path |
|---|---|
| QGIS project | `C:\Users\sharm\LSI_btp.qgz` |
| This log | `C:\Users\sharm\LSI_btp_WORK_ACHIEVED.md` |
| Replication workspace | `C:\Users\sharm\LSI_btp_sikkim_replication\` |
| Processed outputs | `C:\Users\sharm\LSI_btp_sikkim_replication\01_processed\` |
| GitHub pipeline clone | `C:\Users\sharm\lsi-ml-pipeline\` (from 18 Aug session) |
| Reference paper | `C:\Users\sharm\Downloads\sikkim landslide succ refrence 1.pdf` |

---

## 8. Suggested next steps

1. Derive DEM factors in QGIS: slope classes, altitude classes, drainage + distance, TRI, TWI, SPI, STI.
2. Download **GEM faults**, clip, build distance-to-faults.
3. Add a larger landslide inventory (Bhusanket / Bhuvan / Zenodo) and merge with the 22 Aizawl points; split 70/30.
4. Add geology (NGDR or GLiM) and, if possible, Landsat NDVI + LULC.
5. Reclassify all factors, compute FR / FRn, sum to LSI, Natural Breaks (5 classes), ROC/AUC.

Until more inventory and geology arrive, steps 1–2 can proceed with files already on disk.
