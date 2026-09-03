# LSI_btp — Work Achieved

**Project:** Landslide Susceptibility Index (LSI) for Aizawl, Mizoram  
**QGIS project:** `C:\Users\sharm\LSI_btp.qgz`  
**CRS:** EPSG:32646 (WGS 84 / UTM zone 46N)  
**Last saved:** 31 August 2026  
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


### 3.5 Seven terrain factors on the 30 m master grid

All DEM-derived factors were rebuilt onto `01_processed\master_dem_30m.tif` (EPSG:32646, 30 m; extent 461520, 2578020 to 521610, 2700000). Fill sinks, D8 flow accumulation, and a drainage network were generated first.

| Factor | Local raster (not in git) |
|---|---|
| Slope (degrees) | `slope_30m_clip.tif` |
| Altitude | `altitude_30m_clip.tif` |
| Distance to drainages | `distance_to_drainage_30m.tif` |
| TRI | `tri_30m_clip.tif` |
| TWI | `twi_30m_clip.tif` |
| SPI | `spi_30m_clip.tif` |
| STI | `sti_30m_clip.tif` |

Plus `dem_filled_30m.tif`, `flow_accum.tif`, `drainage_network.gpkg`. GeoTIFFs are large (hundreds of MB) and stay local in `LSI_btp_sikkim_replication\01_processed\`.

### 3.6 CHIRPS rainfall (QGIS + GEE, 2010-2018)

Paper used IMD 0.25 deg for 1988-2018. Substitute: CHIRPS daily, mean annual mm.

- GEE script: [`docs/gee/gee_chirps_aizawl_2009_2018.js`](gee/gee_chirps_aizawl_2009_2018.js) (inventory start 2009-10-05; **mean uses complete calendar years 2010-2018** so partial 2009 does not bias the annual total).
- Optional TRMM/IMERG comparison: [`docs/gee/gee_trmm_aizawl_phase1_phase2.js`](gee/gee_trmm_aizawl_phase1_phase2.js).
- Local rasters (not committed): native CHIRPS under `00_drop_raw\03_rainfall_imd\`, warped 30 m mean `01_processed\chirps_mean_annual_30m.tif`.

### 3.7 Paper-break reclasses (6 factors; TRI / SPI / roads caveats)

Integer class rasters `*_reclass_sonker.tif` were burned with Sonker Table 1 breaks:

| Factor | Paper class breaks | Notes |
|---|---|---|
| Slope | 0-17, 17-33, 33-50, 50-67, >67 deg | Applied |
| Altitude | <1500, 1500-3000, 3000-4500, 4500-6000, >6000 m | Aizawl DEM tops ~1906 m, so only the first two classes populate |
| Distance to drainages | 300 / 600 / 900 / 1200 / >1500 m | Applied |
| Earthquake magnitude | 2.60-3.46, 3.46-4.32, 4.32-5.18, 5.18-6.04, 6.04-6.89 | Applied to the IDW surface |
| STI | 0-3.22, 3.22-12.86, 12.86-32.15, 32.15-70.73, 70.73-409.92 | Applied |
| TWI | 1.11-4.65, 4.65-8.19, 8.19-11.73, 11.73-15.27, 15.27-18.81 | Applied |

**Caveats (do not treat these three as 1:1 paper matches):**

- **TRI** — Table 1 classes are 0.01-0.99 (normalized / unitless). QGIS Riley TRI is in elevation-difference units, so paper breaks can dump most Aizawl pixels into one class.
- **SPI** — Table 1 classes run about -3.51 to 13.30 (typical of ln(SPI)). Raw SPI = contributing area x tan(slope) is often orders of magnitude larger.
- **Distance to roads** — paper buffers 300-1500 m. OSM in Aizawl is dense (max distance ~284 m), so almost every pixel falls in the <300 m class. Restrict the network to trunk/primary/secondary/tertiary if FR is uninformative.

FR tables, 5-zone LSI, and ROC/AUC for the 17-factor model have **not** been run yet.

---

## 4. 17 paper factors — status

| # | Factor | Status | Notes |
|---|---|---|---|
| 1 | Rainfall | **Done (CHIRPS substitute)** | GEE 2010-2018 mean annual; 30 m warp in QGIS. Paper used IMD 1988-2018. |
| 2 | Earthquake magnitude | **Done + reclass** | USGS CSV -> IDW surface; paper magnitude breaks |
| 3 | Slope angle | **Done + reclass** | 30 m grid; paper degree breaks |
| 4 | Altitude | **Done + reclass** | 30 m grid; only <1500 and 1500-3000 m populate in Aizawl |
| 5 | Distance to drainages | **Done + reclass** | 30 m proximity; paper 300 m buffers |
| 6 | TRI | **Raster done; reclass caveat** | 30 m Riley TRI; paper 0-1 classes do not transfer cleanly |
| 7 | Geomorphology | **Not started** | Landsat or DEM-derived classes |
| 8 | Geology / lithology | **Not started** | NGDR / GLiM still needed |
| 9 | Soil | **Not started** | Hardest layer |
| 10 | Gravity anomaly | **Optional / skip** | Paper digitised a published map |
| 11 | Distance to faults | **Not started** | GEM Global Active Faults is the easy download |
| 12 | STI | **Done + reclass** | 30 m; paper STI breaks |
| 13 | TWI | **Done + reclass** | 30 m; paper TWI breaks |
| 14 | SPI | **Raster done; reclass caveat** | 30 m raw SPI; paper classes look like ln(SPI) |
| 15 | Distance to roads | **Done; reclass caveat** | OSM 30 m proximity; max ~284 m so paper 300-1500 m bins collapse |
| 16 | LULC | **Not started** | Landsat 8 or WorldCover |
| 17 | NDVI | **Not started** | Local processing script exists (`docs/scripts/process_ndvi_qgis.py`); factor not accepted as finished |

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

- CHIRPS 2010-2018 mean annual is already in QGIS (GEE script in `docs/gee/`).
- IMD 0.25° rainfall: https://imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html (years 1988–2018 to match the paper)
- Landsat 8-9 C2 L2 from [EarthExplorer](https://earthexplorer.usgs.gov/) — Path 136, Row 44, dry season, &lt;10% cloud

### Can be built from data we already have (no download)

DEM hydrology factors (slope, altitude, drainage distance, TRI, STI, TWI, SPI) are done on the 30 m grid. Remaining no-download item: a first-pass geomorphology from the DEM. Gravity anomaly is optional.

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

Still missing for a 17-factor FR stack: **geology, soil, LULC, NDVI, faults, geomorphology**.

1. Download **GEM faults**, clip, build distance-to-faults.
2. Add geology (NGDR or GLiM) and soil if a usable national/global product exists.
3. Finish LULC + NDVI (Landsat 8 Path 136 / Row 44 or WorldCover) and decide whether local NDVI drafts are usable.
4. Fix TRI/SPI reclass (normalize or ln) and optionally thin the OSM network before re-binning roads.
5. Grow the landslide inventory (Bhusanket / Bhuvan / Zenodo) beyond 22 points; split 70/30.
6. Compute FR / FRn, sum to LSI, Natural Breaks (5 classes), ROC/AUC.

No layout PNG/PDF map exports were found to commit; rasters remain local.
