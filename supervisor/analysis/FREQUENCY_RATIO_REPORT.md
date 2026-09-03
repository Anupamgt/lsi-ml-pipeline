# Frequency Ratio analysis — Aizawl, Mizoram (Phase 1 baseline)

**Document for supervisor review — 3-factor topographic FR only.**  
The 17-factor Sonker run (with caveats) is in [`FR_17_FACTOR_REPORT.md`](FR_17_FACTOR_REPORT.md).

**Project:** Landslide Susceptibility Index (LSI), IIT Patna B.Tech  
**Study area:** Aizawl district, Mizoram · CRS: EPSG:32646 (UTM 46N)

This note is the analysis companion to the maps in [`../maps/`](../maps/). It reports **Phase 1**, which is complete, and states clearly what the **17-factor** stack is and is not.

---

## 1. Purpose

Build a landslide susceptibility map for Aizawl using the **Frequency Ratio (FR)** method. Future landslides are assumed to occur under conditions similar to past failures. Each class of each factor is weighted by how much more (or less) landslides concentrate there than would be expected from area alone.

Reference method (applied here to Aizawl, not Sikkim):

> Sonker, I., Tripathi, J.N., Swarnim (2022). Remote sensing and GIS-based landslide susceptibility mapping using frequency ratio method in Sikkim Himalaya. *Quaternary Science Advances* 8, 100067.

Phase 1 uses three topographic factors only. Phase 2 prepares the paper’s 17-factor stack; **FR / LSI / ROC have not yet been computed for those 17 factors**.

---

## 2. Data used in Phase 1

| Item | Source / file | Notes |
|---|---|---|
| Study area | `AIZWAL.shp` | One polygon; layer CRS EPSG:4326, analysis in UTM 46N |
| DEM | Copernicus DSM 30 m, clipped | `P1_09_DEM_Aizawl.tif`; elevation ~20–1906 m |
| Slope | Derived from DEM | Degrees; then 3 classes |
| Aspect | Derived from DEM | 9 compass classes + flat |
| Elevation | DEM | 4 altitude bands |
| Landslides | `Aizawl_landslide_points.gpkg` | **22 points**. First recorded event in this set: 2009-10-05, Chhinga Veng (event_id 1222) |

With only 22 events, class counts are small. FR values are still useful as a **baseline**, but they should not be over-interpreted as a regional model.

---

## 3. Method

### 3.1 Reclassification

Continuous rasters cannot enter the FR equation directly. Slope, aspect, and elevation were sliced into discrete classes (Natural Breaks / practical breaks), then stored as integer GeoTIFFs (`P1_06`–`P1_08`).

### 3.2 Frequency Ratio

For each class \(i\):

\[
FR_i = \frac{N_i / N}{S_i / S}
\]

| Symbol | Meaning |
|---|---|
| \(N_i\) | Landslide points in class \(i\) |
| \(N\) | Total landslides (22) |
| \(S_i\) | Pixel count (area) of class \(i\) |
| \(S\) | Pixel count of the study area |

**Interpretation**

- \(FR > 1\): class is **enriched** in landslides relative to its area (more susceptible)
- \(FR = 1\): landslides in proportion to area
- \(FR < 1\): class is **depleted** (less susceptible)
- \(FR = 0\): no landslides in that class in this inventory

### 3.3 Landslide Susceptibility Index

Each class value on the map was replaced by its FR weight. The three FR rasters were added:

\[
LSI = FR_{\text{slope}} + FR_{\text{aspect}} + FR_{\text{elevation}}
\]

The continuous LSI (`P1_02_LSI_continuous.tif`) was then cut into **five quantile / discrete hazard zones** (`P1_01_LSI_hazard_zones.tif`): Very Low, Low, Moderate, High, Very High.

### 3.4 Validation

A ROC curve was built from the share of area in successively lower hazard zones versus the share of the 22 landslides captured. The **area under the curve (AUC) is 0.77**.

An AUC of 0.5 is no better than chance; 1.0 is perfect separation. 0.77 is a fair–good baseline for a three-factor map on a small inventory. It is **not** comparable to the Sonker et al. Sikkim result (~0.88) because that study used 17 factors and a much larger inventory.

---

## 4. Phase 1 results

### 4.1 Full FR table

Copied from [`../../data/fr_analysis_summary.csv`](../../data/fr_analysis_summary.csv). Landslide % uses \(N = 22\).

| Parameter | Class | Description | Landslides | Landslide % | Pixels | Area % | **FR** | Reading |
|---|---|---|---:|---:|---:|---:|---:|---|
| Aspect | 1 | Flat | 0 | 0.00 | 209 | 0.00 | **0.00** | No events |
| Aspect | 2 | North | 3 | 13.64 | 517,862 | 11.68 | **1.04** | Neutral–slight |
| Aspect | 3 | North-East | 3 | 13.64 | 466,807 | 10.53 | **1.15** | Slight enrichment |
| Aspect | 4 | East | 3 | 13.64 | 494,209 | 11.14 | **1.09** | Slight enrichment |
| Aspect | 5 | South-East | 1 | 4.55 | 465,437 | 10.49 | **0.39** | Depleted |
| Aspect | 6 | **South** | **4** | **18.18** | 515,819 | 11.63 | **1.39** | **Highest aspect FR** |
| Aspect | 7 | South-West | 2 | 9.09 | 476,864 | 10.75 | **0.75** | Depleted |
| Aspect | 8 | **West** | **4** | **18.18** | 530,814 | 11.97 | **1.35** | **High (monsoon-facing)** |
| Aspect | 9 | North-West | 2 | 9.09 | 483,157 | 10.89 | **0.74** | Depleted |
| Elevation | 1 | Very low (< 500 m) | 0 | 0.00 | 1,000,966 | 25.22 | **0.00** | No events in valleys |
| Elevation | 2 | Low (500–800 m) | 2 | 9.09 | 1,015,890 | 25.59 | **0.36** | Depleted |
| Elevation | 3 | Moderate (800–1100 m) | 3 | 13.64 | 1,003,474 | 25.28 | **0.54** | Depleted |
| Elevation | 4 | **High (> 1100 m)** | **17** | **77.27** | 949,092 | 23.91 | **3.24** | **Dominant control** |
| Slope | 1 | Gentle (0–15°) | 4 | 18.18 | 936,735 | 23.60 | **0.86** | Slightly depleted |
| Slope | 2 | Moderate (15–30°) | 6 | 27.27 | 1,018,920 | 25.67 | **1.06** | Neutral |
| Slope | 3 | **Steep (> 30°)** | **12** | **54.55** | 2,013,767 | 50.73 | **1.46** | **Enriched** |

Machine-readable copy: [`fr_class_table_phase1.csv`](fr_class_table_phase1.csv)

### 4.2 What the numbers say

1. **Elevation dominates.** More than three quarters of the 22 landslides sit above 1100 m, in a band that is only ~24% of the map (**FR = 3.24**). Valley floors (< 500 m) have **zero** events in this inventory.
2. **Steep slopes matter, but less than elevation.** Slopes > 30° hold 12 of 22 events (**FR = 1.46**). That class is also half the map, so the ratio is moderate.
3. **South and west aspects** are the most landslide-rich among compass classes (FR 1.39 and 1.35). That is consistent with monsoon exposure on the Mizo Hills, but the counts are only four points each.
4. The **5-zone LSI map** is the sum of these three FR rasters, not a machine-learning prediction.

### 4.3 Maps to look at first

| Priority | File | Why |
|---|---|---|
| 1 | `maps/geotiff/P1_01_LSI_hazard_zones.tif` | Deliverable hazard map |
| 2 | `maps/geotiff/P1_05_Elevation_FR.tif` | Shows the strongest factor |
| 3 | `maps/vectors/Aizawl_landslide_points.gpkg` | Overlay the 22 events on the zones |

JPEG previews of the same names live in `maps/preview/` for viewing on GitHub without QGIS.

---

## 5. Phase 2 status (17 factors)

Sonker et al. used 17 factors, normalized FR (\(FR_n = FR / \max FR\) per factor), \(LSI = \sum FR_n\), Natural Breaks into five zones, and a 70/30 landslide split for ROC.

For Aizawl, the **input rasters are prepared** (see [`SONKER_17_FACTOR_STATUS.md`](SONKER_17_FACTOR_STATUS.md) and `maps/geotiff/S17_*.tif`). The following have **not** been produced yet:

- FR / FRn tables for 17 factors
- 17-factor LSI raster
- 5 Natural Breaks zones from that LSI
- Hold-out ROC/AUC for that model

A 70/30 split of 22 points is only about 15 / 7 events. Any AUC from that split would be **statistically fragile**. Growing the inventory (GSI Bhusanket, Bhuvan, additional field points) should happen before treating a 17-factor ROC as a result.

---

## 6. Limitations (please read with the maps)

| Limitation | Effect |
|---|---|
| \(N = 22\) landslides | Class FR is sensitive to one or two points; several classes have 0–2 events |
| Phase 1 uses only topography | Rainfall, lithology, soil, roads, and hydrology are not in the current LSI |
| Inventory completeness unknown | Absence of points is not proof of stability |
| GLiM geology is one class in Aizawl | Will not discriminate lithology until GSI/NGDR mapping is used |
| OSM roads are dense | Distance-to-roads collapses into the nearest paper class |
| Paper TRI/SPI breaks | Do not transfer cleanly from Sikkim units to the Aizawl rasters |

---

## 7. Suggested next steps

1. Enlarge the landslide inventory, then recompute Phase 1 FR as a check.
2. Fix or drop weak Sonker factors (TRI, SPI, roads, single-class geology) before summing 17 FRn layers.
3. Compute FR / FRn / LSI / 5 zones / ROC on the surviving factors.
4. Keep Phase 1 (AUC 0.77) as the **documented baseline** for comparison.

---

## 8. File pointers

| Role | Path |
|---|---|
| This report | `supervisor/analysis/FREQUENCY_RATIO_REPORT.md` |
| FR numbers (CSV) | `supervisor/analysis/fr_class_table_phase1.csv` |
| 17-factor notes | `supervisor/analysis/SONKER_17_FACTOR_STATUS.md` |
| Hazard GeoTIFF | `supervisor/maps/geotiff/P1_01_LSI_hazard_zones.tif` |
| Navigation | `supervisor/README.md` |
