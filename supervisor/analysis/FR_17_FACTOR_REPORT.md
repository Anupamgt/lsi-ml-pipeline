# Aizawl District Frequency Ratio landslide susceptibility

**Status:** exploratory method transfer, not a finished or publishable map.

This report documents a Frequency Ratio (FR) analysis that applies the workflow of Sonker, Tripathi and Swarnim (2022, *Quaternary Science Advances* 8:100067) — originally developed for Sikkim Himalaya — to Aizawl District, Mizoram, on a 30 m grid. All FR, FRn, LSI, zonation and ROC numbers below were computed from the rasters listed in §11. They were not taken from the Sikkim paper.

---

## 1. Purpose and method

### 1.1 What the paper did

Sonker et al. (2022) mapped landslide susceptibility for Sikkim using 17 thematic factors, a Frequency Ratio model, five Natural Breaks susceptibility zones, and an ROC accuracy claim of 87.8%. Their zone areas were Very High 11.88%, High 15.75%, Medium 25.88%, Low 25.30%, Very Low 21.19%. Factor classes (geology formation names, Sikkim soil texture units, Sikkim rainfall millimetre bins, alpine NDVI breaks, and so on) are **Sikkim map units**. They do not exist as the same classes in Mizoram.

### 1.2 What this run does

The same *algebra* is applied to Aizawl:

\[
\mathrm{FR}_i = \frac{N_i / N}{S_i / S}
\]

where \(N_i\) is the number of *training* landslide cells in class \(i\), \(N\) is the number of training landslide cells, \(S_i\) is the number of study-area cells in class \(i\), and \(S\) is the study-area cell count (AIZWAL polygon ∩ valid Copernicus DEM).

Then, per factor,

\[
\mathrm{FRn}_i = \mathrm{FR}_i / \max_j(\mathrm{FR}_j)
\]

and the landslide susceptibility index is the unweighted sum of FRn across 17 factors:

\[
\mathrm{LSI} = \sum_{k=1}^{17} \mathrm{FRn}_k
\]

A cell with the locally most landslide-associated class in every factor would have LSI = 17. A cell in classes that contain no training landslides would have LSI near 0.

FR > 1 means that class is over-represented among training landslides relative to its area (conventionally called “landslide-prone”). FR < 1 is under-represented. FR = 1 is density-neutral. With \(N = 15\), a single landslide landing in a rare class produces an arbitrarily large FR. That is not a physical measurement; it is a small-sample artefact. See §7 and §9.

### 1.3 Train / test split

Landslide cells were split 70/30 with `numpy.random.default_rng(42)`:

| Split | Cells | Role |
|---|---|---|
| Train | 15 | FR and FRn |
| Test (holdout) | 7 | ROC only |
| All unique landslide cells | 22 | secondary ROC |

FR is **not** computed from the 7 holdout cells. LSI is then evaluated on holdout and on all 22.

**This split is statistically weak.** Seven positives cannot support a credible ROC. The all-22 AUC is optimistic because 15 of those points trained the weights. Neither number should be compared to Sonker’s 87.8% as a like-for-like accuracy claim.

---

## 2. Study area, grid, inventory

### 2.1 Master grid

| Item | Value |
|---|---|
| Raster | `01_processed\master_dem_30m.tif` |
| Size | 2003 × 4066 |
| Resolution | 30 m |
| CRS | EPSG:32646 (WGS 84 / UTM zone 46N) |
| Extent | 461520, 2578020, 521610, 2700000 |
| Project | `C:\Users\sharm\LSI_btp.qgz` |

Study-area mask = rasterized `AIZWAL` (source layer is EPSG:4326, transformed to 32646) ∩ DEM cells with valid elevation. Result: **S = 3,828,235 cells** ≈ **3,445.41 km²**. (A previous clip counted ~3,829,253; the ~1,000-cell difference is polygon rasterization.) About 2,556 mask cells lack at least one factor; those factors contribute FRn = 0 at those pixels.

### 2.2 Landslide inventory

Layer `Aizawl_Points_UTM` in the QGIS project (file `C:\Users\sharm\lsi-ml-pipeline\final_maps\Aizawl_Points_UTM.gpkg`) contains **22 NASA Global Landslide Catalog points**. Despite the layer name, geometries are stored in **EPSG:4326** and were transformed to UTM 46N for sampling.

| Item | Value |
|---|---|
| Points | 22 |
| Unique 30 m cells | 22 (no two points share a pixel) |
| Points on mask | 22 / 22 |
| First event | **2009-10-05**, Chhinga Veng, Aizawl (event_id 1222). Three fatalities, including a child, after rain-triggered collapse of a building occupied by four families. Location accuracy coded **5 km**. |
| Last event in this extract | 2017-06-28, Aizawl–Champhai road |
| Typical location accuracy | 1 km, 5 km, 10 km, 25 km or **50 km** |

These are news-reported events, heavily biased toward roads, towns and fatalities. They are not a complete slope-failure inventory and they are not polygon scarps. Sampling a 5–50 km-accurate point onto a 30 m pixel is, for many records, little better than “somewhere near Aizawl.” That limitation dominates every FR in this file.

---

## 3. The 17 factors

Base directory: `C:\Users\sharm\LSI_btp_sikkim_replication\01_processed\`

Rainfall and gravity arrived as **continuous** rasters and were 5-classed here by **equal-area quantiles** on the study-area mask (not Sikkim millimetre/mGal breaks; QGIS `native:reclassifybynaturalbreaks` was not used). Other factors used the existing Sonker-style integer rasters. Class *labels* in the tables are a mix of (a) empirical min–max of a co-registered continuous source where that source shares the master grid, and (b) Sonker Table 1 names where the raster is categorical (LULC, GLiM, NBSS). Integer class IDs are authoritative; paper bin names that disagree with empirical ranges are caveats, not facts.

### 3.1 Rainfall 5-class (this run)

Source: `chirps_mean_annual_30m.tif` (CHIRPS mean annual precipitation, mm). Valid range on the mask: **1,980–3,021 mm**. Quantile inner edges: 2244.68, 2385.65, 2509.69, 2668.43 mm. Written to `rainfall_reclass_q5.tif`. Each class is 20.00% of the study area by construction.

### 3.2 Gravity 5-class (this run)

Source: `gravity_bouguer_simple_30m_clip.tif` (simple Bouguer ≈ GGMplus gravity disturbance − 0.1119 × H, mGal). Valid range: **−315.7 to +72.7 mGal**. Quantile inner edges: −193.10, −140.77, −96.96, −49.60 mGal. Written to `gravity_reclass_q5.tif`. This is an open-data substitute for the digitized Ansari et al. (2014) Sikkim Bouguer map used by Sonker. It is not GSI terrestrial Bouguer.

### 3.3 Factor status

| # | Factor | Raster | Occupied classes | Dominant class (area) | Caveat |
|---|---|---|---|---|---|
| 1 | Rainfall | `rainfall_reclass_q5.tif` | 5 | each ~20% | Quantile, not Sikkim mm bins. Wettest quintile (Q5) has **0** training landslides. |
| 2 | Earthquake | `earthquake_reclass_sonker.tif` | 2, 3, 4 (no 1, no 5) | class 3 **96.16%** | Sikkim magnitude bins; Aizawl is almost one class. Empty classes 1 and 5. |
| 3 | Slope | `slope_reclass_sonker.tif` | 1–5 | class 2 **50.22%** (empirical ~17–33°) | Empirical ranges do not match a strict 0–15 / 15–25 / … reading. Classes 4–5 (steepest) have **0** training LS. |
| 4 | Altitude | `altitude_reclass_sonker.tif` | **1–2 only** | class 1 **98.83%** (20–1500 m) | Sikkim 221–8586 m bins. Aizawl barely reaches class 2 (1500–1906 m). Classes 3–5 empty. |
| 5 | Dist. drainages | `drainage_dist_reclass_sonker.tif` | 1–5 | class 1 **36.81%** | Empirical distances are hundreds of metres per class, not the 100 m Sikkim labels. |
| 6 | TRI | `tri_reclass_sonker.tif` | 1–5 | class 5 **99.97%** | **Weak.** Roughness saturates. FRn is a near-constant 1. |
| 7 | Geomorphology | `geomorph_reclass_sonker.tif` | 2, 3, 4 | class 4 **50.58%** | Three occupied classes. |
| 8 | Geology | `geology_reclass_sonker_fr.tif` | **1 class** | class 1 **99.996%** | **Weak.** GLiM / Mizoram Surma–Barail siliciclastic basin. Not GSI formation polygons. FR ≡ 1. |
| 9 | Soil | `soil_india_reclass_sonker.tif` | 1, 2, 4 | class 1 **82.96%** | India NBSS, **not** SoilGrids WRB, **not** Sikkim 1:50k texture units. |
| 10 | Gravity | `gravity_reclass_q5.tif` | 5 | ~20% each | Quantile simple Bouguer. Q1 (most negative) and Q5 have 0 training LS. |
| 11 | Dist. faults | `faults_dist_reclass_sonker.tif` | 1–5 | class 5 **74.54%** (>4 km) | Open/GEM faults, not a complete GSI inventory. Class 4 FR is high on 3 training points. |
| 12 | STI | `sti_reclass_sonker.tif` | 1–5 | class 3 **35.55%** | Usable spread. |
| 13 | TWI | `twi_reclass_sonker.tif` | 1–5 | class 2 **54.20%** | Classes 3–5 have 0 training LS. |
| 14 | SPI | `spi_reclass_sonker.tif` | 2–5 (**class 1 empty**) | class 5 **86.34%** | Low SPI classes empty/rare. |
| 15 | Dist. roads | `roads_dist_reclass_sonker.tif` | **1 only** | class 1 **99.97%** | **Weak.** OSM roads vs Sikkim 1 km bins: the whole district is “near a road.” |
| 16 | LULC | `lulc_reclass_sonker.tif` | 1, 3, 4, 5, 6 (no glacier) | Forest **94.46%** | Build-up is 1.63% of area but 10/15 train LS (inventory bias). |
| 17 | NDVI | `ndvi_reclass_sonker.tif` | 1–5 | class 5 **99.49%** | **Weak / explosive FR.** Sikkim alpine breaks; Aizawl forest is class 5. Classes 3–4 are rare urban/cleared pixels that coincide with GLC points. |

---

## 4. Full FR / FRn tables

FR from **15 training cells**. \(S = 3{,}828{,}235\). FRn = FR / max(FR) within that factor. `N_i all` is the count among all 22 landslide cells (not used for FR).

### 4.1 Rainfall (quantile)

| Class | Empirical mm | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 Q1 | 1980–2245 | 20.00 | 3 | 1.000 | 0.375 |
| 2 Q2 | 2245–2386 | 20.00 | 3 | 1.000 | 0.375 |
| 3 Q3 | 2386–2510 | 20.00 | 8 | **2.667** | 1.000 |
| 4 Q4 | 2510–2668 | 20.00 | 1 | 0.333 | 0.125 |
| 5 Q5 | 2668–3021 | 20.00 | 0 | 0.000 | 0.000 |

### 4.2 Earthquake

| Class | Area % | N_i train | FR | FRn |
|---|---|---|---|---|
| 2 | 3.57 | 0 | 0.000 | 0.000 |
| 3 | 96.16 | 15 | **1.040** | 1.000 |
| 4 | 0.23 | 0 | 0.000 | 0.000 |

Empty: class 1, class 5. All 15 training points sit in class 3 because almost the whole district is class 3.

### 4.3 Slope (empirical source range)

| Class | Empirical ° | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 | 0–17 | 24.84 | 3 | 0.805 | 0.567 |
| 2 | 17–33 | 50.22 | 7 | 0.929 | 0.655 |
| 3 | 33–50 | 23.49 | 5 | **1.419** | 1.000 |
| 4 | 50–67 | 1.45 | 0 | 0.000 | 0.000 |
| 5 | 67–73 | 0.007 | 0 | 0.000 | 0.000 |

### 4.4 Altitude

| Class | Empirical m | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 | 20–1500 | 98.83 | 15 | **1.012** | 1.000 |
| 2 | 1500–1906 | 1.17 | 0 | 0.000 | 0.000 |

### 4.5 Distance to drainages (empirical m)

| Class | Empirical m | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 | 0–297 | 36.81 | 1 | 0.181 | 0.058 |
| 2 | 300–598 | 30.59 | 3 | 0.654 | 0.208 |
| 3 | 600–899 | 21.17 | 10 | **3.149** | 1.000 |
| 4 | 900–1199 | 9.03 | 1 | 0.739 | 0.235 |
| 5 | 1200–2697 | 2.41 | 0 | 0.000 | 0.000 |

### 4.6 TRI

| Class | Area % | N_i train | FR | FRn |
|---|---|---|---|---|
| 1–4 combined | 0.026 | 0 | 0.000 | 0.000 |
| 5 | **99.974** | 15 | 1.000 | 1.000 |

### 4.7 Geomorphology

| Class | Area % | N_i train | FR | FRn |
|---|---|---|---|---|
| 2 | 0.001 | 0 | 0.000 | 0.000 |
| 3 | 49.42 | 6 | 0.809 | 0.682 |
| 4 | 50.58 | 9 | **1.186** | 1.000 |

### 4.8 Geology (GLiM / FR raster)

| Class | Area % | N_i train | FR | FRn |
|---|---|---|---|---|
| 1 (single lithology; GLiM ss siliciclastic sedimentary basin) | 99.996 | 15 | 1.000 | 1.000 |

### 4.9 Soil (India NBSS reclass)

| Class | Area % | N_i train | FR | FRn |
|---|---|---|---|---|
| 1 | 82.96 | 13 | **1.045** | 1.000 |
| 2 | 15.95 | 2 | 0.836 | 0.800 |
| 4 | 1.09 | 0 | 0.000 | 0.000 |

### 4.10 Gravity (quantile, mGal)

| Class | Empirical mGal | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 Q1 | −316 to −193 | 20.00 | 0 | 0.000 | 0.000 |
| 2 Q2 | −193 to −141 | 20.00 | 6 | **2.000** | 0.750 |
| 3 Q3 | −141 to −97 | 20.00 | 8 | **2.667** | 1.000 |
| 4 Q4 | −97 to −50 | 20.00 | 1 | 0.333 | 0.125 |
| 5 Q5 | −50 to +73 | 20.00 | 0 | 0.000 | 0.000 |

### 4.11 Distance to faults (empirical m)

| Class | Empirical m | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 | 0–997 | 6.50 | 1 | **1.025** | 0.323 |
| 2 | 1001–1998 | 6.28 | 0 | 0.000 | 0.000 |
| 3 | 2001–3000 | 6.38 | 0 | 0.000 | 0.000 |
| 4 | 3001–3999 | 6.30 | 3 | **3.174** | 1.000 |
| 5 | 4001–20810 | 74.54 | 11 | 0.984 | 0.310 |

Class 4 FR > 1 is three training points in 6.3% of the area. Do not read this as “3–4 km from a fault is the dangerous belt.”

### 4.12 STI

| Class | Empirical | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 | 0.0035–3.22 | 5.35 | 1 | **1.247** | 0.715 |
| 2 | 3.22–12.86 | 33.39 | 6 | **1.198** | 0.687 |
| 3 | 12.86–32.15 | 35.55 | 4 | 0.750 | 0.430 |
| 4 | 32.15–70.73 | 15.29 | 4 | **1.744** | 1.000 |
| 5 | 70.73–7.57e4 | 10.43 | 0 | 0.000 | 0.000 |

### 4.13 TWI

| Class | Empirical | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 | 2.29–4.65 | 35.54 | 4 | 0.750 | 0.555 |
| 2 | 4.65–8.19 | 54.20 | 11 | **1.353** | 1.000 |
| 3 | 8.19–11.73 | 8.28 | 0 | 0.000 | 0.000 |
| 4 | 11.73–15.27 | 1.67 | 0 | 0.000 | 0.000 |
| 5 | 15.27–24.2 | 0.31 | 0 | 0.000 | 0.000 |

### 4.14 SPI

| Class | Empirical | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 2 | 0.03–3.22 | 1.48 | 0 | 0.000 | 0.000 |
| 3 | 3.22–6.58 | 4.86 | 0 | 0.000 | 0.000 |
| 4 | 6.58–9.94 | 7.32 | 4 | **3.644** | 1.000 |
| 5 | 9.94–3.41e7 | 86.34 | 11 | 0.849 | 0.233 |

Class 1 is empty.

### 4.15 Distance to roads

| Class | Area % | N_i train | FR | FRn |
|---|---|---|---|---|
| 1 | 99.966 | 15 | 1.000 | 1.000 |

### 4.16 LULC (Sonker 6-class names)

| Class | Name | Area % | N_i train | N_i all | FR | FRn |
|---|---|---|---|---|---|---|
| 1 | Water | 0.39 | 0 | 0 | 0.000 | 0.000 |
| 3 | Build-up | 1.63 | **10** | 15 | **40.79** | 1.000 |
| 4 | Barren | 0.004 | 0 | 0 | 0.000 | 0.000 |
| 5 | Agriculture | 3.52 | 1 | 1 | **1.895** | 0.046 |
| 6 | Forest | 94.46 | 4 | 6 | 0.282 | 0.007 |

Glacier (class 2) is absent. **10 of 15 training landslides sit in build-up (1.63% of area).** That is catalog geocoding (town/road news), not a statement that forested slopes do not fail.

### 4.17 NDVI (Sonker breaks, tails extended)

| Class | Break | Area % | N_i train | FR | FRn |
|---|---|---|---|---|---|
| 1 Very low | < −0.18 | 0.0005 | 0 | 0.000 | 0.000 |
| 2 Low | −0.18–0.02 | 0.008 | 0 | 0.000 | 0.000 |
| 3 Moderate | 0.02–0.22 | 0.059 | 3 | **340.74** | 1.000 |
| 4 High | 0.22–0.42 | 0.444 | 5 | **75.03** | 0.220 |
| 5 Very high | ≥ 0.42 | 99.49 | 7 | 0.469 | 0.001 |

FR = 340 for NDVI class 3 means: 3 of 15 training points fell in 2,247 cells (0.059% of the district). Those pixels are rare, lower-vegetation (urban/cleared) patches. The number is an artefact. It will dominate LSI wherever a GLC point sits in town.

---

## 5. Classes with FR > 1

Sorted by FR. Treat the top of this list as **inventory artefacts** until a polygon inventory exists.

| Factor | Class | FR | N_i train | Area % |
|---|---|---|---|---|
| NDVI | 3 moderate | 340.74 | 3 | 0.059 |
| NDVI | 4 high | 75.03 | 5 | 0.444 |
| LULC | 3 build-up | 40.79 | 10 | 1.634 |
| SPI | 4 | 3.644 | 4 | 7.319 |
| Faults | 4 | 3.174 | 3 | 6.301 |
| Drainage | 3 | 3.149 | 10 | 21.168 |
| Gravity | 3 Q3 | 2.667 | 8 | 20.000 |
| Rainfall | 3 Q3 | 2.667 | 8 | 20.000 |
| Gravity | 2 Q2 | 2.000 | 6 | 20.000 |
| LULC | 5 agriculture | 1.895 | 1 | 3.517 |
| STI | 4 | 1.744 | 4 | 15.286 |
| Slope | 3 (~33–50°) | 1.419 | 5 | 23.487 |
| TWI | 2 | 1.353 | 11 | 54.204 |
| STI | 1 | 1.247 | 1 | 5.347 |
| STI | 2 | 1.198 | 6 | 33.386 |
| Geomorphology | 4 | 1.186 | 9 | 50.576 |
| Soil | 1 | 1.045 | 13 | 82.961 |
| Earthquake | 3 | 1.040 | 15 | 96.158 |
| Faults | 1 | 1.025 | 1 | 6.502 |
| Altitude | 1 | 1.012 | 15 | 98.830 |
| Roads | 1 | 1.000 | 15 | 99.966 |
| TRI | 5 | 1.000 | 15 | 99.974 |
| Geology | 1 | 1.000 | 15 | 99.996 |

The last four are density-neutral (FR ≈ 1) because the class *is* the study area. They add a constant to LSI and do not discriminate.

More plausible (still fragile) physical signals, if any, sit in the middle of the list: rainfall Q3, gravity Q2–Q3, slope class 3, drainage class 3, STI 4, SPI 4, TWI 2. Each of those still rests on ≤11 training cells.

---

## 6. LSI surface and five zones

LSI = Σ FRn (17 factors). Written to `lsi_frn_sum_30m.tif`.

| Statistic | Value |
|---|---|
| Min | **5.465** |
| Max | **15.592** |
| Mean | **10.451** |
| Std. dev. | 0.959 |
| Theoretical range | 0–17 |

Mean landslide-cell LSI (all 22) = **12.715**. Mean background (8,000 random non-landslide cells) = **10.440**. Holdout mean LSI = **11.510**.

### 6.1 Zonation

Five classes from a 1-D k-means approximation to Natural Breaks on 40,000 sampled LSI values (Jenks DP on the full raster is not practical; QGIS `native:reclassifybynaturalbreaks` was not used). Breaks: **9.144, 10.061, 10.861, 11.800**. Zone 1 = very low. Raster: `lsi_fr_zones_5_30m.tif`.

Equal-area quantile breaks (not used for the raster) were 9.683, 10.194, 10.644, 11.239 for comparison.

| Zone | Name | Cells | Area km² | Area % | LSI range | LS (all 22) | Capture % | Train 15 | Test 7 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Very Low | 275,117 | 247.61 | 7.19 | 5.465–9.143 | 0 | 0.0 | 0 | 0 |
| 2 | Low | 1,036,587 | 932.93 | 27.08 | 9.144–10.060 | 1 | 4.5 | 0 | 1 |
| 3 | Medium | 1,306,797 | 1,176.12 | 34.14 | 10.061–10.861 | 1 | 4.5 | 1 | 0 |
| 4 | High | 903,490 | 813.14 | 23.60 | 10.861–11.800 | 4 | 18.2 | 1 | 3 |
| 5 | Very High | 306,244 | 275.62 | 8.00 | 11.800–15.592 | 16 | **72.7** | 13 | 3 |

High + Very High together: **20 / 22 landslides (90.9%)** in **31.6%** of the district. Very High alone: 72.7% of landslides in 8.0% of area.

That concentration is real in the arithmetic and **largely circular**: LULC build-up and mid NDVI have huge FRn, GLC points are in towns, towns become Very High. It is not independent evidence that 8% of Aizawl is a Sikkim-style “very high” zone.

Sonker’s Sikkim Very High zone was 11.88% of *that* study area with a different inventory. The percentages are not comparable.

---

## 7. ROC / AUC

Positives: landslide-cell LSI. Negatives: **8,000** random mask cells that are not landslide cells (seed 42). AUC from Mann–Whitney P(LSI_ls > LSI_bg) + 0.5 P(tie), and from the trapezoid of the TPR–FPR curve. The two estimators agree.

| Test | n positives | n background | AUC (Mann–Whitney) | AUC (trapezoid) |
|---|---|---|---|---|
| All landslide cells | 22 | 8,000 | **0.892** | 0.891 |
| Holdout 30% | **7** | 8,000 | **0.792** | 0.791 |
| Train 70% (diagnostic only) | 15 | 8,000 | 0.938 | — |

At 5% false-positive rate the all-22 curve already reaches TPR ≈ 0.68; the holdout curve is only ≈ 0.29. With seven positives, a single point moving across a threshold swings TPR by 1/7 ≈ 0.14. **Holdout AUC 0.79 is not a validation of the map.** Train AUC 0.94 is overfitting. All-22 AUC 0.89 re-uses the training points.

Sonker et al. reported 87.8% ROC accuracy in Sikkim with a much larger inventory. That figure must not be cited as the accuracy of this Aizawl raster.

---

## 8. Limitations (read before using the rasters)

1. **n = 22 points.** FR, 70/30 ROC, and “landslide capture by zone” are all under-powered. Several FR > 1 classes rest on one to three points.
2. **Location accuracy 1–50 km** on a 30 m grid. Many FR signals (build-up, moderate NDVI) are geocoding artefacts.
3. **NASA GLC is news-biased** toward roads, towns, and fatalities. Forest and remote failures are missing, so forest FR < 1 is expected even if forested slopes fail.
4. **Four factors are near-constants** in Aizawl: TRI class 5 (99.97%), geology one lithology (100%), roads class 1 (99.97%), NDVI class 5 (99.49%). They do not discriminate. NDVI’s rare classes 3–4 then explode FR.
5. **Sikkim class breaks transferred to Mizoram** (altitude, earthquake, NDVI, TRI, road distance, geology names) are the wrong legend for this landscape.
6. **Geology is GLiM**, not GSI / Bhukosh Mizoram formations. Aizawl is a siliciclastic sedimentary basin; Sikkim’s gneiss/Tethyan/Rangit units do not occur here.
7. **Soil is India NBSS**, not the Sikkim 1:50,000 texture legend and not SoilGrids.
8. **Gravity is satellite-derived simple Bouguer**, not the paper’s digitized terrestrial Bouguer.
9. **Rainfall is CHIRPS quantiles**, not the paper’s Sikkim mm classes.
10. **No conditioning / collinearity control.** Slope, TRI, SPI, STI and TWI are hydrologically related; summing 17 FRn double-counts topography.
11. **LSI is not a probability** and is not calibrated to return period or magnitude.
12. **Do not use these rasters for planning, insurance, or a thesis “final map.”** They document a method transfer and its failure modes.

---

## 9. What would change with better data

| Upgrade | Effect |
|---|---|
| GSI / Bhukosh geology (Bhuban, Boka Bil, Tipam, etc.) | Geology would stop being a constant and could become a real factor. |
| Polygon or high-accuracy point inventory (hundreds of scars, not 22 news points) | FR would stabilize; 70/30 ROC would mean something; LULC/NDVI artefacts would shrink. |
| Aizawl-specific class breaks (Jenks on local slope, NDVI, TRI, rainfall) | End saturation of TRI/NDVI/roads/altitude. |
| GSI terrestrial Bouguer | Gravity would match the paper’s physical quantity. |
| Inventory that includes forested failures | Forest FR would rise; Very High would be less of an urban mask. |
| Factor screening (remove constants; drop collinear DEM derivatives) | LSI would not be a 17-layer sum of noise plus two urban proxies. |

Until those exist, the honest product is this report plus the rasters as **working files**, not a susceptibility map of Aizawl District.

---

## 10. Computation notes

- Script: `02_fr_tables\compute_fr_lsi.py` (PyQGIS / GDAL / numpy).
- Seed 42 for the 70/30 split and the 8,000 background cells.
- Landslide sampling: geotransform column/row, unique cells.
- Zone method: 1-D k-means on a 40,000-cell LSI sample (Jenks approximation).
- QGIS project `C:\Users\sharm\LSI_btp.qgz` group **Sonker 2022 factors** now includes `LSI_FRn_sum_30m`, `LSI_FR_zones_5`, `Rainfall_reclass_q5`, `Gravity_reclass_q5`.
- Elapsed compute ~16 s. SAGA was not used.
- Continuous sources that are **not** on the master grid were skipped for empirical min/max only (`earthquake_magnitude_clip.tif` 1196×2438; `distance_to_roads_clip.tif` 1993×4064). FR still used the Sonker reclass rasters, which are on-grid.

---

## 11. Output files

| File | Contents |
|---|---|
| `01_processed\rainfall_reclass_q5.tif` | Rainfall 5-class quantile |
| `01_processed\gravity_reclass_q5.tif` | Gravity 5-class quantile |
| `01_processed\lsi_frn_sum_30m.tif` | LSI = Σ FRn |
| `01_processed\lsi_fr_zones_5_30m.tif` | Zones 1 (very low) … 5 (very high) |
| `02_fr_tables\fr_results.json` | Full numeric dump (FR, split, ROC points, inventory) |
| `02_fr_tables\fr_class_table.csv` | One row per factor class |
| `02_fr_tables\FR_ANALYSIS_REPORT.md` | This report |
| `02_fr_tables\compute_fr_lsi.py` | Computation script |

---

## 12. Citation of the method paper

Sonker, I., Tripathi, J.N., Swarnim, 2022. Remote sensing and GIS-based landslide susceptibility mapping using frequency ratio method in Sikkim Himalaya. *Quaternary Science Advances* 8, 100067. https://doi.org/10.1016/j.qsa.2022.100067

This Aizawl run is a **replication of the method, not of the Sikkim map**, and it is not a peer-reviewable susceptibility assessment of Mizoram.
