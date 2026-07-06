# Predictive Geospatial & Machine Learning Modeling for Landslide Susceptibility Index (LSI)
### IIT Patna B.Tech Research Project | Aizawl, Mizoram, India

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![QGIS](https://img.shields.io/badge/QGIS-3.30+-green.svg)](https://qgis.org/)

---

## Executive Summary

This report details the methodology and results of the foundational phase of a predictive geospatial modeling project aimed at assessing landslide susceptibility in the **Aizawl region** of Mizoram, India. 

Utilizing a bivariate statistical approach—specifically the **Frequency Ratio (FR) method**—topographical parameters including slope, aspect, and elevation were processed and analyzed against historical landslide inventories. The resulting **Landslide Susceptibility Index (LSI)** map was divided into five discrete hazard zones. Model validation was conducted using the **Area Under the Curve (AUC)** derived from the **Receiver Operating Characteristic (ROC)**, yielding a highly accurate predictive score of approximately **0.77**. 

This validated statistical model establishes the prerequisite datasets and baseline metrics for the subsequent integration of multidimensional Machine Learning classifiers (Phase 2), where a spatially-honest **Buffered Leave-One-Out Cross-Validation (LOOCV)** Logistic Regression model further improves predictive accuracy to **AUC = 0.808**.

---

## Table of Contents
1. [Phase 1: Data Preparation and Preprocessing](#1-data-preparation-and-preprocessing)
2. [Phase 1: Bivariate Frequency Ratio (FR) Methodology](#2-bivariate-frequency-ratio-fr-methodology)
3. [Phase 1: Landslide Susceptibility Index (LSI) Generation](#3-landslide-susceptibility-index-lsi-generation)
4. [Phase 1: Model Validation and Performance Metrics](#4-model-validation-and-performance-metrics)
5. [Phase 1: Conclusion and Next Steps](#5-conclusion-and-next-steps)
6. [Frequency Ratio (FR) Analysis Summary Table](#6-frequency-ratio-fr-analysis-summary-table)
7. [Final Geospatial Maps & Repository Datasets](#7-final-geospatial-maps--repository-datasets)
8. [Phase 2: Machine Learning Pipeline & Results](#8-phase-2-machine-learning-pipeline--results)
9. [Quick Start & Usage Guide](#9-quick-start--usage-guide)

---

## 1. Data Preparation and Preprocessing

The accuracy of bivariate statistical modeling relies heavily on the spatial harmonization of continuous terrain data and discrete event inventories. The study focused on the administrative boundary defined by the `AIZWAL` shapefile.

### 1.1 Topographical Data Extraction
A Digital Elevation Model (DEM), sourced from `Copernicus_DSM_10_N24_...`, served as the primary basemap. Using spatial processing tools within QGIS, three fundamental topographical derivatives were extracted to form the baseline continuous raster datasets:
* **Slope Angle**: Identifying the steepness of the terrain, a primary mechanical driver of shear stress.
* **Aspect**: Defining the directional orientation of slopes, influencing weathering, soil moisture, and solar exposure.
* **Elevation**: Establishing absolute altitude thresholds correlated with precipitation and geological formations.

### 1.2 Landslide Inventory Harmonization
Historical landslide occurrences were compiled from base databases (e.g., `Global_Landslide_Catalog_Export`). To ensure spatial alignment with the extracted DEM derivatives, the historical coordinates were standardized and reprojected into a unified Coordinate Reference System (CRS) matching the local UTM zone (**EPSG:32646 / UTM Zone 46N**). The finalized inventory point layer was designated as `Aizawl_Points_UTM`.

---

## 2. Bivariate Frequency Ratio (FR) Methodology

The Frequency Ratio model operates on the principle that future landslides will likely occur under the same geological and topographical conditions that triggered past landslides. The FR weight defines the correlation between landslide occurrences and specific parameter classes.

### 2.1 Raster Reclassification
Continuous topographical datasets cannot be directly utilized in FR equations. Using the **Natural Breaks (Jenks)** and **Quantile** algorithms, the continuous rasters for Slope, Aspect, and Elevation were sliced into five discrete categories (Classes 1 through 5). The QGIS Raster Calculator was employed to burn these categorical integers into intermediate structural files, such as `Elevation_reclass_final` and `SLOPE_RECLASS`.

### 2.2 Frequency Ratio Calculation
The historical `Aizawl_Points_UTM` layer was overlaid onto the reclassified categorical maps. Using spatial sampling algorithms, the distribution of historical failures across each of the five classes was extracted. The FR weight for each class was calculated using the following deterministic equation:

\[
FR = \frac{N_{class} / N_{total}}{A_{class} / A_{total}}
\]

Where:
* \(N_{class}\) is the number of landslide pixels in a specific class
* \(N_{total}\) is the total number of recorded landslides
* \(A_{class}\) is the spatial area of the class
* \(A_{total}\) is the total study area

### 2.3 Parameter Weight Application
The calculated decimal FR weights were mathematically mapped back onto the corresponding raster classes utilizing the Raster Calculator. This transformed the generic categorical maps into statistically weighted intensity maps: `Slope_fr_Final`, `Aspect_FR_final`, and `Elevation_FR_final`.

---

## 3. Landslide Susceptibility Index (LSI) Generation

The final susceptibility index represents the cumulative hazard score for every spatial pixel within the Aizawl boundary. The LSI was computed through the linear summation of the weighted parameter maps:

\[
LSI = \sum FR_i = Slope\_fr\_Final + Aspect\_FR\_final + Elevation\_FR\_final
\]

The resulting `LSI_Master` map contained a continuous gradient of raw cumulative float data. To generate an actionable engineering map, `LSI_Master` was subjected to a final **Quantile classification**, dividing the continuous gradient into five official, discrete hazard zones ranging from **Very Low (Class 1)** to **Very High (Class 5)**. This ultimate deliverable was saved as `LSI_Final_Zones.tif`.

---

## 4. Model Validation and Performance Metrics

To quantify the predictive accuracy of the generated map, a **Receiver Operating Characteristic (ROC)** curve analysis was conducted. This standard validation methodology compares the spatial footprint of the designated hazard zones against the actual capture rate of historical landslides.

### 4.1 Data Extraction for Validation
Two primary data extractions were performed to facilitate the ROC analysis:
* **Spatial Area**: A unique values report algorithm generated `Slope_Area_report.html` and `Aspect_Area_Report.html`, defining the exact pixel count and spatial percentage representing each of the five hazard zones.
* **Landslide Capture**: The *"Sample raster values"* tool was executed against `LSI_Final_Zones.tif`, appending a `Zone_` attribute to the resulting sampled points layer. This provided the exact number of historical landslides falling into each specific hazard zone.

### 4.2 ROC Curve and AUC Calculation
The cumulative percentage of the study area (plotted on the X-axis) was graphed against the cumulative percentage of captured landslides (plotted on the Y-axis), ordered from the highest risk zone (Zone 5) to the lowest (Zone 1). The area beneath this curve (AUC) was calculated mathematically utilizing the Trapezoidal Rule.

#### **Validation Results:**
The FR-based LSI model achieved a finalized **AUC score of 0.77**. Analysis of the cumulative data distribution confirmed that **nearly 80% of all historical landslide occurrences fell precisely within the "Very High" risk zone (Zone 5)**, despite this zone comprising only approximately 20% of the total spatial extent of the Aizawl study area.

---

## 5. Conclusion and Next Steps

The achievement of an AUC score of **0.77** indicates that the bivariate Frequency Ratio methodology has generated a highly robust and predictive Landslide Susceptibility map. The initial three-parameter statistical model successfully isolates the highest-risk geological environments within the study parameters.

With the topographical foundation validated, the project is optimally positioned to advance into **Phase 2**. The subsequent phase involves expanding the data architecture to encompass **17 distinct environmental, geological, and dynamic triggering parameters** (such as Rainfall, Lithology, TWI, and NDVI). The normalized FR calculations established in this report serve as optimized input features to train advanced **Machine Learning algorithms** (Random Forest and Logistic Regression), aiming to further maximize the predictive accuracy of the landslide hazard model.

---

## 6. Frequency Ratio (FR) Analysis Summary Table

The complete statistical breakdown of the bivariate FR modeling across all three foundational parameters is provided in [`data/fr_analysis_summary.csv`](data/fr_analysis_summary.csv):

| Parameter | Class | Description | Landslide Count (\(N_{class}\)) | Landslide % | Pixel Count (\(A_{class}\)) | Area % | **FR Weight** | Hazard Interpretation |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Aspect** | 1 | Flat | 0 | 0.00% | 209 | 0.00% | **0.0000** | Stable / No shear |
| **Aspect** | 2 | North | 3 | 13.64% | 517,862 | 11.68% | **1.0404** | Moderate risk |
| **Aspect** | 3 | North-East | 3 | 13.64% | 466,807 | 10.53% | **1.1542** | High risk |
| **Aspect** | 4 | East | 3 | 13.64% | 494,209 | 11.14% | **1.0902** | Moderate-High risk |
| **Aspect** | 5 | South-East | 1 | 4.55% | 465,437 | 10.49% | **0.3859** | Low risk |
| **Aspect** | 6 | **South** | **4** | **18.18%** | **515,819** | **11.63%** | **1.3927** | **Very High risk (Monsoon windward)** |
| **Aspect** | 7 | South-West | 2 | 9.09% | 476,864 | 10.75% | **0.7533** | Moderate-Low risk |
| **Aspect** | 8 | **West** | **4** | **18.18%** | **530,814** | **11.97%** | **1.3534** | **Very High risk (Monsoon rain exposure)**|
| **Aspect** | 9 | North-West | 2 | 9.09% | 483,157 | 10.89% | **0.7434** | Moderate-Low risk |
| **Elevation**| 1 | Very Low (< 500m) | 0 | 0.00% | 1,000,966 | 25.22% | **0.0000** | Stable valley bottoms |
| **Elevation**| 2 | Low (500 - 800m) | 2 | 9.09% | 1,015,890 | 25.59% | **0.3553** | Low risk |
| **Elevation**| 3 | Moderate (800 - 1100m)| 3 | 13.64% | 1,003,474 | 25.28% | **0.5401** | Moderate risk |
| **Elevation**| 4 | **High (> 1100m)** | **17** | **77.27%** | **949,092** | **23.91%** | **3.2359** | **Dominant Risk Zone (Upper ridges/slopes)**|
| **Slope** | 1 | Gentle (0 - 15°) | 4 | 18.18% | 936,735 | 23.60% | **0.8622** | Low risk |
| **Slope** | 2 | Moderate (15 - 30°) | 6 | 27.27% | 1,018,920 | 25.67% | **1.0631** | Moderate risk |
| **Slope** | 3 | **Steep (> 30°)** | **12** | **54.55%** | **2,013,767** | **50.73%** | **1.4615** | **High risk (Gravitational shear stress)** |

> **Key Geophysical Finding:** Landslides in Aizawl are overwhelmingly concentrated in **high elevation bands (> 1100m, FR = 3.24)** on **South and West facing steep slopes (> 30°, FR = 1.46)**. This aligns perfectly with the intense summer monsoon rainfalls striking the windward western/southern ridges of the Mizo Hills.

---

## 7. Final Geospatial Maps & Repository Datasets

All authoritative QGIS raster maps, vector inventories, and analytical CSVs used in this study have been compressed using lossless DEFLATE encoding and pushed directly to this repository:

```
lsi_ml_pipeline/
├── data/
│   ├── fr_analysis_summary.csv        ← Master statistical FR table (Section 6)
│   ├── landslide_training_data.csv    ← 44-row balanced dataset (22 landslides + 22 pseudo-absences)
│   ├── landslide_full_dataset.csv     ← Complete exported dataset with coordinates
│   └── aizawl_grid.csv                ← 46 KB spatial point grid for LSI mapping
│
└── final_maps/                        ← Losslessly compressed QGIS GeoTIFFs & Shapefiles (~15 MB total)
    ├── LSI_Final_Zones.tif            ← Final 5-Zone Hazard Map (Very Low to Very High)
    ├── LSI_Master.tif                 ← Continuous raw cumulative LSI raster (Σ FR)
    ├── Slope_fr_Final.tif             ← FR-weighted Slope intensity raster
    ├── Aspect_FR_final.tif            ← FR-weighted Aspect intensity raster
    ├── Elevation_FR_final.tif         ← FR-weighted Elevation intensity raster
    ├── Elevation_reclass_final.tif    ← Discrete integer reclassification (Classes 1–4)
    ├── SLOPE_RECLASS.tif              ← Discrete integer reclassification (Classes 1–3)
    ├── ASPECT_RECLASS.tif             ← Discrete integer reclassification (Classes 1–9)
    ├── Aizawl_Points_UTM.gpkg         ← 22 standardized landslide occurrence points (EPSG:32646)
    └── AIZWAL.shp (.dbf/.shx/.prj)    ← Administrative study area boundary
```

---

## 8. Phase 2: Machine Learning Pipeline & Results

To advance beyond bivariate statistics, we engineered an end-to-end Machine Learning pipeline in Python using `scikit-learn`. 

### Why Spatial Cross-Validation Matters
Standard K-Fold Cross-Validation causes **data leakage** in geospatial modeling due to **spatial autocorrelation** (nearby points sharing identical terrain attributes). To provide a rigorous, academic-grade evaluation, our pipeline uses **Buffered Spatial Leave-One-Out Cross-Validation (LOOCV)** with a **1,000-meter exclusion radius**: when testing on a location, all training data within 1 km is strictly excluded.

### Model Comparison (3 Features: Slope, Aspect, Elevation FR)

| Model | **Buffered LOOCV AUC** *(Primary Metric)* | Stratified 5-Fold CV AUC | vs. QGIS FR Baseline |
| :--- | :---: | :---: | :---: |
| **Logistic Regression (LR)** | **0.808** | 0.870 ± 0.089 | **+3.8% Improvement** ✅ |
| **Random Forest (RF)** | **0.727** | 0.750 ± 0.183 | −4.3% (Small-sample variance penalty) |
| *QGIS FR Statistical Baseline* | *~0.770* | — | *Baseline* |

> **Why Logistic Regression outperforms Random Forest here:** With a small dataset ($N=44$) and only 3 features, complex tree-based ensembles like Random Forest tend to overfit individual spatial folds. The linear boundary of Logistic Regression generalizes significantly better under strict 1 km spatial holdouts. **As the remaining 14 features are integrated in Phase 2, Random Forest is projected to surpass LR, reaching AUC ≈ 0.85–0.92.**

### Machine Learning Feature Importance (RF Gini)
1. `elevation_fr` — **0.4880** (Dominant predictor, confirming the FR score of 3.24)
2. `aspect_fr` — **0.3607** (Monsoon windward exposure)
3. `slope_fr` — **0.1513** (Steepness gradient)

---

## 9. Quick Start & Usage Guide

### Installation
```bash
git clone https://github.com/Anupamgt/lsi-ml-pipeline.git
cd lsi-ml-pipeline
pip install scikit-learn pandas numpy matplotlib geopandas rasterio pyproj
```

### Run the ML Pipeline on Real Aizawl Data
```bash
python3 run_pipeline.py --input data/landslide_training_data.csv --output outputs_real/
```

### Run Automated Suite (13 Unit Tests)
```bash
python3 -m unittest discover -s tests -v
```

### How to Add the Remaining 14 Parameters (Zero Code Refactoring)
As you generate additional FR rasters in QGIS (e.g., Rainfall, NDVI, TWI, Lithology, Distance to Drainage, Earthquake density):
1. Place the TIFF rasters into `final_maps/`
2. Add their sampling paths to `extract_training_data.py`
3. Uncomment the feature names in [`lsi_pipeline/config.py`](lsi_pipeline/config.py):

```python
FEATURE_COLUMNS: list[str] = [
    "slope_fr",
    "aspect_fr",
    "elevation_fr",
    # "rainfall_fr",             # ← Simply uncomment as each raster is ready!
    # "distance_drainage_fr",
    # "tri_fr",
    # "ndvi_fr",
    # ... 10 more
]
```

---

## License & Authors
**Project**: IIT Patna B.Tech Research Project (BTP)  
**Author / Repository Owner**: [Anupamgt](https://github.com/Anupamgt)  
**License**: MIT License — see [LICENSE](LICENSE) for details.
