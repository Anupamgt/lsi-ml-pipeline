# data/

This directory holds input CSV files and Frequency Ratio (FR) lookup tables for the LSI ML pipeline.

## Required files

| File | How to generate / source |
|------|----------------|
| `landslide_training_data.csv` | `python extract_training_data.py` (samples all 17 Sonker class GeoTIFFs → raw FR) — preferred. Or `python generate_sample_data.py --with-negatives` for synthetic tests. |
| `aizawl_grid.csv` | `python extract_training_data.py --grid` (strided sampling of Sonker class rasters → FR). Synthetic fallback: `python generate_sample_data.py --grid`. |
| `fr_class_table_17factor.csv` | Authoritative 17-factor class→FR table (column `FR`). Used by extraction and grid regen. |
| `fr_results_17factor.json` | Full 17-factor FR analysis JSON companion to the class table. |
| `fr_analysis_summary.csv` | **Phase-1 only** (3-parameter Aspect / Elevation / Slope). Kept for historical Section 6 of the main README — not used by the ML pipeline. Prefer `fr_class_table_17factor.csv` for current work. |

## Training schema (17 Sonker FR factors)

`landslide_training_data.csv` columns:

```
x, y,
rainfall_fr, earthquake_fr, slope_fr, elevation_fr, distance_drainage_fr,
tri_fr, geomorphology_fr, geology_fr, soil_fr, gravity_anomaly_fr,
distance_faults_fr, sti_fr, twi_fr, spi_fr, distance_roads_fr,
lulc_fr, ndvi_fr,
target
```

- `x`, `y` — UTM Easting/Northing in metres (**EPSG:32646**)
- `*_fr` — raw Frequency Ratio for each of the 17 active factors (no `aspect_fr`)
- `target` — `1` (landslide) or `0` (pseudo-absence)

Column order matches `FEATURE_COLUMNS` in `lsi_pipeline/config.py` / `feature_registry.py`.

## Grid schema

`aizawl_grid.csv` uses the **same 17 FR columns** plus coordinates, **without** `target`:

```
x, y, <all FEATURE_COLUMNS>
```

## FR lookup tables

| Table | Role |
|-------|------|
| [`fr_class_table_17factor.csv`](fr_class_table_17factor.csv) | **Use this** for class→FR mapping (`factor`, `class`, `FR`). |
| [`fr_analysis_summary.csv`](fr_analysis_summary.csv) | Phase-1 3-factor summary (Aspect / Elevation / Slope). Historical reference only. |
