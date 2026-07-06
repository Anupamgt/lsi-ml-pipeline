# data/

This directory holds input CSV files for the LSI ML pipeline.

## Required files (not committed — generated locally)

| File | How to generate |
|------|----------------|
| `landslide_training_data.csv` | Run `python extract_training_data.py` (requires QGIS FR rasters) OR `python generate_sample_data.py` for synthetic test data |
| `aizawl_grid.csv` | Export a regular point grid from QGIS covering the study area, with all FR feature columns |

## Schema

`landslide_training_data.csv` columns:

```
x, y, slope_fr, aspect_fr, elevation_fr, [additional_fr_features...], target
```

- `x`, `y` — UTM Easting/Northing in metres (EPSG:32646)
- `*_fr` — Frequency Ratio value for each parameter
- `target` — 1 (landslide) or 0 (non-landslide / pseudo-absence)
