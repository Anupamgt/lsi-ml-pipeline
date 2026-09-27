# Aizawl stack — Stage A band constraint

Source stack: `outputs_vlm/aizawl_stack.tif` (21 bands). Stats: `outputs_vlm/aizawl_stack_stats.json`.

## Continuous vs class codes

Of **21** stack bands, only **6 are continuous**:

| Band name | Kind |
|---|---|
| `rainfall_mm` | continuous |
| `gravity_mgal` | continuous |
| `slope_final_deg` | continuous |
| `elevation_m` | continuous |
| `hillshade` | continuous (viz aid) |
| `slope_dem_deg` | derived continuous |

The other **15** bands are **integer class codes** (not continuous physical values, not one-hot encodings):

`earthquake_class`, `slope_class`, `altitude_class`, `drainage_dist_class`, `tri_class`, `geomorphology_class`, `geology_class`, `soil_class`, `faults_dist_class`, `sti_class`, `twi_class`, `spi_class`, `roads_dist_class`, `lulc_class`, `ndvi_class`.

Phase 0 already lists missing continuous rebuilds in `aizawl_stack_stats.json` → `missing_continuous_rebuild` and in `audit.md`. That inventory is sufficient for planning rebuilds.

## Stage A prompt rule

**Stage A prompts must use class names/labels for those 15 class bands** until TWI, SPI, STI, TRI, distance layers, and NDVI are rebuilt as continuous. Do not treat class integers as metres, indices, or NDVI floats in the prompt text.
