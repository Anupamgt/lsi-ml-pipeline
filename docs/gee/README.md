# Google Earth Engine rainfall scripts (Aizawl LSI)

Paste the `.js` files into https://code.earthengine.google.com/ then Run and start the Drive export tasks.

| Script | Product | Window used for FR rainfall |
|---|---|---|
| `gee_chirps_aizawl_2009_2018.js` | CHIRPS daily (`UCSB-CHG/CHIRPS/DAILY`) | Mean annual mm for **complete years 2010-2018** (2009 starts 5 Oct with the first inventory date, so it is skipped unless `INCLUDE_PARTIAL_YEARS` is true) |
| `gee_trmm_aizawl_phase1_phase2.js` | TRMM 3B43 (1998-2018) + IMERG monthly (2019+) | Optional alternative / comparison. Paper used IMD 1988-2018. |
| `gee_trmm_aizawl_phase1_phase2.py` | Same as the TRMM JS | Earth Engine Python API equivalent |

Exports target EPSG:32646 at 30 m on the `master_dem_30m` grid (461520, 2578020 to 521610, 2700000).

**Not in this repo:** the exported GeoTIFFs. After Drive download they live locally at `C:\Users\sharm\LSI_btp_sikkim_replication\00_drop_raw\03_rainfall_imd\` and the warped 30 m mean annual raster is `01_processed\chirps_mean_annual_30m.tif`. Those rasters are hundreds of MB and are not committed.
