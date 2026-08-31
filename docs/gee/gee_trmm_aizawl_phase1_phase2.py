"""
LSI_btp — Aizawl rainfall from Google Earth Engine (Python / earthengine-api)

Same products, dates, unit conversion, and exports as
gee_trmm_aizawl_phase1_phase2.js

Run in Colab or locally after:
  pip install earthengine-api
  earthengine authenticate

Then:
  python gee_trmm_aizawl_phase1_phase2.py

Tasks appear in https://code.earthengine.google.com/ → Tasks
(and in Drive folder LSI_btp_rainfall once you start them).
"""

import ee

ee.Initialize()

# ---------------------------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------------------------
DRIVE_FOLDER = "LSI_btp_rainfall"

PHASE1_START = "1998-01-01"
PHASE1_END = "2019-01-01"  # exclusive; includes all of 2018

PHASE2_START = "2019-01-01"
PHASE2_END = "2025-09-01"

IMERG_LONG_START = "2000-06-01"
IMERG_LONG_END = "2025-09-01"

EXPORT_NATIVE = True
EXPORT_30M = True
EXPORT_CLIM12 = True

WEST, SOUTH, EAST, NORTH = 92.55, 23.31, 93.21, 24.41
BUFFER_DEG = 0.1

TRMM_SCALE_M = 27830
IMERG_SCALE_M = 11132
UTM_CRS = "EPSG:32646"
UTM_SCALE = 30
MAX_PIXELS = int(1e13)

# ---------------------------------------------------------------------------
# AOI
# ---------------------------------------------------------------------------
aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], "EPSG:4326", False)
aoi_buf = aoi.buffer(BUFFER_DEG * 111320)

print("AOI (W,S,E,N):", WEST, SOUTH, EAST, NORTH)
print("Phase 1 TRMM (end exclusive):", PHASE1_START, PHASE1_END)
print("Phase 2 IMERG (end exclusive):", PHASE2_START, PHASE2_END)
print(
    "ASSUMPTION: Phase 1 = paper-overlap TRMM 1998–2018; "
    "Phase 2 = IMERG 2019–latest (TRMM ended 2019-12)."
)


def to_monthly_mm(img):
    """precipitation (mm/hr monthly mean rate) → mm in that calendar month."""
    start = ee.Date(img.get("system:time_start"))
    hours = start.advance(1, "month").difference(start, "hour")
    return (
        img.select("precipitation")
        .multiply(hours)
        .rename("precip_mm")
        .copyProperties(img, ["system:time_start"])
        .set("hours_in_month", hours)
    )


def monthly_collection(col_id, start, end_exclusive):
    return (
        ee.ImageCollection(col_id)
        .select("precipitation")
        .filterDate(start, end_exclusive)
        .filterBounds(aoi_buf)
        .map(to_monthly_mm)
    )


def year_list(start, end_exclusive):
    y0 = ee.Date(start).get("year")
    y1 = ee.Date(end_exclusive).advance(-1, "day").get("year")
    return ee.List.sequence(y0, y1)


def annual_collection(monthly_col, start, end_exclusive):
    def one_year(y):
        y = ee.Number(y)
        y0 = ee.Date.fromYMD(y, 1, 1)
        y1 = y0.advance(1, "year")
        n = monthly_col.filterDate(y0, y1).size()
        return (
            monthly_col.filterDate(y0, y1)
            .sum()
            .rename("annual_mm")
            .set("year", y)
            .set("n_months", n)
            .set("system:time_start", y0.millis())
        )

    return ee.ImageCollection.fromImages(year_list(start, end_exclusive).map(one_year))


def monsoon_collection(monthly_col, start, end_exclusive):
    def one_year(y):
        y = ee.Number(y)
        y0 = ee.Date.fromYMD(y, 1, 1)
        y1 = y0.advance(1, "year")
        return (
            monthly_col.filterDate(y0, y1)
            .filter(ee.Filter.calendarRange(6, 9, "month"))
            .sum()
            .rename("monsoon_mm")
            .set("year", y)
            .set("system:time_start", y0.millis())
        )

    return ee.ImageCollection.fromImages(year_list(start, end_exclusive).map(one_year))


def monthly_climatology_12(monthly_col):
    def one_month(m):
        m = ee.Number(m)
        name = ee.String("mm_").cat(m.format("%02d"))
        return monthly_col.filter(ee.Filter.calendarRange(m, m, "month")).mean().rename(name)

    images = ee.List.sequence(1, 12).map(one_month)
    return ee.ImageCollection.fromImages(images).toBands().rename(
        [
            "mm_01",
            "mm_02",
            "mm_03",
            "mm_04",
            "mm_05",
            "mm_06",
            "mm_07",
            "mm_08",
            "mm_09",
            "mm_10",
            "mm_11",
            "mm_12",
        ]
    )


def print_minmax(img, label, scale):
    stats = img.reduceRegion(
        reducer=ee.Reducer.minMax().combine(reducer2=ee.Reducer.mean(), sharedInputs=True),
        geometry=aoi,
        scale=scale,
        maxPixels=int(1e9),
        bestEffort=True,
    )
    print(label, stats.getInfo())


def export_native(img, prefix, scale):
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=prefix + "_epsg4326",
        folder=DRIVE_FOLDER,
        fileNamePrefix=prefix + "_epsg4326",
        region=aoi_buf,
        crs="EPSG:4326",
        scale=scale,
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
    )
    task.start()
    print("started", prefix + "_epsg4326")


def export_utm30(img, prefix):
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(),
        description=prefix + "_utm46n_30m",
        folder=DRIVE_FOLDER,
        fileNamePrefix=prefix + "_utm46n_30m",
        region=aoi_buf,
        crs=UTM_CRS,
        scale=UTM_SCALE,
        maxPixels=MAX_PIXELS,
        fileFormat="GeoTIFF",
    )
    task.start()
    print("started", prefix + "_utm46n_30m")


def export_pair(img, prefix, native_scale):
    if EXPORT_NATIVE:
        export_native(img, prefix, native_scale)
    if EXPORT_30M:
        export_utm30(img, prefix)


# ---------------------------------------------------------------------------
# Phase 1 — TRMM 3B43, paper-overlap years
# ---------------------------------------------------------------------------
p1_monthly = monthly_collection("TRMM/3B43V7", PHASE1_START, PHASE1_END)
print("Phase 1 TRMM monthly count:", p1_monthly.size().getInfo())

p1_annual = (
    annual_collection(p1_monthly, PHASE1_START, PHASE1_END)
    .mean()
    .clip(aoi_buf)
    .rename("phase1_trmm_mean_annual_mm")
)
p1_monsoon = (
    monsoon_collection(p1_monthly, PHASE1_START, PHASE1_END)
    .mean()
    .clip(aoi_buf)
    .rename("phase1_trmm_mean_jjas_mm")
)
p1_clim12 = monthly_climatology_12(p1_monthly).clip(aoi_buf)

print_minmax(p1_annual, "Phase 1 TRMM mean annual mm/year", TRMM_SCALE_M)
print_minmax(p1_monsoon, "Phase 1 TRMM mean JJAS mm", TRMM_SCALE_M)

# ---------------------------------------------------------------------------
# Phase 2 — IMERG Monthly V07 (explicit TRMM substitute)
# ---------------------------------------------------------------------------
print("PHASE 2 SUBSTITUTION: NASA/GPM_L3/IMERG_MONTHLY_V07 (TRMM/3B43V7 ends 2019-12).")
p2_monthly = monthly_collection("NASA/GPM_L3/IMERG_MONTHLY_V07", PHASE2_START, PHASE2_END)
print("Phase 2 IMERG monthly count:", p2_monthly.size().getInfo())

p2_annual = (
    annual_collection(p2_monthly, PHASE2_START, PHASE2_END)
    .mean()
    .clip(aoi_buf)
    .rename("phase2_imerg_mean_annual_mm")
)
p2_monsoon = (
    monsoon_collection(p2_monthly, PHASE2_START, PHASE2_END)
    .mean()
    .clip(aoi_buf)
    .rename("phase2_imerg_mean_jjas_mm")
)
p2_clim12 = monthly_climatology_12(p2_monthly).clip(aoi_buf)

print_minmax(p2_annual, "Phase 2 IMERG mean annual mm/year", IMERG_SCALE_M)
print_minmax(p2_monsoon, "Phase 2 IMERG mean JJAS mm", IMERG_SCALE_M)

# Optional long IMERG climatology
long_monthly = monthly_collection("NASA/GPM_L3/IMERG_MONTHLY_V07", IMERG_LONG_START, IMERG_LONG_END)
print("IMERG long monthly count:", long_monthly.size().getInfo())
long_annual = (
    annual_collection(long_monthly, IMERG_LONG_START, IMERG_LONG_END)
    .mean()
    .clip(aoi_buf)
    .rename("imerg_long_mean_annual_mm")
)
long_monsoon = (
    monsoon_collection(long_monthly, IMERG_LONG_START, IMERG_LONG_END)
    .mean()
    .clip(aoi_buf)
    .rename("imerg_long_mean_jjas_mm")
)
print_minmax(long_annual, "IMERG long mean annual mm/year", IMERG_SCALE_M)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
export_pair(p1_annual, "phase1_TRMM3B43V7_meanAnnual_mmYr_1998_2018", TRMM_SCALE_M)
export_pair(p1_monsoon, "phase1_TRMM3B43V7_meanJJAS_mm_1998_2018", TRMM_SCALE_M)
export_pair(p2_annual, "phase2_IMERG_V07_meanAnnual_mmYr_2019_2025", IMERG_SCALE_M)
export_pair(p2_monsoon, "phase2_IMERG_V07_meanJJAS_mm_2019_2025", IMERG_SCALE_M)
export_pair(long_annual, "optional_IMERG_V07_meanAnnual_mmYr_2000_2025", IMERG_SCALE_M)
export_pair(long_monsoon, "optional_IMERG_V07_meanJJAS_mm_2000_2025", IMERG_SCALE_M)

if EXPORT_CLIM12:
    export_native(p1_clim12, "phase1_TRMM3B43V7_monthlyClim12_mm_1998_2018", TRMM_SCALE_M)
    export_native(p2_clim12, "phase2_IMERG_V07_monthlyClim12_mm_2019_2025", IMERG_SCALE_M)

print("Exports started. Monitor at https://code.earthengine.google.com/ (Tasks).")
print("Drive folder:", DRIVE_FOLDER)
print(
    "Primary FR factor = mean annual mm/year. "
    "Sonker used IMD 1988–2018; Phase 1 TRMM covers 1998–2018 only."
)
