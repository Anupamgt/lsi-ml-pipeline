/**
 * LSI_btp — Aizawl mean rainfall from CHIRPS daily (Google Earth Engine)
 * Paste into https://code.earthengine.google.com/  →  Run  →  Tasks tab → RUN
 *
 * ---------------------------------------------------------------------------
 * FIRST LANDSLIDE DATE (START)
 * ---------------------------------------------------------------------------
 * START_DATE = 2009-10-05
 *
 * Source (inventory in use, 22 points):
 *   QGIS layer  Aizawl_Points_UTM
 *   file        C:\Users\sharm\lsi-ml-pipeline\final_maps\Aizawl_Points_UTM.gpkg
 *   field       event_date
 *   earliest    10/05/2009 12:00:00 AM
 *   event_id    1222
 *   place       Chhinga Veng, Aizawl  (NASA GLC / PTI; 3 killed)
 *   lon/lat     92.72124394, 23.73478778
 *
 * Same 22 records also live as Analysis_layer and
 * Reprojected_LANDSLIDE_layers_UTM (same event_date field).
 * 00_drop_raw\01_landslide_inventory is empty.
 *
 * NASA GLC worldwide CSV has an older bbox hit on 2007-09-29
 * (NH-45, event_id 285) that is NOT in the 22-point FR inventory.
 * This script uses OUR 22-point layer, not that extra GLC point.
 *
 * END_DATE = 2018-12-31  (user: data till 2018)
 *
 * Incomplete first year: 2009 only has 5 Oct–31 Dec. Yearly totals for
 * 2009 would be ~3 months and would bias mean annual LOW, so the FR
 * raster averages complete calendar years 2010–2018. Toggle
 * INCLUDE_PARTIAL_YEARS if you want 2009 included anyway.
 *
 * Product: UCSB-CHG/CHIRPS/DAILY  band precipitation = mm/day (depth, not rate).
 *   annual_mm(year) = sum of daily mm in that year
 *   mean annual     = mean of those yearly totals
 *   nodata          = CHIRPS fill is already masked in GEE; do not unmask to 0
 * Pentad (UCSB-CHG/CHIRPS/PENTAD) is used only if the daily collection is empty.
 *
 * After Drive download, drop the TIF into:
 *   C:\Users\sharm\LSI_btp_sikkim_replication\00_drop_raw\03_rainfall_imd\
 */

// =============================================================================
// 1. USER SETTINGS
// =============================================================================

var START_DATE = '2009-10-05';  // first Aizawl_Points_UTM event_date
var END_DATE   = '2018-12-31';  // inclusive; GEE filter uses day after
var INCLUDE_PARTIAL_YEARS = false;  // false = skip 2009 (incomplete)
var MIN_DAYS_FOR_COMPLETE_YEAR = 330;

var DRIVE_FOLDER = 'LSI_btp_rainfall';

var EXPORT_CRS   = 'EPSG:32646';
var EXPORT_SCALE = 30;

// master_dem_30m.tif (confirmed in QGIS): 461520,2578020 → 521610,2700000
var MASTER_XMIN = 461520;
var MASTER_YMIN = 2578020;
var MASTER_XMAX = 521610;
var MASTER_YMAX = 2700000;
var MASTER_TRANSFORM = [30, 0, MASTER_XMIN, 0, -30, MASTER_YMAX];  // 2003 x 4066

var MAX_PIXELS = 1e13;

// AIZWAL polygon is EPSG:4326 MultiPolygon (420 vertices). At CHIRPS 0.05°
// (~5.5 km) the district envelope is the practical clip.
// QGIS AIZWAL extent: 92.6238, 23.3117, 93.2111, 24.4129
var WEST  = 92.6238;
var SOUTH = 23.3117;
var EAST  = 93.2111;
var NORTH = 24.4129;
var BUFFER_DEG = 0.02;

var CHIRPS_DAILY  = 'UCSB-CHG/CHIRPS/DAILY';
var CHIRPS_PENTAD = 'UCSB-CHG/CHIRPS/PENTAD';
var CHIRPS_NATIVE_M = 5566;

// =============================================================================
// 2. AOI AND DATES
// =============================================================================

var aoiWgs = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], 'EPSG:4326', false);
var aoiBuf = aoiWgs.buffer(BUFFER_DEG * 111320);
var masterRect = ee.Geometry.Rectangle(
    [MASTER_XMIN, MASTER_YMIN, MASTER_XMAX, MASTER_YMAX], EXPORT_CRS, false);

var start = ee.Date(START_DATE);
var endExclusive = ee.Date(END_DATE).advance(1, 'day');  // include 2018-12-31

Map.centerObject(aoiWgs, 9);
Map.addLayer(aoiWgs, {color: 'black'}, 'AIZWAL bbox (EPSG:4326)', true);
Map.addLayer(masterRect, {color: 'red'}, 'master_dem_30m extent (UTM 46N)', false);

print('START (first landslide)', START_DATE,
      '— Aizawl_Points_UTM.event_date, event_id 1222, Chhinga Veng, 2009-10-05');
print('END (inclusive)', END_DATE);
print('Export CRS/scale', EXPORT_CRS, EXPORT_SCALE);
print('Master grid', MASTER_XMIN, MASTER_YMIN, MASTER_XMAX, MASTER_YMAX);

// =============================================================================
// 3. COLLECTION (daily preferred; pentad fallback)
// =============================================================================

var daily = ee.ImageCollection(CHIRPS_DAILY)
    .select('precipitation')
    .filterDate(start, endExclusive)
    .filterBounds(aoiBuf);

var pentad = ee.ImageCollection(CHIRPS_PENTAD)
    .select('precipitation')
    .filterDate(start, endExclusive)
    .filterBounds(aoiBuf);

var nDaily = daily.size();
print('CHIRPS daily image count (expect ~3300–3400)', nDaily);

// precipitation is already mm for the time step (mm/day or mm/pentad). Sum = mm.
var precip = ee.ImageCollection(ee.Algorithms.If(
    nDaily.gt(0), daily, pentad));
var productUsed = ee.String(ee.Algorithms.If(
    nDaily.gt(0), CHIRPS_DAILY, CHIRPS_PENTAD + ' (FALLBACK — daily empty)'));
print('Product used', productUsed);

// =============================================================================
// 4. YEARLY TOTALS → MEAN ANNUAL  /  MEAN JJAS
// =============================================================================

var y0 = start.get('year');
var y1 = ee.Date(END_DATE).get('year');
var years = ee.List.sequence(y0, y1);

function yearWindow(y) {
  y = ee.Number(y);
  var yStart = ee.Date.fromYMD(y, 1, 1);
  var yEnd   = yStart.advance(1, 'year');
  // clip to [START, END]
  var w0 = ee.Date(ee.Algorithms.If(yStart.millis().lt(start.millis()), start, yStart));
  var w1 = ee.Date(ee.Algorithms.If(yEnd.millis().gt(endExclusive.millis()), endExclusive, yEnd));
  return {y: y, w0: w0, w1: w1};
}

function annualImage(y) {
  var w = yearWindow(y);
  var col = precip.filterDate(w.w0, w.w1);
  var n = col.size();
  return col.sum()
      .rename('annual_mm')
      .set('year', w.y)
      .set('n_images', n)
      .set('system:time_start', w.w0.millis());
}

function monsoonImage(y) {
  y = ee.Number(y);
  var yStart = ee.Date.fromYMD(y, 1, 1);
  var yEnd   = yStart.advance(1, 'year');
  var col = precip
      .filterDate(yStart, yEnd)
      .filter(ee.Filter.calendarRange(6, 9, 'month'));
  return col.sum()
      .rename('monsoon_mm')
      .set('year', y)
      .set('n_images', col.size())
      .set('system:time_start', yStart.millis());
}

var annualCol = ee.ImageCollection.fromImages(years.map(annualImage));
var monsoonCol = ee.ImageCollection.fromImages(years.map(monsoonImage));

var annualComplete = annualCol.filter(ee.Filter.gte('n_images', MIN_DAYS_FOR_COMPLETE_YEAR));
var monsoonComplete = monsoonCol.filter(ee.Filter.gte('n_images', 100));  // ~122 JJAS days

var annualForMean = ee.ImageCollection(ee.Algorithms.If(
    INCLUDE_PARTIAL_YEARS, annualCol, annualComplete));
var monsoonForMean = ee.ImageCollection(ee.Algorithms.If(
    INCLUDE_PARTIAL_YEARS, monsoonCol, monsoonComplete));

print('Years in annual mean (complete unless INCLUDE_PARTIAL_YEARS)',
      annualForMean.aggregate_array('year'));
print('n_images per year (all years, including partial 2009)',
      annualCol.aggregate_array('year'),
      annualCol.aggregate_array('n_images'));

var meanAnnual = annualForMean.mean().clip(aoiBuf).rename('chirps_mean_annual_mm');
var meanMonsoon = monsoonForMean.mean().clip(aoiBuf).rename('chirps_mean_jjas_mm');

// Bilinear resample to 30 m (CHIRPS native is ~0.05°). Looks smooth — expected.
var meanAnnual30 = meanAnnual.resample('bilinear').reproject({
  crs: EXPORT_CRS,
  crsTransform: MASTER_TRANSFORM
});
var meanMonsoon30 = meanMonsoon.resample('bilinear').reproject({
  crs: EXPORT_CRS,
  crsTransform: MASTER_TRANSFORM
});

// =============================================================================
// 5. SANITY CHECK + MAP
// =============================================================================

function printStats(img, label, scale, geom) {
  var s = img.reduceRegion({
    reducer: ee.Reducer.minMax().combine({
      reducer2: ee.Reducer.mean(),
      sharedInputs: true
    }),
    geometry: geom,
    scale: scale,
    maxPixels: 1e9,
    bestEffort: true
  });
  print(label + '  (Aizawl should be ~1500–4000 mm/yr, not 0 and not 50000)', s);
}

printStats(meanAnnual, 'Mean annual mm/year (CHIRPS native scale)', CHIRPS_NATIVE_M, aoiWgs);
printStats(meanMonsoon, 'Mean JJAS mm (CHIRPS native scale)', CHIRPS_NATIVE_M, aoiWgs);

var palAnnual = {
  min: 1500, max: 4000,
  palette: ['#f7fcf0', '#ccebc5', '#7bccc4', '#2b8cbe', '#084081']
};
var palMonsoon = {
  min: 1000, max: 3200,
  palette: ['#ffffcc', '#a1dab4', '#41b6c4', '#2c7fb8', '#253494']
};

Map.addLayer(meanAnnual, palAnnual, 'CHIRPS mean annual mm/year 2010-2018', true);
Map.addLayer(meanMonsoon, palMonsoon, 'CHIRPS mean JJAS mm 2010-2018', false);

print('CHIRPS units: precipitation band is mm/day. Yearly total = sum of daily images.');
print('Nodata: GEE already masks CHIRPS fill values. Do not unmask(0).');
print('After download drop TIF in C:\\Users\\sharm\\LSI_btp_sikkim_replication\\00_drop_raw\\03_rainfall_imd\\');

// =============================================================================
// 6. EXPORTS  (EPSG:32646, 30 m, master-grid transform)
// =============================================================================

var annualName = 'chirps_mean_annual_mm_2009_2018_aizawl_30m_utm46n';
var monsoonName = 'chirps_mean_jjas_mm_2009_2018_aizawl_30m_utm46n';

Export.image.toDrive({
  image: meanAnnual30.toFloat(),
  description: annualName,
  folder: DRIVE_FOLDER,
  fileNamePrefix: annualName,
  region: masterRect,
  crs: EXPORT_CRS,
  crsTransform: MASTER_TRANSFORM,
  maxPixels: MAX_PIXELS,
  fileFormat: 'GeoTIFF'
});

Export.image.toDrive({
  image: meanMonsoon30.toFloat(),
  description: monsoonName,
  folder: DRIVE_FOLDER,
  fileNamePrefix: monsoonName,
  region: masterRect,
  crs: EXPORT_CRS,
  crsTransform: MASTER_TRANSFORM,
  maxPixels: MAX_PIXELS,
  fileFormat: 'GeoTIFF'
});

print('Exports queued. Code Editor → Tasks → RUN each task. Allow Drive access if asked.');
print('Primary FR factor =', annualName + '.tif');
