/**
 * LSI_btp — Aizawl rainfall from Google Earth Engine
 * Sonker et al. (2022) Frequency Ratio replication (applied to Aizawl, Mizoram)
 *
 * Paper rainfall: IMD Pune *daily* gauges, 1988–2018, mapped as a rainfall factor
 * (Fig. 2a). Typical FR use = long-term mean annual rainfall (mm/year).
 *
 * This script does NOT silently replace IMD with TRMM. It fetches satellite
 * precipitation as an alternative when IMD NetCDF is not available.
 *
 * ---------------------------------------------------------------------------
 * PHASE DATES (edit here only)
 * ---------------------------------------------------------------------------
 * Project notes do not date Phase 1 vs Phase 2 as rainfall windows.
 * They define:
 *   Phase 1 = 3-factor terrain FR baseline (slope / aspect / elevation)
 *   Phase 2 = 17-factor Sonker stack + ML (rainfall is factor #1)
 *
 * Rainfall-year assumption used below:
 *   PHASE 1 = TRMM 3B43 climatology overlapping the paper window as far as
 *             TRMM exists: 1998-01-01 → 2018-12-31.
 *             TRMM starts 1998; 1988–1997 cannot be filled with TRMM.
 *   PHASE 2 = GPM IMERG Monthly V07 after TRMM ends, covering recent inventory
 *             years (Zenodo Aizawl events 2016–2025): 2019-01-01 → latest.
 *             Product substitution is labeled in filenames and print() output.
 *
 * Extra (optional): IMERG long climatology 2000-06 → latest as a single
 * consistent 17-factor rainfall raster (0.1°, recommended if you want one map).
 *
 * GEE collections:
 *   TRMM/3B43V7                      1998-01 → 2019-12  (~0.25°, mm/hr)
 *   NASA/GPM_L3/IMERG_MONTHLY_V07    ~2000-06 → present (~0.1°, mm/hr)
 *
 * Unit conversion (BOTH products):
 *   band `precipitation` is a monthly-mean RATE in mm/hour.
 *   mm/month = rate_mm_hr * hours_in_that_month
 *   (hours from image start to start+1 month, so leap Februaries are correct)
 *   annual mm  = sum of 12 monthly totals in that year
 *   mean annual = mean of those yearly sums
 *
 * After Drive download: clip to AIZWAL polygon and snap/resample to the
 * 30 m EPSG:32646 master grid in QGIS (native TRMM stays ~0.25°).
 *
 * How to run: paste into https://code.earthengine.google.com/ → Run
 * Then open the Tasks tab and click RUN on each export.
 */

// =============================================================================
// 1. USER SETTINGS — edit dates / Drive folder here
// =============================================================================

var DRIVE_FOLDER = 'LSI_btp_rainfall';

// Exclusive end dates (GEE filterDate end is exclusive)
var PHASE1_START = '1998-01-01';
var PHASE1_END   = '2019-01-01';   // includes all of 2018 (paper end year)

var PHASE2_START = '2019-01-01';
var PHASE2_END   = '2025-09-01';   // IMERG monthly currently through ~2025-08

// Optional long IMERG climatology (one consistent product for 17-factor FR)
var IMERG_LONG_START = '2000-06-01';
var IMERG_LONG_END   = '2025-09-01';

var EXPORT_NATIVE = true;   // EPSG:4326 at native satellite resolution
var EXPORT_30M    = true;   // optional master-grid-ready EPSG:32646 / 30 m
var EXPORT_CLIM12 = true;   // 12-band mean monthly climatology (mm)

// Aizawl rectangle (WGS84). User box; notes also list west 92.62.
var WEST  = 92.55;
var SOUTH = 23.31;
var EAST  = 93.21;
var NORTH = 24.41;
var BUFFER_DEG = 0.1;

var TRMM_SCALE_M  = 27830;  // catalog pixel size ~0.25°
var IMERG_SCALE_M = 11132;  // catalog pixel size ~0.1°
var UTM_CRS   = 'EPSG:32646';
var UTM_SCALE = 30;
var MAX_PIXELS = 1e13;

// =============================================================================
// 2. AOI
// =============================================================================

var aoi = ee.Geometry.Rectangle([WEST, SOUTH, EAST, NORTH], 'EPSG:4326', false);
var aoiBuf = aoi.buffer(BUFFER_DEG * 111320);  // ~0.1° in metres

Map.centerObject(aoi, 9);
Map.addLayer(aoi,    {color: 'black'}, 'AOI Aizawl box', true);
Map.addLayer(aoiBuf, {color: 'gray'},  'AOI + 0.1 deg buffer', false);

print('AOI (west,south,east,north)', [WEST, SOUTH, EAST, NORTH]);
print('Phase 1 TRMM window (end exclusive)', PHASE1_START, PHASE1_END);
print('Phase 2 IMERG window (end exclusive)', PHASE2_START, PHASE2_END);
print('ASSUMPTION: Phase 1 = paper-overlap TRMM 1998–2018; Phase 2 = IMERG 2019–latest (TRMM ended 2019-12).');

// =============================================================================
// 3. UNIT CONVERSION AND COMPOSITES
// =============================================================================

/**
 * mm/hr (monthly mean rate) → mm in that calendar month.
 * hours = difference(start, start+1 month) so 28/29/30/31-day months are exact.
 */
function toMonthlyMm(img) {
  var start = ee.Date(img.get('system:time_start'));
  var hours = start.advance(1, 'month').difference(start, 'hour');
  return img.select('precipitation')
      .multiply(hours)
      .rename('precip_mm')
      .copyProperties(img, ['system:time_start'])
      .set('hours_in_month', hours);
}

function monthlyCollection(colId, start, endExclusive) {
  return ee.ImageCollection(colId)
      .select('precipitation')
      .filterDate(start, endExclusive)
      .filterBounds(aoiBuf)
      .map(toMonthlyMm);
}

function yearList(start, endExclusive) {
  var y0 = ee.Date(start).get('year');
  var y1 = ee.Date(endExclusive).advance(-1, 'day').get('year');
  return ee.List.sequence(y0, y1);
}

/** One image per year: sum of monthly mm → annual mm */
function annualCollection(monthlyCol, start, endExclusive) {
  return ee.ImageCollection.fromImages(yearList(start, endExclusive).map(function(y) {
    y = ee.Number(y);
    var y0 = ee.Date.fromYMD(y, 1, 1);
    var y1 = y0.advance(1, 'year');
    var n = monthlyCol.filterDate(y0, y1).size();
    return monthlyCol.filterDate(y0, y1)
        .sum()
        .rename('annual_mm')
        .set('year', y)
        .set('n_months', n)
        .set('system:time_start', y0.millis());
  }));
}

/** Mean Jun–Sep total (mm) across years */
function monsoonCollection(monthlyCol, start, endExclusive) {
  return ee.ImageCollection.fromImages(yearList(start, endExclusive).map(function(y) {
    y = ee.Number(y);
    var y0 = ee.Date.fromYMD(y, 1, 1);
    var y1 = y0.advance(1, 'year');
    return monthlyCol
        .filterDate(y0, y1)
        .filter(ee.Filter.calendarRange(6, 9, 'month'))
        .sum()
        .rename('monsoon_mm')
        .set('year', y)
        .set('system:time_start', y0.millis());
  }));
}

/** 12-band image: long-term mean mm for Jan..Dec */
function monthlyClimatology12(monthlyCol) {
  var months = ee.List.sequence(1, 12);
  var images = months.map(function(m) {
    m = ee.Number(m);
    var name = ee.String('mm_').cat(m.format('%02d'));
    return monthlyCol
        .filter(ee.Filter.calendarRange(m, m, 'month'))
        .mean()
        .rename(name);
  });
  return ee.ImageCollection.fromImages(images).toBands()
      .rename(['mm_01','mm_02','mm_03','mm_04','mm_05','mm_06',
               'mm_07','mm_08','mm_09','mm_10','mm_11','mm_12']);
}

function clipBoth(img) {
  return img.clip(aoiBuf);
}

function printMinMax(img, label, geom, scale) {
  var stats = img.reduceRegion({
    reducer: ee.Reducer.minMax().combine({
      reducer2: ee.Reducer.mean(),
      sharedInputs: true
    }),
    geometry: geom,
    scale: scale,
    maxPixels: 1e9,
    bestEffort: true
  });
  print(label + ' min/max/mean (AOI, sanity check ~2000–3500 mm/yr for Aizawl)', stats);
}

function printCount(col, label) {
  print(label + ' image count', col.size());
}

// =============================================================================
// 4. BUILD PHASE COMPOSITES
// =============================================================================

var palAnnual = {
  min: 1500, max: 3500,
  palette: ['#f7fcf0','#ccebc5','#7bccc4','#2b8cbe','#084081']
};
var palMonsoon = {
  min: 1000, max: 2800,
  palette: ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
};

// --- Phase 1: TRMM 3B43 (paper-overlap years) ---
var p1Monthly = monthlyCollection('TRMM/3B43V7', PHASE1_START, PHASE1_END);
printCount(p1Monthly, 'Phase 1 TRMM/3B43V7 monthly');

var p1Annual  = clipBoth(annualCollection(p1Monthly, PHASE1_START, PHASE1_END).mean())
    .rename('phase1_trmm_mean_annual_mm');
var p1Monsoon = clipBoth(monsoonCollection(p1Monthly, PHASE1_START, PHASE1_END).mean())
    .rename('phase1_trmm_mean_jjas_mm');
var p1Clim12  = clipBoth(monthlyClimatology12(p1Monthly));

printMinMax(p1Annual,  'Phase 1 TRMM mean annual mm/year', aoi, TRMM_SCALE_M);
printMinMax(p1Monsoon, 'Phase 1 TRMM mean JJAS mm',        aoi, TRMM_SCALE_M);

Map.addLayer(p1Annual,  palAnnual,  'P1 TRMM mean annual mm/yr 1998-2018', true);
Map.addLayer(p1Monsoon, palMonsoon, 'P1 TRMM mean JJAS mm 1998-2018', false);

// --- Phase 2: IMERG Monthly V07 (TRMM does not cover this window fully) ---
print('PHASE 2 SUBSTITUTION: NASA/GPM_L3/IMERG_MONTHLY_V07 (not TRMM). TRMM/3B43V7 ends 2019-12.');

var p2Monthly = monthlyCollection('NASA/GPM_L3/IMERG_MONTHLY_V07', PHASE2_START, PHASE2_END);
printCount(p2Monthly, 'Phase 2 IMERG_MONTHLY_V07 monthly');

var p2Annual  = clipBoth(annualCollection(p2Monthly, PHASE2_START, PHASE2_END).mean())
    .rename('phase2_imerg_mean_annual_mm');
var p2Monsoon = clipBoth(monsoonCollection(p2Monthly, PHASE2_START, PHASE2_END).mean())
    .rename('phase2_imerg_mean_jjas_mm');
var p2Clim12  = clipBoth(monthlyClimatology12(p2Monthly));

printMinMax(p2Annual,  'Phase 2 IMERG mean annual mm/year', aoi, IMERG_SCALE_M);
printMinMax(p2Monsoon, 'Phase 2 IMERG mean JJAS mm',        aoi, IMERG_SCALE_M);

Map.addLayer(p2Annual,  palAnnual,  'P2 IMERG mean annual mm/yr 2019-latest', false);
Map.addLayer(p2Monsoon, palMonsoon, 'P2 IMERG mean JJAS mm 2019-latest', false);

// --- Optional: long IMERG climatology (best single 17-factor rainfall) ---
var longMonthly = monthlyCollection('NASA/GPM_L3/IMERG_MONTHLY_V07', IMERG_LONG_START, IMERG_LONG_END);
printCount(longMonthly, 'IMERG long climatology monthly');

var longAnnual  = clipBoth(annualCollection(longMonthly, IMERG_LONG_START, IMERG_LONG_END).mean())
    .rename('imerg_long_mean_annual_mm');
var longMonsoon = clipBoth(monsoonCollection(longMonthly, IMERG_LONG_START, IMERG_LONG_END).mean())
    .rename('imerg_long_mean_jjas_mm');

printMinMax(longAnnual, 'IMERG long mean annual mm/year', aoi, IMERG_SCALE_M);
Map.addLayer(longAnnual, palAnnual, 'IMERG long mean annual mm/yr 2000-latest', false);

// =============================================================================
// 5. EXPORTS
// =============================================================================

function exportNative(img, prefix, scale) {
  Export.image.toDrive({
    image: img.toFloat(),
    description: prefix + '_epsg4326',
    folder: DRIVE_FOLDER,
    fileNamePrefix: prefix + '_epsg4326',
    region: aoiBuf,
    crs: 'EPSG:4326',
    scale: scale,
    maxPixels: MAX_PIXELS,
    fileFormat: 'GeoTIFF'
  });
}

function exportUtm30(img, prefix) {
  Export.image.toDrive({
    image: img.toFloat(),
    description: prefix + '_utm46n_30m',
    folder: DRIVE_FOLDER,
    fileNamePrefix: prefix + '_utm46n_30m',
    region: aoiBuf,
    crs: UTM_CRS,
    scale: UTM_SCALE,
    maxPixels: MAX_PIXELS,
    fileFormat: 'GeoTIFF'
  });
}

function exportPair(img, prefix, nativeScale) {
  if (EXPORT_NATIVE) exportNative(img, prefix, nativeScale);
  if (EXPORT_30M)    exportUtm30(img, prefix);
}

exportPair(p1Annual,  'phase1_TRMM3B43V7_meanAnnual_mmYr_1998_2018', TRMM_SCALE_M);
exportPair(p1Monsoon, 'phase1_TRMM3B43V7_meanJJAS_mm_1998_2018',     TRMM_SCALE_M);

exportPair(p2Annual,  'phase2_IMERG_V07_meanAnnual_mmYr_2019_2025', IMERG_SCALE_M);
exportPair(p2Monsoon, 'phase2_IMERG_V07_meanJJAS_mm_2019_2025',     IMERG_SCALE_M);

exportPair(longAnnual,  'optional_IMERG_V07_meanAnnual_mmYr_2000_2025', IMERG_SCALE_M);
exportPair(longMonsoon, 'optional_IMERG_V07_meanJJAS_mm_2000_2025',     IMERG_SCALE_M);

if (EXPORT_CLIM12) {
  exportNative(p1Clim12, 'phase1_TRMM3B43V7_monthlyClim12_mm_1998_2018', TRMM_SCALE_M);
  exportNative(p2Clim12, 'phase2_IMERG_V07_monthlyClim12_mm_2019_2025',  IMERG_SCALE_M);
}

print('Exports queued. Code Editor → Tasks tab → RUN each task.');
print('Drive folder:', DRIVE_FOLDER);
print('Primary FR factor = mean annual mm/year. Sonker used IMD 1988–2018; Phase 1 TRMM covers 1998–2018 only.');
