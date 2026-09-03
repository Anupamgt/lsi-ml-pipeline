# -*- coding: utf-8 -*-
"""Cloud-masked Landsat 8 C2 SR NDVI median composite on the Aizawl master grid.

Run from QGIS MCP execute_code. SAS token is fetched in memory and never written.
"""
import json
import os
import urllib.request

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
gdal.SetConfigOption("CPL_VSIL_CURL_USE_HEAD", "NO")
gdal.SetConfigOption("GDAL_HTTP_MAX_RETRY", "4")
gdal.SetConfigOption("GDAL_HTTP_TIMEOUT", "90")
gdal.SetConfigOption("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".TIF,.tif,.Tiff")

MASTER = r"C:\Users\sharm\LSI_btp_sikkim_replication\01_processed\master_dem_30m.tif"
OUT_DIR = r"C:\Users\sharm\LSI_btp_sikkim_replication\01_processed"
RAW_DIR = r"C:\Users\sharm\LSI_btp_sikkim_replication\00_drop_raw\09_ndvi_landsat"
TMP_DIR = os.path.join(RAW_DIR, "_scene_ndvi")
SEL_JSON = os.path.join(RAW_DIR, "_selected_l8.json")
AIZWAL = r"C:\Users\sharm\lsi-ml-pipeline\final_maps\AIZWAL.shp"

NDVI_OUT = os.path.join(OUT_DIR, "ndvi_30m_clip.tif")
RECLASS_OUT = os.path.join(OUT_DIR, "ndvi_reclass_sonker.tif")

XMIN, YMIN, XMAX, YMAX = 461520.0, 2578020.0, 521610.0, 2700000.0
NX, NY = 2003, 4066
SCALE = 0.0000275
OFFSET = -0.2
NODATA = -9999.0

# Collection 2 QA_PIXEL bits to mask: fill, dilated cloud, cirrus, cloud, cloud shadow
QA_MASK_BITS = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)


def pc_token():
    tok = json.loads(
        urllib.request.urlopen(
            "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2",
            timeout=30,
        )
        .read()
        .decode()
    )
    t = tok.get("token")
    if not t:
        raise RuntimeError("Planetary Computer SAS token empty")
    print("sas_expiry", tok.get("msft:expiry"))
    return t


def vsicurl(href, token):
    return "/vsicurl/" + href + "?" + token


def warp_band(src, dst, resample):
    opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS="EPSG:32646",
        outputBounds=[XMIN, YMIN, XMAX, YMAX],
        width=NX,
        height=NY,
        resampleAlg=resample,
        outputType=gdal.GDT_UInt16,
        dstNodata=0,
        multithread=True,
        creationOptions=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    gdal.Warp(dst, src, options=opts)


def write_float(path, arr, nodata=NODATA):
    ref = gdal.Open(MASTER)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(
        path,
        NX,
        NY,
        1,
        gdal.GDT_Float32,
        options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"],
    )
    ds.SetGeoTransform(ref.GetGeoTransform())
    ds.SetProjection(ref.GetProjection())
    ref = None
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(arr)
    band.FlushCache()
    ds = None


def scene_ndvi(item, token, out_tif):
    red_p = os.path.join(TMP_DIR, item["id"] + "_B4.tif")
    nir_p = os.path.join(TMP_DIR, item["id"] + "_B5.tif")
    qa_p = os.path.join(TMP_DIR, item["id"] + "_QA.tif")
    qa_href = item["hrefs"].get("qa_pixel") or item["hrefs"]["red"].replace("_SR_B4.TIF", "_QA_PIXEL.TIF")
    warp_band(vsicurl(item["hrefs"]["red"], token), red_p, "bilinear")
    warp_band(vsicurl(item["hrefs"]["nir08"], token), nir_p, "bilinear")
    warp_band(vsicurl(qa_href, token), qa_p, "near")

    red = gdal.Open(red_p).ReadAsArray().astype(np.float32)
    nir = gdal.Open(nir_p).ReadAsArray().astype(np.float32)
    qa = gdal.Open(qa_p).ReadAsArray()

    red_sr = red * SCALE + OFFSET
    nir_sr = nir * SCALE + OFFSET
    # QA_PIXEL (not ST_QA). Mask fill/dilated-cloud/cirrus/cloud/shadow.
    # Allow small negative SR from the C2 scale/offset; drop only fill DN=0.
    valid = (
        (red > 0)
        & (nir > 0)
        & ((qa.astype(np.uint32) & QA_MASK_BITS) == 0)
        & ((nir_sr + red_sr) != 0)
    )
    ndvi = np.full(red.shape, np.nan, dtype=np.float32)
    ndvi[valid] = (nir_sr[valid] - red_sr[valid]) / (nir_sr[valid] + red_sr[valid])
    ndvi = np.clip(ndvi, -1.0, 1.0)
    n_ok = int(np.isfinite(ndvi).sum())
    print(
        "scene",
        item["id"],
        "valid_px",
        n_ok,
        "pct",
        round(100.0 * n_ok / ndvi.size, 2),
        "min",
        float(np.nanmin(ndvi)) if n_ok else None,
        "max",
        float(np.nanmax(ndvi)) if n_ok else None,
        "mean",
        float(np.nanmean(ndvi)) if n_ok else None,
    )
    write_float(out_tif, np.where(np.isfinite(ndvi), ndvi, NODATA))
    for p in (red_p, nir_p, qa_p):
        try:
            os.remove(p)
        except OSError:
            pass
    return n_ok


def median_stack(paths):
    stack = []
    for p in paths:
        a = gdal.Open(p).ReadAsArray().astype(np.float32)
        a[a == NODATA] = np.nan
        stack.append(a)
    med = np.nanmedian(np.stack(stack, axis=0), axis=0).astype(np.float32)
    n_valid = int(np.isfinite(med).sum())
    print("median_valid_px", n_valid, "pct", round(100.0 * n_valid / med.size, 2))
    return med


def clip_aizawl(arr):
    tmp = os.path.join(TMP_DIR, "_ndvi_preclip.tif")
    write_float(tmp, np.where(np.isfinite(arr), arr, NODATA))
    clipped = os.path.join(TMP_DIR, "_ndvi_clip.tif")
    opts = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=AIZWAL,
        cropToCutline=False,
        dstNodata=NODATA,
        outputBounds=[XMIN, YMIN, XMAX, YMAX],
        width=NX,
        height=NY,
        dstSRS="EPSG:32646",
        outputType=gdal.GDT_Float32,
        creationOptions=["TILED=YES", "COMPRESS=LZW"],
    )
    gdal.Warp(clipped, tmp, options=opts)
    out = gdal.Open(clipped).ReadAsArray().astype(np.float32)
    out[out == NODATA] = np.nan
    return out


def reclass_sonker(ndvi):
    """Sonker 2022 Table 1 NDVI breaks. Ends extended so Aizawl values outside Sikkim range still classify."""
    cls = np.full(ndvi.shape, NODATA, dtype=np.float32)
    valid = np.isfinite(ndvi)
    # 1 Very low: -0.38 to -0.18  (also < -0.38)
    cls[valid & (ndvi < -0.18)] = 1
    # 2 Low: -0.18 to 0.02
    cls[valid & (ndvi >= -0.18) & (ndvi < 0.02)] = 2
    # 3 Moderate: 0.02 to 0.22
    cls[valid & (ndvi >= 0.02) & (ndvi < 0.22)] = 3
    # 4 High: 0.22 to 0.42
    cls[valid & (ndvi >= 0.22) & (ndvi < 0.42)] = 4
    # 5 Very high: 0.42 to 0.62  (also > 0.62)
    cls[valid & (ndvi >= 0.42)] = 5
    return cls


def grid_check(path):
    ds = gdal.Open(path)
    gt = ds.GetGeoTransform()
    print(
        "grid",
        os.path.basename(path),
        "size",
        ds.RasterXSize,
        ds.RasterYSize,
        "gt",
        gt,
        "crs_ok",
        "32646" in (ds.GetProjection() or ""),
        "match_master",
        ds.RasterXSize == NX
        and ds.RasterYSize == NY
        and abs(gt[0] - XMIN) < 1e-6
        and abs(gt[3] - YMAX) < 1e-6
        and abs(gt[1] - 30) < 1e-9,
    )


def main(limit=None):
    os.makedirs(TMP_DIR, exist_ok=True)
    items = json.load(open(SEL_JSON, encoding="utf-8"))["items"]
    if limit:
        items = items[:limit]
    token = pc_token()
    scene_paths = []
    for item in items:
        out_tif = os.path.join(TMP_DIR, item["id"] + "_ndvi.tif")
        if os.path.exists(out_tif) and os.path.getsize(out_tif) > 1000:
            print("skip existing", item["id"])
            scene_paths.append(out_tif)
            continue
        n_ok = scene_ndvi(item, token, out_tif)
        if n_ok > 0:
            scene_paths.append(out_tif)
        else:
            print("WARN empty", item["id"])
    if not scene_paths:
        raise RuntimeError("no valid scene NDVI rasters")
    med = median_stack(scene_paths)
    clipped = clip_aizawl(med)
    finite = clipped[np.isfinite(clipped)]
    print(
        "clip_stats n",
        finite.size,
        "min",
        float(finite.min()) if finite.size else None,
        "max",
        float(finite.max()) if finite.size else None,
        "mean",
        float(finite.mean()) if finite.size else None,
    )
    write_float(NDVI_OUT, np.where(np.isfinite(clipped), clipped, NODATA))
    cls = reclass_sonker(clipped)
    write_float(RECLASS_OUT, cls)
    grid_check(MASTER)
    grid_check(NDVI_OUT)
    grid_check(RECLASS_OUT)
    # class counts
    for c in range(1, 6):
        print("class", c, "px", int((cls == c).sum()))
    print("DONE", NDVI_OUT, RECLASS_OUT)


if __name__ == "__main__" or True:
    # limit via env-like global set by caller
    _limit = globals().get("PROCESS_LIMIT", None)
    main(limit=_limit)
