import math
import time
import gc
import os
import tempfile
import numpy as np
from numba import njit

try:
    import arcpy
except ImportError:
    arcpy = None

try:
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')
    gdal.SetConfigOption('GDAL_CACHEMAX', '2048')
    _GDAL_OK = True
except ImportError:
    _GDAL_OK = False

try:
    from numba import njit
    _src = globals().get('__file__', '')
    _DZ_CACHE = bool(
        _src
        and '<' not in _src
        and _src.endswith('.py')
        and os.path.isfile(_src)
    )
    _DZ_NUMBA = True
except ImportError:
    _DZ_NUMBA = False
    _DZ_CACHE = False
    def njit(*a, **kw):
        return a[0] if a and callable(a[0]) else (lambda f: f)

_NB_DR   = np.array([-1,-1, 0, 1, 1, 1, 0,-1], dtype=np.int32)
_NB_DC   = np.array([ 0, 1, 1, 1, 0,-1,-1,-1], dtype=np.int32)
_NB_CODE = np.array([64,128, 1, 2, 4,  8,16,32], dtype=np.int32)
_NB_FAC  = np.array([1.0, math.sqrt(2), 1.0, math.sqrt(2),
                     1.0, math.sqrt(2), 1.0, math.sqrt(2)], dtype=np.float64)
_REV_IDX = np.array([4,5,6,7,0,1,2,3], dtype=np.int32)
_DZ_ND   = -9999.0
_STRIP_ROWS = 500

def _make_memmap(shape, dtype, fill=0):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.dzdat')
    tmp.close()
    mm = np.memmap(tmp.name, dtype=dtype, mode='w+', shape=shape)
    mm[:] = fill
    mm.flush()
    return mm, tmp.name


def _dem_to_memmap(dem_path, nodata=_DZ_ND, msg_fn=None):
    if msg_fn is None: msg_fn = print
    robj  = arcpy.Raster(dem_path)
    nrows, ncols = int(robj.height), int(robj.width)
    cs    = float(robj.meanCellWidth)
    nd_val = float(robj.noDataValue) if robj.noDataValue is not None else nodata
    ll    = arcpy.Point(robj.extent.XMin, robj.extent.YMin)
    sr    = arcpy.Describe(dem_path).spatialReference

    elev_mm, elev_f = _make_memmap((nrows, ncols), np.float32, fill=nd_val)
    for i in range(0, nrows, _STRIP_ROWS):
        r1 = min(i + _STRIP_ROWS, nrows)
        tr = r1 - i
        pt = arcpy.Point(robj.extent.XMin, robj.extent.YMax - r1 * cs)
        strip = arcpy.RasterToNumPyArray(robj, pt, ncols, tr, nodata_to_value=nd_val)
        elev_mm[i:r1, :] = strip.astype(np.float32, copy=False)
    elev_mm.flush()
    return elev_mm, elev_f, nrows, ncols, cs, nd_val, ll, sr

@njit(cache=_DZ_CACHE)
def _dz_hlt(he, hf, hk, i, j):
    if he[i] != he[j]:
        return he[i] < he[j]
    return hf[i] < hf[j]

@njit(cache=_DZ_CACHE)
def _dz_hswap(he, hf, hk, hr, hc, i, j):
    he[i], he[j] = he[j], he[i]
    hf[i], hf[j] = hf[j], hf[i]
    hk[i], hk[j] = hk[j], hk[i]
    hr[i], hr[j] = hr[j], hr[i]
    hc[i], hc[j] = hc[j], hc[i]

@njit(cache=_DZ_CACHE)
def _dz_sift_up(he, hf, hk, hr, hc, n):
    i = n - 1
    while i > 0:
        p = (i - 1) >> 1
        if _dz_hlt(he, hf, hk, i, p):
            _dz_hswap(he, hf, hk, hr, hc, i, p)
            i = p
        else: break

@njit(cache=_DZ_CACHE)
def _dz_sift_dn(he, hf, hk, hr, hc, n, s):
    i = s
    while True:
        l, r2, m = 2*i + 1, 2*i + 2, i
        if l < n and _dz_hlt(he, hf, hk, l, m): m = l
        if r2 < n and _dz_hlt(he, hf, hk, r2, m): m = r2
        if m == i: break
        _dz_hswap(he, hf, hk, hr, hc, i, m)
        i = m

@njit(cache=_DZ_CACHE)
def _dz_hpush(he, hf, hk, hr, hc, hn, e, f, k, r, c):
    he[hn] = e; hf[hn] = f; hk[hn] = k; hr[hn] = r; hc[hn] = c
    _dz_sift_up(he, hf, hk, hr, hc, hn + 1)
    return hn + 1

@njit(cache=_DZ_CACHE)
def _dz_hpop(he, hf, hk, hr, hc, hn):
    e0, f0, k0, r0, c0 = he[0], hf[0], hk[0], hr[0], hc[0]
    hn -= 1
    if hn > 0:
        he[0], hf[0], hk[0], hr[0], hc[0] = he[hn], hf[hn], hk[hn], hr[hn], hc[hn]
        _dz_sift_dn(he, hf, hk, hr, hc, hn, 0)
    return e0, f0, k0, r0, c0, hn

@njit(cache=_DZ_CACHE)
def _dz_close(a, b):
    return abs(a - b) <= 1e-9 + 1e-9 * (abs(a) + abs(b))

@njit(cache=_DZ_CACHE)
def _dz_flat_adj(r, c, mr, mc, m_code, midx, elev, fd_arr, nsd_arr, flow_dirs,
                 nrows, ncols, cs, nb_dr, nb_dc, nb_code, nb_fac, rev_idx):
    e_rc  = elev[r, c]
    rev_m = nb_code[rev_idx[midx]]
    best_d   = m_code
    best_nsd = nsd_arr[mr, mc]
    for k in range(8):
        kr, kc = r + nb_dr[k], c + nb_dc[k]
        if not (0 <= kr < nrows and 0 <= kc < ncols): continue
        if not _dz_close(elev[kr, kc], e_rc): continue
        if abs(kr - mr) > 1 or abs(kc - mc) > 1: continue
        if not _dz_close(fd_arr[kr, kc] + cs * nb_fac[k], fd_arr[r, c]): continue
        if (flow_dirs[kr, kc] == nb_code[rev_idx[k]]) and (flow_dirs[mr, mc] == rev_m):
            if nsd_arr[kr, kc] < best_nsd:
                best_nsd = nsd_arr[kr, kc]
                best_d   = nb_code[k]
        elif (flow_dirs[kr, kc] == nb_code[rev_idx[k]]):
            best_d   = nb_code[k]
            best_nsd = nsd_arr[kr, kc]
    flow_dirs[r, c] = best_d
    new_midx = -1
    for k in range(8):
        if nb_code[k] == best_d: new_midx = k; break
    if new_midx >= 0:
        mr2, mc2 = r + nb_dr[new_midx], c + nb_dc[new_midx]
        if 0 <= mr2 < nrows and 0 <= mc2 < ncols:
            nsd_arr[r, c] = (nsd_arr[mr2, mc2] + 1
                             if flow_dirs[mr2, mc2] == nb_code[rev_idx[new_midx]]
                             else 0)

@njit(cache=_DZ_CACHE)
def _dz_core(elev, nodata_mask, closed, flow_dirs, fd_arr, nsd_arr,
             nrows, ncols, cs, or_, oc_, n_outlets,
             dr, dc, code, fac, rev, he, hf, hk, hrr, hcc):
    hn = np.int64(0)
    for i in range(n_outlets):
        hn = _dz_hpush(he, hf, hk, hrr, hcc, hn,
                       elev[or_[i], oc_[i]], 0.0, np.int64(i), or_[i], oc_[i])
    counter = np.int64(n_outlets)
    while hn > 0:
        _, _, _, r, c, hn = _dz_hpop(he, hf, hk, hrr, hcc, hn)
        m_code = flow_dirs[r, c]
        if m_code != 0:
            midx = -1
            for k in range(8):
                if code[k] == m_code: midx = k; break
            if midx >= 0:
                mr, mc = r + dr[midx], c + dc[midx]
                if 0 <= mr < nrows and 0 <= mc < ncols:
                    dz = elev[r, c] - elev[mr, mc]
                    if dz == 0.0:
                        _dz_flat_adj(r, c, mr, mc, m_code, midx, elev, fd_arr, nsd_arr,
                                     flow_dirs, nrows, ncols, cs, dr, dc, code, fac, rev)
                    elif dz > 0.0:
                        e_rc   = elev[r, c]
                        best_s = dz / (cs * fac[midx])
                        best_d = m_code
                        for k in range(8):
                            nr2, nc2 = r + dr[k], c + dc[k]
                            if 0 <= nr2 < nrows and 0 <= nc2 < ncols and not nodata_mask[nr2, nc2]:
                                s2 = (e_rc - elev[nr2, nc2]) / (cs * fac[k])
                                if s2 > best_s: best_s = s2; best_d = code[k]
                        flow_dirs[r, c] = best_d
        e_rc, fd_rc = elev[r, c], fd_arr[r, c]
        for k in range(8):
            nr, nc = r + dr[k], c + dc[k]
            if 0 <= nr < nrows and 0 <= nc < ncols and not closed[nr, nc]:
                closed[nr, nc] = True
                flow_dirs[nr, nc] = code[rev[k]]
                n_elev = elev[nr, nc]
                fd_new = (fd_rc + cs * fac[k]) if _dz_close(n_elev, e_rc) else 0.0
                fd_arr[nr, nc] = fd_new
                hn = _dz_hpush(he, hf, hk, hrr, hcc, hn, n_elev, fd_new, counter, nr, nc)
                counter += 1
    return flow_dirs

def _arr_to_geotiff_gdal(arr, tmp_path, xmin, ymax, cs, nodata_val, wkt_str, msg_fn):
    nrows, ncols = arr.shape
    estimated_bytes = int(nrows) * int(ncols) * 4

    if arr.dtype == np.float32 or arr.dtype == np.float64:
        gdal_dtype = gdal.GDT_Float32
        np_dtype   = np.float32
    else:
        gdal_dtype = gdal.GDT_Int32
        np_dtype   = np.int32

    bigtiff_opt = 'BIGTIFF=YES' if estimated_bytes > 2 * 1024 ** 3 else 'BIGTIFF=IF_SAFER'

    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(
        tmp_path, ncols, nrows, 1, gdal_dtype,
        options=[
            'COMPRESS=DEFLATE',
            'ZLEVEL=2',
            'PREDICTOR=2',
            'TILED=YES',
            'BLOCKXSIZE=512',
            'BLOCKYSIZE=512',
            bigtiff_opt,
            'NUM_THREADS=ALL_CPUS'
        ]
    )
    if ds is None:
        raise RuntimeError(f"GDAL could not create GeoTIFF: {tmp_path}")

    ds.SetGeoTransform((xmin, cs, 0.0, ymax, 0.0, -cs))
    ds.SetProjection(wkt_str)

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata_val))

    CHUNK_ROWS = 512
    for row_start in range(0, nrows, CHUNK_ROWS):
        row_end = min(row_start + CHUNK_ROWS, nrows)
        chunk = arr[row_start:row_end, :].astype(np_dtype, copy=False)
        band.WriteArray(chunk, xoff=0, yoff=row_start)

    band.FlushCache()
    ds.FlushCache()
    ds = None


def _save_raster(arr, path, ll, cs, sr, nodata_val, msg_fn):
    t0 = time.time()
    nrows, ncols = arr.shape

    if _GDAL_OK:
        msg_fn("  [>] Writing GeoTIFF with GDAL...")
        tmp_tif = os.path.join(tempfile.gettempdir(), f"_dz_tmp_{os.getpid()}.tif")
        try:
            wkt  = sr.exportToString() if hasattr(sr, 'exportToString') else ""
            xmin = ll.X
            ymax = ll.Y + nrows * cs
            _arr_to_geotiff_gdal(arr, tmp_tif, xmin, ymax, cs, nodata_val, wkt, msg_fn)
            msg_fn(f"    GDAL write time     : {time.time()-t0:.1f}s")

            tc = time.time()
            msg_fn("  [>] Copying to destination (CopyRaster)...")
            if arcpy.Exists(path):
                arcpy.management.Delete(path)
            p_type = ("32_BIT_FLOAT" if arr.dtype in [np.float32, np.float64]
                      else "32_BIT_SIGNED")
            arcpy.management.CopyRaster(
                in_raster=tmp_tif,
                out_rasterdataset=path,
                pixel_type=p_type,
                nodata_value=nodata_val
            )
            msg_fn(f"    CopyRaster time     : {time.time()-tc:.1f}s")
        finally:
            if os.path.exists(tmp_tif):
                try: os.remove(tmp_tif)
                except: pass

    else:
        msg_fn("  [!] GDAL not available, falling back to arcpy...")
        if arcpy.Exists(path):
            arcpy.management.Delete(path)
        out_r = arcpy.NumPyArrayToRaster(arr, ll, cs, cs, nodata_val)
        arcpy.DefineProjection_management(out_r, sr)
        out_r.save(path)
        del out_r

    msg_fn(f"  [+] Save complete — total: {time.time()-t0:.1f}s")


class Toolbox(object):
    def __init__(self):
        self.label = "FlowDZ"
        self.alias = "FlowDZ"
        self.tools = [FlowDZFlood]


class FlowDZFlood(object):
    def __init__(self):
        self.label = "DZFlood"
        self.canRunInBackground = True

    def getParameterInfo(self):
        # Parameter 0 — Input DEM
        p_dem = arcpy.Parameter(
            "in_dem",
            "Digital Elevation Model",
            "Input",
            "GPRasterLayer",
            "Required"
        )

        # Parameter 1 — Is the DEM already filled?
        p_prefilled = arcpy.Parameter(
            "dem_is_prefilled",
            "DEM is already filled (skip Fill step)",
            "Input",
            "GPBoolean",
            "Optional"
        )
        p_prefilled.value = False  # default: run Fill

        # Parameter 2 — Flow Direction output
        p_fd = arcpy.Parameter(
            "out_flow_dir",
            "Flow Direction",
            "Output",
            "DERasterDataset",
            "Required"
        )

        # Parameter 3 — Flow Accumulation output
        p_fa = arcpy.Parameter(
            "out_flow_acc",
            "Flow Accumulation",
            "Output",
            "DERasterDataset",
            "Required"
        )

        return [p_dem, p_prefilled, p_fd, p_fa]

    def execute(self, parameters, messages):
        in_dem        = parameters[0].valueAsText
        dem_prefilled = parameters[1].value   # True / False / None
        out_fd        = parameters[2].valueAsText
        out_acc       = parameters[3].valueAsText

        def msg(s): arcpy.AddMessage(s)

        t_start = time.time()
        def elapsed(t): return f"{time.time()-t:.1f}s"

        arcpy.CheckOutExtension("Spatial")

        # ¦¦ Step counter bookkeeping ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        # If fill is skipped we still show 5 steps; step 1 just becomes a note.
        skip_fill = bool(dem_prefilled)

        # ¦¦ [1/5] Fill ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        t1 = time.time()
        if skip_fill:
            msg("??? [1/5] Fill step SKIPPED — input DEM is already filled.")
            dem_to_read = in_dem   # use the original DEM directly
            _fill_tmp   = None
        else:
            msg("??? [1/5] Filling input DEM (arcpy.sa.Fill)...")
            from arcpy.sa import Fill
            filled_raster = Fill(in_dem)
            _fill_tmp = os.path.join(arcpy.env.scratchGDB, f"_dz_filled_{os.getpid()}")
            filled_raster.save(_fill_tmp)
            del filled_raster
            gc.collect()
            dem_to_read = _fill_tmp
            msg(f"??? [1/5] DONE ? {elapsed(t1)}")

        # ¦¦ [2/5] Read DEM ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        t2 = time.time()
        msg("??? [2/5] Reading DEM into memory...")
        elev32_mm, e_f, nrows, ncols, cs, nd_v, ll, sr = _dem_to_memmap(dem_to_read, msg_fn=msg)
        elev32 = np.array(elev32_mm)
        del elev32_mm
        gc.collect()
        if _fill_tmp is not None:
            try: arcpy.management.Delete(_fill_tmp)
            except: pass
        msg(f"    Size: {nrows}x{ncols} = {nrows*ncols:,} cells")
        msg(f"    GDAL available: {_GDAL_OK}")
        msg(f"??? [2/5] DONE ? {elapsed(t2)}")

        # ¦¦ [3/5] Masks & outlets ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        t3 = time.time()
        msg("??? [3/5] Building masks and locating outlets...")
        nd_mask = (elev32 == np.float32(nd_v))
        ie = np.zeros((nrows, ncols), bool)
        ie[0,:] = ie[-1,:] = ie[:,0] = ie[:,-1] = True
        pad = np.pad(nd_mask, 1, constant_values=False)
        hnn = np.zeros((nrows, ncols), bool)
        for dr, dc in zip(_NB_DR, _NB_DC):
            hnn |= pad[1+dr:nrows+1+dr, 1+dc:ncols+1+dc]
        om = (~nd_mask) & (ie | hnn)
        or_, oc_ = np.where(om)
        msg(f"    Outlet count: {len(or_):,}")
        msg(f"??? [3/5] DONE ? {elapsed(t3)}")

        # ¦¦ [4/5] Flow Direction ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        t4 = time.time()
        msg("??? [4/5] Computing flow direction (DZ-Flood core)...")
        cl,  cf  = _make_memmap((nrows, ncols), bool,       False)
        fd,  ff  = _make_memmap((nrows, ncols), np.int32,   0)
        fda, fa  = _make_memmap((nrows, ncols), np.float32, 0.0)
        nsd, fn  = _make_memmap((nrows, ncols), np.int32,   0)
        cl[nd_mask] = cl[om] = True
        hucre_sayisi = np.int64(nrows) * ncols
        nt = np.int64(hucre_sayisi * 0.2)
        he,  fhe = _make_memmap((nt,), np.float32, 0)
        hf,  fhf = _make_memmap((nt,), np.float32, 0)
        hk,  fhk = _make_memmap((nt,), np.int64,   0)
        hr,  fhr = _make_memmap((nt,), np.int32,   0)
        hc,  fhc = _make_memmap((nt,), np.int32,   0)

        _dz_core(elev32, nd_mask,
                 np.asarray(cl), np.asarray(fd), np.asarray(fda), np.asarray(nsd),
                 nrows, ncols, cs,
                 or_.astype(np.int32), oc_.astype(np.int32), len(or_),
                 _NB_DR, _NB_DC, _NB_CODE, _NB_FAC, _REV_IDX,
                 np.asarray(he), np.asarray(hf), np.asarray(hk),
                 np.asarray(hr), np.asarray(hc))

        fd_final = np.asarray(fd)
        msg(f"??? [4/5] DONE ? {elapsed(t4)}")

        # ¦¦ [5/5] Save results ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        t5 = time.time()
        msg("??? [5/5] Saving results...")

        msg("    Writing Flow Direction...")
        _save_raster(fd_final, out_fd, ll, cs, sr, 0, msg)

        t5b = time.time()
        msg("    Computing Flow Accumulation (arcpy.sa.FlowAccumulation)...")
        from arcpy.sa import FlowAccumulation
        acc_raster = FlowAccumulation(out_fd)
        acc_raster.save(out_acc)
        del acc_raster
        gc.collect()
        msg(f"    FlowAccumulation    : {time.time()-t5b:.1f}s")
        msg(f"??? [5/5] DONE ? {elapsed(t5)}")

        # ¦¦ Cleanup ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        for f in [e_f, cf, ff, fa, fn, fhe, fhf, fhk, fhr, fhc]:
            try: os.remove(f)
            except: pass

        # ¦¦ Summary ¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦
        msg("")
        msg("==========================================")
        msg(f"  TOTAL TIME: {elapsed(t_start)}")
        msg( "  Step times:")
        if skip_fill:
            msg( "    1) Fill (arcpy.sa)      : SKIPPED")
        else:
            msg(f"    1) Fill (arcpy.sa)      : {elapsed(t1)}")
        msg(f"    2) DEM Read             : {elapsed(t2)}")
        msg(f"    3) Mask & Outlets       : {elapsed(t3)}")
        msg(f"    4) Flow Dir (DZ core)   : {elapsed(t4)}")
        msg(f"    5) Save (FD + FA)       : {elapsed(t5)}")
        msg("==========================================")
        msg("Processing completed successfully.")
