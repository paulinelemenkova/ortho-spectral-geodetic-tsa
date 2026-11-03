#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from scipy.io import netcdf_file

from gbdt_cv import features
from roi import R

A, B = R("transect")


def stations(step_km: float, half_km: float, dx_km: float) -> tuple[np.ndarray, ...]:
    g = Geod(ellps="WGS84")
    az, _, length = g.inv(A[0], A[1], B[0], B[1])
    d = np.arange(0, length / 1000 + 1e-6, step_km)
    lon0, lat0, _ = g.fwd(np.full(d.size, A[0]), np.full(d.size, A[1]), np.full(d.size, az), d * 1000)
    off = np.arange(-half_km, half_km + 1e-6, dx_km)
    lo = np.repeat(lon0, off.size)
    la = np.repeat(lat0, off.size)
    o = np.tile(off, d.size)
    lon, lat, _ = g.fwd(lo, la, np.full(o.size, az + 90.0), o * 1000)
    return d, lon.reshape(d.size, -1), lat.reshape(d.size, -1)


def sample(path_or_arr, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if isinstance(path_or_arr, tuple):
        z, tr, crs = path_or_arr
    else:
        with rasterio.open(path_or_arr) as s:
            z, tr, crs = s.read(1).astype(float), s.transform, s.crs
            if s.nodata is not None:
                z[z == s.nodata] = np.nan
    x, y = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    c, r = ~tr * (x, y)
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    ok = (r >= 0) & (r < z.shape[0]) & (c >= 0) & (c < z.shape[1])
    out = np.full(lon.shape, np.nan)
    out[ok] = z[r[ok], c[ok]]
    return out


def stats(v: np.ndarray, nmin: int = 5) -> np.ndarray:
    n = np.isfinite(v).sum(axis=1)
    with np.errstate(all="ignore"):
        q = np.nanpercentile(v, [50, 25, 75], axis=1).T
    q[n < nmin] = np.nan
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("../data"))
    ap.add_argument("--figs", type=Path, default=Path("../figures"))
    ap.add_argument("--out", type=Path, default=Path("../figures/fig_profiles"))
    ap.add_argument("--half", type=float, default=1.0)
    a = ap.parse_args()
    d, lon, lat = stations(0.1, a.half, 0.03)

    dt = sample(a.data / "thermal" / "dT_med.tif", lon, lat)
    vup = sample(a.figs / "fig_insar_velocity" / "vup_ll.tif", lon, lat)
    paths = sorted((a.data / "composites_v2").glob("composite_*.tif"))
    with rasterio.open(paths[0]) as s:
        tr, crs = s.transform, s.crs
    twi_med = np.nanmedian(np.stack([features(p)[6] for p in paths]), axis=0)
    twi = sample((twi_med, tr, crs), lon, lat)
    with netcdf_file(a.figs / "fig_trend_maps" / "sen_twi.nc", "r", mmap=False) as f:
        glon = f.variables["lon"][:].copy()
        glat = f.variables["lat"][:].copy()
        z = f.variables["z"][:].copy().astype(float)
    res = float(glon[1] - glon[0])
    tr_g = rasterio.transform.from_origin(glon[0] - res / 2, glat[-1] + res / 2, res, res)
    sen = sample((z[::-1], tr_g, "EPSG:4326"), lon, lat)

    cols = [d[:, None], stats(dt), stats(vup), stats(twi), stats(sen)[:, :1]]
    tab = np.hstack(cols)
    a.out.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        a.out / "profile_surface.txt",
        tab,
        fmt="%.4f",
        header="dist_km dT dT_p25 dT_p75 vup vup_p25 vup_p75 twi twi_p25 twi_p75 sen_twi",
    )
    for name, k in (("dT", 1), ("vup", 4), ("twi", 7), ("sen", 10)):
        ok = np.isfinite(tab[:, k])
        print(
            name,
            "covers",
            tab[ok, 0].min(),
            "-",
            tab[ok, 0].max(),
            "km; range",
            np.nanmin(tab[:, k]),
            np.nanmax(tab[:, k]),
        )


if __name__ == "__main__":
    main()
