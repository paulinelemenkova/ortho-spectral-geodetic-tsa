#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy.io import netcdf_file
from scipy.stats import norm, rankdata

from gbdt_cv import features
from roi import R


def sen_mk(v: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = t.size
    i, j = np.triu_indices(n, 1)
    dv = v[j] - v[i]
    slope = np.nanmedian(dv / (t[j] - t[i])[:, None], axis=0)
    s = np.nansum(np.sign(dv), axis=0)
    var = n * (n - 1) * (2 * n + 5) / 18.0
    r = rankdata(v - slope[None, :] * t[:, None], axis=0)
    r -= r.mean(axis=0)
    den = np.sum(r * r, axis=0)
    lim = norm.ppf(0.975) / np.sqrt(n)
    corr = np.ones(v.shape[1])
    for k in range(1, n - 2):
        a = np.sum(r[: n - k] * r[k:], axis=0) / den
        corr += np.where(np.abs(a) > lim, 2 * (n - k) * (n - k - 1) * (n - k - 2) * a, 0) / (
            n * (n - 1) * (n - 2)
        )
    z = np.where(s != 0, (s - np.sign(s)) / np.sqrt(var * np.maximum(corr, 1e-6)), 0.0)
    return slope, 2 * norm.sf(np.abs(z))


def write_nc(path: Path, lon: np.ndarray, lat: np.ndarray, z: np.ndarray) -> None:
    with netcdf_file(path, "w") as f:
        f.createDimension("lon", lon.size)
        f.createDimension("lat", lat.size)
        vx = f.createVariable("lon", "f8", ("lon",))
        vx[:] = lon
        vy = f.createVariable("lat", "f8", ("lat",))
        vy[:] = lat
        vz = f.createVariable("z", "f4", ("lat", "lon"))
        vz[:] = z.astype(np.float32)
        vz._FillValue = np.float32(np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites", type=Path, default=Path("composites_v2"))
    ap.add_argument("--wc", type=Path, default=Path("wc30.u8"))
    ap.add_argument("--extent-mask", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--region", type=float, nargs=4, default=list(R("region")))
    ap.add_argument("--res", type=float, default=0.0003)
    ap.add_argument("--min-years", type=int, default=35)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(a.composites.glob("composite_*.tif"))
    years = np.array([int(p.stem.split("_")[1]) for p in paths], float)
    with rasterio.open(paths[0]) as s:
        tr, crs, h, w = s.transform, s.crs, s.height, s.width
    twi = np.stack([features(p)[6] for p in paths]).reshape(len(paths), -1)
    built = (np.fromfile(a.wc, np.uint8) == 50).reshape(-1)
    ok = (np.isfinite(twi).sum(axis=0) >= a.min_years) & ~built
    slope = np.full(h * w, np.nan)
    pval = np.full(h * w, np.nan)

    v = twi[:, ok]
    v = np.where(np.isfinite(v), v, np.nanmedian(v, axis=0))
    slope[ok], pval[ok] = sen_mk(v, years)
    slope *= 10
    cls = np.full(h * w, np.nan)
    cls[ok] = 3
    cls[ok & (slope < 0) & (pval < 0.05)] = 2
    cls[ok & (slope < 0) & (pval < 0.01)] = 1
    cls[ok & (slope > 0) & (pval < 0.05)] = 4
    cls[ok & (slope > 0) & (pval < 0.01)] = 5
    cls[built] = 6

    stats: dict[str, object] = {"n_years": int(years.size)}
    if a.extent_mask is not None:
        m = np.load(a.extent_mask).reshape(-1) & ok
        stats["extent_cells"] = int(m.sum())
        for k, name in enumerate(("dec01", "dec05", "ns", "inc05", "inc01"), start=1):
            stats[name] = float(np.mean(cls[m] == k))
        stats["median_slope"] = float(np.median(slope[m]))
        stats["p90_slope"] = float(np.percentile(slope[m], 90))
        stats["p10_slope"] = float(np.percentile(slope[m], 10))
    win = ok & ~np.isnan(slope)
    stats["window_median_slope"] = float(np.median(slope[win]))
    stats["window_share_inc05"] = float(np.mean(cls[win] >= 4))
    stats["window_share_dec05"] = float(np.mean(cls[win] <= 2))
    (a.out / "trend_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    x0, x1, y0, y1 = a.region
    glon = np.arange(x0 + a.res / 2, x1, a.res)
    glat = np.arange(y0 + a.res / 2, y1, a.res)
    gx, gy = np.meshgrid(glon, glat)
    ux, uy = Transformer.from_crs(4326, crs, always_xy=True).transform(gx, gy)
    gc, gr = ~tr * (ux, uy)
    idx = np.clip(np.floor(gr).astype(int), 0, h - 1) * w + np.clip(np.floor(gc).astype(int), 0, w - 1)
    write_nc(a.out / "sen_twi.nc", glon, glat, slope[idx])
    write_nc(a.out / "mk_class.nc", glon, glat, cls[idx])


if __name__ == "__main__":
    main()
