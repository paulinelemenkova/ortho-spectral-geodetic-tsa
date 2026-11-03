#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform, transform_bounds
from scipy.ndimage import distance_transform_edt
from scipy.optimize import curve_fit

from roi import R

H, W = R("grid")[3], R("grid")[2]
CRS = R("crs")
BOXES = [R("box_a"), R("box_b")]
SPLIT_LAT = R("split_lat")


def box(tr: rasterio.Affine, b: tuple[float, ...]) -> np.ndarray:
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", CRS, *b, densify_pts=21)
    m = np.zeros((H, W), bool)
    m[
        int((tr.f - y1) // 30) : int((tr.f - y0) // 30) + 1,
        int((x0 - tr.c) // 30) : int((x1 - tr.c) // 30) + 1,
    ] = True
    return m


def anomaly(t: np.ndarray, strata: np.ndarray, bg: np.ndarray) -> np.ndarray:
    out = np.full(t.shape, np.nan)
    for s in np.unique(strata[np.isfinite(t)]):
        cell = (strata == s) & np.isfinite(t)
        ref = t[cell & bg]
        if ref.size < 50:
            continue
        med = np.median(ref)
        mad = 1.4826 * np.median(np.abs(ref - med))
        if mad > 0:
            out[cell] = (t[cell] - med) / mad
    return out


def lag(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(z, 1, constant_values=np.nan)
    s = np.zeros(z.shape)
    n = np.zeros(z.shape)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                v = p[1 + dy : 1 + dy + z.shape[0], 1 + dx : 1 + dx + z.shape[1]]
                ok = np.isfinite(v)
                s += np.where(ok, v, 0)
                n += ok
    return s, n


def decay(d: np.ndarray, a: float, length: float, c: float) -> np.ndarray:
    return a * np.exp(-d / length) + c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites", type=Path, default=Path("../composites"))
    ap.add_argument("--aux", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--years", default="2013-2026")
    ap.add_argument("--dmax", type=float, default=6000.0)
    a = ap.parse_args()
    y0, y1 = (int(v) for v in a.years.split("-"))
    rng = np.random.default_rng(7)

    wc = np.fromfile(a.aux / "wc30.u8", np.uint8).reshape(H, W)
    dem = np.fromfile(a.aux / "dem30.i16", np.int16).reshape(H, W).astype(float)
    with rasterio.open(a.composites / f"composite_{y1}.tif") as s:
        prof, tr = s.profile, s.transform
    trav = box(tr, BOXES[0]) | box(tr, BOXES[1])
    strata = wc.astype(int) * 1000 + (dem // 50).astype(int)
    bg = ~trav & (wc != 80)

    lst, dts = [], []
    for y in range(y0, y1 + 1):
        with rasterio.open(a.composites / f"composite_{y}.tif") as s:
            t = s.read(7).astype(float) / 100
            t[(s.read(10) == 0) | (s.read(7) == s.nodata)] = np.nan
        lst.append(t)
        dts.append(anomaly(t, strata, bg))
    lst_med = np.nanmedian(lst, axis=0)
    dt = np.nanmedian(dts, axis=0)

    ok = np.isfinite(dt)
    z = np.where(ok, (dt - np.nanmean(dt)) / np.nanstd(dt), np.nan)
    s_lag, n_lag = lag(z)
    wz = np.where(n_lag > 0, s_lag / np.maximum(n_lag, 1), np.nan)
    moran = float(np.nansum(z * wz) / np.nansum(z * z))
    local = z * wz
    zv = z[ok]
    ge = np.zeros(z.shape)
    nperm = 199
    for _ in range(nperm):
        draw = rng.choice(zv, size=(8,) + z.shape)
        mask = np.arange(8)[:, None, None] < n_lag[None]
        sim = z * np.where(mask, draw, 0).sum(0) / np.maximum(n_lag, 1)
        ge += np.abs(sim) >= np.abs(local)
    p = (ge + 1) / (nperm + 1)
    lisa = np.zeros(z.shape, np.uint8)
    sig = ok & (p <= 0.01)
    lisa[sig & (z > 0) & (wz > 0)] = 1
    lisa[sig & (z < 0) & (wz < 0)] = 2
    lisa[sig & (z < 0) & (wz > 0)] = 3
    lisa[sig & (z > 0) & (wz < 0)] = 4

    faults = json.loads((a.aux / "faults.json").read_text())
    geoms = []
    for f in faults:
        xs, ys = transform("EPSG:4326", CRS, *zip(*f["xy"], strict=True))
        geoms.append({"type": "LineString", "coordinates": list(zip(xs, ys, strict=True))})
    line = rasterize([(g, 1) for g in geoms], (H, W), transform=tr, all_touched=True)
    dist = distance_transform_edt(line == 0) * 30.0
    _, lat = transform(CRS, "EPSG:4326", [tr.c + 15.0] * H, [tr.f - 15 - 30 * i for i in range(H)])
    latg = np.repeat(np.array(lat)[:, None], W, axis=1)
    segs = {"S": latg < SPLIT_LAT, "N": latg >= SPLIT_LAT}

    rows, fits = [], []
    edges = np.arange(0, a.dmax + 1, 250.0)
    for nm, sm in segs.items():
        m = ok & sm & (dist <= a.dmax)
        d, v = dist[m], dt[m]
        for lo, hi in pairwise(edges):
            b = (d >= lo) & (d < hi)
            if b.sum() >= 20:
                q = np.percentile(v[b], [25, 50, 75])
                rows.append([nm, (lo + hi) / 2, *q, int(b.sum())])
        p0 = (2.0, 1000.0, 0.0)
        bounds = ([0, 50, -5], [50, 20000, 5])
        popt, _ = curve_fit(decay, d, v, p0=p0, bounds=bounds, maxfev=20000)
        boot = []
        for _ in range(500):
            i = rng.integers(0, d.size, d.size)
            try:
                boot.append(curve_fit(decay, d[i], v[i], p0=popt, bounds=bounds, maxfev=5000)[0])
            except RuntimeError:
                continue
        bs = np.array(boot)
        dd = np.linspace(0, a.dmax, 121)
        curves = np.array([decay(dd, *b) for b in bs])
        lo95, hi95 = np.percentile(curves, [2.5, 97.5], axis=0)
        r2 = 1 - np.sum((v - decay(d, *popt)) ** 2) / np.sum((v - v.mean()) ** 2)
        fits.append(
            {
                "segment": nm,
                "A": popt[0],
                "A_lo": np.percentile(bs[:, 0], 2.5),
                "A_hi": np.percentile(bs[:, 0], 97.5),
                "L": popt[1],
                "L_lo": np.percentile(bs[:, 1], 2.5),
                "L_hi": np.percentile(bs[:, 1], 97.5),
                "c": popt[2],
                "c_lo": np.percentile(bs[:, 2], 2.5),
                "c_hi": np.percentile(bs[:, 2], 97.5),
                "r2": r2,
                "n": int(d.size),
                "curve_d": dd.tolist(),
                "curve": decay(dd, *popt).tolist(),
                "lo95": lo95.tolist(),
                "hi95": hi95.tolist(),
            }
        )

    a.out.mkdir(parents=True, exist_ok=True)
    prof.update(count=1, dtype="float32", nodata=np.nan, compress="deflate")
    for nm, arr in (("lst_med", lst_med), ("dT_med", dt), ("dist_fault", dist)):
        with rasterio.open(a.out / f"{nm}.tif", "w", **prof) as dst:
            dst.write(arr.astype(np.float32), 1)
    prof.update(dtype="uint8", nodata=None)
    with rasterio.open(a.out / "lisa.tif", "w", **prof) as dst:
        dst.write(lisa, 1)
    with open(a.out / "decay_bins.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["segment", "d", "q25", "median", "q75", "n"])
        wr.writerows(rows)
    (a.out / "decay_fit.json").write_text(json.dumps(fits))
    with open(a.out / "thermal_stats.txt", "w") as fh:
        fh.write(f"years {y0}-{y1}; Moran I {moran:.3f}\n")
        for k, v in zip(("HH", "LL", "LH", "HL"), range(1, 5), strict=True):
            fh.write(f"LISA {k}: {int((lisa == v).sum())} cells\n")
        for nm, m in (
            ("box_a", box(tr, BOXES[0])),
            ("box_b", box(tr, BOXES[1])),
        ):
            fh.write(
                f"{nm}: median dT {np.nanmedian(dt[m]):.2f}, p90 {np.nanpercentile(dt[m], 90):.2f}, "
                f"median LST {np.nanmedian(lst_med[m]):.1f} C, HH cells {int((lisa[m] == 1).sum())}\n"
            )
        fh.write(f"window: median LST {np.nanmedian(lst_med):.1f} C\n")
        for f in fits:
            fh.write(
                f"{f['segment']}: A {f['A']:.2f} [{f['A_lo']:.2f}, {f['A_hi']:.2f}], "
                f"L {f['L']:.0f} m [{f['L_lo']:.0f}, {f['L_hi']:.0f}], c {f['c']:.2f} "
                f"[{f['c_lo']:.2f}, {f['c_hi']:.2f}], R2 {f['r2']:.3f}, n {f['n']}\n"
            )
    print((a.out / "thermal_stats.txt").read_text())


if __name__ == "__main__":
    main()
