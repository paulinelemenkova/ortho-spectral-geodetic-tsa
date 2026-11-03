#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform

from roi import R

UPLAND_MIN_Z = 600.0
POINTS = R("probes")


warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def read_h5(path: Path, name: str, band: int | None = None) -> np.ndarray:
    try:
        import h5py
    except ImportError:
        with rasterio.open(f'HDF5:"{path}"://{name}') as src:
            return np.asarray(src.read(band if band is not None else 1))
    with h5py.File(path, "r") as f:
        ds = f[name]
        return np.asarray(ds[band - 1] if band is not None else ds[:])


def h5_dates(path: Path, n: int) -> list[dt.date]:
    raw = path.read_bytes()
    hit = re.search(rb"(?:(?:19|20)\d{2}[01]\d[0-3]\d){%d}" % n, raw)
    if hit is None:
        raise RuntimeError(f"no date list of length {n} in {path}")
    s = hit.group().decode()
    utc = dt.UTC
    return [
        dt.datetime.strptime(s[k : k + 8], "%Y%m%d").replace(tzinfo=utc).date() for k in range(0, 8 * n, 8)
    ]


def to_ll(arr: np.ndarray, src: rasterio.DatasetReader, path: Path) -> None:
    b = src.bounds
    lo0, la0 = transform(src.crs, "EPSG:4326", [b.left, b.right], [b.bottom, b.top])
    res = 0.0008
    w = int(np.ceil((max(lo0) - min(lo0)) / res))
    h = int(np.ceil((max(la0) - min(la0)) / res))
    dst_tr = from_origin(min(lo0), max(la0), res, res)
    out = np.full((h, w), np.nan, np.float32)
    reproject(
        arr.astype(np.float32),
        out,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_tr,
        dst_crs="EPSG:4326",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    prof = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": dst_tr,
        "nodata": np.nan,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(out, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mintpy", type=Path, default=Path("mintpy_asc"))
    ap.add_argument("--out", type=Path, default=Path("."))
    args = ap.parse_args()
    m = args.mintpy
    args.out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(m / "velocity.tif") as src:
        v = src.read(1).astype(np.float64) * 1000.0
        with rasterio.open(m / "maskTempCoh.tif") as mk:
            ok = (mk.read(1) > 0) & np.isfinite(v) & (v != 0)
        geo = m / "inputs" / "geometryGeo.h5"
        z = read_h5(geo, "height").astype(np.float64)
        inc = np.deg2rad(read_h5(geo, "incidenceAngle").astype(np.float64))
        up = ok & (z > UPLAND_MIN_Z)
        ref = float(np.median(v[up]))
        vlos = np.where(ok, v - ref, np.nan)
        vup = vlos / np.cos(inc)
        to_ll(vlos, src, args.out / "vlos_ll.tif")
        to_ll(vup, src, args.out / "vup_ll.tif")

        floor = ok & (z < 250)
        stats = {
            "reference_los_mm_yr": ref,
            "upland_pixels": int(up.sum()),
            "upland_mad_mm_yr": float(np.median(np.abs(v[up] - ref))),
            "valid_fraction": float(ok.mean()),
            "incidence_median_deg": float(np.rad2deg(np.median(inc[ok]))),
            "floor_vlos_median": float(np.nanmedian(vlos[floor])),
            "floor_vup_median": float(np.nanmedian(vup[floor])),
            "vup_min": float(np.nanmin(vup)),
            "area_vup_lt_m10_km2": float(np.sum(vup < -10) * 0.08**2),
        }

        rows: dict[str, list[float]] = {}
        tsf = m / "timeseries_demErr.h5"
        with rasterio.open(f'HDF5:"{tsf}"://timeseries') as f:
            n_dates = f.count
        dates = h5_dates(tsf, n_dates)
        ix = {}
        for lab, (lo, la, _) in POINTS.items():
            x, y = transform("EPSG:4326", src.crs, [lo], [la])
            i, j = src.index(x[0], y[0])
            ix[lab] = (i, j)
            stats[f"{lab}_vup"] = float(np.nanmedian(vup[i - 1 : i + 2, j - 1 : j + 2]))
        for k in range(n_dates):
            d = read_h5(tsf, "timeseries", k + 1).astype(np.float64) * 1000.0
            d -= np.median(d[up])
            for lab, (i, j) in ix.items():
                blk = d[i - 1 : i + 2, j - 1 : j + 2]
                msk = ok[i - 1 : i + 2, j - 1 : j + 2]
                c = float(np.cos(inc[i, j]))
                val = float(np.median(blk[msk])) / c if msk.any() else np.nan
                rows.setdefault(lab, []).append(val)

    with open(args.out / "points.txt", "w") as fh:
        for lab, (lo, la, desc) in POINTS.items():
            fh.write(f"{lo} {la} {lab} {desc}\n")
    with open(args.out / "ts_points.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["date", *POINTS])
        for k, d in enumerate(dates):
            wr.writerow([d.isoformat(), *(f"{rows[lab][k]:.2f}" for lab in POINTS)])
    with open(args.out / "insar_stats.txt", "w") as fh:
        for key, val in stats.items():
            fh.write(f"{key} {val:.4f}\n" if isinstance(val, float) else f"{key} {val}\n")
    print((args.out / "insar_stats.txt").read_text())


if __name__ == "__main__":
    main()
