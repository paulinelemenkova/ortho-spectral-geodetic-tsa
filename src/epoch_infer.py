#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.io import netcdf_file
from scipy.ndimage import binary_dilation, label

from gbdt_cv import features
from roi import R

EPOCHS = (1985, 1995, 2005, 2015, 2025)
RB = R("box_b")
CELL_HA = 0.09


def twi_counts(folder: Path) -> np.ndarray:
    n = None
    for p in sorted(folder.glob("composite_*.tif")):
        t = features(p)[6]
        n = (t > 0.20).astype(int) if n is None else n + (t > 0.20)
    assert n is not None
    return n


def write_nc(path: Path, lon: np.ndarray, lat: np.ndarray, z: np.ndarray) -> None:
    with netcdf_file(path, "w") as f:
        f.createDimension("lon", lon.size)
        f.createDimension("lat", lat.size)
        vx = f.createVariable("lon", "f8", ("lon",))
        vx[:] = lon
        vx.units = "degrees_east"
        vy = f.createVariable("lat", "f8", ("lat",))
        vy[:] = lat
        vy.units = "degrees_north"
        vz = f.createVariable("z", "i1", ("lat", "lon"))
        vz[:] = z.astype(np.int8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites", type=Path, default=Path("composites_v2"))
    ap.add_argument("--model", type=Path, default=Path("model_lgbm.txt"))
    ap.add_argument("--wc", type=Path, default=Path("wc30.u8"))
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--region", type=float, nargs=4, default=list(R("region")))
    ap.add_argument("--res", type=float, default=0.0002)
    ap.add_argument("--min-years", type=int, default=3)
    ap.add_argument("--min-ha", type=float, default=2.0)
    ap.add_argument("--buffer", type=float, default=150.0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(a.composites / "composite_2025.tif") as s:
        tr, crs, h, w = s.transform, s.crs, s.height, s.width
    wc = np.fromfile(a.wc, np.uint8).reshape(h, w)
    built = wc == 50
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = tr * (cols + 0.5, rows + 0.5)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(xs, ys)

    lab, _ = label(twi_counts(a.composites) >= a.min_years, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel()) * CELL_HA
    size[0] = 0
    core = np.isin(lab, np.flatnonzero(size >= a.min_ha))
    r = round(a.buffer / 30)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    ra = binary_dilation(core, (yy**2 + xx**2) <= r * r) & ~built
    rb = (lon >= RB[0]) & (lon <= RB[2]) & (lat >= RB[1]) & (lat <= RB[3]) & ~built

    bst = lgb.Booster(model_file=str(a.model))
    x0, x1, y0, y1 = a.region
    glon = np.arange(x0 + a.res / 2, x1, a.res)
    glat = np.arange(y0 + a.res / 2, y1, a.res)
    gx, gy = np.meshgrid(glon, glat)
    ux, uy = Transformer.from_crs(4326, crs, always_xy=True).transform(gx, gy)
    gc, gr = ~tr * (ux, uy)
    gc = np.clip(np.floor(gc).astype(int), 0, w - 1)
    gr = np.clip(np.floor(gr).astype(int), 0, h - 1)
    rows_out = []
    for yr in EPOCHS:
        x = features(a.composites / f"composite_{yr}.tif")
        ok = np.all(np.isfinite(x), axis=0)
        cls = np.zeros((h, w), np.uint8)
        cls[ok] = bst.predict(x[:, ok].T).argmax(axis=1) + 1
        cls[built] = 6
        write_nc(a.out / f"class_{yr}.nc", glon, glat, cls[gr, gc])
        for site, m in (("a", ra), ("b", rb)):
            rows_out.append(
                [yr, site] + [round(float(((cls == k) & m).sum() * CELL_HA), 2) for k in range(1, 6)]
            )
    with open(a.out / "class_areas.csv", "w", newline="") as fh:
        cw = csv.writer(fh)
        cw.writerow(
            [
                "year",
                "site",
                "active_white",
                "inactive_grey",
                "water_covered",
                "iron_stained",
                "other",
            ]
        )
        cw.writerows(rows_out)

    cs = plt.contour(glon, glat, ra[gr, gc].astype(float), levels=[0.5])
    with open(a.out / "extent.gmt", "w") as fh:
        fh.writelines(">\n" + "".join(f"{p[0]:.6f} {p[1]:.6f}\n" for p in seg) for seg in cs.allsegs[0])
    print(
        f"core {core.sum() * CELL_HA:.1f} ha, extent_a {ra.sum() * CELL_HA:.1f} ha,"
        f" box_b {rb.sum() * CELL_HA:.1f} ha"
    )
    for row in rows_out:
        print(*row)


if __name__ == "__main__":
    main()
