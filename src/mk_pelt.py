#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from itertools import pairwise
from pathlib import Path

import lightgbm as lgb
import numpy as np
import rasterio
from pyproj import Transformer
from scipy.ndimage import binary_dilation, label
from scipy.stats import norm

from gbdt_cv import features
from roi import R

RB = R("box_b")
CELL_HA = 0.09
SENSOR = {"LT05": "5", "LE07": "7", "LC08": "8", "LC09": "9"}


def mann_kendall(x: np.ndarray) -> dict[str, float]:
    n = x.size
    t = np.arange(n, dtype=float)
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18
    slope = sen(t, x)[0]
    r = np.argsort(np.argsort(x - slope * t)).astype(float)
    r -= r.mean()
    acf = [np.sum(r[: n - k] * r[k:]) / np.sum(r * r) for k in range(1, n)]
    lim = norm.ppf(0.975) / np.sqrt(n)
    corr = 1 + 2 / (n * (n - 1) * (n - 2)) * sum(
        (n - k) * (n - k - 1) * (n - k - 2) * a for k, a in enumerate(acf, start=1) if abs(a) > lim
    )
    var *= max(corr, 1e-6)
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    tau = s / (0.5 * n * (n - 1))
    return {
        "S": float(s),
        "tau": float(tau),
        "z": float(z),
        "p": float(2 * norm.sf(abs(z))),
        "nstar_factor": float(corr),
    }


def sen(
    t: np.ndarray, x: np.ndarray, alpha: float = 0.05, var_factor: float = 1.0
) -> tuple[float, float, float, float]:
    i, j = np.triu_indices(t.size, 1)
    d = np.sort((x[j] - x[i]) / (t[j] - t[i]))
    n, m = t.size, d.size
    c = norm.ppf(1 - alpha / 2) * np.sqrt(var_factor * n * (n - 1) * (2 * n + 5) / 18)
    lo = int(np.clip(np.floor((m - c) / 2), 0, m - 1))
    hi = int(np.clip(np.ceil((m + c) / 2), 0, m - 1))
    b = float(np.median(d))
    return b, float(d[lo]), float(d[hi]), float(np.median(x - b * t))


def penalty(x: np.ndarray) -> float:
    sigma = 1.4826 * np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) / np.sqrt(2)
    return float(2 * sigma**2 * np.log(x.size))


def pelt(x: np.ndarray, pen: float, min_seg: int) -> list[int]:
    n = x.size
    cs, cs2 = np.r_[0, np.cumsum(x)], np.r_[0, np.cumsum(x * x)]

    def cost(a: int, b: int) -> float:
        m = b - a
        return float(cs2[b] - cs2[a] - (cs[b] - cs[a]) ** 2 / m)

    f = np.full(n + 1, np.inf)
    f[0] = -pen
    last = np.zeros(n + 1, int)
    cand = [0]
    for b in range(min_seg, n + 1):
        vals = [(f[a] + cost(a, b) + pen, a) for a in cand if b - a >= min_seg]
        if vals:
            f[b], last[b] = min(vals)
        cand = [a for a in cand if b - a < min_seg or f[a] + cost(a, b) <= f[b]]
        if b + 1 - min_seg >= min_seg:
            cand.append(b + 1 - min_seg)
    cps, b = [], n
    while b > 0:
        b = last[b]
        if b > 0:
            cps.append(int(b))
    return sorted(cps)


def masks(folder: Path, wc: np.ndarray, labels: np.ndarray, tr, crs) -> tuple[np.ndarray, np.ndarray]:
    built = wc == 50
    n = None
    for p in sorted(folder.glob("composite_*.tif")):
        t = features(p)[6]
        n = (t > 0.20).astype(int) if n is None else n + (t > 0.20)
    assert n is not None
    lab, _ = label(n >= 3, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel()) * CELL_HA
    size[0] = 0
    core = np.isin(lab, np.flatnonzero(size >= 2.0))
    yy, xx = np.mgrid[-5:6, -5:6]
    ra = binary_dilation(core, (yy**2 + xx**2) <= 25) & ~built
    h, w = wc.shape
    cols, rows = np.meshgrid(np.arange(w), np.arange(h))
    xs, ys = tr * (cols + 0.5, rows + 0.5)
    lon, lat = Transformer.from_crs(crs, 4326, always_xy=True).transform(xs, ys)
    inside = (lon >= RB[0]) & (lon <= RB[2]) & (lat >= RB[1]) & (lat <= RB[3])
    rb = inside & np.isin(labels, (1, 3, 4))
    return ra, rb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--composites", type=Path, default=Path("composites_v2"))
    ap.add_argument("--model", type=Path, default=Path("model_lgbm.txt"))
    ap.add_argument("--wc", type=Path, default=Path("wc30.u8"))
    ap.add_argument("--labels", type=Path, default=Path("labels30.tif"))
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--min-seg", type=int, default=3)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(a.composites / "composite_2025.tif") as s:
        tr, crs, h, w = s.transform, s.crs, s.height, s.width
    wc = np.fromfile(a.wc, np.uint8).reshape(h, w)
    with rasterio.open(a.labels) as s:
        labels = s.read(1)
    ra, rb = masks(a.composites, wc, labels, tr, crs)
    bst = lgb.Booster(model_file=str(a.model))

    rows = []
    for p in sorted(a.composites.glob("composite_*.tif")):
        x = features(p)
        with rasterio.open(p) as s:
            ids = s.tags().get("scenes", "").split(",")
        ok = np.all(np.isfinite(x), axis=0)
        cls = np.zeros((h, w), np.uint8)
        cls[ok] = bst.predict(x[:, ok].T).argmax(axis=1) + 1
        twi, blue, red = x[6], x[0], x[2]
        rows.append(
            {
                "year": int(p.stem.split("_")[1]),
                "sensors": "".join(sorted({SENSOR[i[:4]] for i in ids if i[:4] in SENSOR})),
                "ra_twi": round(float(np.nanmean(twi[ra])), 5),
                "ra_white_ha": round(float(((cls == 1) & ra).sum() * CELL_HA), 2),
                "ra_water_ha": round(float(((cls == 3) & ra).sum() * CELL_HA), 2),
                "rb_twi": round(float(np.nanmean(twi[rb])), 5),
                "rb_rb": round(float(np.nanmean(red[rb] / blue[rb])), 4),
            }
        )
    with open(a.out / "annual_series.csv", "w", newline="") as fh:
        cw = csv.DictWriter(fh, fieldnames=list(rows[0]))
        cw.writeheader()
        cw.writerows(rows)

    years = np.array([r["year"] for r in rows], float)
    res: dict[str, object] = {
        "n_years": int(years.size),
        "ra_extent_ha": float(ra.sum() * CELL_HA),
        "rb_cells": int(rb.sum()),
    }
    for var in ("ra_twi", "ra_white_ha", "ra_water_ha", "rb_twi", "rb_rb"):
        v = np.array([r[var] for r in rows], float)
        mk = mann_kendall(v)
        b, lo, hi, b0 = sen(years, v, var_factor=max(mk["nstar_factor"], 1.0))
        pen = penalty(v)
        cps = pelt(v, pen, a.min_seg)
        bounds = [0, *cps, v.size]
        segs = [
            {
                "from": int(years[s]),
                "to": int(years[e - 1]),
                "mean": float(v[s:e].mean()),
            }
            for s, e in pairwise(bounds)
        ]
        res[var] = {
            **mk,
            "sen": b,
            "sen_lo": lo,
            "sen_hi": hi,
            "intercept": b0,
            "penalty": pen,
            "change_years": [int(years[c]) for c in cps],
            "segments": segs,
        }
        print(var, json.dumps(res[var]))
    (a.out / "trends.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
