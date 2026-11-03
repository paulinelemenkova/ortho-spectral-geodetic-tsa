#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from rasterio.windows import Window

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
PLATFORMS = ("landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9")
SLC_OFF = "2003-05-31"
SR_BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
NODATA = -32768

ROY_A = np.array([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172], np.float32)
ROY_B = np.array([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071], np.float32)
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "VSI_CACHE": "TRUE",
}
_TOKEN: dict[str, Any] = {}


def http_json(url: str, payload: dict[str, Any] | None = None, tries: int = 5) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=40) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 + 3 * k)
    raise RuntimeError(f"request failed: {url}")


def sign(href: str) -> str:
    if "t" not in _TOKEN or time.time() - _TOKEN["t"] > 1800:
        _TOKEN.update(tok=http_json(SAS_URL)["token"], t=time.time())
    return f"{href}?{_TOKEN['tok']}"


def search(bbox: list[float], year: int, max_cloud: float) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "collections": ["landsat-c2-l2"],
        "intersects": {
            "type": "Polygon",
            "coordinates": [
                [
                    [bbox[0], bbox[1]],
                    [bbox[2], bbox[1]],
                    [bbox[2], bbox[3]],
                    [bbox[0], bbox[3]],
                    [bbox[0], bbox[1]],
                ]
            ],
        },
        "datetime": f"{year}-06-01T00:00:00Z/{year}-09-30T23:59:59Z",
        "limit": 250,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    items = http_json(STAC_URL, body).get("features", [])
    keep = []
    for it in items:
        p = it["properties"]
        if p.get("platform") in PLATFORMS and p.get("landsat:collection_category") == "T1":
            keep.append(it)

    best: dict[tuple[str, str], dict[str, Any]] = {}
    for it in keep:
        key = (it["properties"]["platform"], it["properties"]["datetime"][:10])
        if key not in best or it["properties"]["eo:cloud_cover"] < best[key]["properties"]["eo:cloud_cover"]:
            best[key] = it
    return list(best.values())


def make_grid(bbox: list[float], crs: str, res: float = 30.0) -> tuple[Affine, int, int]:
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    x0, y0 = math.floor(x0 / res) * res, math.floor(y0 / res) * res
    x1, y1 = math.ceil(x1 / res) * res, math.ceil(y1 / res) * res
    return from_origin(x0, y1, res, res), int((x1 - x0) / res), int((y1 - y0) / res)


def box_mask(boxes: list[list[float]], crs: str, tr: Affine, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), bool)
    for b in boxes:
        x0, y0, x1, y1 = transform_bounds("EPSG:4326", crs, *b, densify_pts=21)
        c0, c1 = max(0, int((x0 - tr.c) // tr.a)), min(w, math.ceil((x1 - tr.c) / tr.a))
        r0, r1 = max(0, int((tr.f - y1) // tr.a)), min(h, math.ceil((tr.f - y0) / tr.a))
        m[r0:r1, c0:c1] = True
    return m


def read(href: str, crs: str, tr: Affine, w: int, h: int) -> np.ndarray:
    for k in range(3):
        try:
            with rasterio.Env(**GDAL_ENV), rasterio.open(sign(href)) as src:
                st = src.transform
                dx, dy = (tr.c - st.c) / st.a, (tr.f - st.f) / st.e
                if (
                    str(src.crs) == crs
                    and st.a == tr.a
                    and st.e == tr.e
                    and dx == round(dx)
                    and dy == round(dy)
                ):
                    win = Window(int(dx), int(dy), w, h)
                    return src.read(1, window=win, boundless=True, fill_value=0)
            with (
                rasterio.Env(**GDAL_ENV),
                rasterio.open(sign(href)) as src,
                WarpedVRT(
                    src,
                    crs=crs,
                    transform=tr,
                    width=w,
                    height=h,
                    resampling=Resampling.nearest,
                ) as vrt,
            ):
                return vrt.read(1)
        except rasterio.errors.RasterioIOError:
            time.sleep(3 + 5 * k)
    raise RuntimeError(f"read failed: {href}")


def qa_bits(qa: np.ndarray) -> dict[str, np.ndarray]:
    q = qa.astype(np.uint16)
    return {
        "fill": (q & 1) > 0,
        "dcloud": ((q >> 1) & 1) > 0,
        "cloud": ((q >> 3) & 1) > 0,
        "shadow": ((q >> 4) & 1) > 0,
    }


def twi(sr: np.ndarray) -> np.ndarray:
    b, g, r = sr[0], sr[1], sr[2]
    mx = np.fmax(np.fmax(b, g), r)
    mn = np.fmin(np.fmin(b, g), r)
    with np.errstate(all="ignore"):
        return (b + g + r) / 3 * (1 - (mx - mn) / mx)


def qmap(x: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    ux, last = np.unique(qx[::-1], return_index=True)
    qx, qy = ux, qy[::-1][last]
    y = np.interp(x, qx, qy)
    y = np.where(x < qx[0], qy[0] + (x - qx[0]), y)
    y = np.where(x > qx[-1], qy[-1] + (x - qx[-1]), y)
    return np.where((x > qx[-1]) & (x >= 1.0), qy[-1], y)


def harmonise(sr: np.ndarray, platform: str, bright: dict[str, Any] | None) -> np.ndarray:
    if platform not in ("landsat-4", "landsat-5", "landsat-7"):
        return sr
    sr = ROY_A[:, None, None] + ROY_B[:, None, None] * sr
    if bright is None:
        return sr
    w = np.nan_to_num(np.clip((twi(sr) - 0.05) / 0.05, 0, 1))
    steps = ["l7l8"] if platform == "landsat-7" else ["l5l7", "l7l8"]
    out = sr.copy()
    for j in range(6):
        x = sr[j]
        for st in steps:
            x = qmap(
                x,
                np.asarray(bright[st]["q_src"][j]),
                np.asarray(bright[st]["q_ref"][j]),
            )
        out[j] = sr[j] + w * (x - sr[j])
    return out.astype(np.float32)


def reuse(year: int, folder: Path) -> list[dict[str, Any]]:
    with open(folder / f"scenes_{year}.csv") as fh:
        ids = [r["id"] for r in csv.DictReader(fh) if r["used"] == "1"]
    if not ids:
        return []
    cache = folder / "items.json"
    if cache.exists():
        items = json.loads(cache.read_text())
    else:
        body = {"collections": ["landsat-c2-l2"], "ids": ids, "limit": 100}
        items = {it["id"]: it for it in http_json(STAC_URL, body).get("features", [])}
    return [items[i] for i in ids if i in items]


def process_year(year: int, cfg: dict[str, Any], args: argparse.Namespace, pool: ThreadPoolExecutor) -> None:
    win = cfg["windows"]["core"]
    crs = cfg["crs"]
    tr, w, h = make_grid(win["bbox"], crs)
    inbox = box_mask(win["screen_exclude"], crs, tr, w, h)
    items = reuse(year, args.scene_list) if args.scene_list else search(win["bbox"], year, args.max_cloud)
    sign("")
    items.sort(
        key=lambda it: (
            it["properties"]["platform"] == "landsat-7" and it["properties"]["datetime"][:10] > SLC_OFF,
            it["properties"]["eo:cloud_cover"],
        )
    )
    log: list[dict[str, Any]] = []

    def qa_of(it: dict[str, Any]) -> np.ndarray:
        return read(it["assets"]["qa_pixel"]["href"], crs, tr, w, h)

    qas = list(pool.map(qa_of, items))
    chosen: list[tuple[dict[str, Any], np.ndarray]] = []
    for it, qa in zip(items, qas, strict=True):
        b = qa_bits(qa)
        valid = ~b["fill"] & ~inbox
        clear = valid & ~b["dcloud"] & ~b["cloud"] & ~b["shadow"]
        frac = float(clear.sum() / max(valid.sum(), 1))
        cover = float(valid.sum() / (~inbox).sum())
        ok = args.scene_list is not None or (
            frac >= args.min_clear and cover >= 0.5 and len(chosen) < args.per_year
        )
        log.append(
            {
                "id": it["id"],
                "date": it["properties"]["datetime"][:10],
                "platform": it["properties"]["platform"],
                "scene_cloud": it["properties"]["eo:cloud_cover"],
                "clear_frac": round(frac, 3),
                "valid_frac": round(cover, 3),
                "used": int(ok),
            }
        )
        if ok:
            chosen.append((it, qa))

    def lwir_key(it: dict[str, Any]) -> str:
        return "lwir11" if "lwir11" in it["assets"] else "lwir"

    jobs = []
    for it, _ in chosen:
        for band in (*SR_BANDS, lwir_key(it), "qa"):
            jobs.append((it, band))
    arrays = list(pool.map(lambda j: read(j[0]["assets"][j[1]]["href"], crs, tr, w, h), jobs))
    nb = len(SR_BANDS) + 2
    sr_stack, lst_stack, stqa_stack = [], [], []
    for k, (it, qa) in enumerate(chosen):
        arr = arrays[k * nb : (k + 1) * nb]
        b = qa_bits(qa)
        bad = b["fill"] | b["shadow"] | (~inbox & (b["dcloud"] | b["cloud"]))
        sr = np.stack(arr[:6]).astype(np.float32) * 2.75e-5 - 0.2
        bad |= np.any(np.stack(arr[:6]) == 0, axis=0)
        sr = harmonise(sr, it["properties"]["platform"], args.bright)
        sr[:, bad] = np.nan
        sr_stack.append(sr)
        st = arr[6].astype(np.float32) * 0.00341802 + 149.0 - 273.15
        stqa = arr[7].astype(np.float32) * 0.01

        stbad = bad | (arr[6] == 0) | (arr[7] < 0) | (~inbox & (stqa > args.stqa_max))
        st[stbad] = np.nan
        stqa[bad | (arr[7] < 0)] = np.nan
        lst_stack.append(st)
        stqa_stack.append(stqa)

    out = np.full((10, h, w), NODATA, np.int16)
    if chosen:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            srm = np.nanmedian(np.stack(sr_stack), axis=0)
            lstm = np.nanmedian(np.stack(lst_stack), axis=0)
            qam = np.nanmedian(np.stack(stqa_stack), axis=0)
        nsr = np.sum(~np.isnan(np.stack(sr_stack)[:, 0]), axis=0)
        nlst = np.sum(~np.isnan(np.stack(lst_stack)), axis=0)
        for i in range(6):
            v = np.clip(np.round(srm[i] * 10000), -32767, 32767)
            out[i] = np.where(np.isnan(srm[i]), NODATA, v).astype(np.int16)
        out[6] = np.where(np.isnan(lstm), NODATA, np.round(lstm * 100)).astype(np.int16)
        out[7] = np.where(np.isnan(qam), NODATA, np.round(qam * 100)).astype(np.int16)
        out[8] = nsr.astype(np.int16)
        out[9] = nlst.astype(np.int16)
    prof = {
        "driver": "GTiff",
        "dtype": "int16",
        "count": 10,
        "width": w,
        "height": h,
        "crs": crs,
        "transform": tr,
        "nodata": NODATA,
        "compress": "deflate",
        "predictor": 2,
        "zlevel": 9,
        "tiled": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with rasterio.open(args.out / f"composite_{year}.tif", "w", **prof) as dst:
        dst.write(out)
        names = (
            "blue",
            "green",
            "red",
            "nir",
            "swir1",
            "swir2",
            "lst",
            "st_qa",
            "n_sr",
            "n_lst",
        )
        for i, nm in enumerate(names, 1):
            dst.set_band_description(i, nm)
        dst.update_tags(
            year=year,
            scenes=",".join(it["id"] for it, _ in chosen),
            harmonisation="TM/ETM+ to OLI, Roy et al. 2016 OLS"
            + ("; bright-surface quantile mapping" if args.bright else ""),
            stqa_max_K=args.stqa_max,
        )
    with open(args.out / f"scenes_{year}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(
            fh,
            fieldnames=[
                "id",
                "date",
                "platform",
                "scene_cloud",
                "clear_frac",
                "valid_frac",
                "used",
            ],
        )
        wr.writeheader()
        wr.writerows(log)
    nsr_med = int(np.median(out[8])) if chosen else 0
    print(
        f"{year}: candidates {len(items)}, used {len(chosen)} "
        f"({','.join(sorted({it['properties']['platform'][-1] for it, _ in chosen}))}), "
        f"median clear obs {nsr_med}, grid {w}x{h}",
        flush=True,
    )


def parse_years(text: str) -> list[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--years", required=True, help="e.g. 1984-1988")
    ap.add_argument("--out", type=Path, default=Path("composites"))
    ap.add_argument("--per-year", type=int, default=8)
    ap.add_argument("--max-cloud", type=float, default=60.0)
    ap.add_argument("--min-clear", type=float, default=0.80)
    ap.add_argument("--stqa-max", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--bright-map", type=Path, help="bright_map.json (quantile mapping)")
    ap.add_argument("--scene-list", type=Path, help="folder with scenes_<YYYY>.csv")
    args = ap.parse_args()
    args.bright = json.loads(args.bright_map.read_text()) if args.bright_map else None
    cfg = json.loads(args.config.read_text())
    with ThreadPoolExecutor(args.workers) as pool:
        for year in parse_years(args.years):
            t0 = time.time()
            process_year(year, cfg, args, pool)
            print(f"   {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
