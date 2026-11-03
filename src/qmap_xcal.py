#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

from roi import R

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token/landsat-c2-l2"
X0, Y0, W, H = R("grid")
CRS = R("crs")
WIN = R("window")
BOXES = [R("box_a"), R("box_b")]
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
ROY_A = np.array([0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172], np.float32)
ROY_B = np.array([0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071], np.float32)
RADSAT = {"landsat-5": (0, 1, 2), "landsat-7": (0, 1, 2), "landsat-8": (1, 2, 3)}
ENV = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "GDAL_HTTP_MAX_RETRY": "4"}
TOKEN = ""


def post(body: dict[str, Any]) -> Any:
    req = urllib.request.Request(STAC, json.dumps(body).encode(), {"Content-Type": "application/json"})
    for k in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except OSError:
            time.sleep(2 + 2 * k)
    raise RuntimeError("STAC")


def box(b: tuple[float, ...]) -> np.ndarray:
    x0, y0, x1, y1 = transform_bounds("EPSG:4326", CRS, *b, densify_pts=21)
    m = np.zeros((H, W), bool)
    m[
        int((Y0 - y1) // 30) : int((Y0 - y0) // 30) + 1,
        int((x0 - X0) // 30) : int((x1 - X0) // 30) + 1,
    ] = True
    return m


INBOX = box(BOXES[0]) | box(BOXES[1])
PBOX = box(BOXES[0])


def read(href: str) -> np.ndarray:
    for k in range(4):
        try:
            with rasterio.Env(**ENV), rasterio.open(f"{href}?{TOKEN}") as s:
                w = from_bounds(X0, Y0 - H * 30, X0 + W * 30, Y0, s.transform)
                w = w.round_offsets().round_lengths()
                return s.read(1, window=w, boundless=True, fill_value=0)
        except (OSError, ValueError):
            time.sleep(2 + 3 * k)
    raise RuntimeError(href)


def scenes(platform: str, start: str, end: str) -> dict[str, dict[str, Any]]:
    body = {
        "collections": ["landsat-c2-l2"],
        "bbox": list(WIN),
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 500,
        "query": {
            "eo:cloud_cover": {"lt": 10},
            "platform": {"eq": platform},
            "landsat:wrs_path": {"eq": R("wrs")[:3]},
            "landsat:wrs_row": {"eq": R("wrs")[3:]},
            "landsat:collection_category": {"eq": "T1"},
        },
    }
    out = {}
    for it in post(body)["features"]:
        d = it["properties"]["datetime"][:10]
        if 6 <= int(d[5:7]) <= 9:
            out[d] = it
    return out


def by_ids(sensor: str, dates: list[str]) -> dict[str, dict[str, Any]]:
    ids = [f"{sensor}_L2SP_{R('wrs')}_{d.replace('-', '')}_02_T1" for d in dates]
    out: dict[str, dict[str, Any]] = {}
    for k in range(0, len(ids), 50):
        body = {"collections": ["landsat-c2-l2"], "ids": ids[k : k + 50], "limit": 100}
        for it in post(body)["features"]:
            if it["properties"].get("eo:cloud_cover", 100) < 10:
                out[it["properties"]["datetime"][:10]] = it
    return out


def pairs_of(a: dict[str, Any], b: dict[str, Any]) -> list[tuple[str, str]]:
    out = []
    for db in sorted(b):
        for off in (-8, 8):
            da = (dt.date.fromisoformat(db) + dt.timedelta(days=off)).isoformat()
            if da in a:
                out.append((da, db))
                break
    return out


KEYS = (*BANDS, "qa_pixel", "qa_radsat")


def load_all(items: dict[str, dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    jobs = [(d, k) for d in items for k in KEYS]
    with ThreadPoolExecutor(24) as pool:
        arr = list(pool.map(lambda j: read(items[j[0]]["assets"][j[1]]["href"]), jobs))
    got: dict[str, list[np.ndarray]] = {}
    for (d, _), a in zip(jobs, arr, strict=True):
        got.setdefault(d, []).append(a)
    return {d: unpack(items[d], got[d]) for d in items}


def unpack(it: dict[str, Any], arrs: list[np.ndarray]) -> dict[str, np.ndarray]:
    q = arrs[6].astype(np.uint16)
    bad = ((q & 1) > 0) | (((q >> 4) & 1) > 0) | np.any(np.stack(arrs[:6]) == 0, axis=0)
    cloud = (((q >> 1) & 1) > 0) | (((q >> 3) & 1) > 0)
    bad |= cloud & ~INBOX
    sr = np.stack(arrs[:6]).astype(np.float32) * 2.75e-5 - 0.2
    sat = np.zeros((H, W), bool)
    for bit in RADSAT.get(it["properties"]["platform"], ()):
        sat |= ((arrs[7].astype(np.uint16) >> bit) & 1) > 0
    return {"sr": sr, "bad": bad, "sat": sat}


def twi(sr: np.ndarray) -> np.ndarray:
    b, g, r = sr[0], sr[1], sr[2]
    mx = np.maximum(np.maximum(b, g), r)
    mn = np.minimum(np.minimum(b, g), r)
    with np.errstate(all="ignore"):
        return (b + g + r) / 3 * (1 - (mx - mn) / mx)


def roy(sr: np.ndarray) -> np.ndarray:
    return ROY_A[:, None, None] + ROY_B[:, None, None] * sr


QS = np.linspace(0.5, 99.5, 100)


def transfer(x: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    ux, last = np.unique(qx[::-1], return_index=True)
    qx, qy = ux, qy[::-1][last]
    y = np.interp(x, qx, qy)
    y = np.where(x < qx[0], qy[0] + (x - qx[0]), y)
    y = np.where(x > qx[-1], qy[-1] + (x - qx[-1]), y)
    return np.where((x > qx[-1]) & (x >= 1.0), qy[-1], y)


def apply(sr: np.ndarray, qx: np.ndarray, qy: np.ndarray) -> np.ndarray:
    w = np.nan_to_num(np.clip((twi(sr) - 0.05) / 0.05, 0, 1))
    out = sr.copy()
    for j in range(sr.shape[0]):
        out[j] = sr[j] + w * (transfer(sr[j], qx[j], qy[j]) - sr[j])
    return out


def curves(src: np.ndarray, ref: np.ndarray, m: np.ndarray) -> np.ndarray:
    return np.array([[np.percentile(src[j][m], QS), np.percentile(ref[j][m], QS)] for j in range(6)])


def pair_data(
    pairs: list[tuple[str, str]],
    data: dict[str, dict[str, np.ndarray]],
    ka: str,
    kb: str,
) -> tuple[list[tuple[str, str]], np.ndarray, list[dict[str, Any]]]:
    keep: list[tuple[str, str]] = []
    cur: list[np.ndarray] = []
    rec: list[dict[str, Any]] = []
    for da, db in pairs:
        a, b = data[f"{ka}_{da}"], data[f"{kb}_{db}"]
        m = ~a["bad"] & ~b["bad"] & INBOX
        if (m & PBOX).sum() < 0.6 * PBOX.sum():
            continue
        keep.append((da, db))
        cur.append(curves(roy(a["sr"]), b["sr"] if kb == "8" else roy(b["sr"]), m))
        rec.append(
            {
                "a": da,
                "b": db,
                "valid_box": round(float((m & PBOX).sum() / PBOX.sum()), 3),
                "sat_box_a": round(float(a["sat"][m & PBOX].mean()), 4),
                "sat_box_b": round(float(b["sat"][m & PBOX].mean()), 4),
            }
        )
    return keep, np.array(cur), rec


def run(mode: str, out: Path, pair_file: Path | None = None) -> None:
    ka, kb, pa, pb_, t0, t1 = (
        ("7", "8", "landsat-7", "landsat-8", "2013-04-01", "2022-04-30")
        if mode == "l7l8"
        else ("5", "7", "landsat-5", "landsat-7", "1999-06-01", "2011-11-30")
    )
    if pair_file:
        tok = pair_file.read_text().split()
        iso = [(f"{t[:4]}-{t[4:6]}-{t[6:8]}", f"{t[9:13]}-{t[13:15]}-{t[15:17]}") for t in tok]
        sen = {"5": "LT05", "7": "LE07", "8": "LC08"}
        sa = by_ids(sen[ka], [a for a, _ in iso])
        sb = by_ids(sen[kb], [b for _, b in iso])
        pairs = [(a, b) for a, b in iso if a in sa and b in sb]
    else:
        sa, sb = scenes(pa, t0, t1), scenes(pb_, t0, t1)
        pairs = pairs_of(sa, sb)
    print(f"{pa} {len(sa)}, {pb_} {len(sb)}, pairs {len(pairs)}", flush=True)
    items = {f"{ka}_{p[0]}": sa[p[0]] for p in pairs} | {f"{kb}_{p[1]}": sb[p[1]] for p in pairs}
    data = load_all(items)
    print("read", round(time.time() - T0), flush=True)
    keep, cur, rec = pair_data(pairs, data, ka, kb)
    med = np.median(cur, axis=0)
    for k, (da, db) in enumerate(keep):
        o = np.median(np.delete(cur, k, axis=0), axis=0)
        a, b = data[f"{ka}_{da}"], data[f"{kb}_{db}"]
        m = ~a["bad"] & ~b["bad"] & PBOX
        src = roy(a["sr"])
        ref = b["sr"] if kb == "8" else roy(b["sr"])
        cor = apply(src, o[:, 0], o[:, 1])
        for nm, x in (("ref", ref), ("roy", src), ("corr", cor)):
            t = twi(x)[m]
            rec[k][f"area_ha_{nm}"] = round(float((t > 0.20).sum()) * 0.09, 2)
            rec[k][f"twi_med_{nm}"] = round(float(np.nanmedian(t)), 4)
    res = {
        "mode": mode,
        "bands": BANDS,
        "q": QS.tolist(),
        "q_src": med[:, 0].round(4).tolist(),
        "q_ref": med[:, 1].round(4).tolist(),
        "q_src_iqr": np.percentile(cur[:, :, 0], [25, 75], axis=0).round(4).tolist(),
        "q_ref_iqr": np.percentile(cur[:, :, 1], [25, 75], axis=0).round(4).tolist(),
        "pairs": rec,
    }
    (out / f"bright_{mode}.json").write_text(json.dumps(res))
    print("done", round(time.time() - T0), flush=True)


T0 = time.time()


def main() -> None:
    global TOKEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("l7l8", "l5l7"), default="l7l8")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--pairs", type=Path, help="file of YYYYMMDD-YYYYMMDD pairs (skip search)")
    args = ap.parse_args()
    for k in range(5):
        try:
            with urllib.request.urlopen(SAS, timeout=60) as r:
                TOKEN = json.load(r)["token"]
            break
        except OSError:
            time.sleep(2 + 2 * k)
    args.out.mkdir(parents=True, exist_ok=True)
    run(args.mode, args.out, args.pairs)


if __name__ == "__main__":
    main()
