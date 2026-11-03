#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{}"
COLLECTION = {"landsat": "landsat-c2-l2", "sentinel2": "sentinel-2-l2a"}
RESOLUTION = {"landsat": 30.0, "sentinel2": 10.0}
QA_BAND = {"landsat": "qa_pixel", "sentinel2": "SCL"}
LANDSAT_PLATFORMS = ("landsat-4", "landsat-5", "landsat-7", "landsat-8", "landsat-9")
L7_SLC_FAILURE = "2003-05-31"
SNAP = 30.0
SCREEN_FACTOR = 8
SCREEN_WORKERS = 6
HTTP_TIMEOUT = 20
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_TIMEOUT": "30",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "VSI_CACHE": "TRUE",
}
CAT_FIELDS = [
    "window",
    "sensor",
    "year",
    "date",
    "platform",
    "item_id",
    "scene_cloud",
    "clear_frac",
    "selected",
    "flag",
    "folder",
]

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def http_json(url: str, payload: dict[str, Any] | None = None, tries: int = 4) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    last: Exception | None = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as err:
            if 400 <= err.code < 500 and err.code != 429:
                raise
            last = err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            last = err
        time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"request failed after {tries} tries: {url}: {last}")


class TokenCache:
    def __init__(self) -> None:
        self._tokens: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, collection: str) -> str:
        with self._lock:
            tok = self._tokens.get(collection)
            if tok is None or time.time() - tok[1] > 1800:
                res = http_json(SAS_URL.format(collection))
                tok = (str(res["token"]), time.time())
                self._tokens[collection] = tok
            return tok[0]

    def sign(self, href: str, collection: str) -> str:
        return f"{href}?{self.get(collection)}"


TOKENS = TokenCache()


def stac_search(
    collection: str, bbox: list[float], start: str, end: str, max_cloud: float
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "collections": [collection],
        "bbox": bbox,
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "limit": 250,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    items: list[dict[str, Any]] = []
    url = STAC_URL
    while True:
        res = http_json(url, body)
        items.extend(res.get("features", []))
        nxt = [lk for lk in res.get("links", []) if lk.get("rel") == "next"]
        if not nxt:
            return items
        url = nxt[0]["href"]
        body = nxt[0].get("body", body)


@dataclass(frozen=True)
class Grid:
    crs: str
    transform: Affine
    width: int
    height: int

    def coarsen(self, factor: int) -> Grid:
        res = self.transform.a * factor
        return Grid(
            self.crs,
            from_origin(self.transform.c, self.transform.f, res, res),
            max(1, math.ceil(self.width / factor)),
            max(1, math.ceil(self.height / factor)),
        )


def make_grid(bbox: list[float], crs: str, res: float) -> Grid:
    xmin, ymin, xmax, ymax = transform_bounds("EPSG:4326", crs, *bbox, densify_pts=21)
    xmin, ymin = math.floor(xmin / SNAP) * SNAP, math.floor(ymin / SNAP) * SNAP
    xmax, ymax = math.ceil(xmax / SNAP) * SNAP, math.ceil(ymax / SNAP) * SNAP
    width, height = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    return Grid(crs, from_origin(xmin, ymax, res, res), width, height)


def screen_weights(grid: Grid, exclude: list[list[float]]) -> np.ndarray:
    use = np.ones((grid.height, grid.width), dtype=bool)
    res = grid.transform.a
    for box in exclude:
        x0, y0, x1, y1 = transform_bounds("EPSG:4326", grid.crs, *box, densify_pts=21)
        c0 = max(0, int((x0 - grid.transform.c) // res))
        c1 = min(grid.width, int(math.ceil((x1 - grid.transform.c) / res)))
        r0 = max(0, int((grid.transform.f - y1) // res))
        r1 = min(grid.height, int(math.ceil((grid.transform.f - y0) / res)))
        use[r0:r1, c0:c1] = False
    return use


def read_to_grid(href: str, grid: Grid) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.Env(**GDAL_ENV), rasterio.open(href) as src:
        with WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=Resampling.nearest,
        ) as vrt:
            arr = vrt.read(1)
            return arr, {"dtype": str(arr.dtype), "nodata": vrt.nodata}


def clear_mask(sensor: str, qa: np.ndarray) -> np.ndarray:
    if sensor == "landsat":
        q = qa.astype(np.uint16)

        def bit(b: int) -> np.ndarray:
            return ((q >> b) & 1).astype(bool)

        good = bit(6) | bit(7) | bit(5)
        bad = bit(0) | bit(1) | bit(3) | bit(4)
        return good & ~bad
    return np.isin(qa, (4, 5, 6, 7, 11))


def asset_key(sensor: str, item: dict[str, Any], band: str) -> str:
    assets = item["assets"]
    if sensor == "landsat" and band == "lwir":
        return "lwir11" if "lwir11" in assets else "lwir"
    if sensor == "landsat" and band == "st_qa":
        return "qa"
    return band


def latest_processing(sensor: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if sensor != "sentinel2":
        return items
    best: dict[str, dict[str, Any]] = {}
    for it in items:
        stem, _, processed = str(it["id"]).rpartition("_")
        if stem not in best or processed > str(best[stem]["id"]).rpartition("_")[2]:
            best[stem] = it
    return list(best.values())


def item_date(item: dict[str, Any]) -> str:
    return str(item["properties"]["datetime"])[:10]


def platform_of(sensor: str, item: dict[str, Any]) -> str:
    p = str(item["properties"].get("platform", sensor)).lower()
    return p.replace("sentinel-", "S").replace("landsat-", "L")


def usable(sensor: str, item: dict[str, Any]) -> bool:
    props = item["properties"]
    if sensor == "landsat":
        return (
            props.get("platform") in LANDSAT_PLATFORMS
            and props.get("landsat:collection_category", "T1") == "T1"
        )
    return True


def cache_item(out: Path, collection: str, item: dict[str, Any]) -> None:
    d = out / "_items" / collection
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{item['id']}.json").write_text(json.dumps(item))


def load_item(out: Path, collection: str, item_id: str) -> dict[str, Any] | None:
    f = out / "_items" / collection / f"{item_id}.json"
    return json.loads(f.read_text()) if f.exists() else None


def read_csv(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {(r["window"], r["sensor"], r["item_id"]): r for r in csv.DictReader(fh)}


def write_csv(path: Path, rows: dict[tuple[str, str, str], dict[str, str]]) -> None:
    ordered = sorted(rows.values(), key=lambda r: (r["window"], r["sensor"], r["date"], r["item_id"]))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=CAT_FIELDS)
        wr.writeheader()
        wr.writerows(ordered)
    tmp.replace(path)


@dataclass
class Job:
    window: str
    sensor: str
    year: int
    grid: Grid
    bands: list[str]
    per_year: int
    bbox: list[float]
    exclude: list[list[float]]


def screen(job: Job, item: dict[str, Any], weights: np.ndarray) -> float:
    coll = COLLECTION[job.sensor]
    href = TOKENS.sign(item["assets"][QA_BAND[job.sensor]]["href"], coll)
    qa, _ = read_to_grid(href, job.grid.coarsen(SCREEN_FACTOR))
    return float(clear_mask(job.sensor, qa)[weights].mean())


def select_year(
    job: Job,
    cfg: dict[str, Any],
    out: Path,
    candidates: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    coll = COLLECTION[job.sensor]
    weights = screen_weights(job.grid.coarsen(SCREEN_FACTOR), job.exclude)
    rows: list[dict[str, str]] = []
    for season, tag in (
        (cfg["season"], ""),
        (cfg.get("season_fallback", cfg["season"]), "fallback_season"),
    ):
        items = [
            it
            for it in stac_search(
                coll,
                job.bbox,
                f"{job.year}-{season[0]}",
                f"{job.year}-{season[1]}",
                cfg["max_scene_cloud"],
            )
            if usable(job.sensor, it)
        ]
        if job.sensor == "landsat":
            normal = [
                it
                for it in items
                if not (it["properties"]["platform"] == "landsat-7" and item_date(it) >= L7_SLC_FAILURE)
            ]
            items = normal or items
        items = latest_processing(job.sensor, items)
        done = {r["item_id"] for r in rows}
        items = [it for it in items if it["id"] not in done]
        with ThreadPoolExecutor(SCREEN_WORKERS) as ex:
            fracs = list(ex.map(lambda it: _safe_screen(job, it, weights), items))
        for it, frac in zip(items, fracs, strict=True):
            if frac is None:
                continue
            cache_item(out, coll, it)
            slc = it["properties"].get("platform") == "landsat-7" and item_date(it) >= L7_SLC_FAILURE
            rows.append(
                {
                    "window": job.window,
                    "sensor": job.sensor,
                    "year": str(job.year),
                    "date": item_date(it),
                    "platform": platform_of(job.sensor, it),
                    "item_id": it["id"],
                    "scene_cloud": f"{it['properties'].get('eo:cloud_cover', float('nan')):.1f}",
                    "clear_frac": f"{frac:.4f}",
                    "selected": "0",
                    "flag": ";".join(x for x in (tag, "l7_slc_off" if slc else "") if x),
                    "folder": "",
                }
            )
        if any(float(r["clear_frac"]) >= cfg["min_clear"] for r in rows):
            break

    best: dict[str, dict[str, str]] = {}
    for r in rows:
        if r["date"] not in best or float(r["clear_frac"]) > float(best[r["date"]]["clear_frac"]):
            best[r["date"]] = r
    ranked = sorted(best.values(), key=lambda r: (-float(r["clear_frac"]), r["date"]))
    chosen = [r for r in ranked if float(r["clear_frac"]) >= cfg["min_clear"]][: job.per_year]
    if not chosen and ranked:
        chosen = ranked[:1]
        chosen[0]["flag"] = ";".join(x for x in (chosen[0]["flag"], "below_min_clear") if x)
    for r in chosen:
        r["selected"] = "1"
        r["folder"] = f"{job.window}/{job.sensor}/{r['date'].replace('-', '')}_{r['platform']}"
    for r in rows:
        candidates[(r["window"], r["sensor"], r["item_id"])] = r
    log(
        f"[{job.window}/{job.sensor}/{job.year}] {len(rows)} screened, {len(chosen)} selected: "
        + ", ".join(f"{r['date']} {r['platform']} {float(r['clear_frac']):.2f}" for r in chosen)
    )
    return chosen


def _safe_screen(job: Job, item: dict[str, Any], weights: np.ndarray) -> float | None:
    try:
        return screen(job, item, weights)
    except Exception as err:
        log(f"  screening failed for {item['id']}: {err}")
        return None


def download(job_grid: Grid, sensor: str, bands: list[str], row: dict[str, str], out: Path) -> bool:
    coll = COLLECTION[sensor]
    item = load_item(out, coll, row["item_id"])
    if item is None:
        log(f"  no cached item for {row['item_id']}; rerun with --rescreen")
        return False
    folder = out / row["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    for band in bands:
        dst = folder / f"{band}.tif"
        if dst.exists():
            continue
        key = asset_key(sensor, item, band)
        if key not in item["assets"]:
            log(f"  {row['item_id']}: asset {key} missing")
            continue
        arr, info = read_to_grid(TOKENS.sign(item["assets"][key]["href"], coll), job_grid)
        profile = {
            "driver": "GTiff",
            "width": job_grid.width,
            "height": job_grid.height,
            "count": 1,
            "dtype": info["dtype"],
            "crs": job_grid.crs,
            "transform": job_grid.transform,
            "nodata": info["nodata"],
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }
        tmp = dst.with_suffix(".part.tif")
        with rasterio.open(tmp, "w", **profile) as ds:
            ds.write(arr, 1)
        tmp.replace(dst)
    props = item["properties"]
    meta = {
        "item_id": item["id"],
        "datetime": props.get("datetime"),
        "platform": props.get("platform"),
        "scene_cloud": props.get("eo:cloud_cover"),
        "clear_frac_window": row["clear_frac"],
        "flag": row["flag"],
        "crs": job_grid.crs,
        "resolution_m": job_grid.transform.a,
        "processing_baseline": props.get("s2:processing_baseline", ""),
        "wrs_path_row": f"{props.get('landsat:wrs_path', '')}/{props.get('landsat:wrs_row', '')}",
        "mgrs_tile": props.get("s2:mgrs_tile", ""),
    }
    (folder / "metadata.txt").write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n")
    log(f"  downloaded {row['folder']}")
    return True


def parse_years(text: str | None, lo: int, hi: int) -> list[int]:
    if not text:
        return list(range(lo, hi + 1))
    a, _, b = text.partition("-")
    years = range(int(a), int(b or a) + 1)
    return [y for y in years if lo <= y <= hi]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="windows.json", type=Path)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--windows", help="comma-separated window names (default: all)")
    ap.add_argument("--sensors", help="comma-separated: landsat,sentinel2 (default: all configured)")
    ap.add_argument("--years", help="single year or range, e.g. 2023 or 1984-2026")
    ap.add_argument("--workers", type=int, default=8, help="parallel scene downloads")
    ap.add_argument("--dry-run", action="store_true", help="screen and select only, no downloads")
    ap.add_argument(
        "--rescreen",
        action="store_true",
        help="screen again years already in catalogue.csv",
    )
    args = ap.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    cat_path, cand_path = args.out / "catalogue.csv", args.out / "candidates.csv"
    catalogue, candidates = read_csv(cat_path), read_csv(cand_path)
    want_w = set(args.windows.split(",")) if args.windows else None
    want_s = set(args.sensors.split(",")) if args.sensors else None

    jobs: list[Job] = []
    for wname, w in cfg["windows"].items():
        if want_w and wname not in want_w:
            continue
        for sensor, s in w["sensors"].items():
            if want_s and sensor not in want_s:
                continue
            grid = make_grid(w["bbox"], cfg["crs"], RESOLUTION[sensor])
            log(f"grid {wname}/{sensor}: {grid.width} x {grid.height} cells at {RESOLUTION[sensor]:.0f} m")
            for year in parse_years(args.years, *s["years"]):
                jobs.append(
                    Job(
                        wname,
                        sensor,
                        year,
                        grid,
                        s["bands"],
                        s["per_year"],
                        w["bbox"],
                        w.get("screen_exclude", []),
                    )
                )

    downloads: list[tuple[Grid, str, list[str], dict[str, str]]] = []
    for job in jobs:
        done = [
            r
            for r in catalogue.values()
            if r["window"] == job.window and r["sensor"] == job.sensor and r["year"] == str(job.year)
        ]
        if done and not args.rescreen:
            chosen = done
        else:
            try:
                chosen = select_year(job, cfg, args.out, candidates)
            except Exception as err:
                log(f"[{job.window}/{job.sensor}/{job.year}] search failed: {err}")
                continue
            for k in [
                k
                for k, r in catalogue.items()
                if r["window"] == job.window and r["sensor"] == job.sensor and r["year"] == str(job.year)
            ]:
                del catalogue[k]
            for r in chosen:
                catalogue[(r["window"], r["sensor"], r["item_id"])] = r
            write_csv(cat_path, catalogue)
            write_csv(cand_path, candidates)
        downloads.extend((job.grid, job.sensor, job.bands, r) for r in chosen)

    if args.dry_run:
        log(f"dry run: {len(downloads)} scenes selected; nothing downloaded")
        return 0
    ok = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(download, *d, args.out) for d in downloads]
        for fut in futures:
            try:
                ok += bool(fut.result())
            except Exception as err:
                log(f"  download failed: {err}")
    log(f"done: {ok}/{len(downloads)} scenes complete in {args.out}")
    return 0 if ok == len(downloads) else 1


if __name__ == "__main__":
    sys.exit(main())
