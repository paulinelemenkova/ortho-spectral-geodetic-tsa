#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from roi import R

WINDOW = R("window")
MARGIN_DEG = 0.03
START = "2016-01-01"
JOB_PREFIX = R("job_prefix", "sbas")
LAYERS = ("unw_phase", "corr", "conncomp", "dem", "lv_theta", "lv_phi", "water_mask")


def job_name(bursts: list[str]) -> str:
    return f"{JOB_PREFIX}_{bursts[0]}"


def window_wkt(margin: float = 0.0) -> str:
    x0, y0, x1, y1 = WINDOW
    x0, y0, x1, y1 = x0 - margin, y0 - margin, x1 + margin, y1 + margin
    return f"POLYGON(({x0} {y0},{x1} {y0},{x1} {y1},{x0} {y1},{x0} {y0}))"


def burst_search(end: str) -> list[Any]:
    import asf_search as asf

    return list(
        asf.geo_search(
            dataset=asf.DATASET.SLC_BURST,
            intersectsWith=window_wkt(),
            start=START,
            end=end,
            polarization="VV",
        )
    )


def cmd_search(args: argparse.Namespace) -> None:
    from shapely import wkt
    from shapely.geometry import shape

    win = wkt.loads(window_wkt())
    res = burst_search(args.end)
    groups: dict[str, list[Any]] = defaultdict(list)
    for r in res:
        groups[r.properties["burst"]["fullBurstID"]].append(r)
    rows = []
    for bid, items in groups.items():
        p = items[0].properties
        geom = shape(items[0].geometry)
        dates = sorted({it.properties["startTime"][:10] for it in items})
        rows.append(
            {
                "burst_id": bid,
                "direction": p["flightDirection"],
                "path": p["pathNumber"],
                "n_dates": len(dates),
                "first": dates[0],
                "last": dates[-1],
                "window_covered": round(geom.intersection(win).area / win.area, 3),
            }
        )
    rows.sort(key=lambda r: (r["direction"], -r["window_covered"], -r["n_dates"]))
    with open(args.csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(res)} burst granules, {len(rows)} burst IDs -> {args.csv}")
    for r in rows:
        print(
            f"  {r['burst_id']:<18} {r['direction']:<10} path {r['path']:>3}  "
            f"{r['n_dates']:>4} dates {r['first']}..{r['last']}  "
            f"window covered {100 * r['window_covered']:.0f} %"
        )
    for d in ("ASCENDING", "DESCENDING"):
        full = [r for r in rows if r["direction"] == d and r["window_covered"] >= 0.99]
        if full:
            best = max(full, key=lambda r: r["n_dates"])
            print(f"recommended {d.lower()}: --burst {best['burst_id']}")
        else:
            print(
                f"no single {d.lower()} burst covers the whole window: give the two "
                "adjacent burst IDs of one path with the largest coverage"
            )


def granules_by_date(burst_ids: list[str], end: str) -> dict[str, list[str]]:
    per_burst: dict[str, dict[str, str]] = {b: {} for b in burst_ids}
    for r in burst_search(end):
        bid = r.properties["burst"]["fullBurstID"]
        if bid in per_burst:
            per_burst[bid][r.properties["startTime"][:10]] = r.properties["sceneName"]
    common = set.intersection(*(set(v) for v in per_burst.values()))
    return {d: [per_burst[b][d] for b in burst_ids] for d in sorted(common)}


def make_pairs(dates: list[str], neighbours: int, max_days: int) -> list[tuple[str, str]]:
    t = [dt.date.fromisoformat(d) for d in dates]
    pairs = []
    for i in range(len(t)):
        for j in range(i + 1, min(i + 1 + neighbours, len(t))):
            if (t[j] - t[i]).days <= max_days:
                pairs.append((dates[i], dates[j]))
    return pairs


def cmd_submit(args: argparse.Namespace) -> None:
    import hyp3_sdk as sdk

    by_date = granules_by_date(args.burst, args.end)
    dates = list(by_date)
    if args.dry_season:
        dates = [d for d in dates if 5 <= int(d[5:7]) <= 10]
    pairs = make_pairs(dates, args.neighbours, args.max_days)
    print(f"bursts {args.burst}: {len(dates)} dates {dates[0]}..{dates[-1]}, {len(pairs)} pairs")
    gaps = [
        (a, b)
        for a, b in itertools.pairwise(dates)
        if (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days > args.max_days
    ]
    if gaps:
        print(f"warning: {len(gaps)} gaps longer than {args.max_days} days break the network:")
        for a, b in gaps[:10]:
            print(f"   {a} -> {b}")
    if args.dry_run:
        print("dry run: nothing submitted")
        return
    hyp3 = sdk.HyP3(prompt="password")
    print("credits available:", hyp3.check_credits())
    prepared = []
    for d1, d2 in pairs:
        g1, g2 = by_date[d1], by_date[d2]
        if len(args.burst) == 1:
            job = sdk.HyP3.prepare_insar_isce_burst_job(
                g1[0],
                g2[0],
                name=job_name(args.burst),
                apply_water_mask=False,
                looks=args.looks,
            )
        else:
            job = sdk.HyP3.prepare_insar_isce_multi_burst_job(
                g1,
                g2,
                name=job_name(args.burst),
                apply_water_mask=False,
                looks=args.looks,
            )
        prepared.append(job)
    for k in range(0, len(prepared), 200):
        batch = hyp3.submit_prepared_jobs(prepared[k : k + 200])
        print(f"submitted {k + len(batch)} / {len(prepared)}")
    print("credits left:", hyp3.check_credits())


def cmd_status(args: argparse.Namespace) -> None:
    import hyp3_sdk as sdk

    jobs = sdk.HyP3(prompt="password").find_jobs(name=job_name(args.burst))
    count: dict[str, int] = defaultdict(int)
    for j in jobs:
        count[j.status_code] += 1
    print(dict(count), "total", len(jobs))
    failed = [j for j in jobs if j.status_code == "FAILED"]
    for j in failed[:10]:
        print("failed:", j.job_parameters.get("granules"))


def fetch(url: str, dst: Path) -> None:
    import urllib.request

    tmp = dst.with_suffix(dst.suffix + ".part")
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            tmp.rename(dst)
            return
        except OSError:
            if k == 3:
                raise


def cmd_download(args: argparse.Namespace) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import hyp3_sdk as sdk

    args.out.mkdir(parents=True, exist_ok=True)
    jobs = sdk.HyP3(prompt="password").find_jobs(name=job_name(args.burst))
    done = [j for j in jobs if j.succeeded()]
    have = {p.name for p in args.out.glob("*.zip")}
    todo = [(f["url"], args.out / f["filename"]) for j in done for f in j.files if f["filename"] not in have]
    print(
        f"{len(done)} of {len(jobs)} jobs succeeded; {len(todo)} files to download "
        f"to {args.out} with {args.workers} parallel downloads"
    )
    with ThreadPoolExecutor(args.workers) as pool:
        for k, _ in enumerate(pool.map(lambda t: fetch(*t), todo), 1):
            if k % 25 == 0 or k == len(todo):
                print(f"  {k} / {len(todo)}", flush=True)


def cmd_clip(args: argparse.Namespace) -> None:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    zips = sorted(args.zips.glob("*.zip"))
    if not zips:
        sys.exit(f"no zip files in {args.zips}")
    args.out.mkdir(parents=True, exist_ok=True)
    grid: dict[str, Any] = {}
    for k, zp in enumerate(zips, 1):
        stem = zp.stem
        with zipfile.ZipFile(zp) as zf:
            members = {Path(n).name: n for n in zf.namelist() if n.endswith(".tif")}
            for layer in LAYERS:
                name = next((n for n in members if n.endswith(f"_{layer}.tif")), None)
                if name is None:
                    continue
                dst = args.out / stem / name
                if dst.exists():
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                with (
                    zf.open(members[name]) as fh,
                    rasterio.MemoryFile(fh.read()) as mem,
                    mem.open() as src,
                ):
                    if not grid:
                        res = src.res[0]
                        x0, y0, x1, y1 = WINDOW
                        m = MARGIN_DEG
                        bx0, by0, bx1, by1 = transform_bounds(
                            "EPSG:4326", src.crs, x0 - m, y0 - m, x1 + m, y1 + m
                        )
                        bx0, by1 = bx0 // res * res, -(-by1 // res) * res
                        w = int(-(-(bx1 - bx0) // res))
                        h = int(-(-(by1 - by0) // res))
                        grid = {
                            "crs": src.crs,
                            "transform": from_origin(bx0, by1, res, res),
                            "width": w,
                            "height": h,
                        }
                    rs = Resampling.nearest
                    with WarpedVRT(src, resampling=rs, **grid) as vrt:
                        prof = vrt.profile
                        prof.update(driver="GTiff", compress="deflate", tiled=False)
                        with rasterio.open(dst, "w", **prof) as out:
                            out.write(vrt.read())
            for meta in (n for n in zf.namelist() if n.endswith(".txt")):
                (args.out / stem / Path(meta).name).write_bytes(zf.read(meta))
        if k % 25 == 0 or k == len(zips):
            print(f"  clipped {k} / {len(zips)}")


def cmd_template(args: argparse.Namespace) -> None:
    c = args.clip.resolve()
    x0, y0, x1, y1 = WINDOW
    print(f"""## smallbaselineApp template
mintpy.load.processor        = hyp3
mintpy.load.unwFile          = {c}/*/*_unw_phase.tif
mintpy.load.corFile          = {c}/*/*_corr.tif
mintpy.load.connCompFile     = {c}/*/*_conncomp.tif
mintpy.load.demFile          = {c}/*/*_dem.tif
mintpy.load.incAngleFile     = {c}/*/*_lv_theta.tif
mintpy.load.azAngleFile      = {c}/*/*_lv_phi.tif
mintpy.load.waterMaskFile    = {c}/*/*_water_mask.tif
mintpy.subset.lalo           = {y0 - 0.02}:{y1 + 0.02},{x0 - 0.02}:{x1 + 0.02}
mintpy.network.coherenceBased = yes
mintpy.network.minCoherence  = 0.3
mintpy.reference.lalo        = auto
mintpy.unwrapError.method    = no
mintpy.troposphericDelay.method = no
mintpy.deramp                = no
mintpy.topographicResidual   = yes
mintpy.timeseriesInv.weightFunc = var
mintpy.networkInversion.maskDataset = no
mintpy.networkInversion.minTempCoh = 0.7
mintpy.save.kmz              = no
mintpy.save.hdfEos5          = no
## export:
##   save_gdal.py velocity.h5 -d velocity -o velocity.tif
##   save_gdal.py velocity.h5 -d velocityStd -o velocityStd.tif
##   save_gdal.py temporalCoherence.h5 -o temporalCoherence.tif
##   save_gdal.py maskTempCoh.h5 -o maskTempCoh.tif
##   timeseries_demErr.h5 (or timeseries.h5), inputs/geometryGeo.h5, pic/network.pdf
""")


def main() -> None:
    today = dt.datetime.now(tz=dt.UTC).date().isoformat()
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--end", default=today)
    s.add_argument("--csv", default="bursts.csv")
    s = sub.add_parser("submit")
    s.add_argument("--burst", nargs="+", required=True, help="full burst ID(s), same path")
    s.add_argument("--end", default=today)
    s.add_argument("--neighbours", type=int, default=3, help="pairs per date (default 3)")
    s.add_argument("--max-days", type=int, default=48, help="longest pair (default 48 days)")
    s.add_argument("--looks", default="20x4", choices=["20x4", "10x2", "5x1"])
    s.add_argument("--dry-season", action="store_true", help="May-October dates only")
    s.add_argument("--dry-run", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--burst", nargs="+", required=True, help="burst ID(s) as in submit")
    s = sub.add_parser("download")
    s.add_argument("--burst", nargs="+", required=True, help="burst ID(s) as in submit")
    s.add_argument("--out", type=Path, default=Path("hyp3_zips"))
    s.add_argument("--workers", type=int, default=8, help="parallel downloads")
    s = sub.add_parser("clip")
    s.add_argument("--zips", type=Path, default=Path("hyp3_zips"))
    s.add_argument("--out", type=Path, default=Path("hyp3_clip"))
    s = sub.add_parser("template")
    s.add_argument("--clip", type=Path, default=Path("hyp3_clip"))
    args = ap.parse_args()
    {
        "search": cmd_search,
        "submit": cmd_submit,
        "status": cmd_status,
        "download": cmd_download,
        "clip": cmd_clip,
        "template": cmd_template,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
