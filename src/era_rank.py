#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def climate_table(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["date"])
    d["hy"] = d["date"].dt.year + (d["date"].dt.month >= 10)
    p = d.groupby("hy")["precip"].sum()
    js = d[d["date"].dt.month.between(6, 9)]
    t = js.groupby(js["date"].dt.year)["tmean"].mean()
    c = pd.DataFrame({"p0": p, "p1": p.shift(1), "t_js": t})
    c["p01"] = c["p0"] + c["p1"]
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=Path, default=Path("../data/trends/annual_series.csv"))
    ap.add_argument("--climate", type=Path, default=Path("../figures/fig_climate/climate_daily.csv"))
    ap.add_argument("--out", type=Path, default=Path("../data/trends"))
    a = ap.parse_args()
    s = pd.read_csv(a.series).set_index("year")
    c = climate_table(a.climate)
    j = s.join(c, how="inner")
    rows = []
    for var in ("ra_twi", "ra_white_ha", "ra_water_ha"):
        for cv in ("p0", "p1", "p01", "t_js"):
            for form in ("raw", "diff"):
                x, y = j[var], j[cv]
                if form == "diff":
                    x, y = x.diff(), y.diff()
                ok = np.isfinite(x) & np.isfinite(y)
                r, p = spearmanr(x[ok], y[ok])
                rows.append(
                    {
                        "series": var,
                        "climate": cv,
                        "form": form,
                        "rho": round(float(r), 3),
                        "p": round(float(p), 4),
                        "n": int(ok.sum()),
                    }
                )
    out = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "climate_link.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
