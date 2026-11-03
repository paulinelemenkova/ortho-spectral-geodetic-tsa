#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import rasterio
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

FEATURES = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "TWI",
    "MNDWI",
    "CR",
    "NDVI",
)
CLASSES = (1, 2, 3, 4, 5)
RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": None,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "n_jobs": -1,
}
LGB_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 10,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "class_weight": "balanced",
    "verbose": -1,
}


def features(path: Path) -> np.ndarray:
    with rasterio.open(path) as s:
        r = s.read(list(range(1, 7))).astype(np.float64)
        r[r == s.nodata] = np.nan
    r /= 1e4
    b, g, red, nir, s1, s2 = r
    mx = np.fmax(np.fmax(b, g), red)
    mn = np.fmin(np.fmin(b, g), red)
    with np.errstate(all="ignore"):
        twi = (b + g + red) / 3 * (1 - (mx - mn) / mx)
        mndwi = (g - s1) / (g + s1)
        cr = s1 / s2
        ndvi = (nir - red) / (nir + red)
    return np.stack([*r, twi, mndwi, cr, ndvi])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=Path("labels30.tif"))
    ap.add_argument("--composite", type=Path, default=Path("composite_2025.tif"))
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--n-other", type=int, default=3000)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    with rasterio.open(a.labels) as s:
        lab = s.read(1)
    x = features(a.composite)
    ok = np.all(np.isfinite(x), axis=0) & (lab > 0)
    rows, cols = np.nonzero(ok)
    y = lab[rows, cols].astype(int)
    keep = np.flatnonzero(y != 5)
    other = np.flatnonzero(y == 5)
    keep = np.sort(np.concatenate([keep, rng.choice(other, a.n_other, replace=False)]))
    rows, cols, y = rows[keep], cols[keep], y[keep]
    X = x[:, rows, cols].T
    groups = (rows // a.block) * 1000 + cols // a.block
    print(
        "cells per class",
        {c: int((y == c).sum()) for c in CLASSES},
        "blocks",
        len(np.unique(groups)),
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof = {"rf": np.zeros_like(y), "lgbm": np.zeros_like(y)}
    for tr, te in cv.split(X, y, groups):
        rf = RandomForestClassifier(random_state=42, **RF_PARAMS).fit(X[tr], y[tr])
        gb = lgb.LGBMClassifier(random_state=42, **LGB_PARAMS).fit(X[tr], y[tr])
        oof["rf"][te] = rf.predict(X[te])
        oof["lgbm"][te] = gb.predict(X[te])

    res: dict[str, object] = {
        "n": {str(c): int((y == c).sum()) for c in CLASSES},
        "blocks": len(np.unique(groups)),
        "block_m": a.block * 30,
        "rf_params": RF_PARAMS,
        "lgbm_params": LGB_PARAMS,
    }
    for m, p in oof.items():
        f1 = f1_score(y, p, labels=list(CLASSES), average=None)
        res[m] = {
            "oa": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, labels=list(CLASSES), average="macro")),
            "f1": {str(c): float(v) for c, v in zip(CLASSES, f1, strict=True)},
        }
        cm = confusion_matrix(y, p, labels=list(CLASSES))
        np.savetxt(a.out / f"confusion_{m}.csv", cm, fmt="%d", delimiter=",")
    r_ok, g_ok = oof["rf"] == y, oof["lgbm"] == y
    b01, b10 = int((r_ok & ~g_ok).sum()), int((~r_ok & g_ok).sum())
    pval = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    res["mcnemar"] = {"rf_only": b01, "lgbm_only": b10, "p": float(pval)}

    rf = RandomForestClassifier(random_state=42, **RF_PARAMS).fit(X, y)
    gb = lgb.LGBMClassifier(random_state=42, **LGB_PARAMS).fit(X, y)
    joblib.dump(rf, a.out / "model_rf.joblib", compress=3)
    gb.booster_.save_model(str(a.out / "model_lgbm.txt"))
    nf = len(FEATURES)
    contrib = gb.predict(X, pred_contrib=True).reshape(len(y), len(CLASSES), nf + 1)
    shap = contrib[:, :, :nf]
    with open(a.out / "shap_mean.csv", "w") as fh:
        fh.write("feature," + ",".join(f"class{c}" for c in CLASSES) + "\n")
        for j, f in enumerate(FEATURES):
            vals = [np.abs(shap[:, k, j]).mean() for k in range(len(CLASSES))]
            fh.write(f + "," + ",".join(f"{v:.5f}" for v in vals) + "\n")
    idx = np.sort(rng.choice(len(y), min(2000, len(y)), replace=False))
    hdr = ["class"] + [f"x_{f}" for f in FEATURES] + [f"s1_{f}" for f in FEATURES]
    tab = np.column_stack([y[idx], X[idx], shap[idx, 0, :]])
    np.savetxt(
        a.out / "shap_sample.csv",
        tab,
        delimiter=",",
        header=",".join(hdr),
        comments="",
        fmt="%.5f",
    )
    (a.out / "accuracy.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("rf", "lgbm", "mcnemar")}, indent=1))


if __name__ == "__main__":
    main()
