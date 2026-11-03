# ortho-spectral-geodetic-tsa

**Orthogonal Spectral, Thermal and Geodetic Evidence Streams: A Reproducible Python Framework for Quantile-Harmonised Archive Analytics, Explainable Surface-State Learning and Kinematic Attribution**

![python](https://img.shields.io/badge/python-3.11%2B-informational) ![lint](https://img.shields.io/badge/lint-ruff-informational) ![types](https://img.shields.io/badge/types-mypy-informational) ![status](https://img.shields.io/badge/status-pre--release-lightgrey)

## Abstract

An end-to-end, ROI-agnostic framework for dense optical/TIR/C-band time-series analytics: it ingests STAC-indexed C2L2 SR/ST and L2A stacks, enforces ETM+→OLI radiometric continuity through a high-albedo QM layered on OLS cross-calibration, and builds QA-screened annual composites on a fixed UTM lattice. Surface states are inferred by spatially blocked, class-balanced RF/GBDT ensembles with McNemar-tested skill and TreeSHAP attribution. The temporal layer combines Hamed–Rao-corrected MK, Theil–Sen with variance-scaled CI and PELT (L2 cost, BIC-type penalty), at aggregate and per-pixel level. LST is stratified into robust *z*-anomalies (LULC × hypsometric band), regressed against structural distance via a bootstrapped exponential kernel, and tested with global Moran's *I* and LISA. Kinematics come from burst-wise SBAS (HyP3/ISCE2 → MintPy), LOS→v<sub>up</sub> projection and stable-reference demeaning. Swath statistics along arbitrary geodesics and Spearman coupling with ERA5/ERA5-Land complete the pipeline.

## Modules

| Module | Stage | Core operators |
|---|---|---|
| `stac_pull.py` | L0→L1 | STAC/SAS query, QA_PIXEL/SCL clear-sky ranking, NN warp to common grid |
| `c2l2_compose.py` | L1→L2 | annual median compositing, ST scaling, exclusion-aware cloud masking |
| `qmap_xcal.py` | L2 | ETM+→OLI OLS + bright-tail QM, TWI-gated fading |
| `gbdt_cv.py` | L3 | RF vs LightGBM, StratifiedGroupKFold, McNemar, `pred_contrib` SHAP |
| `epoch_infer.py` | L3 | multi-epoch inference, morphological extent, class-area accounting |
| `mk_pelt.py` | L4 | MK-HR, Sen + CI, PELT segmentation of annual series |
| `pix_theilsen.py` | L4 | per-cell MK/Sen, significance classes, geographic regridding |
| `lst_rz.py` | L4 | robust *z* ΔT, EDT distance, exp-decay fit, Moran's *I*, LISA |
| `era_rank.py` | L4 | hydrological-year aggregation, Spearman ρ raw/Δ¹ |
| `sbas_stage.py` | K0→K1 | burst search, SBAS network design, HyP3 submit/clip, MintPy template |
| `los2up.py` | K2 | LOS→v<sub>up</sub>, upland referencing, probe time series |
| `swath_xs.py` | S | geodesic swath sampling, median/IQR profiles |
| `roi.py` | — | runtime ROI descriptor resolution |

## Execution

```bash
python -m pip install -e .            # [sar] extra for the K-branch
export ROI_CFG=/path/to/roi.json      # schema: cfg/roi.example.json
python src/stac_pull.py --config windows.json    # schema: cfg/windows.example.json
python src/<module>.py -h
```

ROI geometry, lattice and acquisition descriptors are resolved only at runtime. The repository contains no site-specific constants.

## Conventions

WGS84 bounding boxes are given as (λ<sub>0</sub>, φ<sub>0</sub>, λ<sub>1</sub>, φ<sub>1</sub>). Grids use 30 m cells with nearest-neighbour resampling. Slopes are given per decade and velocities in mm a<sup>−1</sup>. Code is checked with `ruff` and `mypy`.

## Status and licence

Pre-release, accompanying a manuscript under peer review. © 2026 Polina Lemenkova. All rights reserved; no licence to use, copy, modify or redistribute is granted until a licence is assigned at release (see `LICENSE`).
