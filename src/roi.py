#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PATH = Path(os.environ.get("ROI_CFG", "roi.json"))
_CFG: dict[str, Any] = json.loads(_PATH.read_text()) if _PATH.exists() else {}


def _freeze(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return {k: _freeze(x) for k, x in v.items()}
    return v


def R(key: str, default: Any = None) -> Any:
    if key in _CFG:
        return _freeze(_CFG[key])
    if default is None:
        raise KeyError(f"{key!r} not defined in {_PATH}")
    return default
