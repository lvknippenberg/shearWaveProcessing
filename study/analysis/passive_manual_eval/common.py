"""Shared paths and helpers for the evaluation of the manual passive reading.

Everything here READS the shearWaveProcessing repo (``swp`` package, configs, view labels) and the
per-folder outputs on Z:; nothing is written to either. All analysis scripts work on a frozen
SNAPSHOT (``00_snapshot.py``) so a rerun on the same snapshot gives the same numbers, and a rerun
after more reading only needs a new snapshot.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True                       # never drop .pyc files into the repo

_HERE = Path(__file__).resolve().parent
# the repo this folder lives in (study/analysis/passive_manual_eval), or SWP_REPO
REPO = Path(os.environ.get("SWP_REPO", _HERE.parents[2] if (_HERE.parents[2] / "src" / "swp").is_dir()
                           else r"D:/Luuk van Knippenberg/Github/shearWaveProcessing"))
sys.path.insert(0, str(REPO / "src"))
ROOT = Path(os.environ.get("SWP_RAW_DATA", "Z:/raw_data"))
EVAL = Path(__file__).resolve().parent
SNAPS = EVAL / "snapshots"
CACHE = EVAL / "cache"
OUT = EVAL / "results"

LABELS = ("MVC", "AVC", "AK", "other")


def snapshot_dir(name: str | None = None) -> Path:
    """The named snapshot, or the newest one."""
    if name:
        p = SNAPS / name
        if not p.is_dir():
            raise SystemExit(f"no snapshot {p}")
        return p
    snaps = sorted(d for d in SNAPS.glob("*") if (d / "MANIFEST.json").exists())
    if not snaps:
        raise SystemExit("no snapshot yet: run 00_snapshot.py")
    return snaps[-1]


def out_dir(snap: Path, part: str) -> Path:
    d = OUT / snap.name / part
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as fh:
        return json.load(fh)


def tables(snap: Path):
    """The collected tables of a snapshot as pandas DataFrames (windows, lines, folders, log)."""
    import pandas as pd
    t = {}
    for n in ("folders", "windows", "lines", "log", "retest", "review"):
        f = snap / "tables" / f"{n}.csv"
        if f.exists():
            t[n] = pd.read_csv(f)
    return t


def st_path(snap: Path, subject: str, folder: str, window: int) -> Path:
    return snap / "data" / subject / folder / f"st_win{int(window)}.npz"


def load_st(path):
    """st_win<i>.npz -> dict(views=[dict(name, quantity, data (t,r), r (m), t (s))], bmode...)."""
    d = np.load(path, allow_pickle=False)
    n = int(d["n_views"])
    views = [dict(name=str(d[f"v{j}_name"]), quantity=str(d[f"v{j}_quantity"]),
                  data=np.asarray(d[f"v{j}_data"], float), r=np.asarray(d[f"v{j}_r"], float),
                  t=np.asarray(d[f"v{j}_t"], float)) for j in range(n)]
    return dict(views=views, bmode_u8=d["bmode_u8"], bmode_extent=np.asarray(d["bmode_extent"]),
                line_x_mm=np.asarray(d["line_x_mm"]), line_z_mm=np.asarray(d["line_z_mm"]))


def view(st, name):
    for v in st["views"]:
        if v["name"] == name:
            return v
    raise KeyError(name)


def savefig(fig, path, dpi=130):
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def subj_short(s: str) -> str:
    """C000000012 -> C12."""
    return "C" + str(int(s[1:])) if s.startswith("C") and s[1:].isdigit() else s
