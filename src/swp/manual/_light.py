"""zea-free access to the few library pieces the interactive editors need.

``swp.acquisition`` and ``swp.mline`` import zea (and with it torch) in their package
``__init__``, which costs ~15-60 s on this machine. The modules the editors use - the trigger log
and the line transfer - only need numpy/scipy, so they are loaded here straight from their files,
under private names, without running the package ``__init__``. The worker (which needs the full
pipeline anyway) imports the normal way.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]


def _load(private_name, relpath):
    if private_name in sys.modules:
        return sys.modules[private_name]
    spec = importlib.util.spec_from_file_location(private_name, _SRC / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[private_name] = mod
    spec.loader.exec_module(mod)
    return mod


def triggerlog():
    return _load("_swp_light_triggerlog", "acquisition/triggerlog.py")


def rrcheck():
    """swp.acquisition.rrcheck, whose ``from .triggerlog import`` needs a parent package: a bare
    stand-in package pointing at the acquisition directory (its zea-importing __init__ never runs)."""
    import types
    pkg = "_swp_light_acq"
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [str(_SRC / "acquisition")]
        sys.modules[pkg] = m
    import importlib
    return importlib.import_module(f"{pkg}.rrcheck")


def transfer():
    return _load("_swp_light_transfer", "mline/transfer.py")


# Same display as the GIFs and the September M-line prompts (swp.mline.select._display_8bit):
# white point at the 99.9th percentile of the in-sector envelope, range down to its 5th percentile
# (clamped to 25-90 dB), gamma2 tone curve. Constants mirror swp.acquisition.gifs
# (tests/test_manual.py checks they still match).
DEFAULT_CURVE = "gamma2"
HI_PCT, LO_PCT = 99.9, 5.0
DR_LIMITS = (25.0, 90.0)


def display_8bit(env):
    from ..viz.tonecurves import apply_curve

    env = np.asarray(env, dtype=np.float32)
    inside = env[env > 0]
    if inside.size == 0:
        return np.zeros(env.shape, np.uint8)
    ref = max(float(np.percentile(inside, HI_PCT)), 1e-12)
    floor = float(np.percentile(inside, LO_PCT))
    dr = 20.0 * np.log10(ref / floor) if floor > 0 else DR_LIMITS[1]
    dr = float(np.clip(dr, *DR_LIMITS))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(env / ref + 1e-12)
    norm = np.clip((db + dr) / dr, 0.0, 1.0)
    return (apply_curve(norm, DEFAULT_CURVE) * 255).astype(np.uint8)


def order_points(points_xz, anchor=None):
    """Order clicked points along their principal axis, starting at the end nearest ``anchor``
    (as swp.mline.select._order_points: click order does not matter, s = 0 at the first click)."""
    p = np.asarray(points_xz, dtype=float)
    if p.shape[0] <= 2:
        ordered = p.copy()
    else:
        centred = p - p.mean(axis=0)
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
        ordered = p[np.argsort(centred @ vt[0])]
    if anchor is not None and ordered.shape[0] >= 2:
        a = np.asarray(anchor, dtype=float)
        if np.hypot(*(ordered[0] - a)) > np.hypot(*(ordered[-1] - a)):
            ordered = ordered[::-1]
    return ordered
