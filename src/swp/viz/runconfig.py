"""Unified, config-driven setup for the iq2sws pipeline.

A single YAML config fully describes a run for **either phantom or in-vivo** data — only the
``data`` / ``mline`` / ``focus`` sections differ between datasets; the ``pipeline`` section is
identical.  This module turns a config dict into the concrete objects the pipeline needs
(:class:`Acquisition`, :class:`MLine`, :class:`PushFocus`, :class:`PipelineConfig`).

Config units are human-friendly: lengths in mm (keys ending ``_mm``) or µm (``_um``) and
frequencies in Hz.  ``_mm``/``_um`` keys are converted to the SI (``_m``) keys the filters expect.
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np
import yaml

from .core.acquisition import Acquisition
from .core.geometry import PushFocus
from .io import load_acquisition, load_mline
from .io.ecg import load_ecg
from .mline import MLine, horizontal_mline, mline_from_points
from .pipeline import PipelineConfig, Step


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _convert_params(params: Optional[dict]) -> dict:
    """Convert mm/µm-suffixed keys to the SI (_m) keys the filters/estimators expect."""
    out = {}
    for k, v in (params or {}).items():
        if k.endswith("_mm"):
            out[k[:-3] + "_m"] = float(v) * 1e-3
        elif k.endswith("_um"):
            out[k[:-3] + "_m"] = float(v) * 1e-6
        else:
            out[k] = v
    return out


# --- path resolution -------------------------------------------------------
def hdf5_path(cfg: dict, m: int) -> str:
    d = cfg["data"]
    return os.path.join(d["root"], d.get("grid_dir", ""), d["file"].format(m=m))


def mline_npz_path(cfg: dict, m: int) -> str:
    return os.path.join(cfg["data"]["root"], cfg["mline"]["npz"].format(m=m))


def ecg_path(cfg: dict) -> Optional[str]:
    ecg = cfg["data"].get("ecg")
    return os.path.join(cfg["data"]["root"], ecg) if ecg else None


# --- object builders -------------------------------------------------------
def load_acq(cfg: dict, m: int) -> Acquisition:
    return load_acquisition(hdf5_path(cfg, m))


def build_mline(cfg: dict, acq: Acquisition, m: int) -> MLine:
    mc = cfg["mline"]
    if mc["type"] == "manual":
        points, ns_file = load_mline(mline_npz_path(cfg, m))
        return mline_from_points(points, mc.get("n_samples", ns_file))
    if mc["type"] == "horizontal":
        return horizontal_mline(acq.x, mc["depth_mm"] * 1e-3, mc.get("n_samples", 250))
    raise ValueError(f"unknown mline type {mc['type']!r}")


def _focus_from_xz(acq: Acquisition, px: float, pz: float) -> PushFocus:
    ix = int(np.argmin(np.abs(acq.x - px)))
    iz = int(np.argmin(np.abs(acq.z - pz)))
    return PushFocus(x=float(acq.x[ix]), z=float(acq.z[iz]), ix=ix, iz=iz)


def build_focus(cfg: dict, acq: Acquisition) -> Optional[PushFocus]:
    """Resolve the ARF push focus.

    ``stored`` -> use the push location written into the HDF5 (``acq.push_x/z``), falling
    back to the config ``x_mm``/``z_mm`` for older files that predate it. ``fixed`` -> the
    config ``(x_mm, z_mm)`` point. ``auto``/``vertical`` -> None (the pipeline auto-detects
    the focus; for ``vertical`` the wave origin ``r0`` is set from ``push_x_m`` instead).
    """
    fc = cfg.get("focus", {"mode": "auto"})
    mode = fc.get("mode", "auto")
    if mode == "stored":
        px = acq.push_x if acq.push_x is not None else fc.get("x_mm", 0.0) * 1e-3
        pz = acq.push_z if acq.push_z is not None else fc.get("z_mm", 40.0) * 1e-3
        return _focus_from_xz(acq, px, pz)
    if mode == "fixed":
        return _focus_from_xz(acq, fc.get("x_mm", 0.0) * 1e-3, fc.get("z_mm", 40.0) * 1e-3)
    return None


def _steps(filter_list) -> list:
    return [Step(f["name"], _convert_params(f.get("params"))) for f in (filter_list or [])]


def resolve_push_x_m(cfg: dict, acq: Optional[Acquisition] = None) -> Optional[float]:
    """Lateral x of the vertical ARF push line, for the origin ``r0`` (M-line crossing).

    ``stored`` mode prefers the push x written into the HDF5 (``acq.push_x``); ``vertical``
    uses the config ``x_mm``; both fall back to the config ``x_mm``. Other modes -> None
    (``r0`` is then the M-line point nearest the 2-D focus).
    """
    fc = cfg.get("focus", {})
    mode = fc.get("mode")
    if mode == "stored":
        if acq is not None and acq.push_x is not None:
            return float(acq.push_x)
        return fc.get("x_mm", 0.0) * 1e-3
    if mode == "vertical":
        return fc.get("x_mm", 0.0) * 1e-3
    return None


def build_pipeline_config(cfg: dict, overrides: Optional[dict] = None,
                          acq: Optional[Acquisition] = None) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from the ``pipeline`` + ``mline`` config sections.

    ``overrides`` (dict) shallow-overrides top-level pipeline keys (used by experiments to vary
    one axis at a time), e.g. ``{"estimator": "kasai", "field_filters": [...]}``. Pass ``acq``
    so a ``stored`` focus can anchor ``r0`` on the push location saved in the HDF5.
    """
    p = dict(cfg["pipeline"])
    if overrides:
        p.update(overrides)
    mc = cfg["mline"]
    return PipelineConfig(
        estimator=p.get("estimator", "loupas"),
        estimator_params=_convert_params(p.get("estimator_params")),
        mode=p.get("mode", "relative_to_reference"),
        quantity=p.get("quantity", "displacement"),
        iq_filters=_steps(p.get("iq_filters")),
        field_filters=_steps(p.get("field_filters")),
        directional=p.get("directional", True),
        speed=p.get("speed", "ttp_ransac"),
        drop_first=p.get("drop_first", 1),
        mline_offsets=mc.get("offsets", 5),
        mline_offset_step_m=mc.get("offset_step_mm", None) and mc["offset_step_mm"] * 1e-3,
        push_x_m=resolve_push_x_m(cfg, acq),
    )


def cardiac_phases(cfg: dict) -> Optional[np.ndarray]:
    path = ecg_path(cfg)
    if path and os.path.exists(path):
        return load_ecg(path).push_cardiac_phases()
    return None


def measurements(cfg: dict) -> list:
    m = cfg.get("run", {}).get("measurement", 0)
    return list(m) if isinstance(m, (list, tuple)) else [m]


def discover_measurements(cfg: dict) -> List[int]:
    """Every measurement index present in the data folder, by matching the ``data.file`` template.

    Globs ``data.root/grid_dir`` for files whose name matches ``file`` with ``{m}`` replaced by an
    integer, and returns the sorted indices.  This is what the core ``run.py`` batches over.
    """
    d = cfg["data"]
    token = "\x00"
    pat = re.escape(d["file"].format(m=token)).replace(token, r"(\d+)")
    rx = re.compile("^" + pat + "$")
    folder = os.path.join(d["root"], d.get("grid_dir", ""))
    if not os.path.isdir(folder):
        return []
    out = {int(mo.group(1)) for f in os.listdir(folder) if (mo := rx.match(f))}
    return sorted(out)


def bands(cfg: dict) -> Optional[List[tuple]]:
    """Multi-band confirmation view: ``run.bands`` as a list of (f_lo, f_hi) Hz, or None."""
    b = cfg.get("run", {}).get("bands")
    return [(float(lo), float(hi)) for lo, hi in b] if b else None


def outdir(cfg: dict, root: str) -> str:
    od = cfg.get("run", {}).get("outdir", "results/run")
    return od if os.path.isabs(od) else os.path.join(root, od)
