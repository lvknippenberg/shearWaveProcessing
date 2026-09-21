"""Stage 3: run the field estimator on a real passive event window.

Reuses the existing processing front end unchanged - ``run_pipeline`` already returns the
filtered 2-D field as ``res.field`` with ``res.times`` - so this is a change of *estimator*, not
of data.

The key output for validation is not the speed on its own but its **projection onto the M-line
that was drawn by hand for the same window**:

    predicted apparent speed along the line = c_field / cos(theta_field - theta_line)

A 1-D hand pick measures the apparent speed, which is the true speed divided by the cosine of the
angle between the wave and the line. Comparing the projection rather than the speed therefore
tests ``c`` and ``theta`` jointly, and is strictly stronger than comparing speeds: an estimate
that gets the speed right and the direction wrong fails it.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from .roi import ROI, ensure_roi, roi_from_mline
from .structure import Grid, aggregate, estimate_field, interference_indicator

__all__ = ["WindowEstimate", "estimate_window", "project_onto_mline"]


@dataclass
class WindowEstimate:
    folder: str
    window: int
    label: str
    view: str
    speed: float                  # true speed [m/s], direction-resolved
    theta_deg: float              # propagation direction in the image plane
    coherence: float
    direction_spread_deg: float
    interference: float           # envelope modulation - CANDIDATE gate, see structure.py
    n_voxels: int
    frac_kept: float
    roi_length_mm: float
    roi_thickness_mm: float
    projected: float = float("nan")      # apparent speed along the stored M-line
    mline_theta_deg: float = float("nan")
    hand_speed: float = float("nan")     # the hand-drawn apparent speed, if one exists

    def as_row(self):
        return dataclasses.asdict(self)


def mline_direction_deg(mline):
    """Direction of an M-line in the same convention as the estimator: atan2(dz, dx)."""
    x, z = np.asarray(mline.x, float), np.asarray(mline.z, float)
    return float(np.degrees(np.arctan2(z[-1] - z[0], x[-1] - x[0])))


def project_onto_mline(speed, theta_deg, mline_theta_deg, max_factor=8.0):
    """Apparent speed a 1-D estimator on that line should see: ``c / cos(dtheta)``.

    Capped, because at grazing incidence the projection diverges and a huge predicted value
    carries no information - it only says the wave is nearly perpendicular to the line.
    """
    d = np.radians(theta_deg - mline_theta_deg)
    cosd = abs(np.cos(d))
    if cosd < 1.0 / max_factor:
        return float(speed * max_factor)
    return float(speed / cosd)


def estimate_window(folder, config, window, view_name=None, roi=None, pad_ms=20.0,
                    roi_thickness_mm=12.0, min_coherence=0.5, part="left"):
    """Field estimate for one detected event window of one folder.

    Args:
        view_name: substring of the processing view to use; default is the first one.
        roi: an :class:`ROI`; if None it is loaded or derived from the stored M-line.
        part: which stored M-line part to project onto ("full"/"left"/"right").
    """
    from .. import passive as P
    from ..viz.pipeline import run_pipeline

    cfg, p = P._paths(folder, config)
    n_samples = cfg["mline"].get("n_samples", 250)
    st, windows = P.read_windows(p["windows_json"])
    if st is None or window >= len(windows):
        raise SystemExit(f"window {window} not found in {p['windows_json']}")
    w = windows[window]

    acq = P.load_acq(folder, config)
    views = P._build_views(cfg, acq)
    if view_name:
        views = [v for v in views if view_name.lower() in v[0].lower()] or views
    vname, vcfg = views[0]

    pad = pad_ms * 1e-3
    i0 = P._frame_at_time(acq.t, w.t0 - pad)
    i1 = P._frame_at_time(acq.t, w.t1 + pad) + 1
    acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])

    ml_full = P._load_line(P._window_npz(p["mlines"], window), n_samples)
    if roi is None:
        roi, _ = ensure_roi(folder, config, thickness_mm=roi_thickness_mm)
        if roi is None:
            roi = roi_from_mline(ml_full, thickness_mm=roi_thickness_mm)

    # The M-line itself is only needed to define the ROI and to project onto; the estimate does
    # not use its direction.
    res = run_pipeline(acq_w, ml_full, vcfg, focus=None)
    field = np.asarray(res.field, float)
    grid = Grid(dz=float(acq.dz), dx=float(acq.dx), dt=float(1.0 / acq.prf))

    est = estimate_field(field, grid)
    mask2d = roi.mask(acq.x, acq.z)
    if mask2d.sum() < 16:
        raise SystemExit("ROI covers too few pixels of the field")
    s = aggregate(est, mask=mask2d[None, :, :], min_coherence=min_coherence)
    length_mm, thick_mm = roi.extent_mm()

    out = WindowEstimate(
        folder=str(folder), window=int(window), label=str(w.label or "?"), view=vname,
        speed=float(s["speed"]), theta_deg=float(s["theta_deg"]),
        coherence=float(s["coherence"]),
        direction_spread_deg=float(s["direction_spread_deg"]),
        interference=float(interference_indicator(field, est)),
        n_voxels=int(s["n_voxels"]), frac_kept=float(s["frac_kept"]),
        roi_length_mm=length_mm, roi_thickness_mm=thick_mm)

    # --- projection onto the hand-drawn line, for validation -------------------------------
    try:
        from passive_mline_split import split_line                    # study-side helper

        from ..viz.mline import mline_from_points
        ml = (ml_full if part == "full"
              else mline_from_points(split_line(ml_full, n_samples)[part], n_samples))
    except Exception:                                                 # noqa: BLE001
        ml = ml_full
    out.mline_theta_deg = mline_direction_deg(ml)
    if np.isfinite(out.speed):
        out.projected = project_onto_mline(out.speed, out.theta_deg, out.mline_theta_deg)
    return out, est, roi
