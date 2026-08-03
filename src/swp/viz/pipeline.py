"""Config-driven, composable processing pipeline.

A pipeline is *data*: choose an estimator, an ordered list of IQ-space and field-space
filters (each a ``(name, params)`` step), optional directional space-time filtering, and a
speed method.  This makes method/filter combinations trivial to enumerate and compare
(see ``archive/experiments.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .core.acquisition import Acquisition
from .core.geometry import detect_push_focus, PushFocus
from .estimators import ESTIMATORS
from .filters import IQ_FILTERS, FIELD_FILTERS
from .filters.context import FilterCtx
from .filters.directional import outward_spacetime
from .mline.mline import MLine
from .speed import build_spacetime, SpaceTime, SPEED_METHODS, SpeedResult


@dataclass
class Step:
    name: str
    params: dict = field(default_factory=dict)


@dataclass
class PipelineConfig:
    estimator: str = "loupas"
    estimator_params: dict = field(default_factory=dict)
    mode: str = "relative_to_reference"      # or "frame_to_frame"
    quantity: str = "displacement"           # "displacement" or "velocity"
    iq_filters: List[Step] = field(default_factory=list)
    field_filters: List[Step] = field(default_factory=list)
    directional: bool = False                # outward directional filter on the space-time
    speed: str = "radon"
    drop_first: int = 1                      # drop corrupt first frame(s)
    mline_offsets: int = 5
    mline_offset_step_m: Optional[float] = None
    push_x_m: Optional[float] = None         # vertical push at this lateral x: r0 = M-line crossing there

    def label(self) -> str:
        ff = "+".join(s.name for s in self.field_filters) or "none"
        iqf = "+".join(s.name for s in self.iq_filters) or "none"
        d = "+dir" if self.directional else ""
        return (f"{self.estimator}/{self.quantity}|iq:{iqf}|f:{ff}{d}|{self.speed}")


@dataclass
class PipelineResult:
    st: SpaceTime
    speed: SpeedResult
    r0: float
    focus: PushFocus
    config: PipelineConfig
    field: np.ndarray                        # the M-line-sampled quantity's source field
    times: np.ndarray


def _r0_along_mline(mline: MLine, focus: PushFocus) -> float:
    d = np.hypot(mline.x - focus.x, mline.z - focus.z)
    return float(mline.r[int(np.argmin(d))])


def _r0_lateral_crossing(mline: MLine, x0: float) -> float:
    """Arc length where the M-line crosses the vertical push line x = x0 (the physical wave
    origin for a vertical ARF push: the intersection is depth-independent)."""
    d = mline.x - x0
    sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
    if sign_change.size:
        i = sign_change[0]
        denom = d[i] - d[i + 1]
        frac = d[i] / denom if denom != 0 else 0.0
        return float(mline.r[i] + frac * (mline.r[i + 1] - mline.r[i]))
    return float(mline.r[int(np.argmin(np.abs(d)))])


def run_pipeline(acq: Acquisition, mline: MLine, cfg: PipelineConfig,
                 focus: Optional[PushFocus] = None) -> PipelineResult:
    # --- IQ-space filters ---
    iq = acq.iq
    for step in cfg.iq_filters:
        iq = IQ_FILTERS[step.name](iq, **step.params)

    # --- displacement estimation ---
    est = ESTIMATORS[cfg.estimator]
    kwargs = dict(dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod, prf=acq.prf,
                  mode=cfg.mode, **cfg.estimator_params)
    if cfg.mode == "relative_to_reference":
        kwargs["reference"] = acq.ref_iq
    res = est(iq, **kwargs)

    if cfg.quantity == "velocity":
        f_all, t_all = res.velocity, 0.5 * (acq.t[:-1] + acq.t[1:])
    elif cfg.quantity == "acceleration":
        f_all = np.diff(res.velocity, axis=0) * acq.prf   # d(velocity)/dt [m/s^2]
        t_all = acq.t[1:-1]
    else:
        f_all, t_all = res.displacement, acq.t
    k = cfg.drop_first
    fld, times = f_all[k:], t_all[k:]

    # --- push focus (auto) ---
    if focus is None:
        focus = detect_push_focus(res.displacement, acq.x, acq.z)

    # --- field-space filters ---
    ctx = FilterCtx(dz=acq.dz, dx=acq.dx, prf=acq.prf, t=times, x=acq.x, z=acq.z,
                    focus_ix=focus.ix, focus_x=focus.x)
    # reference trajectory for reference-based filters (computed on demand)
    from .filters import REFERENCE_FILTERS
    if any(s.name in REFERENCE_FILTERS for s in cfg.field_filters):
        ref_res = est(acq.ref_iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod,
                      prf=acq.prf, mode="relative_to_reference", reference=acq.ref_iq,
                      **cfg.estimator_params)
        ctx.ref_disp = ref_res.displacement
        if acq.t_ref is not None:
            ctx.t_ref = np.asarray(acq.t_ref, float)
        else:   # phantom has references but no timestamps: synthesize a pre-push axis
            n_ref = acq.ref_iq.shape[0]
            ctx.t_ref = (np.arange(n_ref) - n_ref) / acq.prf
    for step in cfg.field_filters:
        fld = FIELD_FILTERS[step.name](fld, ctx, **step.params)

    # --- M-line sampling -> space-time ---
    step_m = cfg.mline_offset_step_m if cfg.mline_offset_step_m is not None else acq.dz * 4
    st = build_spacetime(fld, acq.z, acq.x, mline, times, quantity=cfg.quantity,
                         n_offsets=cfg.mline_offsets, offset_step_m=step_m)

    if cfg.push_x_m is not None:
        r0 = _r0_lateral_crossing(mline, cfg.push_x_m)
    else:
        r0 = _r0_along_mline(mline, focus)

    # --- optional directional (outward) filtering on the space-time ---
    if cfg.directional:
        st = SpaceTime(outward_spacetime(st.data, st.r, r0), st.r, st.t, st.quantity)

    # --- speed estimation ---
    speed = SPEED_METHODS[cfg.speed](st, r0)
    return PipelineResult(st=st, speed=speed, r0=r0, focus=focus, config=cfg,
                          field=fld, times=times)
