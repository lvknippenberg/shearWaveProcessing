"""One continuous slow-time record across the push: reference + tracking on a uniform time axis.

The tracking ensemble alone is ~16 ms long, and a 75-120 Hz high-pass corner has an impulse
response of ~8-13 ms, so zero-phase filtering on the tracking block alone is dominated by edge
transients. The reference frames (~10 ms before the push) are the natural extension. This module
joins them:

1. frame-to-frame displacement over the concatenated IQ stack ``[reference, tracking[drop:]]``,
   cumulated so the record is continuous through the push (no reference-relative phase wrap - each
   step is one frame interval, except the single step that spans the push);
2. the displacement is interpolated onto a uniform grid anchored on the tracking frame times
   (the push interval, ~1 ms, is not a whole number of frames);
3. the result can be filtered like any field and then cropped back to the tracking window.

Requires reference timestamps that include the push interval (``acq.meta['push_gap_s']`` set;
see ``scripts/retrofit_push_gap.py``) - with the old synthetic ones the reference block sits
~1 ms too close and the join is wrong.

Caveat measured 2026-09-24 (``study/analysis/push_gap_check.py``): in vivo the speckle correlation
from the last reference frame to the first tracking frames is ~0.8, against ~0.95 within either
block over the same time span (the phantom shows no such drop). The one displacement step that
spans the push is therefore noisier than the rest; the frame-to-frame steps on either side are not
affected.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass
class ContinuousRecord:
    field: np.ndarray        # (n_t, nz, nx) displacement [m] or velocity [m/s] on the uniform grid
    t: np.ndarray            # (n_t,) seconds; t = 0 at the first kept tracking frame's time origin
    first_tracking: int      # index of the first tracking sample on the grid
    quantity: str


def continuous_record(acq, estimator, estimator_kwargs: dict, quantity: str = "displacement",
                      drop_first: int = 1, n_pre: int | None = None) -> ContinuousRecord:
    """Estimate ``quantity`` over reference + tracking as one uniformly sampled record.

    ``estimator`` is a frame-to-frame estimator (e.g. ``loupas_displacement``); ``n_pre`` limits
    how many reference frames are used (default all).
    """
    if acq.ref_iq is None or acq.t_ref is None:
        raise ValueError("continuous_record needs reference frames and their timestamps")
    if not (acq.meta or {}).get("push_gap_s"):
        warnings.warn("reference timestamps do not include the push interval "
                      "(no custom/push_gap_s); run scripts/retrofit_push_gap.py", stacklevel=2)
    ref, t_ref = acq.ref_iq, np.asarray(acq.t_ref, float)
    if n_pre is not None:
        ref, t_ref = ref[-n_pre:], t_ref[-n_pre:]
    trk, t_trk = acq.iq[drop_first:], np.asarray(acq.t, float)[drop_first:]
    stack = np.concatenate([ref, trk], axis=0)
    t_all = np.concatenate([t_ref, t_trk])
    est = estimator(stack, mode="frame_to_frame", **estimator_kwargs)
    disp = est.displacement                                  # cumulative, (n_all, nz, nx)

    dt = 1.0 / acq.prf
    k0 = int(np.ceil((t_all[0] - t_trk[0]) / dt - 1e-9))     # negative: grid steps before tracking
    t_u = t_trk[0] + np.arange(k0, int(round((t_trk[-1] - t_trk[0]) / dt)) + 1) * dt
    field = _interp_time(disp, t_all, t_u)
    if quantity == "velocity":
        field = np.gradient(field, dt, axis=0)
    elif quantity == "acceleration":
        field = np.gradient(np.gradient(field, dt, axis=0), dt, axis=0)
    return ContinuousRecord(field=field, t=t_u, first_tracking=-k0, quantity=quantity)


def _interp_time(x: np.ndarray, t: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Linear interpolation along axis 0 of a (n, ...) array (vectorised over the rest)."""
    idx = np.clip(np.searchsorted(t, t_new) - 1, 0, len(t) - 2)
    w = ((t_new - t[idx]) / (t[idx + 1] - t[idx])).reshape((-1,) + (1,) * (x.ndim - 1))
    return (1.0 - w) * x[idx] + w * x[idx + 1]
