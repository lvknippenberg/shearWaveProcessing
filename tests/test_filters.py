"""Temporal filters, the continuous reference+tracking record and the Giannantonio motion filter."""
import dataclasses

import numpy as np
import pytest

from conftest import PRF, make_acquisition
from swp.viz.filters import FIELD_FILTERS
from swp.viz.filters.context import FilterCtx


def ctx_for(n, nz=4, nx=6, t=None):
    return FilterCtx(dz=4e-4, dx=4e-4, prf=PRF, t=np.arange(n) / PRF if t is None else t,
                     x=(np.arange(nx) - nx // 2) * 4e-4, z=np.arange(nz) * 4e-4, focus_x=0.0)


def test_bandpass_keeps_inband_zero_phase_and_removes_drift():
    n = 400
    t = np.arange(n) / PRF
    sig = np.sin(2 * np.pi * 300 * t)
    field = (sig + 5 + 20 * t)[:, None, None] * np.ones((1, 4, 6))
    out = FIELD_FILTERS["temporal_bandpass"](field, ctx_for(n), f_lo=75, f_hi=750, order=3)[:, 0, 0]
    mid = slice(100, 300)
    assert np.corrcoef(out[mid], sig[mid])[0, 1] > 0.99            # no phase shift
    assert abs(out[mid].std() / sig[mid].std() - 1) < 0.1
    assert abs(out[mid].mean()) < 0.05


def test_continuous_record_is_continuous_across_the_push_gap():
    """Constant wall velocity through reference, a 1.17 ms push gap and tracking: the joined
    displacement must be one straight line on the uniform grid."""
    from swp.viz.estimators import loupas_displacement
    from swp.viz.slowtime import continuous_record
    v, gap = 0.01, 1170e-6
    n_ref, n_trk = 39, 59
    t = np.arange(n_trk) / PRF
    t_ref = (np.arange(n_ref) - n_ref) / PRF - gap
    acq = make_acquisition(np.broadcast_to((v * t)[:, None, None], (n_trk, 12, 10)).copy(),
                           np.broadcast_to((v * t_ref)[:, None, None], (n_ref, 12, 10)).copy(),
                           t=t, t_ref=t_ref, meta={"push_gap_s": gap})
    kw = dict(dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod, prf=acq.prf)
    rec = continuous_record(acq, loupas_displacement, kw, quantity="displacement", drop_first=1)
    d = np.median(rec.field, axis=(1, 2))
    slope = np.polyfit(rec.t, d, 1)[0]
    assert slope == pytest.approx(v, rel=0.02)
    assert np.max(np.abs(np.diff(d) - v / PRF)) < 0.2 * v / PRF    # no step at the gap
    assert rec.t[rec.first_tracking] == pytest.approx(t[1])
    assert rec.t[0] < 0


def test_continuous_record_warns_without_push_gap():
    from swp.viz.estimators import loupas_displacement
    from swp.viz.slowtime import continuous_record
    acq = make_acquisition(np.zeros((20, 8, 6)), np.zeros((10, 8, 6)), t_ref=(np.arange(10) - 10) / PRF)
    with pytest.warns(UserWarning, match="push interval"):
        continuous_record(acq, loupas_displacement, dict(dz=acq.dz, dx=acq.dx, c=acq.c,
                                                         f_demod=acq.f_demod, prf=acq.prf))


def test_giannantonio_removes_motion_and_keeps_the_wave():
    """Quadratic wall motion + a transient inside the excluded window: the fit uses only the
    pre-push and late samples, so the motion goes and the transient stays."""
    t = np.arange(-40, 60) / PRF
    motion = 30e-6 + 400e-6 * t + 2.0 * t ** 2
    wave = 3e-6 * np.exp(-((t - 4e-3) / 1e-3) ** 2)
    field = (motion + wave)[:, None, None] * np.ones((1, 4, 6))
    field[:, :, :] = (motion[:, None, None] + wave[:, None, None])
    ctx = ctx_for(len(t), t=t)
    ctx.x = np.zeros(6)                                  # every column at the push: window 0..3 ms+
    out = FIELD_FILTERS["giannantonio_motion_filter"](field, ctx, order=2, c_min=1.5, t_wave=8e-3)
    col = out[:, 0, 0]
    late = t > 8e-3
    assert np.abs(col[late]).max() < 0.2e-6                     # motion removed where fitted
    assert col[np.argmin(np.abs(t - 4e-3))] == pytest.approx(3e-6, rel=0.25)   # wave kept


def test_giannantonio_needs_prepush_samples():
    with pytest.raises(ValueError, match="pre-push"):
        FIELD_FILTERS["giannantonio_motion_filter"](np.zeros((10, 2, 2)), ctx_for(10, 2, 2))
