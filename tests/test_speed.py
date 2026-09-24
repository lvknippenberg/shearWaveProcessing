"""Speed estimators and the directional filter on synthetic space-times with a known speed.

Ported from iq2sws/tests/test_speed_synthetic.py (lost when the code moved here), plus the passive
signed slant stack and the normalised Radon added on 2026-09-24.
"""
import numpy as np
import pytest

from swp.viz.speed.spacetime import SpaceTime
from swp.viz.speed.tof import slant_stack_speed, tof_xcorr_speed, ttp_ransac_speed
from swp.viz.filters.directional import outward_spacetime, directional_spacetime
from swp.viz import metrics


def make_wave(c=3.0, r0=0.020, prf=3703.7, T=58, R=250, span=0.040, w=0.0015, noise=0.0,
              direction="outward", seed=0):
    t = np.arange(T) / prf
    r = np.linspace(0, span, R)
    d = np.abs(r - r0)
    if direction == "outward":
        st = np.exp(-((d[None, :] - c * t[:, None]) / w) ** 2)
    else:
        st = np.exp(-((d[None, :] - (0.018 - c * t[:, None])) / w) ** 2)
    if noise:
        st = st + noise * np.random.default_rng(seed).standard_normal(st.shape)
    return SpaceTime(st, r, t, "displacement"), r0


def one_way(c=3.0, T=120, prf=925.9, R=250, span=0.040, w=0.004):
    """A passive-style wave crossing the whole line in one direction (sign of c)."""
    t = np.arange(T) / prf
    r = np.linspace(0, span, R)
    pos = r if c > 0 else span - r
    return SpaceTime(np.exp(-((pos[None, :] - abs(c) * (t[:, None] - 0.02)) / w) ** 2), r, t, "velocity")


@pytest.mark.parametrize("fn", [slant_stack_speed, tof_xcorr_speed, ttp_ransac_speed])
def test_recover_speed_clean(fn):
    st, r0 = make_wave(c=3.0)
    assert abs(fn(st, r0).c_mean - 3.0) < 0.4


def test_recover_speed_noisy():
    st, r0 = make_wave(c=4.5, noise=0.4, seed=1)
    assert abs(slant_stack_speed(st, r0).c_mean - 4.5) < 0.6


def _l2(a):
    a = a - a.mean()
    return float((a ** 2).sum())


def test_outward_is_selective():
    out_st, r0 = make_wave(c=3.0, direction="outward")
    in_st, _ = make_wave(c=3.0, direction="inward")
    kept_out = _l2(outward_spacetime(out_st.data, out_st.r, r0)) / _l2(out_st.data)
    kept_in = _l2(outward_spacetime(in_st.data, in_st.r, r0)) / _l2(in_st.data)
    assert kept_in < 0.2 and kept_out > 3 * kept_in


@pytest.mark.xfail(strict=True, reason="the Tukey taper (2026-08-06 fix) is applied before the FFT "
                                       "and never undone: a clean outward wave keeps only ~55 % of "
                                       "its energy, lost at the record start (where the wave leaves "
                                       "r0) and the M-line ends")
def test_outward_preserves_outward_energy():
    out_st, r0 = make_wave(c=3.0, direction="outward")
    assert _l2(outward_spacetime(out_st.data, out_st.r, r0)) / _l2(out_st.data) > 0.8


def test_single_direction_nulls_opposite_lobe():
    """The 2026-08-06 fix: keep='pos' must remove most of the -r lobe (it kept ~55 % before)."""
    st, r0 = make_wave(c=3.0)
    left = st.r < r0 - 3e-3
    kept = directional_spacetime(st.data, "pos")
    assert _l2(kept[:, left]) / _l2(st.data[:, left]) < 0.3


@pytest.mark.parametrize("c", [2.5, -3.5])
def test_signed_slant_stack_passive(c):
    st = one_way(c)
    _, got = metrics.slant_stack_speed(st, None, remove_flat=False)
    assert np.sign(got) == np.sign(c) and abs(abs(got) - abs(c)) / abs(c) < 0.2


@pytest.mark.parametrize("c", [2.5, -3.5])
def test_normalized_radon_recovers_plane_wave(c):
    st = one_way(c)
    res = metrics.normalized_radon_speed(st)
    assert np.sign(res["speed"]) == np.sign(c)
    assert abs(abs(res["speed"]) - abs(c)) / abs(c) < 0.15
    assert res["tracking"] > 2.0            # the line sits on the wave, well above noise (~1)


def test_line_tracking_on_vs_off_wave():
    st = one_way(3.0)
    on = metrics.normalized_radon_speed(st)
    assert metrics.line_tracking(st, on["t0"], on["speed"]) > \
        3 * metrics.line_tracking(st, on["t0"] + 0.05, on["speed"])
