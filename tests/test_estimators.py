"""Displacement estimators on phase-encoded synthetic IQ with a known axial displacement."""
import numpy as np
import pytest

from conftest import C, F_DEMOD, PRF, DZ, DX, speckle, phase_encode
from swp.viz.estimators import loupas_displacement, kasai_displacement, ESTIMATORS

WRAP = C / (4 * F_DEMOD)          # 98.6 um: reference-relative phase wraps beyond +/- this


def _kw(**extra):
    return dict(dz=DZ, dx=DX, c=C, f_demod=F_DEMOD, prf=PRF, **extra)


@pytest.mark.parametrize("shape", ["box", "gaussian"])
def test_frame_to_frame_recovers_ramp(shape):
    d = np.linspace(0, 20e-6, 30)                           # 20 um ramp
    iq = phase_encode(speckle(40, 30), d)
    res = loupas_displacement(iq, mode="frame_to_frame", kernel_x_m=1e-3, kernel_shape=shape, **_kw())
    got = np.median(res.displacement, axis=(1, 2))
    assert np.allclose(got, d, atol=0.3e-6)
    assert np.allclose(np.median(res.velocity, axis=(1, 2)), np.diff(d) * PRF, rtol=0.05)


def test_relative_to_reference_matches_truth():
    d = np.linspace(0, 20e-6, 30)
    iq0 = speckle(40, 30)
    res = loupas_displacement(phase_encode(iq0, d), mode="relative_to_reference",
                              reference=phase_encode(iq0, np.zeros(5)), **_kw())
    assert np.allclose(np.median(res.displacement, axis=(1, 2)), d, atol=0.3e-6)


def test_reference_relative_wraps_where_frame_to_frame_does_not():
    """Why in-vivo displacement relative to the reference fails: 150 um of wall motion is past
    the 98.6 um wrap limit. Frame-to-frame steps stay small and the cumulative sum is right."""
    d = np.linspace(0, 150e-6, 60)
    iq0 = speckle(40, 30)
    iq = phase_encode(iq0, d)
    f2f = loupas_displacement(iq, mode="frame_to_frame", **_kw())
    rel = loupas_displacement(iq, mode="relative_to_reference", reference=iq0, **_kw())
    assert abs(np.median(f2f.displacement[-1]) - 150e-6) < 2e-6
    assert np.median(rel.displacement[-1]) < 0          # wrapped to the other side
    assert WRAP < 150e-6


def test_kasai_equals_loupas_without_local_frequency():
    iq = phase_encode(speckle(40, 30), np.linspace(0, 10e-6, 12))
    a = kasai_displacement(iq, **_kw()).displacement
    b = loupas_displacement(iq, local_frequency=False, **_kw()).displacement
    assert np.allclose(a, b)


def test_registry_has_cfwi_and_core_estimators():
    for k in ("loupas", "kasai", "xcorr", "rf_ncc", "cfwi"):
        assert k in ESTIMATORS


def test_cfwi_lights_up_moving_tissue_only():
    """A region oscillating at 6 cm/s peak passes the 2 cm/s clutter filter; a static one not."""
    n, nz, nx = 200, 20, 20
    t = np.arange(n) / 925.9
    disp = np.zeros((n, nz, nx))
    disp[:, :, 10:] = (0.06 / (2 * np.pi * 60)) * np.sin(2 * np.pi * 60 * t)[:, None, None]
    iq = phase_encode(speckle(nz, nx), disp)
    res = ESTIMATORS["cfwi"](iq, dz=DZ, dx=DX, c=C, f_demod=F_DEMOD, prf=925.9)
    env = res.displacement[20:-20]
    assert env[:, :, 10:].mean() > 5 * env[:, :, :10].mean()
