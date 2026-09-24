"""Shared synthetic data for the test suite.

Everything here is synthetic with a known answer, so the tests run in seconds without any
acquisition on disk. Real-data regression checks live in ``study/analysis`` and are not run here.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "scripts"):
    if os.path.join(_REPO, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_REPO, _p))

C = 1540.0
F_DEMOD = 3.90625e6          # our pulse-inversion 2nd-harmonic demodulation frequency
PRF = 3703.7
DZ = DX = 0.394e-3


def speckle(nz, nx, seed=0, corr_z=2.0):
    """Baseband complex speckle, axially correlated so the lag-1 axial phase averages ~0
    (otherwise the Loupas local-frequency correction sees a random carrier)."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    re = gaussian_filter(rng.standard_normal((nz, nx)), (corr_z, 0.7))
    im = gaussian_filter(rng.standard_normal((nz, nx)), (corr_z, 0.7))
    return re + 1j * im


def phase_encode(iq0, disp):
    """IQ frames whose phase encodes axial displacement ``disp`` (n, nz, nx) or (n,) [m].

    Matches the estimators' convention d = c * angle(conj(a) b) / (4 pi f)."""
    disp = np.asarray(disp, float)
    if disp.ndim == 1:
        disp = disp[:, None, None]
    return iq0[None] * np.exp(1j * 4 * np.pi * F_DEMOD * disp / C)


def make_acquisition(disp_trk, disp_ref=None, t=None, t_ref=None, seed=0, **kw):
    from swp.viz.core.acquisition import Acquisition
    n, nz, nx = disp_trk.shape
    iq0 = speckle(nz, nx, seed)
    x = (np.arange(nx) - nx // 2) * DX
    z = 0.040 + np.arange(nz) * DZ
    ref = None if disp_ref is None else phase_encode(iq0, disp_ref)
    return Acquisition(iq=phase_encode(iq0, disp_trk), ref_iq=ref, x=x, z=z,
                       t=np.arange(n) / PRF if t is None else t, prf=PRF, f_demod=F_DEMOD,
                       f0=F_DEMOD / 2, c=C, dz=DZ, dx=DX, t_ref=t_ref, push_x=0.0,
                       push_z=float(z[nz // 2]), **kw)


def outward_wave(c=3.0, r0=0.0, n=58, x=None, width=3e-3, amp=5e-6):
    """Axial displacement (n, nx) of a Gaussian pulse travelling away from r0 at speed c."""
    t = np.arange(n) / PRF
    d = np.abs(x - r0)
    return amp * np.exp(-((d[None, :] - c * t[:, None] - 2e-3) / width) ** 2)


@pytest.fixture
def rng():
    return np.random.default_rng(1234)
