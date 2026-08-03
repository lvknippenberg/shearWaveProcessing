"""Windowed complex-IQ normalized cross-correlation (speckle tracking).

The precision benchmark for small-displacement tracking; benefits from the axially-refined
fine grid (small ``dz``) to resolve sub-sample shifts.  Integer-lag search + parabolic
sub-sample interpolation on the normalized-correlation magnitude, vectorised over lags.
See docs/literature_review.md sec. 2.3.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .common import DisplacementResult, complex_uniform_filter1d, samples_for_length


def _windowed_energy(x2: np.ndarray, w: int) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x2, size=w, axis=0, mode="nearest")


def _pair_displacement(ref: np.ndarray, cur: np.ndarray, dz: float, c: float, f_demod: float,
                       window_m: float, max_disp_m: float,
                       zero_lag_penalty: float = 0.05) -> np.ndarray:
    """Axial displacement (nz, nx) [m] aligning ``cur`` to ``ref`` (hybrid xcorr).

    Two stages, as in autocorrelation-guided cross-correlation:
      1. **Envelope** normalized cross-correlation gives the *integer* sample lag,
         unambiguously (the demodulated carrier makes complex-correlation magnitude
         cycle-hop; the envelope has a single peak).  This resolves large displacements.
      2. The **complex phase** of the windowed correlation at that lag gives the sub-sample
         residual: d_sub = c*phase / (4 pi f_demod).  ARF displacements here are << dz, so
         this phase term dominates -- the fine grid alone cannot resolve them geometrically.
    """
    nz, nx = ref.shape
    w = max(3, samples_for_length(window_m, dz))
    L = max(1, samples_for_length(max_disp_m, dz))
    eref_sig, cur_sig = np.abs(ref), np.abs(cur)
    eref = _windowed_energy(eref_sig ** 2, w)
    lags = np.arange(-L, L + 1)
    ncc = np.empty((lags.size, nz, nx))
    for i, l in enumerate(lags):
        shifted = np.roll(cur_sig, -l, axis=0)
        s = _windowed_energy(eref_sig * shifted, w)
        ecur = _windowed_energy(shifted ** 2, w)
        ncc[i] = s / np.sqrt(eref * ecur + 1e-20)
    # bias toward zero lag: a non-zero integer lag must beat lag 0 by penalty*|lag|.
    # Sub-dz / sub-wavelength displacements (as in ARF) then stay at lag 0 and are
    # resolved purely by the phase term below, rather than hopping on envelope noise.
    ncc = ncc - zero_lag_penalty * np.abs(lags)[:, None, None]
    ipk = np.argmax(ncc, axis=0)                       # (nz, nx) integer-lag index
    int_lag = lags[ipk]                                # integer sample shift
    # complex correlation at the (per-pixel) integer lag -> phase sub-sample
    zc = np.arange(nz)[:, None]
    xc = np.arange(nx)[None, :]
    cur_shift = np.take_along_axis(
        np.stack([np.roll(cur, -l, axis=0) for l in lags], axis=0),
        ipk[None], axis=0)[0]
    s_cplx = complex_uniform_filter1d(np.conj(ref) * cur_shift, w, axis=0)
    phase = np.angle(s_cplx)
    d_sub = c * phase / (4.0 * np.pi * f_demod)
    return int_lag * dz + d_sub


def xcorr_displacement(iq: np.ndarray, dz: float, dx: float, c: float, f_demod: float,
                       prf: float, window_m: float = 1.5e-3, max_disp_m: float = 0.3e-3,
                       mode: str = "relative_to_reference",
                       reference: Optional[np.ndarray] = None) -> DisplacementResult:
    """Estimate axial displacement/velocity by windowed IQ cross-correlation.

    ``window_m`` is the axial correlation window (~1-2 wavelengths); ``max_disp_m`` the
    half search range (phase-wrap-free up to this displacement).
    """
    iq = np.asarray(iq)
    n_frames, nz, nx = iq.shape
    dt = 1.0 / prf

    if mode == "relative_to_reference":
        if reference is None:
            raise ValueError("relative_to_reference requires `reference`")
        ref = np.asarray(reference)
        ref = ref.mean(axis=0) if ref.ndim == 3 else ref
        disp = np.stack([_pair_displacement(ref, iq[n], dz, c, f_demod, window_m, max_disp_m)
                         for n in range(n_frames)], axis=0)
        velocity = np.diff(disp, axis=0) / dt
    elif mode == "frame_to_frame":
        steps = np.stack([_pair_displacement(iq[n], iq[n + 1], dz, c, f_demod, window_m, max_disp_m)
                          for n in range(n_frames - 1)], axis=0)
        velocity = steps / dt
        disp = np.concatenate([np.zeros((1, nz, nx)), np.cumsum(steps, axis=0)], axis=0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    fc = np.full((nz, nx), float(f_demod))
    return DisplacementResult(displacement=disp, velocity=velocity, fc=fc,
                              mode=mode, dt=dt, estimator="xcorr")
