"""Windowed **RF** normalized cross-correlation (classic speckle tracking on radio-frequency data).

Unlike :mod:`loupas` / :mod:`kasai` (phase of baseband IQ) and :mod:`xcorr` (complex-IQ NCC), this
estimator tracks the **real RF** signal directly: the broadband carrier gives a sharp correlation
peak whose sub-sample location (parabolic interpolation) is the displacement. It is the precision
benchmark for displacement estimation, and needs the **fine axial grid** produced by re-beamforming
the RF locally around the M-line (``swp.acquisition.finegrid``): on the coarse (~lambda/2) IQ grid
there are too few samples per wavelength for the correlation peak to be well localized.

Input ``rf`` is a real (n_frames, nz, nx) volume (the ``iq`` argument name is kept only so the
estimator registry can call it uniformly; pass a real array). See docs/literature_review.md sec 2.3.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .common import DisplacementResult, samples_for_length


def _ncc_lags(ref: np.ndarray, cur: np.ndarray, w: int, L: int) -> np.ndarray:
    """Windowed NCC of two real (nz, nx) RF frames over axial lags [-L, L] -> (2L+1, nz, nx)."""
    from scipy.ndimage import uniform_filter1d

    def wsum(a):                                   # moving sum over the axial window
        return uniform_filter1d(a, size=w, axis=0, mode="nearest") * w

    e_ref = wsum(ref * ref)
    lags = np.arange(-L, L + 1)
    out = np.empty((lags.size, ref.shape[0], ref.shape[1]), dtype=np.float64)
    for i, l in enumerate(lags):
        shifted = np.roll(cur, -l, axis=0)
        cross = wsum(ref * shifted)
        e_cur = wsum(shifted * shifted)
        out[i] = cross / np.sqrt(e_ref * e_cur + 1e-20)
    return out


def _parabolic(ncc: np.ndarray) -> np.ndarray:
    """Sub-sample peak offset (in samples) by parabolic fit around the integer NCC peak.

    ncc : (2L+1, nz, nx). Returns (nz, nx) sub-sample shift = int_peak_lag + delta, where the
    integer peak lag is measured from the centre (zero-lag) of the correlation window.
    """
    nL, nz, nx = ncc.shape
    ipk = np.argmax(ncc, axis=0)                   # (nz, nx) index of peak in [0, 2L]
    ipk_c = np.clip(ipk, 1, nL - 2)
    zc = np.arange(nz)[:, None]
    xc = np.arange(nx)[None, :]
    y0 = ncc[ipk_c - 1, zc, xc]
    y1 = ncc[ipk_c, zc, xc]
    y2 = ncc[ipk_c + 1, zc, xc]
    denom = (y0 - 2 * y1 + y2)
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (y0 - y2) / denom, 0.0)
    delta = np.clip(delta, -1.0, 1.0)
    L = (nL - 1) // 2
    return (ipk.astype(np.float64) - L) + delta    # signed sub-sample lag from zero


def rf_ncc_displacement(iq: np.ndarray, dz: float, dx: float, c: float, f_demod: float,
                        prf: float, window_m: float = 1.0e-3, max_disp_m: float = 0.2e-3,
                        mode: str = "relative_to_reference",
                        reference: Optional[np.ndarray] = None) -> DisplacementResult:
    """Axial displacement/velocity by windowed **RF** normalized cross-correlation.

    ``iq`` is a **real** (n_frames, nz, nx) RF volume (fine axial grid). ``window_m`` is the axial
    correlation window (~1-2 wavelengths), ``max_disp_m`` the half search range. A positive lag =
    tissue moved away from the transducer (increasing depth).
    """
    rf = np.asarray(iq).real.astype(np.float64)
    n_frames, nz, nx = rf.shape
    dt = 1.0 / prf
    w = max(3, samples_for_length(window_m, dz))
    L = max(1, samples_for_length(max_disp_m, dz))

    def pair(a, b):                                # sub-sample lag (nz,nx) aligning b onto a, in metres
        return _parabolic(_ncc_lags(a, b, w, L)) * dz

    if mode == "relative_to_reference":
        if reference is None:
            raise ValueError("relative_to_reference requires `reference`")
        ref = np.asarray(reference).real
        ref = ref.mean(axis=0) if ref.ndim == 3 else ref
        disp = np.stack([pair(ref, rf[n]) for n in range(n_frames)], axis=0)
        velocity = np.diff(disp, axis=0) / dt
    elif mode == "frame_to_frame":
        steps = np.stack([pair(rf[n], rf[n + 1]) for n in range(n_frames - 1)], axis=0)
        velocity = steps / dt
        disp = np.concatenate([np.zeros((1, nz, nx)), np.cumsum(steps, axis=0)], axis=0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    fc = np.full((nz, nx), float(f_demod))
    return DisplacementResult(displacement=disp, velocity=velocity, fc=fc, mode=mode, dt=dt,
                              estimator="rf_ncc")
