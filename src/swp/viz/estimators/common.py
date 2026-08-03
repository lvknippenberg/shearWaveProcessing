"""Shared numerics and result type for displacement estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DisplacementResult:
    """Axial motion estimates on the tracking grid (SI units)."""
    displacement: np.ndarray       # (n_frames, nz, nx) [m], cumulative vs frame 0 or reference
    velocity: np.ndarray           # (n_frames-1, nz, nx) [m/s], particle velocity
    fc: np.ndarray                 # (nz, nx) [Hz] center frequency used (or scalar-filled)
    mode: str
    dt: float                      # slow-time interval [s]
    estimator: str = "loupas"

    @property
    def displacement_um(self) -> np.ndarray:
        return self.displacement * 1e6

    @property
    def velocity_mm_s(self) -> np.ndarray:
        return self.velocity * 1e3


def complex_uniform_filter1d(z: np.ndarray, size: int, axis: int) -> np.ndarray:
    """Moving-average a complex array along one axis (scipy's filter is real-only)."""
    if size <= 1:
        return z
    from scipy.ndimage import uniform_filter1d
    r = uniform_filter1d(z.real, size=size, axis=axis, mode="nearest")
    i = uniform_filter1d(z.imag, size=size, axis=axis, mode="nearest")
    return r + 1j * i


def samples_for_length(length_m: float, spacing_m: float, minimum: int = 1) -> int:
    """Odd-ish kernel length in samples for a desired physical extent."""
    n = int(round(length_m / spacing_m))
    return max(minimum, n)


def phase_to_displacement(phase: np.ndarray, fc, c: float) -> np.ndarray:
    """Axial displacement [m] from a slow-time phase shift.

    d = c * phase / (4 * pi * fc).  ``fc`` may be a scalar or broadcastable array.
    """
    return c * phase / (4.0 * np.pi * fc)


def local_center_frequency(iq: np.ndarray, f_demod: float, c: float, dz: float,
                           kernel_z: int = 1) -> np.ndarray:
    """Loupas local center frequency from the fast-time (axial) lag-1 autocorrelation.

    For baseband IQ demodulated at ``f_demod``, the residual axial phase progression
    gives the local deviation from ``f_demod``.  Returns an (nz, nx) array.

    iq : (n_frames, nz, nx) complex
    """
    fs = c / (2.0 * dz)                      # equivalent fast-time sampling frequency
    r10 = np.conj(iq[:, :-1, :]) * iq[:, 1:, :]   # (n_frames, nz-1, nx)
    r10 = r10.mean(axis=0)                   # ensemble-average over slow time -> (nz-1, nx)
    if kernel_z > 1:
        r10 = complex_uniform_filter1d(r10, kernel_z, axis=0)
    resid = np.angle(r10) * fs / (2.0 * np.pi)    # (nz-1, nx)
    # pad the last axial row so the map is (nz, nx)
    resid = np.vstack([resid, resid[-1:, :]])
    fc = f_demod + resid
    # guard against pathological near-zero / negative estimates
    fc = np.where(np.abs(fc) < 0.25 * f_demod, f_demod, fc)
    return fc
