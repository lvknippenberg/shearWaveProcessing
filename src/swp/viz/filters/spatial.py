"""Spatial and temporal smoothing / band-pass (displacement-space)."""
from __future__ import annotations

import numpy as np

from .context import FilterCtx


def _odd(n: int) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def spatial_smooth(field: np.ndarray, ctx: FilterCtx = None,
                   sigma_z_m: float = 0.3e-3, sigma_x_m: float = 0.6e-3) -> np.ndarray:
    """Per-frame **Gaussian** smoothing with physical sigmas (kept small to avoid blurring the
    wavefront and biasing speed)."""
    from scipy.ndimage import gaussian_filter
    sz = max(0.0, sigma_z_m / ctx.dz)
    sx = max(0.0, sigma_x_m / ctx.dx)
    return gaussian_filter(field, sigma=(0.0, sz, sx), mode="nearest")


def spatial_median(field: np.ndarray, ctx: FilterCtx = None,
                   size_z_m: float = 0.4e-3, size_x_m: float = 0.8e-3) -> np.ndarray:
    """Per-frame 2-D **median** smoothing (edge-preserving, removes speckle/outliers).

    Kernel sizes are given in metres and converted to (odd) sample counts.  Unlike the Gaussian
    it keeps sharp wavefront edges while suppressing salt-and-pepper noise -- useful for the noisy
    velocity maps.
    """
    from scipy.ndimage import median_filter
    kz = _odd(size_z_m / ctx.dz)
    kx = _odd(size_x_m / ctx.dx)
    return median_filter(field, size=(1, kz, kx), mode="nearest")


def temporal_moving_mean(field: np.ndarray, ctx: FilterCtx = None, window: int = 3) -> np.ndarray:
    """Moving-average across ``window`` frames (slow-time low-pass); reduces frame-to-frame noise
    at the cost of temporal resolution."""
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(field, size=max(1, int(window)), axis=0, mode="nearest")


def temporal_moving_median(field: np.ndarray, ctx: FilterCtx = None, window: int = 3) -> np.ndarray:
    """Moving **median** across ``window`` frames; rejects transient outlier frames while keeping
    the wavefront edge sharper than a moving mean."""
    from scipy.ndimage import median_filter
    return median_filter(field, size=(_odd(window), 1, 1), mode="nearest")


def temporal_bandpass(field: np.ndarray, ctx: FilterCtx = None,
                      f_lo: float = 40.0, f_hi: float = 800.0, order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth temporal band-pass along the frame axis to isolate the wave's
    frequency content (also useful before phase-velocity estimation)."""
    from scipy.signal import butter, filtfilt
    ny = 0.5 * ctx.prf
    lo, hi = f_lo / ny, min(f_hi / ny, 0.99)
    n = field.shape[0]
    if lo <= 0:
        b, a = butter(order, hi, btype="lowpass")
    else:
        b, a = butter(order, [lo, hi], btype="bandpass")
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return field - field.mean(axis=0, keepdims=True)
    return filtfilt(b, a, field, axis=0)
