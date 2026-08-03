"""M3 motion-compensation filters (displacement-space).

See docs/literature_review.md sec. 3.  Phantom data has no cardiac motion, so these are
near-identity there; they matter for in-vivo septal data.
"""
from __future__ import annotations

import numpy as np

from .context import FilterCtx


def polynomial_drift(field: np.ndarray, ctx: FilterCtx = None, order: int = 2,
                     fit_frac: float = 1.0) -> np.ndarray:
    """Subtract a low-order polynomial in slow time per pixel (Giannantonio-style detrend).

    A transient shear wave is not captured by a low-order polynomial, so subtracting the
    fit removes slow bulk/physiological motion while preserving the wave.  ``order`` is the
    polynomial order; ``fit_frac`` optionally restricts the fit to the first fraction of
    frames (e.g. pre-wave) if set < 1.
    """
    n = field.shape[0]
    tt = np.arange(n, dtype=float)
    nfit = max(order + 1, int(round(fit_frac * n)))
    V = np.vander(tt, order + 1)                       # (n, order+1)
    Vf = V[:nfit]
    flat = field.reshape(n, -1)
    coef, *_ = np.linalg.lstsq(Vf, flat[:nfit], rcond=None)
    trend = V @ coef
    return (flat - trend).reshape(field.shape)


def temporal_highpass(field: np.ndarray, ctx: FilterCtx = None, fc_hz: float = 80.0,
                      order: int = 2) -> np.ndarray:
    """Zero-phase Butterworth temporal high-pass along the frame axis.

    Removes low-frequency bulk motion below ``fc_hz``.  Corner must sit below the wave's
    temporal content or slow (diastolic) waves are attenuated.
    """
    from scipy.signal import butter, filtfilt
    prf = ctx.prf
    wn = fc_hz / (0.5 * prf)
    wn = min(max(wn, 1e-3), 0.99)
    b, a = butter(order, wn, btype="highpass")
    n = field.shape[0]
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return field - field.mean(axis=0, keepdims=True)
    return filtfilt(b, a, field, axis=0)


def reference_motion_compensation(field: np.ndarray, ctx: FilterCtx = None, order: int = 2,
                                  use_last_frac: float = 1.0, anchor: bool = False) -> np.ndarray:
    """Estimate cardiac motion from the pre-push reference frames and subtract it.

    The reference frames contain pure cardiac motion (no shear wave).  Per pixel we fit a
    low-order polynomial to the reference displacement trajectory ``ctx.ref_disp`` over the
    reference times ``ctx.t_ref`` (both negative, before the push at t=0), **extrapolate** it
    onto the tracking timestamps ``ctx.t``, and subtract -- leaving the ARF response plus
    residual noise.  This is the Giannantonio pre-push extrapolation filter (review ref [8]);
    unlike ``temporal_highpass`` / ``polynomial_drift`` it *measures* the motion rather than
    assuming a cutoff, and the fit never sees the wave so the ARF transient is untouched.

    Requires ``field`` to be **displacement** on the tracking grid (n_frames, nz, nx) and
    ``ctx.ref_disp`` / ``ctx.t_ref`` populated (done by the pipeline for in-vivo data).
    ``order`` is the polynomial order (2 = captures acceleration); ``use_last_frac`` optionally
    restricts the fit to the last fraction of the reference window (nearest the push).

    ``anchor`` (velocity-anchored linear mode): fit only a line to the last ``use_last_frac`` of
    the reference window, then predict continuity from the *measured* last reference displacement,
    ``pred(t) = u_ref[-1] + v·(t − t_ref[-1])`` with ``v`` the fitted cardiac velocity. This uses
    only the near-push velocity and pins the prediction to the true pre-push position, which is far
    less noise-sensitive than free polynomial extrapolation (``order`` is ignored when ``anchor``).
    """
    if ctx is None or ctx.ref_disp is None or ctx.t_ref is None:
        raise ValueError("reference_motion_compensation requires ctx.ref_disp and ctx.t_ref "
                         "(populated by the pipeline for in-vivo data with reference frames)")
    ref = ctx.ref_disp                 # (n_ref, nz, nx)
    t_ref = np.asarray(ctx.t_ref, float)
    t = np.asarray(ctx.t, float)
    n_ref = ref.shape[0]
    scale = 1e-3                        # fit in ms for conditioning; push at t=0 is the origin
    shape = field.shape

    if anchor:
        k = max(2, int(round(use_last_frac * n_ref)))
        tr = t_ref[-k:] / scale
        refk = ref[-k:].reshape(k, -1)
        A = np.vstack([tr, np.ones_like(tr)]).T          # linear design
        coef, *_ = np.linalg.lstsq(A, refk, rcond=None)  # [slope, intercept] per pixel
        vel = coef[0]                                     # cardiac velocity per pixel [m/ms]
        u_last = ref[-1].reshape(-1)                      # measured last-reference displacement
        pred = (u_last[None, :] + vel[None, :] * ((t[:, None] - t_ref[-1]) / scale))
        return field - pred.reshape(shape)

    k = max(order + 1, int(round(use_last_frac * n_ref)))
    tr = t_ref[-k:]
    refk = ref[-k:].reshape(k, -1)
    Ar = np.vander(tr / scale, order + 1)
    coef, *_ = np.linalg.lstsq(Ar, refk, rcond=None)     # (order+1, npix)
    At = np.vander(t / scale, order + 1)
    pred = (At @ coef).reshape(shape)                    # predicted cardiac disp during tracking
    return field - pred


def adaptive_highpass(field: np.ndarray, ctx: FilterCtx = None, base_fc: float = 40.0,
                      gain: float = 60.0, max_fc: float = 180.0) -> np.ndarray:
    """Per-pixel temporal high-pass whose corner adapts to the local cardiac-motion strength,
    estimated from the reference frames.

    Where the reference frames show strong cardiac motion, use a higher corner (remove more
    low-frequency content); where they are quiet, keep a low corner so slow (diastolic) shear
    waves survive — the reference-informed answer to the "fixed cutoff" concern.  Corner per
    pixel: ``fc = clip(base_fc + gain·(s − 1), base_fc, max_fc)`` with ``s`` the pixel's mean
    reference speed normalised to the median.  Applied via a first-order Butterworth-like mask
    ``H(f) = f² / (f² + fc²)`` in the temporal FFT.
    """
    if ctx is None or ctx.ref_disp is None:
        raise ValueError("adaptive_highpass requires ctx.ref_disp (in-vivo reference frames)")
    prf = ctx.prf
    ref_vel = np.abs(np.diff(ctx.ref_disp, axis=0)).mean(axis=0)   # (nz, nx) mean |ref velocity|
    s = ref_vel / (np.median(ref_vel) + 1e-20)
    fc = np.clip(base_fc + gain * (s - 1.0), base_fc, max_fc)       # (nz, nx) corner [Hz]
    n = field.shape[0]
    freqs = np.fft.rfftfreq(n, d=1.0 / prf)[:, None, None]          # (nf,1,1)
    F = np.fft.rfft(field, axis=0)
    mask = freqs ** 2 / (freqs ** 2 + fc[None, :, :] ** 2 + 1e-20)
    return np.fft.irfft(F * mask, n=n, axis=0)


def axial_strain(field: np.ndarray, ctx: FilterCtx = None, smooth: int = 3) -> np.ndarray:
    """Axial gradient (strain / strain-rate), translation-invariant wavefront emphasis.

    Returns d(field)/dz [1/m if field is displacement].  Insensitive to spatially-uniform
    bulk translation, a robust cross-check for the propagating wave.
    """
    dz = ctx.dz
    g = np.gradient(field, dz, axis=1)
    if smooth > 1:
        from scipy.ndimage import uniform_filter1d
        g = uniform_filter1d(g, smooth, axis=1, mode="nearest")
    return g
