"""Wave speed and direction from the spatial gradient of the temporal phase.

Why this and not the structure tensor
-------------------------------------
``structure.py`` is mathematically correct and passes its synthetic sweep at 0.7 % error, but it
failed on real data: it locked onto axial speckle and reported 0.2-0.7 m/s travelling into depth
(``docs/field_estimator_plan.md``, Stage 3). The cause is that it assumes the **spatial gradient
is dominated by the wave**, whereas a real axial displacement field is dominated by static
speckle amplitude structure. Adding a speckle model to the synthetic generator reproduces the
failure exactly: c = 3.01 -> 0.32 m/s, theta = 0 -> -72 deg.

This estimator works on a quantity speckle leaves alone. Take the temporal Fourier component at
the wave's dominant frequency,

    U(x, z) = A(x, z) exp( i phi(x, z) )

where speckle and attenuation live in the real amplitude ``A`` and the wave vector lives in the
phase. Then

    grad U / U = grad(ln A) + i grad(phi)        so      k = Im( grad U / U )

and the amplitude term falls entirely in the real part. Numerically the division is done as

    k = Im( < conj(U) grad U >_W ) / < |U|^2 >_W

which is an amplitude-weighted average of the phase gradient over a window: stable where the
signal is weak, and needing no phase unwrapping because the complex field is differentiated
directly rather than its angle.

    c = 2 pi f / |k|            theta = atan2(kz, kx)

Being frequency-resolved is not incidental - the measured dispersion on a real AVC window runs
from 1.4 m/s at 7 Hz to 9.3 m/s at 28 Hz, so a single number is meaningless without a frequency
attached, which the structure tensor could not provide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .structure import Grid

__all__ = ["PhaseEstimate", "dominant_frequency", "phase_gradient_speed", "aggregate_phase"]


@dataclass
class PhaseEstimate:
    speed: np.ndarray        # (n_z, n_x) [m/s]
    theta: np.ndarray        # (n_z, n_x) [rad]
    kz: np.ndarray
    kx: np.ndarray
    weight: np.ndarray       # <|U|^2>, the natural amplitude weight
    coherence: np.ndarray    # [0, 1] consistency of the phase gradient within the window
    f_hz: float
    grid: Grid
    margin_vox: tuple = (0, 0)

    def interior(self):
        m = np.zeros(self.speed.shape, bool)
        mz, mx = (min(v, s // 2 - 1) for v, s in zip(self.margin_vox, self.speed.shape))
        m[mz:self.speed.shape[0] - mz or None, mx:self.speed.shape[1] - mx or None] = True
        return m


def dominant_frequency(field, grid, band_hz=(5.0, 60.0)):
    """Energy-weighted centroid of the temporal spectrum inside ``band_hz``."""
    f = np.asarray(field, float)
    n = f.shape[0]
    F = np.fft.rfft(f - f.mean(axis=0), axis=0)
    freqs = np.fft.rfftfreq(n, grid.dt)
    P = (np.abs(F) ** 2).sum(axis=(1, 2))
    m = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not m.any() or P[m].sum() <= 0:
        return float(np.mean(band_hz))
    return float((freqs[m] * P[m]).sum() / P[m].sum())


def _complex_at(field, grid, f_hz, taper=True):
    """Temporal Fourier component of the field at one frequency -> complex (n_z, n_x).

    Uses the exact ``exp(-i 2 pi f t)`` projection rather than an FFT bin, so the frequency does
    not have to land on the grid - it is a short window and the bins are ~7 Hz apart.
    """
    f = np.asarray(field, float)
    n = f.shape[0]
    t = np.arange(n) * grid.dt
    w = np.hanning(n) if taper else np.ones(n)
    kern = (w * np.exp(-2j * np.pi * f_hz * t))[:, None, None]
    return (f * kern).sum(axis=0)


def phase_gradient_speed(field, grid, f_hz=None, band_hz=(5.0, 60.0),
                         window_space_mm=2.0, deriv_space_mm=0.0):
    """Speed/direction map from the phase gradient at one temporal frequency."""
    from scipy.ndimage import gaussian_filter

    if f_hz is None:
        f_hz = dominant_frequency(field, grid, band_hz)
    U = _complex_at(field, grid, f_hz)

    if deriv_space_mm > 0:
        sz = max(deriv_space_mm * 1e-3 / grid.dz, 0.5)
        sx = max(deriv_space_mm * 1e-3 / grid.dx, 0.5)
        Ur = gaussian_filter(U.real, (sz, sx), mode="nearest")
        Ui = gaussian_filter(U.imag, (sz, sx), mode="nearest")
        U = Ur + 1j * Ui
    dUz = np.gradient(U, grid.dz, axis=0)
    dUx = np.gradient(U, grid.dx, axis=1)

    sw = (max(window_space_mm * 1e-3 / grid.dz, 0.8),
          max(window_space_mm * 1e-3 / grid.dx, 0.8))

    def smooth_c(a):
        return (gaussian_filter(a.real, sw, mode="nearest")
                + 1j * gaussian_filter(a.imag, sw, mode="nearest"))

    num_z = smooth_c(np.conj(U) * dUz)
    num_x = smooth_c(np.conj(U) * dUx)
    den = gaussian_filter(np.abs(U) ** 2, sw, mode="nearest") + 1e-30

    kz = np.imag(num_z) / den
    kx = np.imag(num_x) / den
    kmag = np.hypot(kz, kx)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = 2.0 * np.pi * f_hz / kmag
    speed = np.where(kmag > 1e-9, speed, np.inf)

    # Phase-gradient consistency: |<conj(U) grad U>| / <|U| |grad U|>. One if the phase advances
    # uniformly across the window, near zero if the gradient direction is scattered.
    gmag = gaussian_filter(np.abs(U) * np.hypot(np.abs(dUz), np.abs(dUx)), sw,
                           mode="nearest") + 1e-30
    coh = np.hypot(np.abs(num_z), np.abs(num_x)) / gmag

    return PhaseEstimate(speed=speed, theta=np.arctan2(kz, kx), kz=kz, kx=kx,
                         weight=den, coherence=np.clip(coh, 0.0, 1.0), f_hz=float(f_hz),
                         grid=grid, margin_vox=tuple(int(np.ceil(2.0 * s)) for s in sw))


def aggregate_phase(est, mask=None, min_coherence=0.3, min_weight_pct=60.0,
                    speed_bounds=(0.2, 20.0), interior_only=True):
    """Amplitude-weighted summary over the voxels that pass the gates."""
    sel = np.isfinite(est.speed)
    sel &= (est.speed >= speed_bounds[0]) & (est.speed <= speed_bounds[1])
    sel &= est.coherence >= min_coherence
    if min_weight_pct > 0:
        sel &= est.weight >= np.percentile(est.weight, min_weight_pct)
    if interior_only:
        sel &= est.interior()
    if mask is not None:
        sel &= np.broadcast_to(mask, est.speed.shape)

    out = dict(f_hz=est.f_hz, n_px=int(sel.sum()), frac_kept=float(sel.mean()))
    if not sel.any():
        out.update(speed=np.nan, theta_deg=np.nan, coherence=np.nan,
                   direction_spread_deg=np.nan)
        return out
    w = est.weight[sel] * est.coherence[sel]
    s = est.speed[sel]
    order = np.argsort(s)
    cw = np.cumsum(w[order]) / w.sum()
    out["speed"] = float(s[order][np.searchsorted(cw, 0.5)])
    out["speed_iqr"] = (float(s[order][np.searchsorted(cw, 0.25)]),
                        float(s[order][np.searchsorted(cw, 0.75)]))
    ang = 2.0 * est.theta[sel]
    mx, my = np.average(np.cos(ang), weights=w), np.average(np.sin(ang), weights=w)
    out["theta_deg"] = float(np.degrees(0.5 * np.arctan2(my, mx)))
    R = float(np.hypot(mx, my))
    out["direction_spread_deg"] = float(np.degrees(
        0.5 * np.sqrt(max(-2.0 * np.log(max(R, 1e-12)), 0.0))))
    out["coherence"] = float(np.average(est.coherence[sel], weights=w))
    return out
