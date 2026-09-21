"""Local wave speed and propagation direction from the 3-D structure tensor.

The idea
--------
For a locally plane wave ``u(x, z, t) = f(k . r - omega t)`` the 3-D gradient

    grad_3 u = (du/dx, du/dz, du/dt)  is parallel to  (kx, kz, -omega)

so the local orientation of the space-time volume *is* the wave vector. Averaging the outer
product over a small window gives the structure tensor ``J = < grad_3 u grad_3 u^T >``, whose
dominant eigenvector estimates that direction. From it,

    theta = atan2(vz, vx)                      propagation direction
    c     = |vt| / hypot(vx, vz)               speed  [ (rad/s) / (rad/m) = m/s ]

**Gradients must be in physical units** - divided by dx, dz, dt - before the tensor is formed,
or the speed is wrong by the pixel aspect ratio. That is the most likely silent bug in this
module and is what the synthetic stage exists to catch.

Why this is not defeated by the geometry that defeats the slant stack
---------------------------------------------------------------------
The slant stack needs the wave to move measurably *across the aperture*: over 21 mm a 170-290 mm
wave moves ~5 ms out of a ~60 ms period, which is why it fails. The structure tensor never
measures displacement across an aperture; it measures the local gradient orientation, which for
these waves is a perfectly ordinary quantity - at 4 m/s and 16 Hz the wave vector is
``|k| = 2 pi f / c ~ 25 rad/m`` against ``omega ~ 100 rad/s``, a ratio of about 0.25. Nothing is
degenerate. What limits it is noise in the *spatial* gradient, which the synthetic sweep measures.

Confidence
----------
The eigenvalues give a planarity measure for free:

    coherence = (lambda1 - lambda2) / (lambda1 + lambda2)

near 1 for a single plane wave, near 0 for noise, two crossing waves, or bulk motion. This is the
quality gate the 1-D path never had - and it must be validated as one, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Grid", "FieldEstimate", "estimate_field", "aggregate"]


@dataclass(frozen=True)
class Grid:
    """Physical sampling of a field ``(n_t, n_z, n_x)``."""
    dz: float            # axial pixel [m]
    dx: float            # lateral pixel [m]
    dt: float            # frame period [s]

    @property
    def dz_mm(self):
        return self.dz * 1e3

    @property
    def dx_mm(self):
        return self.dx * 1e3

    @property
    def dt_ms(self):
        return self.dt * 1e3


@dataclass
class FieldEstimate:
    """Per-voxel outputs; all arrays share the shape of the input field."""
    speed: np.ndarray            # [m/s]
    theta: np.ndarray            # [rad], atan2(kz, kx); +x is lateral, +z is depth
    coherence: np.ndarray        # [0, 1]
    energy: np.ndarray           # trace of the tensor - how much gradient there is at all
    grid: Grid
    margin_vox: tuple = (0, 0, 0)   # (t, z, x) voxels the averaging window reaches past the edge

    def interior(self):
        """Voxels far enough from every edge that the averaging window is fully inside.

        Not cosmetic. The window is averaged with ``mode="nearest"``, so within about 2 sigma of
        a face the tensor is contaminated by replicated samples, which biases the speed HIGH.
        Measured on a noiseless plane wave in a 12 mm wall: +14.6 % at theta = 60 deg over the
        whole volume against +8.0 % over the interior, and in a 24 mm wall +4.7 % against +0.5 %.
        """
        m = np.zeros(self.speed.shape, bool)
        mt, mz, mx = (min(v, s // 2 - 1) for v, s in zip(self.margin_vox, self.speed.shape))
        m[mt:self.speed.shape[0] - mt or None,
          mz:self.speed.shape[1] - mz or None,
          mx:self.speed.shape[2] - mx or None] = True
        return m

    def summary(self, mask=None, min_coherence=0.5, min_energy_pct=50.0):
        return aggregate(self, mask=mask, min_coherence=min_coherence,
                         min_energy_pct=min_energy_pct)


def _gaussian_derivative(field, sigma_vox, axis):
    """Derivative along one axis, smoothed - far less noisy than a bare finite difference."""
    from scipy.ndimage import gaussian_filter1d

    out = field
    for ax, s in enumerate(sigma_vox):
        order = 1 if ax == axis else 0
        if s > 0:
            out = gaussian_filter1d(out, sigma=s, axis=ax, order=order, mode="nearest")
        elif order == 1:
            out = np.gradient(out, axis=ax)
    return out


def estimate_field(field, grid, sigma_space_mm=0.8, sigma_time_ms=2.0,
                   window_space_mm=1.2, window_time_ms=8.0):
    """Structure-tensor speed / direction / coherence for a field ``(n_t, n_z, n_x)``.

    Args:
        field: the filtered displacement or velocity field.
        grid: :class:`Grid` giving ``dz``, ``dx``, ``dt`` in SI.
        sigma_space_mm, sigma_time_ms: smoothing applied when taking the derivatives.
        window_space_mm, window_time_ms: the window the outer product is averaged over. It must
            be larger than the derivative scale, and is what sets the spatial resolution of the
            result. The default 1.2 mm is deliberately small: the septal wall is only ~10-12 mm
            thick and edge voxels must be discarded (see FieldEstimate.interior), so a 3 mm
            window leaves just 2 % of the wall usable against 23 % at 1.2 mm - at no cost in
            accuracy (measured 2026-09-21, synthetic plane wave).
    """
    f = np.asarray(field, float)
    if f.ndim != 3:
        raise ValueError(f"expected (n_t, n_z, n_x), got {f.shape}")
    from scipy.ndimage import gaussian_filter

    # derivative scales, in voxels, per axis (t, z, x)
    sd = (max(sigma_time_ms * 1e-3 / grid.dt, 0.6),
          max(sigma_space_mm * 1e-3 / grid.dz, 0.6),
          max(sigma_space_mm * 1e-3 / grid.dx, 0.6))
    # physical gradients: divide by the sample spacing of that axis
    gt = _gaussian_derivative(f, sd, 0) / grid.dt
    gz = _gaussian_derivative(f, sd, 1) / grid.dz
    gx = _gaussian_derivative(f, sd, 2) / grid.dx

    # window for averaging the outer product
    sw = (max(window_time_ms * 1e-3 / grid.dt, 0.8),
          max(window_space_mm * 1e-3 / grid.dz, 0.8),
          max(window_space_mm * 1e-3 / grid.dx, 0.8))

    def win(a):
        return gaussian_filter(a, sigma=sw, mode="nearest")

    # J is symmetric; order the axes (x, z, t) so the eigenvector reads (vx, vz, vt)
    Jxx, Jzz, Jtt = win(gx * gx), win(gz * gz), win(gt * gt)
    Jxz, Jxt, Jzt = win(gx * gz), win(gx * gt), win(gz * gt)

    J = np.empty(f.shape + (3, 3), float)
    J[..., 0, 0], J[..., 1, 1], J[..., 2, 2] = Jxx, Jzz, Jtt
    J[..., 0, 1] = J[..., 1, 0] = Jxz
    J[..., 0, 2] = J[..., 2, 0] = Jxt
    J[..., 1, 2] = J[..., 2, 1] = Jzt

    evals, evecs = np.linalg.eigh(J)              # ascending
    lam1, lam2 = evals[..., 2], evals[..., 1]
    v = evecs[..., 2]                             # dominant eigenvector (vx, vz, vt)

    # eigh returns a sign-ambiguous vector; (kx, kz, -omega) has a negative time component for
    # omega > 0, so fix the sign by requiring vt <= 0.
    flip = v[..., 2] > 0
    v = np.where(flip[..., None], -v, v)

    vx, vz, vt = v[..., 0], v[..., 1], v[..., 2]
    k_mag = np.hypot(vx, vz)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.abs(vt) / k_mag
        coherence = (lam1 - lam2) / (lam1 + lam2)
    speed = np.where(k_mag > 1e-12, speed, np.inf)
    coherence = np.nan_to_num(coherence, nan=0.0, posinf=0.0, neginf=0.0)
    theta = np.arctan2(vz, vx)
    return FieldEstimate(speed=speed, theta=theta, coherence=np.clip(coherence, 0.0, 1.0),
                         energy=lam1 + lam2 + evals[..., 0], grid=grid,
                         margin_vox=tuple(int(np.ceil(2.0 * s_)) for s_ in sw))


def aggregate(est, mask=None, min_coherence=0.5, min_energy_pct=50.0,
              speed_bounds=(0.2, 20.0), interior_only=True):
    """Coherence-weighted summary over the voxels that pass the gates.

    Direction is averaged as a **doubled angle**: a wavefront and its reverse describe the same
    line of propagation, so averaging theta directly would cancel two voxels that agree.
    """
    sel = np.isfinite(est.speed)
    sel &= (est.speed >= speed_bounds[0]) & (est.speed <= speed_bounds[1])
    sel &= est.coherence >= min_coherence
    if min_energy_pct > 0:
        thr = np.percentile(est.energy, min_energy_pct)
        sel &= est.energy >= thr
    if interior_only:
        sel &= est.interior()          # edge voxels bias the speed high; see FieldEstimate.interior
    if mask is not None:
        sel &= np.broadcast_to(mask, est.speed.shape)

    out = dict(n_voxels=int(sel.sum()), frac_kept=float(sel.mean()))
    if not sel.any():
        out.update(speed=np.nan, speed_iqr=np.nan, theta_deg=np.nan,
                   coherence=np.nan, direction_spread_deg=np.nan)
        return out
    w = est.coherence[sel]
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
    out["direction_spread_deg"] = float(np.degrees(0.5 * np.sqrt(max(-2.0 * np.log(max(R, 1e-12)),
                                                                    0.0))))
    out["coherence"] = float(np.average(est.coherence[sel], weights=w))
    return out


def interference_indicator(field, est=None):
    """Spatial modulation of the wave envelope - a candidate detector for interference.

    **Why this exists.** The coherence measure above does *not* detect two co-propagating waves
    crossing the ROI: measured on synthetic fields, a single wave scores 0.995 and two waves 60
    degrees apart score 0.992, while the reported speed is corrupted from 3.0 to 3.5-4.4 m/s. The
    reason is that a superposition of two plane waves is still *locally* a well-oriented
    structure, so the local orientation the tensor measures is well defined - it simply is not
    either of the two wave vectors. Coherence does catch pure noise (0.38) and counter-propagating
    waves (0.94), but not the co-propagating case.

    Interference instead shows up as standing spatial structure in the amplitude. This returns the
    coefficient of variation of the peak-amplitude map over the ROI. Synthetic values: single wave
    0.008, two crossing at 60 deg 0.025-0.035, counter-propagating 0.115, noise 0.168.

    **Do not trust this on real data without validating it there.** In vivo the envelope varies
    for reasons that have nothing to do with interference - attenuation, speckle, varying
    coupling, wall curvature - any of which could swamp the effect. It is recorded here as the
    most promising candidate found at Stage 1, not as a working gate.
    """
    f = np.asarray(field, float)
    env = np.abs(f).max(axis=0)
    if est is not None:
        m = est.interior().any(axis=0)
        env = env[m]
    env = env[np.isfinite(env)]
    if env.size < 4 or env.mean() <= 0:
        return float("nan")
    return float(env.std() / env.mean())
