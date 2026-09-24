"""Transfer an M-line drawn on one B-mode buffer to another (e.g. focused buffer 3 -> diverging buffer 4).

The buffers share one image geometry: on the resolution phantom, wire targets agree to < 35 um
between buffers 1, 3 and 4 (``study/analysis/buffer_registration_phantom.py``). In vivo they are
acquired in different heartbeats, so the anatomy moves between the frame a line is drawn on and the
buffer-4 data it is applied to (median 3 mm, up to ~12 mm, between the phase-matched buffer-3
frame and buffer 4). This module estimates that motion locally - in a box around the line - as a
rigid transform (translation, optionally a small rotation about the box centre) and applies it to
the line.

Images are compared at anatomy scale, not speckle scale: log-envelope band-passed between
``fine_mm`` and ``coarse_mm`` (difference of Gaussians), which removes the speckle pattern (not
shared between a focused and a diverging-wave image, nor between beats) and the depth/gain trend
(which otherwise pins the registration at zero), then standardised and Hann-tapered. Translation
comes from phase correlation (sub-pixel, ``upsample_factor=10``, iterated to remove the taper's
bias towards zero); the optional rotation from a grid search on the correlation. The estimate is
repeated over an ensemble of box margins (and, when the caller passes several target images, e.g.
buffer-4 envelopes averaged over different windows) and the median transform is used; the ensemble
Two checks decide whether a mapping is trusted (:meth:`TransferResult.reliable`):

* ``agree``: the fraction of ensemble members whose line lies within 1 mm of the consensus (a single
  outlying member - one box catching another structure - does not fail an otherwise consistent
  estimate);
* ``known_err_mm``: the source image is moved by known shifts and the moved line mapped again; it
  must land where the unmoved one did. This catches an ambiguous correlation surface (several
  similar peaks), where every member can consistently pick the same wrong peak.

Coordinates are (x, z) in mm throughout; the transform maps a point p of the source image onto the
target image as ``T(p) = c + R(angle) (p - c) + (dx, dz)``, with c the box centre.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import affine_transform, gaussian_filter


@dataclass
class Transform:
    dx: float = 0.0          # mm
    dz: float = 0.0          # mm
    angle: float = 0.0       # deg, positive = counter-clockwise in the (x right, z down) image
    cx: float = 0.0          # rotation centre (mm)
    cz: float = 0.0
    corr: float = float("nan")    # correlation of the anatomy images after the transform
    corr0: float = float("nan")   # ... before (identity)

    def apply(self, pts):
        """Map (k, 2) (x, z) mm points from the source image onto the target image."""
        pts = np.asarray(pts, float)
        a = np.radians(self.angle)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        c = np.array([self.cx, self.cz])
        return c + (pts - c) @ R.T + np.array([self.dx, self.dz])


@dataclass
class TransferResult:
    points: np.ndarray                 # mapped (k, 2) (x, z) mm
    transform: Transform               # the median transform that was applied
    spread_mm: float                   # median over members of their median distance to the consensus line
    agree: float = float("nan")        # fraction of members within 1 mm of the consensus line
    known_err_mm: float = float("nan")     # worst known-shift re-mapping error (median over the line)
    spread_rms_mm: float = float("nan")    # RMS over members (outlier-sensitive; for reference)
    ensemble: list = field(default_factory=list)   # every individual Transform

    @property
    def shift_mm(self):
        return float(np.hypot(self.transform.dx, self.transform.dz))

    def reliable(self, min_agree=0.6, max_known_err_mm=1.0):
        """Most of the ensemble agrees on the line position AND known shifts of the source are
        recovered (a single, well-defined correlation peak). An unchecked result (``check=False``)
        is judged on agreement alone."""
        ok = np.isfinite(self.agree) and self.agree >= min_agree
        if np.isfinite(self.known_err_mm):
            ok = ok and self.known_err_mm <= max_known_err_mm
        return bool(ok)


def resample(img, x, z, gx, gz):
    """Image (z, x) on axes x, z [mm] resampled to the grid gx, gz (NaN outside)."""
    f = RegularGridInterpolator((z, x), img, bounds_error=False, fill_value=np.nan)
    Z, X = np.meshgrid(gz, gx, indexing="ij")
    return f(np.stack([Z, X], -1))


def anatomy(env, pix, fine_mm=1.0, coarse_mm=6.0):
    """Log-envelope band-passed to anatomy scale and standardised (no taper). NaN (outside the
    sector) is filled with the median before filtering."""
    d = 20 * np.log10(env / np.nanmax(env) + 1e-6)
    d = np.where(np.isfinite(d), d, np.nanmedian(d))
    d = gaussian_filter(d, fine_mm / pix) - gaussian_filter(d, coarse_mm / pix)
    return (d - d.mean()) / (d.std() + 1e-12)


def warp(env, x, z, t: Transform):
    """Image (z, x) on axes x, z [mm] moved by transform t: out(q) = env(t^-1 q), same axes."""
    Z, X = np.meshgrid(z, x, indexing="ij")
    a = np.radians(t.angle)
    Rinv = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
    q = np.stack([X, Z], -1) - np.array([t.cx + t.dx, t.cz + t.dz])
    p = q @ Rinv.T + np.array([t.cx, t.cz])
    f = RegularGridInterpolator((z, x), env, bounds_error=False, fill_value=np.nan)
    return f(np.stack([p[..., 1], p[..., 0]], -1))


def _taper(d):
    return d * np.outer(np.hanning(d.shape[0]), np.hanning(d.shape[1]))


def _rotate(img, angle_deg):
    """Rotate a (z, x) image about its centre by angle_deg, same convention as Transform."""
    if angle_deg == 0:
        return img
    a = np.radians(angle_deg)
    # forward rotation on (x, z) is [[c,-s],[s,c]]; on (z, x) index order it is [[c, s],[-s, c]].
    # affine_transform needs the inverse (output index -> input index).
    Minv = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    ctr = (np.array(img.shape) - 1) / 2.0
    return affine_transform(img, Minv, offset=ctr - Minv @ ctr, order=1, mode="nearest")


def register_rigid(ref, mov, pix, angles=(0.0,), max_shift_mm=15.0, angle_tol=0.01):
    """Rigid transform (about the image centre) that moves anatomy image ``mov`` onto ``ref``.

    Both are :func:`anatomy` outputs on the same grid (spacing ``pix`` mm). ``angles`` is the
    rotation search grid [deg] (default: translation only); of the angles whose correlation is
    within ``angle_tol`` of the best, the smallest is taken. Returns a Transform with the centre
    left at 0 (the caller sets it) - dx, dz, angle, corr, corr0.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as nd_shift
    R = _taper(ref)

    def corr_of(m):
        return float(np.corrcoef(R.ravel(), _taper(m).ravel())[0, 1])

    def at(angle):
        # The taper (same window on both images) biases a single phase-correlation estimate towards
        # zero shift; re-registering after applying the current estimate removes that bias.
        rot = _rotate(mov, angle)
        s = np.zeros(2)
        for _ in range(4):
            ds, _, _ = phase_cross_correlation(R, _taper(nd_shift(rot, s, order=1, mode="nearest")),
                                               upsample_factor=10, normalization=None)
            s = s + ds
            if np.hypot(*s) * pix > max_shift_mm:
                return None
            if np.hypot(*ds) * pix < 0.05:
                break
        return s, corr_of(nd_shift(rot, s, order=1, mode="nearest"))

    tried = {}
    for ang in np.atleast_1d(angles):
        r = at(float(ang))
        if r is not None:
            tried[float(ang)] = r
    if not tried:
        return Transform(corr=float("nan"), corr0=corr_of(mov))
    # refine on a 0.25 deg grid around the best coarse angle
    step = float(np.diff(np.atleast_1d(angles))[0]) if np.size(angles) > 1 else 0.0
    if step > 0.25:
        a0 = max(tried, key=lambda a: tried[a][1])
        for ang in np.arange(a0 - step + 0.25, a0 + step - 0.24, 0.25):
            r = at(float(ang))
            if r is not None:
                tried[float(ang)] = r
    # The correlation is nearly flat in the angle on in-vivo anatomy (a few hundredths over 10-20
    # deg), so the maximum alone is noise-driven: take the smallest rotation within angle_tol of it.
    cmax = max(r[1] for r in tried.values())
    ang = min((a for a, r in tried.items() if r[1] >= cmax - angle_tol), key=abs)
    s, c = tried[ang]
    return Transform(dx=float(s[1] * pix), dz=float(s[0] * pix), angle=ang, corr=c, corr0=corr_of(mov))


def transfer_line(points, src, dst, pix=0.394, margins=(8.0, 12.0, 16.0),
                  angles=(0.0,), fine_mm=1.0, coarse_mm=6.0, max_shift_mm=15.0, angle_tol=0.01,
                  check=True, check_shifts=((2.0, -1.5), (-3.0, 2.0))):
    """Map an M-line drawn on image ``src`` onto image ``dst``.

    points   (k, 2) line points (x, z) mm, in src coordinates
    src      (env, x_mm, z_mm): the envelope the line was drawn on and its axes
    dst      (env, x_mm, z_mm), or a list of them (e.g. buffer 4 averaged over different windows
             around the event): each is registered separately and adds to the ensemble
    margins  box margins [mm] around the line's bounding box; one estimate per margin per dst
    angles   rotation search grid [deg]. Default translation only: on in-vivo cardiac images the
             anatomy-scale correlation hardly changes with rotation within a ~25 mm box (a few
             hundredths over 10-20 deg), so a free rotation fits noise and drags the translation
             with it. Pass e.g. ``np.arange(-10, 10.01, 1)`` to allow it (regularised by angle_tol).
    check    re-map after moving the source by each of ``check_shifts`` (dx, dz) mm and report the
             worst error as ``known_err_mm`` (3x the cost)

    Returns a TransferResult: the line moved by the median transform of the ensemble.
    """
    kw = dict(pix=pix, margins=margins, angles=angles, fine_mm=fine_mm, coarse_mm=coarse_mm,
              max_shift_mm=max_shift_mm, angle_tol=angle_tol)
    pts = np.asarray(points, float)
    dsts = dst if isinstance(dst, (list, tuple)) and isinstance(dst[0], (list, tuple)) else [dst]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    ens = []
    for m in margins:
        # square-ish box centred on the line (odd pixel counts keep the centre on a pixel)
        c = (lo + hi) / 2
        half = (hi - lo) / 2 + m
        nx, nz = (2 * np.ceil(half / pix) + 1).astype(int)
        gx = c[0] + (np.arange(nx) - (nx - 1) / 2) * pix
        gz = c[1] + (np.arange(nz) - (nz - 1) / 2) * pix
        S = anatomy(resample(src[0], src[1], src[2], gx, gz), pix, fine_mm, coarse_mm)
        for d in dsts:
            D = anatomy(resample(d[0], d[1], d[2], gx, gz), pix, fine_mm, coarse_mm)
            t = register_rigid(D, S, pix, angles=angles, max_shift_mm=max_shift_mm, angle_tol=angle_tol)
            t.cx, t.cz = float(c[0]), float(c[1])
            ens.append(t)
    ok = [t for t in ens if np.isfinite(t.corr)]
    if not ok:
        return TransferResult(points=pts.copy(), transform=Transform(), spread_mm=float("nan"), ensemble=ens)
    mapped = np.stack([t.apply(pts) for t in ok])            # (n_ens, k, 2)
    # median transform: median of the parameters (common centre = line box centre)
    c = (lo + hi) / 2
    med = Transform(dx=float(np.median([t.apply(c[None])[0, 0] - c[0] for t in ok])),
                    dz=float(np.median([t.apply(c[None])[0, 1] - c[1] for t in ok])),
                    angle=float(np.median([t.angle for t in ok])), cx=float(c[0]), cz=float(c[1]),
                    corr=float(np.median([t.corr for t in ok])), corr0=float(np.median([t.corr0 for t in ok])))
    out = med.apply(pts)
    dev = np.median(np.sqrt(((mapped - out[None]) ** 2).sum(-1)), axis=1)      # per member
    rms = float(np.median(np.sqrt(((mapped - out[None]) ** 2).sum(-1).mean(0))))
    known = float("nan")
    if check:
        errs = []
        for sdx, sdz in check_shifts:
            t = Transform(dx=sdx, dz=sdz)
            rk = transfer_line(t.apply(pts), (warp(src[0], src[1], src[2], t), src[1], src[2]), dst,
                               check=False, **kw)
            errs.append(float(np.median(np.hypot(*(rk.points - out).T))))
        known = max(errs)
    return TransferResult(points=out, transform=med, spread_mm=float(np.median(dev)),
                          agree=float(np.mean(dev <= 1.0)), spread_rms_mm=rms, known_err_mm=known,
                          ensemble=ens)
