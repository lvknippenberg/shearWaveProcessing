"""Anatomical M-line: the expected shear-wave propagation path.

An M-line is a poly-line in the (x, z) imaging plane, resampled to ``n_samples`` points
with an arc-length coordinate ``r``.  The wavefield is sampled along it (and, optionally,
along parallel copies offset along the normal, then averaged) to build the space-time plot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MLine:
    x: np.ndarray      # (n_samples,) lateral position [m]  (the resampled spline)
    z: np.ndarray      # (n_samples,) axial position [m]
    r: np.ndarray      # (n_samples,) arc length from first point [m]
    nx_hat: np.ndarray = None  # (n_samples, 2) unit normal (x, z), for offset averaging
    points: np.ndarray = None  # (k, 2) original clicked anchors (x, z) [m], for display

    @property
    def n_samples(self) -> int:
        return self.x.size

    def offset_lines(self, n_offsets: int, offset_step_m: float):
        """The parallel offset M-lines used for averaging: list of (x, z) arrays (metres),
        each shifted along the local normal by a multiple of ``offset_step_m`` (centred on 0).
        Mirrors :func:`sample_along_mline`."""
        if n_offsets <= 1 or offset_step_m <= 0 or self.nx_hat is None:
            return [(self.x, self.z)]
        k = (n_offsets - 1) / 2.0
        offs = (np.arange(n_offsets) - k) * offset_step_m
        return [(self.x + o * self.nx_hat[:, 0], self.z + o * self.nx_hat[:, 1]) for o in offs]


def _resample_spline(points_xz: np.ndarray, n_samples: int):
    """Fit a spline **through** the clicked (x,z) anchors and resample it at ``n_samples`` points of
    **constant arc-length spacing**.

    Matches the interactive selector (``swp.mline.select.fit_spline``): a cubic (or lower-degree for
    few points) interpolating spline, ``s=0`` so it passes through every anchor. The spline is first
    sampled densely in its parameter ``u``; the cumulative arc length of that dense curve is then
    inverted so the returned points are equally spaced in *distance along the curve* (not in ``u``),
    which is what the downstream space-time sampling and speed fits assume. Falls back to a
    piecewise-linear resample if the spline fit fails (e.g. duplicate/collinear anchors).
    """
    p = np.asarray(points_xz, dtype=np.float64)
    if p.shape[0] < 2:
        raise ValueError("need >= 2 anchor points for an M-line")

    def _linear():
        seg = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
        r_knot = np.concatenate([[0.0], np.cumsum(seg)])
        r = np.linspace(0.0, r_knot[-1], n_samples)
        return np.interp(r, r_knot, p[:, 0]), np.interp(r, r_knot, p[:, 1]), r

    if p.shape[0] == 2:
        return _linear()                                   # a spline through 2 points is the line
    try:
        from scipy.interpolate import splprep, splev
        k = min(3, p.shape[0] - 1)
        tck, _ = splprep([p[:, 0], p[:, 1]], k=k, s=0.0)
        u_dense = np.linspace(0.0, 1.0, max(2000, 8 * n_samples))
        xd, zd = splev(u_dense, tck)
        xd, zd = np.asarray(xd), np.asarray(zd)
        s_dense = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xd), np.diff(zd)))])
        r = np.linspace(0.0, s_dense[-1], n_samples)       # equal arc-length targets
        u_at_r = np.interp(r, s_dense, u_dense)            # invert arc length -> parameter u
        x, z = splev(u_at_r, tck)
        return np.asarray(x), np.asarray(z), r
    except Exception:
        return _linear()


def _unit_normals(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-sample in-plane unit normal to the tangent (rotate tangent by 90 deg)."""
    tx = np.gradient(x)
    tz = np.gradient(z)
    tnorm = np.hypot(tx, tz) + 1e-12
    tx, tz = tx / tnorm, tz / tnorm
    # normal = (-tz, tx)
    return np.stack([-tz, tx], axis=1)


def mline_from_points(points_xz: np.ndarray, n_samples: int = 250) -> MLine:
    """Build an M-line from manual anchor points (as stored in the ``.npz`` files).

    Uses a constant-arc-length **spline** through the anchors (see :func:`_resample_spline`), so the
    processed line is the same smooth curve the picker drew - and the samples are equally spaced in
    distance along it, as the space-time / speed steps assume.
    """
    x, z, r = _resample_spline(points_xz, n_samples)
    return MLine(x=x, z=z, r=r, nx_hat=_unit_normals(x, z),
                 points=np.asarray(points_xz, dtype=np.float64))


def horizontal_mline(x_axis: np.ndarray, depth_m: float, n_samples: int = 250) -> MLine:
    """Horizontal M-line at a fixed depth spanning the lateral field (phantom default)."""
    x = np.linspace(x_axis[0], x_axis[-1], n_samples)
    z = np.full(n_samples, float(depth_m))
    r = x - x[0]
    nx_hat = np.tile([0.0, 1.0], (n_samples, 1))  # normal is axial
    return MLine(x=x, z=z, r=r, nx_hat=nx_hat)


def _bilinear_sample(frame: np.ndarray, z_axis: np.ndarray, x_axis: np.ndarray,
                     zq: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Bilinear sample a single (nz, nx) frame at physical query points."""
    iz = np.interp(zq, z_axis, np.arange(z_axis.size))
    ix = np.interp(xq, x_axis, np.arange(x_axis.size))
    from scipy.ndimage import map_coordinates
    return map_coordinates(frame, np.vstack([iz, ix]), order=1, mode="nearest")


def sample_along_mline(
    field: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    mline: MLine,
    n_offsets: int = 1,
    offset_step_m: float = 0.0,
    agg: str = "mean",
) -> np.ndarray:
    """Sample a (n_frames, nz, nx) field along the M-line.

    If ``n_offsets > 1``, also sample ``n_offsets`` parallel copies shifted by multiples of
    ``offset_step_m`` along the local normal (symmetric about the line) and combine them with
    ``agg`` (``"mean"`` or ``"median"``) -- a cheap M-line-tailored spatial denoiser (median is
    more robust to per-line speckle outliers).

    Returns ``(n_frames, n_samples)``.  Vectorised: a single ``map_coordinates`` call over the
    whole (frame, z, x) volume samples every offset copy and frame at once (fast enough for large
    parameter searches); the frame axis uses integer coordinates so no inter-frame interpolation.
    """
    from scipy.ndimage import map_coordinates
    n_frames = field.shape[0]
    ns = mline.n_samples

    if n_offsets <= 1 or offset_step_m <= 0:
        offsets = np.array([0.0])
    else:
        k = (n_offsets - 1) / 2.0
        offsets = (np.arange(n_offsets) - k) * offset_step_m

    zq = np.concatenate([mline.z + off * mline.nx_hat[:, 1] for off in offsets])
    xq = np.concatenate([mline.x + off * mline.nx_hat[:, 0] for off in offsets])
    iz = np.interp(zq, z_axis, np.arange(z_axis.size))     # fractional axial index
    ix = np.interp(xq, x_axis, np.arange(x_axis.size))     # fractional lateral index

    p = iz.size                                            # n_offsets * ns
    fcoord = np.repeat(np.arange(n_frames), p)
    coords = np.vstack([fcoord, np.tile(iz, n_frames), np.tile(ix, n_frames)])
    vals = map_coordinates(field, coords, order=1, mode="nearest")
    stack = vals.reshape(n_frames, offsets.size, ns)
    return np.median(stack, axis=1) if agg == "median" else stack.mean(axis=1)
