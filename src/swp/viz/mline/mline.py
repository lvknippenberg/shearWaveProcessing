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
    x: np.ndarray      # (n_samples,) lateral position [m]
    z: np.ndarray      # (n_samples,) axial position [m]
    r: np.ndarray      # (n_samples,) arc length from first point [m]
    nx_hat: np.ndarray = None  # (n_samples, 2) unit normal (x, z), for offset averaging

    @property
    def n_samples(self) -> int:
        return self.x.size


def _resample_polyline(points_xz: np.ndarray, n_samples: int):
    """Resample an (k,2) (x,z) poly-line to ``n_samples`` equal-arc-length points."""
    p = np.asarray(points_xz, dtype=np.float64)
    seg = np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))
    r_knot = np.concatenate([[0.0], np.cumsum(seg)])
    r = np.linspace(0.0, r_knot[-1], n_samples)
    x = np.interp(r, r_knot, p[:, 0])
    z = np.interp(r, r_knot, p[:, 1])
    return x, z, r


def _unit_normals(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Per-sample in-plane unit normal to the tangent (rotate tangent by 90 deg)."""
    tx = np.gradient(x)
    tz = np.gradient(z)
    tnorm = np.hypot(tx, tz) + 1e-12
    tx, tz = tx / tnorm, tz / tnorm
    # normal = (-tz, tx)
    return np.stack([-tz, tx], axis=1)


def mline_from_points(points_xz: np.ndarray, n_samples: int = 250) -> MLine:
    """Build an M-line from manual anchor points (as stored in the ``.npz`` files)."""
    x, z, r = _resample_polyline(points_xz, n_samples)
    return MLine(x=x, z=z, r=r, nx_hat=_unit_normals(x, z))


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
) -> np.ndarray:
    """Sample a (n_frames, nz, nx) field along the M-line.

    If ``n_offsets > 1``, also sample ``n_offsets`` parallel copies shifted by multiples of
    ``offset_step_m`` along the local normal (symmetric about the line) and average them --
    a cheap M-line-tailored spatial denoiser.

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
    return vals.reshape(n_frames, offsets.size, ns).mean(axis=1)
