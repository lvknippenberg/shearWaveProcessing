"""Geometry helpers: push-focus detection and robust display scaling."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PushFocus:
    x: float   # lateral position [m]
    z: float   # depth [m]
    ix: int    # lateral index
    iz: int    # axial index


def detect_push_focus(
    displacement: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    frames=(1, 2, 3, 4, 5),
    x_halfwidth_m: float = 0.010,
    z_range_m=(0.044, 0.060),
    smooth_sigma: float = 1.5,
) -> PushFocus:
    """Locate the ARF push focus as the peak early axial displacement in a central ROI.

    Field-of-view edges are excluded (they are low-SNR and dominate naive maxima) and the
    early-displacement map is Gaussian-smoothed so a single noisy pixel cannot win.  The
    first tracking frame is typically skipped by the caller (it can phase-wrap).

    displacement : (n_frames, nz, nx) [m], preferably relative-to-reference (drift-free).
    """
    from scipy.ndimage import gaussian_filter
    early = np.abs(displacement[list(frames)]).mean(axis=0)
    if smooth_sigma > 0:
        early = gaussian_filter(early, smooth_sigma)
    xm = np.abs(x) < x_halfwidth_m
    zm = (z >= z_range_m[0]) & (z <= z_range_m[1])
    masked = np.full_like(early, -np.inf)
    zz, xx = np.ix_(zm, xm)
    masked[zz, xx] = early[zz, xx]
    iz, ix = np.unravel_index(np.argmax(masked), masked.shape)
    return PushFocus(x=float(x[ix]), z=float(z[iz]), ix=int(ix), iz=int(iz))


def robust_clim(field: np.ndarray, roi_mask: np.ndarray = None, pct: float = 98.0) -> float:
    """Symmetric color limit from a percentile of |field| (optionally within an ROI).

    ``roi_mask`` (when given) is applied along the last axis if its length matches
    ``field.shape[-1]``; otherwise it is applied to the flattened array.
    """
    a = np.abs(field)
    if roi_mask is not None:
        if roi_mask.ndim == 1 and roi_mask.size == a.shape[-1]:
            a = a[..., roi_mask]
        else:
            a = a[roi_mask]
    return float(np.percentile(a, pct)) + 1e-12
