"""M4 directional (k, omega) filtering: separate outward-/inward- and left/right-travelling
waves to suppress reflections and the counter-propagating wavefront.

See docs/literature_review.md sec. 4.1.  Operates on a 2-D (time, position) image; the sign
convention that selects +position-travelling waves is verified empirically on phantom data.
"""
from __future__ import annotations

import numpy as np

from .context import FilterCtx

# Half-plane of the 2-D FFT (over axes t, r) that retains +r-travelling waves.
# With numpy's exp(-2 pi i(...)) convention a wave g(r - c t), c>0, occupies ft*fr < 0.
_POS_TRAVEL_PRODUCT_SIGN = -1.0


def directional_spacetime(data: np.ndarray, keep: str = "pos") -> np.ndarray:
    """Keep +position ("pos") or -position ("neg") travelling waves in a (T, R) image."""
    if keep not in ("pos", "neg", "all"):
        raise ValueError(keep)
    T, R = data.shape
    ft = np.fft.fftfreq(T)[:, None]
    fr = np.fft.fftfreq(R)[None, :]
    prod = ft * fr
    F = np.fft.fft2(data)
    if keep == "pos":
        mask = (prod * _POS_TRAVEL_PRODUCT_SIGN) > 0
    elif keep == "neg":
        mask = (prod * _POS_TRAVEL_PRODUCT_SIGN) < 0
    else:
        return data
    mask = mask | (prod == 0)   # keep DC / axis lines to avoid ringing
    return np.real(np.fft.ifft2(F * mask))


def outward_spacetime(data: np.ndarray, r: np.ndarray, r0: float) -> np.ndarray:
    """Keep only waves travelling *away* from the origin ``r0`` along the M-line.

    For r >= r0 keep +r-travelling waves; for r < r0 keep -r-travelling waves.  Both halves
    are filtered over the full image (avoiding sub-image edge effects) then stitched.
    """
    pos = directional_spacetime(data, "pos")
    neg = directional_spacetime(data, "neg")
    out = np.where((r >= r0)[None, :], pos, neg)
    return out


def directional_field(field: np.ndarray, ctx: FilterCtx = None, mode: str = "outward",
                      keep: str = "pos") -> np.ndarray:
    """Apply directional filtering along (time, lateral-x) at each depth of a (F, z, x) field.

    mode="outward" splits at ``ctx.focus_ix`` (push column) and keeps waves moving away from
    it; mode="pos"/"neg" keeps a single global direction.
    """
    F, nz, nx = field.shape
    out = np.empty_like(field)
    if mode == "outward":
        if ctx is None or ctx.focus_ix is None:
            raise ValueError("outward mode needs ctx.focus_ix")
        r = np.arange(nx, dtype=float)
        r0 = float(ctx.focus_ix)
        for iz in range(nz):
            out[:, iz, :] = outward_spacetime(field[:, iz, :], r, r0)
    else:
        for iz in range(nz):
            out[:, iz, :] = directional_spacetime(field[:, iz, :], keep)
    return out
