"""Per-pixel, per-transmit inclusion windows for the widebeam buffers.

The widebeam buffers (1 and 5; 6 is not converted in this study) fire 21 beams 4 deg apart
from a **virtual source 123 mm behind the array**, so one beam opens only ~9.3-11.6 deg: at
100 mm depth about **5 of the 21 transmits** actually insonify a given pixel. The beamformer
nevertheless summed all 21 into every pixel, and in vivo the 16 that did not reach it
contribute as much amplitude as the 5 that did (``docs/widebeam_bmode_reconstruction.md``) -
badly-focused, off-axis, reverberation-contaminated energy that lands in the echo-free
regions.

This module builds the weight map that restricts each transmit to the pixels inside its own
transmit cone, for zea's ``Beamform(enable_aligned_apodization=True)``. The cone is computed
from the geometry actually stored with the data (``transmit_origins``, ``polar_angles``,
``focus_distances``, ``tx_apodizations``, ``probe_geometry``) - nothing is assumed about the
sequence, so it is equally correct for any widebeam or diverging-wave buffer.

Not for buffers 2 and 4: they feed the displacement estimators, whose phase must not be
touched.
"""

from __future__ import annotations

import numpy as np

# Window over the normalised off-axis angle u = phi / (scale * phi_max).
WINDOWS = {
    "rect": lambda u: (np.abs(u) <= 1.0).astype(np.float32),
    "hann": lambda u: np.where(np.abs(u) <= 1.0,
                               0.5 * (1.0 + np.cos(np.pi * np.clip(u, -1, 1))), 0.0
                               ).astype(np.float32),
    "tukey": lambda u: np.where(
        np.abs(u) <= 1.0,
        0.5 * (1.0 + np.cos(np.pi * np.clip((np.abs(u) - 0.5) / 0.5, 0.0, 1.0))), 0.0
    ).astype(np.float32),
}

# Default cone width, as a multiple of the geometric half-opening angle.
#
# **1.0 is the adopted value** (``BufferSpec.tx_window`` for buffers 1 and 5, 2026-09-21), and
# this constant matches it so the script and the pipeline agree: +4.5 to +5.9 dB dynamic range
# and +1.1 dB on the specular-arc contrast metric that tracks false lateral structure, for
# +0.49% lateral -6 dB width, 95% CI [+0.19, +1.26] over 66 wire targets paired per target.
# That cost is real but an order of magnitude below the ~5% charged by every rejected
# alternative (nearest-k, receive taper, pfield).
#
# 1.5 is the conservative alternative: +3.3 to +4.5 dB for a lateral change of -0.01%,
# CI [-0.16, +0.15] - provably nothing. It was the first recommendation; the width came down
# to 1.0 once clutter, not dynamic range, turned out to be what the buffer was losing.
#
# Below ~0.75 the cone starts cutting into the beam and resolution goes with it; above ~3 the
# gain is given back.
DEFAULT_SCALE = 1.0
DEFAULT_WINDOW = "rect"


def transmit_angles(params, grid):
    """Signed off-axis angle of every pixel about each transmit's virtual source.

    Args:
        params: zea ``Parameters`` for the buffer (full transmit set).
        grid (np.ndarray): ``(nz, nx, 3)`` pixel coordinates in metres.

    Returns:
        tuple: ``(phi, phi_max)`` - ``phi`` of shape ``(n_tx, nz, nx)`` in radians, and the
        geometric half-opening angle of each beam, shape ``(n_tx,)``.
    """
    d = dict(params)
    theta = np.asarray(d["polar_angles"], np.float64)
    origin = np.asarray(d["transmit_origins"], np.float64)
    focus = np.asarray(d["focus_distances"], np.float64)
    apod = np.asarray(d["tx_apodizations"], np.float64)
    probe = np.asarray(d["probe_geometry"], np.float64)

    direction = np.stack([np.sin(theta), np.cos(theta)], axis=1)          # (n_tx, 2)
    # Virtual source: origin + focus * beam direction (focus < 0 => behind the array).
    vs = origin[:, [0, 2]] + focus[:, None] * direction                   # (n_tx, 2)
    base = np.arctan2(direction[:, 0], direction[:, 1])

    px = np.asarray(grid[..., 0], np.float64)
    pz = np.asarray(grid[..., -1], np.float64)
    phi = np.empty((theta.size,) + px.shape, np.float32)
    phi_max = np.zeros(theta.size)
    for i in range(theta.size):
        ang = np.arctan2(px - vs[i, 0], pz - vs[i, 1]) - base[i]
        phi[i] = np.arctan2(np.sin(ang), np.cos(ang))
        # Half-opening: the widest angle any ACTIVE element subtends at the virtual source.
        sel = apod[i] > 0
        ea = np.arctan2(probe[sel, 0] - vs[i, 0], probe[sel, 2] - vs[i, 1]) - base[i]
        phi_max[i] = np.abs(np.arctan2(np.sin(ea), np.cos(ea))).max()
    return phi, phi_max


def transmit_window(params, grid, window=DEFAULT_WINDOW, scale=DEFAULT_SCALE):
    """Weight map ``(n_tx, nz, nx)``: which pixels each transmit contributes to."""
    if window not in WINDOWS:
        raise ValueError(f"unknown window {window!r}; choose from {sorted(WINDOWS)}")
    phi, phi_max = transmit_angles(params, grid)
    return WINDOWS[window](phi / (phi_max[:, None, None] * float(scale)))


def flat_window(params, grid, window=DEFAULT_WINDOW, scale=DEFAULT_SCALE):
    """The same map flattened to ``(n_pix, n_tx)``, ready for zea.

    ``Parameters.flat_aligned_apodization`` is a computed property (it only exists for
    scanline imaging), so it cannot be assigned: put this into the dict returned by
    ``pipeline.prepare_parameters(params)`` instead, under that key, and build the pipeline
    with ``Beamform(enable_aligned_apodization=True)``.
    """
    w = transmit_window(params, grid, window, scale)
    n_tx, nz, nx = w.shape
    return np.moveaxis(w, 0, -1).reshape(nz * nx, n_tx).astype(np.float32)


def coverage(w):
    """Effective number of transmits contributing per pixel, ``(sum w)^2 / sum w^2``."""
    s1 = w.sum(axis=0)
    s2 = (w ** 2).sum(axis=0)
    return np.where(s2 > 0, s1 ** 2 / np.maximum(s2, 1e-20), 0.0)
