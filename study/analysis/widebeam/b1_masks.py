"""Per-pixel / per-transmit weight maps (which pixels each transmit contributes to).

Every rule returns ``W`` of shape ``(n_tx, nz, nx)``, float32.  The composited image is

    I(p) = sum_tx W[tx, p] * S[tx, p]   /   N(p)

with the normalisation ``N`` chosen separately (see :func:`compose`), because the
normalisation is a real degree of freedom: an unnormalised sum darkens the sector edges
(fewer transmits cover them), ``sum W`` equalises brightness but amplifies noise where
coverage is thin, and ``sqrt(sum W^2)`` equalises the NOISE level instead.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------------------------------------ windows
def _rect(u):
    return (np.abs(u) <= 1.0).astype(np.float32)


def _hann(u):
    return np.where(np.abs(u) <= 1.0, 0.5 * (1.0 + np.cos(np.pi * np.clip(u, -1, 1))), 0.0
                    ).astype(np.float32)


def _tukey(u, taper=0.5):
    """Flat to |u| = 1-taper, cosine-tapered to 0 at |u| = 1."""
    a = np.abs(u)
    flat = 1.0 - taper
    w = np.ones_like(a, np.float32)
    if taper > 0:
        ramp = np.clip((a - flat) / taper, 0.0, 1.0)
        w = (0.5 * (1.0 + np.cos(np.pi * ramp))).astype(np.float32)
    return np.where(a <= 1.0, w, 0.0).astype(np.float32)


def _gauss(u):
    """Gaussian with sigma = 1 in units of u (no hard cut-off, truncated at 3 sigma)."""
    return np.where(np.abs(u) <= 3.0, np.exp(-0.5 * u ** 2), 0.0).astype(np.float32)


WINDOWS = {"rect": _rect, "hann": _hann, "gauss": _gauss,
           "tukey25": lambda u: _tukey(u, 0.25), "tukey50": lambda u: _tukey(u, 0.50),
           "tukey75": lambda u: _tukey(u, 0.75)}


# -------------------------------------------------------------------- rules
def w_all(geom):
    """Baseline: every transmit contributes to every pixel (the current pipeline)."""
    return np.ones((geom.n_tx,) + geom.shape, np.float32)


def w_cone(geom, window="rect", scale=1.0, apod_thresh=0.0):
    """Angular cone about each beam axis, width = ``scale`` x the geometric half-opening.

    ``apod_thresh=0`` uses the full 80-element aperture cone (+/-4.7 to 5.8 deg);
    ``0.5`` uses the -6 dB effective aperture of the Hann transmit apodization.
    """
    phi = geom.phi
    pm = geom.phi_max(apod_thresh)[:, None, None] * scale
    return WINDOWS[window](phi / pm)


def w_fixed(geom, window="rect", half_deg=6.0):
    """Angular window of a FIXED half-width, independent of the per-transmit geometry."""
    return WINDOWS[window](geom.phi / np.radians(half_deg))


def w_nearest_k(geom, k=3):
    """The k transmits whose beam axis is angularly closest to the pixel."""
    a = np.abs(geom.phi)
    order = np.argsort(a, axis=0)[:k]                       # (k, nz, nx)
    w = np.zeros_like(a, np.float32)
    nz, nx = geom.shape
    ii, jj = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
    for r in range(k):
        w[order[r], ii, jj] = 1.0
    return w


def w_pfield(geom, pfield):
    """Weight by the simulated transmit pressure field (zea ``compute_pfield``)."""
    p = np.asarray(pfield, np.float32)
    return p / (p.max() + 1e-20)


def w_pfield_mask(geom, pfield, db=-6.0):
    """Hard mask at ``db`` below each transmit's own peak of the simulated field."""
    p = np.asarray(pfield, np.float32)
    thr = 10 ** (db / 20.0) * p.reshape(p.shape[0], -1).max(axis=1)[:, None, None]
    return (p >= thr).astype(np.float32)


# --------------------------------------------------------------- composition
def compose(stack, w, norm="none", floor=0.15, chunk=4):
    """Compound the per-transmit stack under weight map ``w``.

    Args:
        stack: ``(n_tx, n_frames, nz, nx)`` complex64 (memmap is fine).
        w: ``(n_tx, nz, nx)`` float32.
        norm: ``"none"`` (plain coherent sum), ``"sum"`` (divide by sum of weights -
            equalises BRIGHTNESS), or ``"rms"`` (divide by sqrt(sum w^2) - equalises the
            NOISE level, so speckle statistics stay comparable across the sector).
        floor: fraction of the map's maximum below which the normaliser is clamped, so
            pixels no transmit reaches are not amplified into noise.

    Returns:
        ``(n_frames, nz, nx)`` complex64.
    """
    n_tx, n_fr, nz, nx = stack.shape
    out = np.zeros((n_fr, nz, nx), np.complex64)
    for s in range(0, n_tx, chunk):
        blk = np.asarray(stack[s:s + chunk])
        out += np.einsum("tzx,tfzx->fzx", w[s:s + chunk], blk)
    if norm == "sum":
        n = w.sum(axis=0)
    elif norm == "rms":
        n = np.sqrt((w ** 2).sum(axis=0))
    else:
        return out
    n = np.maximum(n, floor * n.max())
    return (out / n).astype(np.complex64)


def compose_incoherent(stack, w, chunk=4):
    """Envelope (incoherent) compounding under the same weight map."""
    n_tx, n_fr, nz, nx = stack.shape
    out = np.zeros((n_fr, nz, nx), np.float32)
    for s in range(0, n_tx, chunk):
        blk = np.asarray(stack[s:s + chunk])
        out += np.einsum("tzx,tfzx->fzx", w[s:s + chunk], np.abs(blk))
    return out


def coverage(w):
    """Effective number of transmits contributing per pixel: (sum w)^2 / sum w^2."""
    s1 = w.sum(axis=0)
    s2 = (w ** 2).sum(axis=0)
    return np.where(s2 > 0, s1 ** 2 / np.maximum(s2, 1e-20), 0.0)
