"""The catalogue of buffer-1 reconstruction rules under test.

Each entry is (label, weight-map builder, normalisation).  The weight map decides WHICH
PIXELS each transmit contributes to; the normalisation decides how the per-pixel transmit
count is compensated.
"""
from __future__ import annotations

import numpy as np

import b1_masks as M


def catalogue(geom, include=None):
    """Return an ordered dict {label: (w, norm)}."""
    C = {}

    def add(label, w, norm="none"):
        if include is None or any(k in label for k in include):
            C[label] = (w, norm)

    # ---- baseline: every transmit into every pixel -------------------------
    add("all-21 (pipeline)", M.w_all(geom))

    # ---- hard geometric cone: only pixels the transmit actually insonified --
    for s in (0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        add(f"cone rect x{s:g}", M.w_cone(geom, "rect", s))
    # ---- same cone, tapered edges -----------------------------------------
    for s in (1.0, 1.5, 2.0, 3.0):
        add(f"cone hann x{s:g}", M.w_cone(geom, "hann", s))
    for s in (1.0, 1.5, 2.0):
        add(f"cone tukey50 x{s:g}", M.w_cone(geom, "tukey50", s))
    for s in (0.5, 1.0, 1.5):
        add(f"cone gauss s{s:g}", M.w_cone(geom, "gauss", s))
    # ---- the -6 dB effective aperture cone --------------------------------
    for s in (1.0, 1.5, 2.0):
        add(f"cone6dB rect x{s:g}", M.w_cone(geom, "rect", s, apod_thresh=0.5))

    # ---- rank-based: the k angularly nearest transmits ---------------------
    for k in (1, 2, 3, 5, 7, 11):
        add(f"nearest-{k}", M.w_nearest_k(geom, k))

    # ---- fixed angular half-width, ignoring the per-transmit geometry ------
    for d in (3.0, 5.0, 8.0, 12.0, 20.0):
        add(f"fixed rect {d:g}deg", M.w_fixed(geom, "rect", d))
    for d in (5.0, 8.0, 12.0):
        add(f"fixed hann {d:g}deg", M.w_fixed(geom, "hann", d))

    return C


NORMALISATIONS = ("none", "sum", "rms")
