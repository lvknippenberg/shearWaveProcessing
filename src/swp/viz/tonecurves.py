"""Display tone ("gamma"/S-) curves mapping normalised log-compressed value -> displayed grey.

The B-mode pipeline log-compresses the envelope to dB and then maps dB linearly to grey. That
linear step is the default here (``"linear"``); a non-linear tone curve redistributes contrast
without touching the underlying dynamic range - lifting mid-grey tissue, crushing the noise
floor, or both.

.. note::

   The five ``gamma1..gamma5`` curves are **digitised by eye from the Verasonics "Default gamma
   curves" plot**, not read from the acquisition - the ``.mat`` stores no gamma curves, and
   ``Resource.DisplayWindow.Colormap`` is a strictly linear grey ramp (verified: max deviation
   from linear 1e-16). They reproduce the *character* of each curve (toe position, mid-tone
   slope, shoulder) but are not the exact Verasonics coefficients. If the real control points can
   be exported, drop them in here and the shapes become exact.

All curves map [0, 1] -> [0, 1] and are applied **after** the dB -> [0, 1] normalisation.
"""
from __future__ import annotations

import numpy as np

# Control points (input, output) read off the reference plot.
_CURVES = {
    # 1: near-linear, gentle shoulder, tops out ~0.90.
    "gamma1": ([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
               [0.00, 0.09, 0.18, 0.28, 0.38, 0.47, 0.56, 0.65, 0.73, 0.82, 0.90]),
    # 2: short toe, then slightly steeper than linear through the mid-tones.
    "gamma2": ([0.00, 0.07, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 1.00],
               [0.00, 0.00, 0.07, 0.20, 0.34, 0.50, 0.58, 0.66, 0.74, 0.83, 0.97]),
    # 3: longer toe, steep mid-tones, clear shoulder - a classic S.
    "gamma3": ([0.00, 0.08, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00],
               [0.00, 0.00, 0.10, 0.22, 0.36, 0.50, 0.64, 0.78, 0.87, 0.93, 0.97]),
    # 4: toe, fast rise, a plateau around mid-grey, then a second rise - double-S.
    "gamma4": ([0.00, 0.08, 0.20, 0.30, 0.40, 0.45, 0.50, 0.60, 0.70, 0.75, 0.85, 1.00],
               [0.00, 0.00, 0.17, 0.33, 0.44, 0.47, 0.50, 0.62, 0.76, 0.82, 0.88, 0.97]),
    # 5: strongest S - heavy toe, late steep rise, early shoulder.
    "gamma5": ([0.00, 0.08, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00],
               [0.00, 0.00, 0.09, 0.14, 0.21, 0.33, 0.42, 0.52, 0.63, 0.70, 0.80, 0.87, 0.91]),
}

NAMES = ["linear", *sorted(_CURVES)]


def apply_curve(x, name="linear"):
    """Map ``x`` in [0, 1] through a tone curve. ``"linear"`` is the identity (current default)."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    if name in (None, "linear"):
        return x
    if name not in _CURVES:
        raise ValueError(f"unknown curve {name!r}; choose from {NAMES}")
    xi, yi = _CURVES[name]
    return np.interp(x, xi, yi).astype(np.float32)


def curve_table(n=256):
    """``(x, {name: y})`` sampled on a common grid, for plotting the curves."""
    x = np.linspace(0.0, 1.0, n)
    return x, {name: apply_curve(x, name) for name in NAMES}
