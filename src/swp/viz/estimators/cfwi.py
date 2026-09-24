"""Clutter filter wave imaging (CFWI) - Salles et al., IEEE UFFC 2019; Espeland et al., JACC CI 2024.

Not a displacement estimator. A mechanical wave passing through the wall briefly raises the local
tissue velocity; a slow-time high-pass ("clutter") filter on the IQ whose cutoff sits just below
that velocity lets the moving tissue through and suppresses the slower background, so the
*envelope* of the filtered IQ lights up as the wave passes. Salles et al. report ~30 % better
spatio-temporal resolution and ~40 % better consistency than tissue Doppler; Espeland et al. use
it in PLAX/apical views for aortic-valve-closure and atrial-kick waves with a 2 cm/s cutoff
(91 Hz at their settings) followed by a temporal derivative.

Steps (Salles Fig. 1): (1) high-pass the IQ in slow time, (2) envelope, (3) spatio-temporal
smoothing, (4) temporal derivative. Steps 1-2 and 4 happen here; step 3 is the pipeline's usual
spatial / temporal filters, which run on whichever quantity is selected.

To fit the pipeline it returns a :class:`DisplacementResult` whose fields are **not** tissue
motion: ``displacement`` = filtered envelope (arbitrary units), ``velocity`` = its temporal
derivative (the CFWI map Espeland draws slopes on). Select ``quantity: velocity`` for CFWI.

The cutoff is set as a velocity: ``f_c = 2 v_c f_demod / c`` (Doppler). At our buffer-4 settings
(3.9 MHz demodulation, ~926 Hz frame rate) 2 cm/s is 101 Hz against a 463 Hz Nyquist, the
velocity aliasing limit being 9.1 cm/s.
"""
from __future__ import annotations

import numpy as np

from .common import DisplacementResult


def cfwi_envelope(iq: np.ndarray, dz: float, dx: float, c: float, f_demod: float, prf: float,
                  cutoff_velocity: float = 0.02, order: int = 2, log: bool = False,
                  mode: str = "frame_to_frame", reference=None, **_ignored) -> DisplacementResult:
    """CFWI on an (n_frames, nz, nx) complex IQ ensemble. ``mode``/``reference`` are accepted for
    registry compatibility and ignored."""
    from scipy.signal import butter, filtfilt
    iq = np.asarray(iq)
    fc = 2.0 * cutoff_velocity * f_demod / c
    wn = float(np.clip(fc / (0.5 * prf), 1e-3, 0.99))
    b, a = butter(order, wn, btype="highpass")
    if iq.shape[0] > 3 * max(len(a), len(b)):
        hp = filtfilt(b, a, iq.real, axis=0) + 1j * filtfilt(b, a, iq.imag, axis=0)
    else:
        hp = iq - iq.mean(axis=0, keepdims=True)
    env = np.abs(hp)
    if log:
        env = 20.0 * np.log10(env + 1e-12)
    dt = 1.0 / prf
    deriv = np.diff(env, axis=0) / dt
    n, nz, nx = env.shape
    return DisplacementResult(displacement=env, velocity=deriv,
                              fc=np.full((nz, nx), float(f_demod)), mode="cfwi", dt=dt,
                              estimator="cfwi")
