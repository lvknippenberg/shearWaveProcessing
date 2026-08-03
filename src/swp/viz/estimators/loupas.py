"""Loupas 2-D autocorrelator for axial tissue displacement/velocity.

Loupas et al., IEEE UFFC 1995 -- evaluates the full Doppler equation with a
2-D autocorrelation: slow-time lag-1 phase for motion, fast-time lag-1 phase for a
local center-frequency correction.  Setting ``local_frequency=False`` reduces this to
a Kasai-style estimator using the fixed demodulation frequency.

See docs/literature_review.md sec. 2.1-2.2.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .common import (
    DisplacementResult,
    complex_uniform_filter1d,
    local_center_frequency,
    phase_to_displacement,
    samples_for_length,
)


def loupas_displacement(
    iq: np.ndarray,
    dz: float,
    dx: float,
    c: float,
    f_demod: float,
    prf: float,
    kernel_z_m: float = 1.0e-3,
    kernel_x_m: float = 0.0,
    local_frequency: bool = True,
    mode: str = "frame_to_frame",
    reference: Optional[np.ndarray] = None,
) -> DisplacementResult:
    """Estimate axial displacement/velocity from a complex IQ ensemble.

    Parameters
    ----------
    iq : (n_frames, nz, nx) complex
        Post-push tracking ensemble.
    dz, dx, c, f_demod, prf
        Scan parameters (see :class:`~iq2sws.core.acquisition.Acquisition`).
    kernel_z_m, kernel_x_m : float
        Axial / lateral averaging window in metres (variance vs resolution trade-off).
    local_frequency : bool
        If True, Loupas local center-frequency correction; else fixed ``f_demod`` (Kasai-like).
    mode : {"frame_to_frame", "relative_to_reference"}
        Differential (cumulative velocity) or absolute vs a reference frame/ensemble.
    reference : (nz, nx) or (n_ref, nz, nx) complex, optional
        Reference for ``relative_to_reference`` (ensemble is averaged in complex).

    Returns
    -------
    DisplacementResult
    """
    iq = np.asarray(iq)
    n_frames, nz, nx = iq.shape
    dt = 1.0 / prf

    kz = samples_for_length(kernel_z_m, dz) if kernel_z_m > 0 else 1
    kx = samples_for_length(kernel_x_m, dx) if kernel_x_m > 0 else 1

    if local_frequency:
        fc = local_center_frequency(iq, f_demod, c, dz, kernel_z=kz)
    else:
        fc = np.full((nz, nx), float(f_demod))

    if mode == "frame_to_frame":
        prod = np.conj(iq[:-1]) * iq[1:]                 # (n_frames-1, nz, nx)
    elif mode == "relative_to_reference":
        if reference is None:
            raise ValueError("mode='relative_to_reference' requires `reference`")
        ref = np.asarray(reference)
        if ref.ndim == 3:
            ref = ref.mean(axis=0)                        # complex-average the ensemble
        prod = np.conj(ref)[None, :, :] * iq             # (n_frames, nz, nx)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    prod = complex_uniform_filter1d(prod, kz, axis=1)
    if kx > 1:
        prod = complex_uniform_filter1d(prod, kx, axis=2)

    phase = np.angle(prod)
    disp_step = phase_to_displacement(phase, fc[None, :, :], c)

    if mode == "frame_to_frame":
        velocity = disp_step / dt                        # (n_frames-1, nz, nx) [m/s]
        cum = np.cumsum(disp_step, axis=0)               # displacement vs frame 0
        displacement = np.concatenate([np.zeros((1, nz, nx)), cum], axis=0)
    else:
        displacement = disp_step                         # (n_frames, nz, nx) vs reference
        velocity = np.diff(displacement, axis=0) / dt

    return DisplacementResult(
        displacement=displacement, velocity=velocity, fc=fc, mode=mode, dt=dt
    )
