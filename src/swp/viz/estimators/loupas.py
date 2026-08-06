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


def _axial_smooth_complex(prod, kz, kx):
    p = complex_uniform_filter1d(prod, kz, axis=1)
    if kx > 1:
        p = complex_uniform_filter1d(p, kx, axis=2)
    return p


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
    adaptive_kernel: bool = False,
    kernel_z_max_m: float = 2.0e-3,
    mode: str = "frame_to_frame",
    reference: Optional[np.ndarray] = None,
) -> DisplacementResult:
    """Estimate axial displacement/velocity from a complex IQ ensemble.

    Parameters
    ----------
    iq : (n_frames, nz, nx) complex
        Post-push tracking ensemble.
    dz, dx, c, f_demod, prf
        Scan parameters, taken from the acquisition (Verasonics): ``c`` speed of sound,
        ``f_demod`` demodulation/centre frequency (phase->displacement scaling), ``prf`` tracking
        frame rate, ``dz`` axial pixel pitch (set by the RF sampling rate / beamforming grid). These
        are **not** free tuning knobs - they must match the data.
    kernel_z_m, kernel_x_m : float
        Axial / lateral averaging window in metres (variance vs resolution trade-off) - the main
        tuning knobs.
    local_frequency : bool
        If True, Loupas local center-frequency correction; else fixed ``f_demod`` (Kasai-like).
    adaptive_kernel : bool
        If True, size the axial kernel per pixel by the **local RF coherence**: use ``kernel_z_m``
        where the signal is coherent (preserve resolution) and grow toward ``kernel_z_max_m`` where it
        is not (reduce noise). Coherence ``C = |<P>| / <|P|>`` of the slow-time correlation ``P`` is
        computed at the base kernel; the displacement is blended ``C·d(small) + (1-C)·d(large)``.
    kernel_z_max_m : float
        Maximum axial kernel for the adaptive mode (used where coherence is low).
    mode : {"frame_to_frame", "relative_to_reference"}
        Differential (cumulative velocity) or absolute vs a reference frame/ensemble.
    reference : (nz, nx) or (n_ref, nz, nx) complex, optional
        Reference for ``relative_to_reference`` (ensemble is averaged in complex).

    Note on **complex-correlation weighting**: taking the angle of the *spatially-averaged complex*
    product ``<conj(ref)·iq>`` (rather than averaging per-sample phases) already weights each sample
    by its magnitude, so higher-SNR samples dominate the estimate - the standard robust Loupas form.

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

    if adaptive_kernel:
        from scipy.ndimage import uniform_filter1d
        kz_max = samples_for_length(kernel_z_max_m, dz) if kernel_z_max_m > 0 else kz
        kz_max = max(kz_max, kz)
        p_small = _axial_smooth_complex(prod, kz, kx)
        p_large = _axial_smooth_complex(prod, kz_max, kx)
        # local coherence in [0,1] at the base kernel: magnitude of the averaged correlation
        # over the average of the magnitudes (1 = fully coherent, 0 = noise).
        num = np.abs(p_small)
        den = uniform_filter1d(np.abs(prod), kz, axis=1, mode="nearest") + 1e-20
        coh = np.clip(num / den, 0.0, 1.0)
        d_small = phase_to_displacement(np.angle(p_small), fc[None, :, :], c)
        d_large = phase_to_displacement(np.angle(p_large), fc[None, :, :], c)
        disp_step = coh * d_small + (1.0 - coh) * d_large
    else:
        prod = _axial_smooth_complex(prod, kz, kx)
        disp_step = phase_to_displacement(np.angle(prod), fc[None, :, :], c)

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
