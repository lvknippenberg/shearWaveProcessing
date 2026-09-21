"""Synthetic wave fields with known ground truth, for validating the field estimator.

Stage 1 of ``docs/field_estimator_plan.md``: if the estimator cannot recover a wave it was
*given*, it will not recover one from real data. Everything here is sampled on the real
acquisition grid (dz = dx = 0.39 mm, dt = 1.08 ms) and passed through the same processing the
passive pipeline applies, so a pass here means something.

Geometry convention matches the beamformed field ``(n_t, n_z, n_x)``: ``x`` is lateral, ``z`` is
depth, ``theta = atan2(kz, kx)`` so theta = 0 is a wave travelling in +x (along the wall) and
theta = 90 deg one travelling into depth.
"""
from __future__ import annotations

import numpy as np

from .structure import Grid

__all__ = ["default_grid", "coords", "plane_wave", "curved_wave", "dispersive_packet",
           "two_waves", "add_noise", "add_bulk_motion", "apply_pipeline_smoothing",
           "speckle_amplitude", "apply_speckle", "axial_clutter"]

# The real passive acquisition: S5-1 buffer 4.
DZ = DX = 0.3945e-3        # m
DT = 1.0 / 925.9259        # s  (PRF 926 Hz)


def default_grid():
    return Grid(dz=DZ, dx=DX, dt=DT)


def coords(grid, wall_mm=12.0, length_mm=40.0, duration_ms=110.0):
    """(t [s], z [m], x [m]) axes for a septal-wall-sized ROI."""
    nz = max(4, int(round(wall_mm * 1e-3 / grid.dz)))
    nx = max(4, int(round(length_mm * 1e-3 / grid.dx)))
    nt = max(8, int(round(duration_ms * 1e-3 / grid.dt)))
    return (np.arange(nt) * grid.dt, np.arange(nz) * grid.dz, np.arange(nx) * grid.dx)


def _packet(phase_s, f_hz, n_cycles=2.5):
    """A Gaussian-windowed sinusoid: one valve-closure burst, not an endless sine."""
    sigma = n_cycles / (2.0 * np.pi * f_hz)
    return np.exp(-0.5 * (phase_s / sigma) ** 2) * np.sin(2.0 * np.pi * f_hz * phase_s)


def _mesh(t, z, x):
    return (t[:, None, None], z[None, :, None], x[None, None, :])


def plane_wave(grid, c=3.0, theta_deg=0.0, f_hz=16.0, amplitude=200e-6,
               t0_ms=45.0, n_cycles=2.5, axes=None):
    """A plane packet crossing the ROI at known speed and direction.

    ``c`` [m/s], ``theta_deg`` measured from +x towards +z. ``t0_ms`` is the arrival time at the
    ROI centre, so the packet sits inside the window whatever the direction.
    """
    t, z, x = axes if axes is not None else coords(grid)
    T, Z, X = _mesh(t, z, x)
    th = np.radians(theta_deg)
    kx, kz = np.cos(th), np.sin(th)
    # distance along the propagation direction, measured from the ROI centre
    d = (X - x.mean()) * kx + (Z - z.mean()) * kz
    phase = (T - t0_ms * 1e-3) - d / c
    return amplitude * _packet(phase, f_hz, n_cycles), (t, z, x)


def curved_wave(grid, c=3.0, source_mm=(-25.0, 6.0), f_hz=16.0, amplitude=200e-6,
                t0_ms=45.0, n_cycles=2.5, axes=None):
    """A packet radiating from a point source outside the ROI: the wavefront is curved.

    This is the realistic case - a valve plane is a source at a finite distance, so the direction
    varies across the ROI. The estimator should track that variation rather than average it away.
    """
    t, z, x = axes if axes is not None else coords(grid)
    T, Z, X = _mesh(t, z, x)
    sx, sz = source_mm[0] * 1e-3, source_mm[1] * 1e-3
    r = np.hypot(X - sx, Z - sz)
    phase = (T - t0_ms * 1e-3) - (r - np.hypot(x.mean() - sx, z.mean() - sz)) / c
    return amplitude * _packet(phase, f_hz, n_cycles), (t, z, x)


def dispersive_packet(grid, c_ref=3.0, f_ref=16.0, power=0.5, theta_deg=0.0,
                      amplitude=200e-6, t0_ms=45.0, f_lo=6.0, f_hi=36.0, n_f=25, axes=None):
    """A packet whose phase speed varies with frequency, ``c(f) = c_ref (f/f_ref)**power``.

    ``power = 0.5`` is the square-root dispersion of a flexural (Lamb A0) plate mode, which is
    what the measured 1.4 -> 9.3 m/s rise between 7 and 28 Hz on a real AVC window looks like.
    There is no single true speed here; the estimator should land near the phase speed at the
    band's dominant frequency, and that is what the validation checks it against.
    """
    t, z, x = axes if axes is not None else coords(grid)
    T, Z, X = _mesh(t, z, x)
    th = np.radians(theta_deg)
    d = (X - x.mean()) * np.cos(th) + (Z - z.mean()) * np.sin(th)
    freqs = np.linspace(f_lo, f_hi, n_f)
    env = np.exp(-0.5 * ((freqs - f_ref) / (0.35 * f_ref)) ** 2)      # spectral envelope
    out = np.zeros(T.shape + np.broadcast(Z, X).shape[1:] if False else
                   np.broadcast(T, Z, X).shape)
    for f, w in zip(freqs, env):
        c_f = c_ref * (f / f_ref) ** power
        out += w * np.sin(2.0 * np.pi * f * ((T - t0_ms * 1e-3) - d / c_f))
    out *= np.exp(-0.5 * ((T - t0_ms * 1e-3) / (2.5 / (2 * np.pi * f_ref))) ** 2)
    out /= np.abs(out).max() + 1e-30
    # the speed the estimator should recover: the phase speed at the dominant frequency
    return amplitude * out, (t, z, x), c_ref


def two_waves(grid, c=(3.0, 5.0), theta_deg=(0.0, 70.0), f_hz=(16.0, 16.0),
              amplitude=(200e-6, 200e-6), t0_ms=(45.0, 50.0), axes=None):
    """Two plane packets crossing at once - the case where coherence *must* drop.

    There is no single correct speed or direction here. This is the negative control: if the
    coherence gate cannot tell this apart from a single clean wave, it is not a quality gate.
    """
    t, z, x = axes if axes is not None else coords(grid)
    a, _ = plane_wave(grid, c[0], theta_deg[0], f_hz[0], amplitude[0], t0_ms[0], axes=(t, z, x))
    b, _ = plane_wave(grid, c[1], theta_deg[1], f_hz[1], amplitude[1], t0_ms[1], axes=(t, z, x))
    return a + b, (t, z, x)


def add_noise(field, snr_db, rng=None):
    """White Gaussian noise at a given SNR relative to the field's RMS."""
    rng = np.random.default_rng(rng)
    sig = float(np.sqrt((np.asarray(field) ** 2).mean()))
    n = sig / (10.0 ** (snr_db / 20.0))
    return field + rng.normal(0.0, n, size=np.shape(field))


def add_bulk_motion(field, axes, amplitude_ratio=1.0, f_hz=3.0, tilt_per_mm=0.0):
    """Add low-frequency, near-uniform motion - the real contaminant of these panels.

    ``tilt_per_mm`` adds a small spatial gradient so the bulk term is not perfectly uniform,
    which is the harder and more realistic case.
    """
    t, z, x = axes
    T, Z, X = _mesh(t, z, x)
    sig = float(np.sqrt((np.asarray(field) ** 2).mean()))
    ramp = 1.0 + tilt_per_mm * ((X - x.mean()) * 1e3)
    return field + amplitude_ratio * sig * np.sqrt(2.0) * ramp * np.sin(2 * np.pi * f_hz * T)


def apply_pipeline_smoothing(field, grid, sigma_z_mm=0.6, sigma_x_mm=1.2, temporal_window=3,
                             band_hz=None):
    """The same smoothing the passive views apply, so the test is like-for-like."""
    from scipy.ndimage import gaussian_filter, uniform_filter1d

    out = np.asarray(field, float)
    if band_hz is not None:
        n = out.shape[0]
        F = np.fft.rfft(out, axis=0)
        f = np.fft.rfftfreq(n, grid.dt)
        F[(f < band_hz[0]) | (f > band_hz[1])] = 0
        out = np.fft.irfft(F, n=n, axis=0)
    out = gaussian_filter(out, sigma=(0.0, sigma_z_mm * 1e-3 / grid.dz,
                                      sigma_x_mm * 1e-3 / grid.dx), mode="nearest")
    if temporal_window and temporal_window > 1:
        out = uniform_filter1d(out, int(temporal_window), axis=0, mode="nearest")
    return out


def speckle_amplitude(grid, axes, corr_z_mm=0.6, corr_x_mm=1.2, contrast=1.0, rng=0):
    """A static, spatially correlated amplitude field - the confound that broke Stage 3.

    Real axial displacement fields are not a clean scalar wave. Loupas returns only the AXIAL
    component, estimated from the phase of a speckle pattern, so the recovered field carries
    strong static spatial structure at the speckle correlation length (~0.5-1 mm axially at
    5 MHz). That structure dominates the spatial gradient, which is precisely the quantity the
    structure tensor assumes is dominated by the wave - so the tensor locked onto it and reported
    near-axial propagation at 0.2-0.7 m/s on real data.

    Returned as a multiplicative amplitude map in roughly ``[1 - contrast, 1 + contrast]``, to be
    applied to a wave field. A phase-based estimator should be insensitive to it; an
    amplitude-gradient-based one should not. That difference is the whole point of the test.
    """
    from scipy.ndimage import gaussian_filter

    _, z, x = axes
    r = np.random.default_rng(rng)
    a = r.normal(size=(z.size, x.size))
    a = gaussian_filter(a, sigma=(max(corr_z_mm * 1e-3 / grid.dz, 0.5),
                                  max(corr_x_mm * 1e-3 / grid.dx, 0.5)), mode="wrap")
    a /= a.std() + 1e-30
    return 1.0 + contrast * np.clip(a, -2.5, 2.5) / 2.5


def apply_speckle(field, amp):
    """Multiply a wave field by a static amplitude map ``(n_z, n_x)``."""
    return np.asarray(field, float) * np.asarray(amp, float)[None, :, :]


def axial_clutter(field, axes, grid, ratio=0.5, corr_z_mm=0.4, f_hz=8.0, rng=1):
    """Static-in-space, slowly varying axial banding - reverberation-like contamination.

    Distinct from speckle: this one *moves in time*, so it survives a temporal band-pass and adds
    a near-vertical structure in depth that a spatial-gradient method will happily interpret as a
    very slow wave travelling into depth. It reproduces the +/-90 degree direction pile-up seen on
    real data.
    """
    from scipy.ndimage import gaussian_filter1d

    t, z, x = axes
    r = np.random.default_rng(rng)
    band = r.normal(size=(z.size, 1))
    band = gaussian_filter1d(band, sigma=max(corr_z_mm * 1e-3 / grid.dz, 0.5), axis=0,
                             mode="nearest")
    band /= band.std() + 1e-30
    sig = float(np.sqrt((np.asarray(field) ** 2).mean()))
    wave = np.sin(2 * np.pi * f_hz * t)[:, None, None]
    return field + ratio * sig * band[None, :, :] * wave
