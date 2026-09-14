"""Control: does the 1.284 deg "peak" survive when there is no transmit structure at all?

Every composite in nearest_k_composite.py reported a peak at EXACTLY 1.284 deg - including
nearest-1, where each pixel comes from a single transmit and cross-transmit interference cannot
exist by construction. A periodicity that is identical across reconstructions that share no
mechanism is a warning sign about the metric, not a discovery about the images.

The suspect is the polar resampling inside `angular_ripple`: it samples a Cartesian grid at
nearest-neighbour onto a polar (r, theta) grid. That stair-steps, and the stair pattern has its
own angular periodicity set by the grid pitch - nothing to do with the transmits.

Controls, in increasing severity:
  1. synthetic fully-developed speckle with NO transmit structure whatsoever
  2. the real image, resampled with bilinear interpolation instead of nearest-neighbour
  3. the real image, with the angular sampling density changed

If (1) peaks at 1.284 deg, the peak is an artefact of the measurement and every statement in this
investigation that rests on "the image peak is not at the transmit pitch" has to be withdrawn.
"""
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phantom_psf as P

LINE_SPACING_DEG = 1.1111
RNG = np.random.default_rng(0)


def ripple_spec(env, co, order=0, n_th=1501, th_max=34.0, r_lo=0.045, r_hi=0.085, n_r=200):
    """angular_ripple, but with a selectable interpolation order and sampling density."""
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo, r_hi, n_r)
    R, TH = np.meshgrid(r, th, indexing="ij")
    fj = (R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)
    fi = (R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)
    if order == 0:
        pol = env[np.clip(fi.astype(int), 0, nz - 1), np.clip(fj.astype(int), 0, nx - 1)]
    else:
        pol = map_coordinates(env, [fi.ravel(), fj.ravel()], order=order,
                              mode="nearest").reshape(fi.shape)
    pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
    prof = pol.mean(axis=0)
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth)
    sel = (freq > 0.3) & (freq < 2.0)
    tot = spec[(freq > 0.05) & (freq < 5)].sum()
    fl = 1.0 / LINE_SPACING_DEG
    frac = spec[(freq > fl * 0.9) & (freq < fl * 1.1)].sum() / tot
    return 1.0 / freq[sel][np.argmax(spec[sel])], np.sqrt(frac) * rip.std() * 100, rip.std() * 100


env, co = P.load_env(P.RECONS[0][0])
nz, nx = env.shape
sector = env > 0

print("--- control 1: synthetic speckle, NO transmit structure -------------------")
peaks = []
for t in range(8):
    g = (RNG.standard_normal((nz, nx)) + 1j * RNG.standard_normal((nz, nx))).astype(np.complex64)
    # smooth to roughly the measured PSF (1.4 mm lateral, 1.0 mm axial) to make it look like
    # fully developed speckle at the right scale
    g = gaussian_filter(g.real, (2.0, 3.0)) + 1j * gaussian_filter(g.imag, (2.0, 3.0))
    synth = np.abs(g).astype(np.float32) * sector
    pk, amp, rms = ripple_spec(synth, co)
    peaks.append(pk)
    print(f"  realisation {t}: peak {pk:6.3f} deg   ripple@line {amp:5.2f}%   rms {rms:5.1f}%")
print(f"  -> peaks: median {np.median(peaks):.3f} deg, "
      f"{sum(abs(p - 1.284) < 0.02 for p in peaks)}/{len(peaks)} land on 1.284 deg")

print("\n--- control 2: real image, interpolation order --------------------------")
for order, lab in ((0, "nearest-neighbour (as used)"), (1, "bilinear"), (3, "cubic spline")):
    pk, amp, rms = ripple_spec(env, co, order=order)
    print(f"  {lab:28s} peak {pk:6.3f} deg   ripple@line {amp:5.2f}%   rms {rms:5.1f}%")

print("\n--- control 3: real image, angular sampling density ---------------------")
for n_th in (751, 1501, 3001, 6001):
    pk, amp, rms = ripple_spec(env, co, order=1, n_th=n_th)
    print(f"  n_th = {n_th:5d} (bilinear)      peak {pk:6.3f} deg   ripple@line {amp:5.2f}%")

print("\n--- control 4: real image, radial band (bilinear) -----------------------")
for lo, hi in ((0.045, 0.085), (0.035, 0.065), (0.065, 0.105), (0.030, 0.105)):
    pk, amp, rms = ripple_spec(env, co, order=1, r_lo=lo, r_hi=hi)
    print(f"  r = {lo * 1e3:3.0f}-{hi * 1e3:3.0f} mm                peak {pk:6.3f} deg   "
          f"ripple@line {amp:5.2f}%")
