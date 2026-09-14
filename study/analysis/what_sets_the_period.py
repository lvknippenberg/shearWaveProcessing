"""What actually sets the 1.284 deg period? It is not the transmit lattice.

nearest_k_composite.py found the peak at 1.284 deg for EVERY composite including nearest-1, where
each pixel comes from one transmit and cross-transmit interference cannot exist. ripple_metric_
control.py showed the peak is not a resampling artefact. So the period is set by something that is
not the transmit pitch and not the compounding rule.

Candidates, each with a test that can falsify it:

  A/B  transmit pitch          - decimate the transmits to a 2.222 / 3.333 deg pitch. If the peak
                                 is on the transmit lattice it must move. (S5d argued it cannot be,
                                 on spectral grounds; this tests it directly.)
  C    mosaic strip width      - nearest-1 with decimated transmits, i.e. wider strips.
  D    single transmit         - the angular structure inside ONE transmit's own region, with no
                                 compounding of any kind.
  E    the phantom's wires     - a resolution phantom has wires on a REGULAR lattice, which in
                                 polar coordinates produces its own angular periodicity. Mask the
                                 detected targets out and see whether the peak survives. This one
                                 is phantom-specific, which is why the in-vivo cross-check matters.
"""
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phantom_psf as P

LINE_SPACING_DEG = 1.1111
STEER_DEG = np.arange(73) * 1.1111111 - 40.0
APEX_M = -0.0121


def spec(env, co, r_lo=0.045, r_hi=0.085, n_th=1501, th_max=34.0, mask=None):
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo, r_hi, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip(((R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    pol = env[ii, jj].astype(np.float64)
    if mask is not None:                      # drop masked pixels from the radial average
        keep = ~mask[ii, jj]
        pol = np.where(keep, pol, np.nan)
        med = np.nanmedian(pol, axis=1, keepdims=True)
        pol = pol / (med + 1e-12)
        prof = np.nanmean(pol, axis=0)
        prof = np.where(np.isfinite(prof), prof, np.nanmean(prof))
    else:
        pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
        prof = pol.mean(axis=0)
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    s = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    f = np.fft.rfftfreq(rip.size, d=dth)
    sel = (f > 0.3) & (f < 2.0)
    return 1.0 / f[sel][np.argmax(s[sel])], rip.std() * 100


stack = np.load(os.path.join(HERE, "phantom_per_tx.npy"))
env_std, co = P.load_env(P.RECONS[0][0])
x, z = co[..., 0], co[..., -1]
pix_deg = np.degrees(np.arctan2(x, z - APEX_M))

print(f"{'test':52s} {'peak':>8s} {'rms':>8s}")
print("-" * 72)
pk, rms = spec(env_std, co)
print(f"{'reference: stored standard':52s} {pk:7.3f}d {rms:7.1f}%")

print("\n-- A/B: decimate the transmits (change the pitch) --")
for step in (1, 2, 3, 4):
    idx = np.arange(0, 73, step)
    img = np.abs(np.sum(stack[idx], axis=0))
    pk, rms = spec(img, co)
    print(f"{f'coherent, every {step} transmit(s) -> {len(idx)} tx, pitch {1.1111 * step:.3f} deg':52s}"
          f" {pk:7.3f}d {rms:7.1f}%")

print("\n-- C: nearest-1 mosaic, varying strip width --")
nz, nx = pix_deg.shape
ii, jj = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
for step in (1, 2, 3):
    idx = np.arange(0, 73, step)
    near = idx[np.argmin(np.abs(pix_deg[None] - STEER_DEG[idx][:, None, None]), axis=0)]
    img = np.abs(stack[near, ii, jj])
    pk, rms = spec(img, co)
    print(f"{f'nearest-1, strips {1.1111 * step:.3f} deg wide':52s} {pk:7.3f}d {rms:7.1f}%")

print("\n-- D: a single transmit, inside its own region --")
for t in (30, 36, 42):
    img = np.abs(stack[t])
    band = abs(STEER_DEG[t])
    pk, rms = spec(img, co, th_max=min(band + 3.3, 34.0))
    print(f"{f'transmit {t} alone (steer {STEER_DEG[t]:+.2f} deg)':52s} {pk:7.3f}d {rms:7.1f}%")

print("\n-- E: mask out the phantom's wire targets --")
targets = P.find_targets(env_std, *P.axes_mm(co)[::-1][::-1])
tm = np.zeros(env_std.shape, bool)
for iz, ix in targets:
    tm[iz, ix] = True
for grow in (0, 6, 14, 30):
    m = binary_dilation(tm, iterations=grow) if grow else tm
    pk, rms = spec(env_std, co, mask=m)
    print(f"{f'{len(targets)} targets masked, dilated {grow} px ({m.mean():.1%} of grid)':52s}"
          f" {pk:7.3f}d {rms:7.1f}%")
