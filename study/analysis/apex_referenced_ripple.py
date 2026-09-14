"""The 1.284 deg peak is the 1.111 deg transmit pitch measured about the wrong centre.

Every angular-ripple measurement in this investigation resampled the image about the ARRAY ORIGIN
(z = 0), using theta = arctan2(x, z). But the sector is steered from a VIRTUAL APEX 12.1 mm behind
the array face (`CenterTransmit.mat`: Region.Shape.Position(3) = -24.57 lambda). A structure that
is periodic at 1.1111 deg in apex-angle is NOT periodic at 1.1111 deg in origin-angle.

For a point at distance Ra from the apex, small-angle,

    theta_origin / theta_apex  ~  Ra / (Ra - 12.1 mm)

which over the 45-85 mm depth band is about 1.15-1.19. So a 1.1111 deg pitch is measured at
~1.28 deg. That is the "unexplained" peak, and it also explains why decimating the transmits moved
it proportionally (2.222 -> 2.520, 3.333 -> saturated).

This script measures both ways and checks the prediction.
"""
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phantom_psf as P

APEX_M = -0.0121
LINE_SPACING_DEG = 1.1111


def spec(env, co, apex_m, r_lo, r_hi, n_th=2001, th_max=30.0):
    """Angular ripple about a chosen centre. r_lo/r_hi are DEPTHS in m, converted per centre."""
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    # radius measured from the chosen centre so the band covers the same depths either way
    r = np.linspace(r_lo - apex_m, r_hi - apex_m, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    Xq = R * np.sin(TH)
    Zq = R * np.cos(TH) + apex_m
    jj = np.clip(((Xq - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip(((Zq - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    pol = env[ii, jj].astype(np.float64)
    pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
    prof = pol.mean(axis=0)
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    s = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    f = np.fft.rfftfreq(rip.size, d=dth)
    tot = s[(f > 0.05) & (f < 5)].sum()
    fl = 1.0 / LINE_SPACING_DEG
    frac = s[(f > fl * 0.9) & (f < fl * 1.1)].sum() / tot
    sel = (f > 0.3) & (f < 2.5)
    return 1.0 / f[sel][np.argmax(s[sel])], np.sqrt(frac) * rip.std() * 100, rip.std() * 100


r_lo, r_hi = 0.045, 0.085
mid = (r_lo + r_hi) / 2
print(f"predicted origin/apex angle ratio at {mid * 1e3:.0f} mm: "
      f"{(mid - APEX_M) / mid:.3f}  ->  {LINE_SPACING_DEG * (mid - APEX_M) / mid:.3f} deg")
print(f"(measured peak with the origin-referenced metric: 1.284 deg)\n")

print(f"{'reconstruction':30s} {'centre':>10s} {'peak':>8s} {'@1.111':>8s} {'rms':>7s}")
print("-" * 68)
for name, lab in P.RECONS:
    try:
        env, co = P.load_env(name)
    except (OSError, KeyError):
        continue
    for apex, cl in ((0.0, "origin"), (APEX_M, "APEX")):
        pk, amp, rms = spec(env, co, apex, r_lo, r_hi)
        print(f"{lab if cl == 'origin' else '':30s} {cl:>10s} {pk:7.3f}d {amp:7.2f}% {rms:6.1f}%")

# the per-transmit stack: decimating the pitch should now track 1.1111 x step exactly
stack_path = os.path.join(HERE, "phantom_per_tx.npy")
if os.path.isfile(stack_path):
    stack = np.load(stack_path)
    env0, co = P.load_env(P.RECONS[0][0])
    print(f"\n{'decimation test (apex-referenced)':30s} {'expected':>10s} {'peak':>8s}")
    print("-" * 52)
    for step in (1, 2, 3):
        img = np.abs(np.sum(stack[np.arange(0, 73, step)], axis=0))
        pk, _, _ = spec(img, co, APEX_M, r_lo, r_hi)
        print(f"{f'pitch {1.1111 * step:.3f} deg':30s} {1.1111 * step:9.3f}d {pk:7.3f}d")
