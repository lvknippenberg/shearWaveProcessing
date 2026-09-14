"""Re-measure the nearest-k ladder with the APEX-referenced ripple metric, and retest the
field correction now that the period is known to match.

The origin-referenced metric used everywhere earlier in this investigation smears a structure that
is periodic in apex-angle across a varying origin-angle period, which destroys the spectral peak
and understates the amplitude by ~10x. Everything here uses the apex.

Two questions:
  1. does reconstructing from fewer transmits remove the artefact? (the nearest-k ladder)
  2. the synthesised transmit field is ANTI-correlated with the image at the same period - so does
     MULTIPLYING by it flatten the ripple where dividing made it worse?
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
STEER_DEG = np.arange(73) * 1.1111111 - 40.0


def spec(env, co, r_lo=0.045, r_hi=0.085, n_th=2001, th_max=30.0, apex_m=APEX_M):
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo - apex_m, r_hi - apex_m, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip((((R * np.cos(TH) + apex_m) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
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
    return np.sqrt(frac) * rip.std() * 100, 1.0 / f[sel][np.argmax(s[sel])]


stack = np.load(os.path.join(HERE, "phantom_per_tx.npy"))
env_std, co = P.load_env(P.RECONS[0][0])
targets = P.find_targets(env_std, *P.axes_mm(co)[::-1][::-1])
pix_deg = np.degrees(np.arctan2(co[..., 0], co[..., -1] - APEX_M))
nz, nx = pix_deg.shape
ii, jj = np.meshgrid(np.arange(nz), np.arange(nx), indexing="ij")
order = np.argsort(np.abs(pix_deg[None] - STEER_DEG[:, None, None]), axis=0)


def report(env, lab):
    amp, pk = spec(env, co)
    rows = P.measure(env, co, targets)
    if rows:
        lat = np.median([r["lat"] for r in rows])
        cnr = np.median([r["cnr"] for r in rows])
        print(f"{lab:30s} {amp:8.2f}% {pk:8.3f}d {lat:8.2f}mm {cnr:7.1f}dB")
    else:
        print(f"{lab:30s} {amp:8.2f}% {pk:8.3f}d   (no targets)")
    return amp


print(f"{'composite':30s} {'@pitch':>9s} {'peak':>9s} {'lateral':>9s} {'CNR':>8s}")
print("-" * 70)
acc = np.zeros((nz, nx), np.complex64)
for k in range(1, 74):
    acc += stack[order[k - 1], ii, jj]
    if k in (1, 2, 3, 6, 12, 73):
        report(np.abs(acc), f"nearest-{k}" + ("  (= region rule)" if k == 6 else ""))
report(np.sum(np.abs(stack), axis=0), "all-73 incoherent")
for name, lab in P.RECONS[1:]:
    try:
        e, c = P.load_env(name)
    except (OSError, KeyError):
        continue
    report(e, lab)

print("\n-- field correction: divide vs MULTIPLY (period now known to match) --")
S = np.load(os.path.join(HERE, "txfield_harm.npy"))
ins = env_std > 0
S = S / np.median(S[ins])
S = np.clip(S, 10 ** (-12 / 20), 10 ** (12 / 20))
base = report(env_std, "standard (reference)")
for g in (0.25, 0.5, 1.0):
    report(env_std / (S ** g), f"/ harmonic field g={g:.2f}")
for g in (0.25, 0.5, 1.0):
    report(env_std * (S ** g), f"* harmonic field g={g:.2f}")
