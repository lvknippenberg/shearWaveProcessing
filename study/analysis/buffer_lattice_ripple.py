"""Is the line artefact in buffers 1 and 4 too? Measured with one consistent definition.

Ground truth from the converted parameters rather than from delay-ramp estimates:

    buffer 1  21 widebeams   polar_angles -40..+40 deg, pitch 4.0000 deg, focus -123.2 mm
    buffer 3  73 focused     polar_angles -40..+40 deg, pitch 1.1111 deg, focus  +78.9 mm
    buffer 4   2 diverging   polar_angles  -6..+6  deg, pitch 12.000 deg, focus  -12.3 mm

`polar_angles` is already in the sector-apex convention - buffer 3's 1.1111 deg matches the region
geometry in `CenterTransmit.mat` exactly - so all three are measured about the same apex
(-12.1 mm), which is the probe's sector apex, not each transmit's focus.

One definition for all of them, so the numbers are comparable:
* angle about the sector apex (the CORRECTION at the top of the doc);
* detrend window = 9x the buffer's own pitch, which is what the fixed 10 deg window happened to be
  for buffer 3 - a fixed window would strip out buffer 1's 4 deg structure and flatter it;
* ripple amplitude in a +/-12% band around the buffer's own pitch, plus where the peak actually is.

Buffer 4 has TWO transmits. Two is not a lattice: there is no periodic structure to find, and any
"peak" is whatever the search band's edge happens to be. Reported for completeness only.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
sys.path.insert(0, r"D:/Luuk van Knippenberg/Github/shearWaveProcessing/src")

import h5py
import numpy as np
from scipy.ndimage import uniform_filter1d

from zea import File

OUT = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\output"
APEX_M = -0.0121
BAND = (0.045, 0.085)


def load(name):
    with h5py.File(os.path.join(OUT, name), "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        iq = np.asarray(g["values"][:])
        co = np.asarray(g["coordinates"])
    return np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).mean(axis=0), co


def measure(env, co, pitch, apex_m=APEX_M, n_th=2401):
    x, z = co[..., 0], co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    # keep the polar samples inside the imaged sector at the shallowest radius
    r_lo, r_hi = BAND[0] - apex_m, BAND[1] - apex_m
    th_max = min(28.0, np.degrees(np.arcsin(min(abs(x0), abs(x1)) / r_hi)) * 0.95)
    th = np.radians(np.linspace(-th_max, th_max, n_th))
    r = np.linspace(r_lo, r_hi, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip((((R * np.cos(TH) + apex_m) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    pol = env[ii, jj].astype(np.float64)
    pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
    prof = pol.mean(axis=0)
    dth = np.degrees(th[1] - th[0])
    win = min(9.0 * pitch, th_max)                      # 9 periods, as buffer 3's 10 deg was
    trend = uniform_filter1d(prof, int(round(win / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    s = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    f = np.fft.rfftfreq(rip.size, d=dth)
    tot = s[(f > 1.0 / (4 * pitch)) & (f < 5)].sum()
    fl = 1.0 / pitch
    amp_at = np.sqrt(s[(f > fl * 0.88) & (f < fl * 1.12)].sum() / max(tot, 1e-30)) * rip.std() * 100
    sel = (f > 1.0 / (2.5 * pitch)) & (f < 1.0 / (0.5 * pitch))
    pk = 1.0 / f[sel][np.argmax(s[sel])]
    amp_pk = np.sqrt(s[(f > 1 / pk * 0.88) & (f < 1 / pk * 1.12)].sum() / max(tot, 1e-30)) \
        * rip.std() * 100
    return amp_at, pk, amp_pk, rip.std() * 100, 2 * th_max


CASES = [
    ("buffer 1 widebeam (21 tx)", "CombinedData_buffer1_iq.hdf5", 4.0000),
    ("buffer 1 REFoCUS adjoint", "CombinedData_buffer1_refocus-adjoint_iq.hdf5", 4.0000),
    ("buffer 3 focused (73 tx)", "CombinedData_buffer3_iq.hdf5", 1.1111),
    ("buffer 3 REFoCUS adjoint", "CombinedData_buffer3_refocus-adjoint_iq.hdf5", 1.1111),
    ("buffer 4 diverging (2 tx)", "CombinedData_buffer4_iq.hdf5", 12.0000),
]
print(f"{'':30s} {'pitch':>7s} {'@pitch':>9s} {'peak':>9s} {'@peak':>8s} {'rms':>7s} {'span':>7s}")
print("-" * 86)
for lab, fn, pitch in CASES:
    try:
        env, co = load(fn)
    except OSError:
        print(f"{lab:30s}  (not available)")
        continue
    a_at, pk, a_pk, rms, span = measure(env, co, pitch)
    print(f"{lab:30s} {pitch:6.3f}d {a_at:8.2f}% {pk:8.3f}d {a_pk:7.2f}% {rms:6.1f}% {span:6.1f}d")

# buffer 1's peak keeps landing at ~2x its pitch. With pulse inversion the two polarities are
# summed per beam, so an imbalance between them would alternate beam-to-beam and show up at 2x.
print("\nbuffer 1: checking the 2x-pitch (8 deg) component against the 4 deg one")
env, co = load("CombinedData_buffer1_iq.hdf5")
for p, lab in ((4.0, "at the transmit pitch"), (8.0, "at twice the pitch")):
    a_at, pk, a_pk, rms, _ = measure(env, co, p)
    print(f"  {lab:26s} pitch {p:4.1f}d -> {a_at:6.2f}%   (peak found {pk:6.3f}d)")
