"""Why is the buffer-2 (active tracking) B-mode so noisy, and is frame 0 distorted?

Compares buffer 2 (tracking, 1 transmit/frame, small ROI) against buffer 4 (passive, 2
transmits/frame, full FOV) on like-for-like measures:
  * per-frame mean envelope (is frame 0 an outlier -> push contamination?)
  * speckle SNR = mean/std of the envelope in a homogeneous patch (Rayleigh speckle -> ~1.91)
  * frame-to-frame correlation (tracking quality)
  * pixel size vs speckle size (is it just zoom?)
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
REPO = r"D:\Luuk van Knippenberg\Github\shearWaveProcessing"
sys.path.insert(0, os.path.join(REPO, "src"))

import h5py
import numpy as np

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\output"
B2 = os.path.join(FOLDER, "CombinedData_buffer2_meas0_iq.hdf5")
B4 = os.path.join(FOLDER, "CombinedData_buffer4_iq.hdf5")


def load(path, n=None):
    with h5py.File(path, "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        v = g["values"]
        iq = np.asarray(v[:n] if n else v[:])
        co = np.asarray(g["coordinates"])
        ref = None
        if "custom" in f and "reference_iq" in f["custom"]:
            ref = np.asarray(f["custom"]["reference_iq"])
    return iq, co, ref


def env(iq):
    return np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)


print("=" * 78)
iq2, co2, ref2 = load(B2)
e2 = env(iq2)
print(f"buffer 2 (tracking) IQ {iq2.shape}   reference {None if ref2 is None else ref2.shape}")
print(f"  per-frame mean envelope, first 8: "
      f"{np.array2string(e2.mean(axis=(1, 2))[:8], precision=1)}")
m = e2.mean(axis=(1, 2))
print(f"  frame0 / median(frames1..) = {m[0] / np.median(m[1:]):.3f}")
if ref2 is not None:
    er = env(ref2)
    print(f"  reference mean envelope (last 3): "
          f"{np.array2string(er.mean(axis=(1, 2))[-3:], precision=1)}")
    print(f"  frame0 / median(reference) = {m[0] / np.median(er.mean(axis=(1, 2))):.3f}")

iq4, co4, _ = load(B4, n=60)
e4 = env(iq4)
print(f"\nbuffer 4 (passive) IQ {iq4.shape}")


def speckle_snr(e, name):
    """mean/std in sliding homogeneous blocks; fully developed Rayleigh speckle -> 1.91."""
    fr = e[len(e) // 2]
    h, w = fr.shape
    bs = 16
    vals = []
    for i in range(0, h - bs, bs):
        for j in range(0, w - bs, bs):
            b = fr[i:i + bs, j:j + bs]
            if b.mean() > 0.15 * fr.mean():        # skip anechoic/shadow blocks
                vals.append(b.mean() / (b.std() + 1e-20))
    v = np.array(vals)
    print(f"  {name}: speckle SNR mean/std = {np.median(v):.2f} "
          f"(n={v.size} blocks; Rayleigh ideal 1.91)")
    return np.median(v)


print("\nspeckle statistics (mid frame):")
speckle_snr(e2, "buffer 2 tracking")
speckle_snr(e4, "buffer 4 passive ")


def frame_corr(iq, name, k=1):
    a = iq[:-k, ..., 0] + 1j * iq[:-k, ..., 1]
    b = iq[k:, ..., 0] + 1j * iq[k:, ..., 1]
    num = np.abs((a * np.conj(b)).sum(axis=(1, 2)))
    den = np.sqrt((np.abs(a) ** 2).sum(axis=(1, 2)) * (np.abs(b) ** 2).sum(axis=(1, 2)))
    c = num / (den + 1e-20)
    print(f"  {name}: frame-to-frame |corr| median {np.median(c):.4f}  "
          f"first 5 {np.array2string(c[:5], precision=3)}")
    return c


print("\nframe-to-frame complex correlation (tracking quality):")
frame_corr(iq2, "buffer 2 tracking")
frame_corr(iq4, "buffer 4 passive ")

# pixel / FOV comparison -> is it just zoom?
def extent(co, name):
    x = co[..., 0] if co.shape[-1] >= 2 else None
    z = co[..., -1]
    print(f"  {name}: grid {co.shape[:2]}  x [{np.nanmin(x)*1e3:7.1f},{np.nanmax(x)*1e3:7.1f}] mm"
          f"  z [{np.nanmin(z)*1e3:6.1f},{np.nanmax(z)*1e3:6.1f}] mm")
    dz = (np.nanmax(z) - np.nanmin(z)) / (co.shape[0] - 1)
    dx = (np.nanmax(x) - np.nanmin(x)) / (co.shape[1] - 1)
    print(f"            pixel {dz*1e6:.0f} x {dx*1e6:.0f} um")


print("\ngeometry:")
extent(co2, "buffer 2")
extent(co4, "buffer 4")
