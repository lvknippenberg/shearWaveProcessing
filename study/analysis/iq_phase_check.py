"""Phase-level comparison of two beamformed outputs of the same folder (e.g. Windows vs Linux).

``linux_validation.py compare`` reports relRMS / correlation per dataset; this answers the follow-up
question that matters for shear-wave processing: is a difference a constant phase offset (harmless -
displacement uses frame-to-frame phase) or does it change the displacement?

Per compared output it prints, for buffer 4 and one buffer-2 push:
  * global phase offset and its spread over space and over time (0 = constant),
  * phase offset versus depth (a depth trend points at geometry / sound speed / demodulation),
  * lag-1 (frame-to-frame) phase difference = Kasai displacement proxy: median / p99 |diff| and
    correlation,
  * envelope ratio and log-envelope correlation.
Only in-sector pixels above the median envelope are used.

2026-09-17 results on C000000001 (reference = Windows ``output``):
  old server zea 44208e0b (JAX): buffer-4 lag-1 corr 0.21, phase offset 0 -> 57 deg with depth.
  zea 8c2699fd, JAX:   lag-1 median 0.002 deg, p99 0.06 deg, corr 0.9999.
  zea 8c2699fd, torch: identical (0.0000 deg, corr 1.000000).

Usage:
    python study/analysis/iq_phase_check.py --folder <measurement folder> output_linux output_linux_torch
"""
import argparse

import h5py
import numpy as np

KEY = "tracks/track_0/data/beamformed_data/values"


def load(path, sl):
    with h5py.File(path, "r") as f:
        v = f[KEY][sl].astype(np.float64)
    return v[..., 0] + 1j * v[..., 1]


def report(name, a, b):
    env_a, env_b = np.abs(a), np.abs(b)
    mask = env_a > np.percentile(env_a, 50)
    dphi = np.angle(b * np.conj(a))
    mean_vec = np.exp(1j * dphi).mean(axis=0)
    t_spread = 1 - np.abs(mean_vec)
    glob = np.angle(np.exp(1j * dphi)[:, mask[0]].mean())
    print(f"{name}: global phase offset {np.degrees(glob):7.2f} deg, "
          f"circular spread over space {1 - abs(np.exp(1j * dphi)[:, mask[0]].mean()):.3f}, "
          f"median temporal spread per pixel {np.median(t_spread[mask[0]]):.4f}")
    zc = np.exp(1j * dphi).mean(axis=(0, 2))
    print("   phase offset vs depth (deg, every 40 rows):", np.round(np.degrees(np.angle(zc[::40])), 1))
    ka = np.angle(a[1:] * np.conj(a[:-1]))
    kb = np.angle(b[1:] * np.conj(b[:-1]))
    m2 = mask[1:] & mask[:-1]
    d = np.angle(np.exp(1j * (kb - ka)))[m2]
    print(f"   lag-1 phase (displacement proxy): median |diff| {np.degrees(np.median(np.abs(d))):.4f} deg, "
          f"p99 {np.degrees(np.percentile(np.abs(d), 99)):.4f} deg; corr {np.corrcoef(ka[m2], kb[m2])[0, 1]:.6f}")
    print(f"   envelope: median ratio {np.median(env_b[mask] / env_a[mask]):.4f}, "
          f"log corr {np.corrcoef(np.log(env_a[mask]), np.log(env_b[mask]))[0, 1]:.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--ref-name", default="output")
    ap.add_argument("--push", type=int, default=5, help="buffer-2 measurement to check")
    ap.add_argument("outputs", nargs="+", help="output dir names inside the folder")
    a = ap.parse_args()
    F = a.folder
    for out in a.outputs:
        print(f"=== {a.ref_name} vs {out} ===")
        report("buffer4 frames 400-430",
               load(f"{F}/{a.ref_name}/CombinedData_buffer4_iq.hdf5", np.s_[400:430]),
               load(f"{F}/{out}/CombinedData_buffer4_iq.hdf5", np.s_[400:430]))
        name = f"CombinedData_buffer2_meas{a.push}_iq.hdf5"
        report(f"buffer2 meas{a.push} frames 0-30",
               load(f"{F}/{a.ref_name}/{name}", np.s_[0:30]), load(f"{F}/{out}/{name}", np.s_[0:30]))


if __name__ == "__main__":
    main()
