"""Per-subject envelope levels for the three montage sets, to choose a shared normalisation.

The delivered montages log-compress each clip against its OWN maximum, so brightness is not
comparable between tiles. To fix that we need one reference level - but only if the sets are
physically comparable in the first place. Two things to check:

  1. spread of per-subject maxima within a set (how much a single scale would darken/clip);
  2. the gain offset between buffer-3 standard and REFoCUS - REFoCUS applies H^H with a ramp
     filter, which is NOT unit gain, so an absolute comparison between them may just be showing
     the decode gain rather than image quality.
"""
import glob
import os

import h5py
import numpy as np

SETS = {
    "buffer1": r"Z:\raw_data\C*\*\output\CombinedData_buffer1_iq.hdf5",
    "buffer3_standard": r"Z:\raw_data\C*\*\output\CombinedData_buffer3_iq.hdf5",
    "buffer3_refocus": r"Z:\raw_data\C*\*\output\CombinedData_buffer3_refocus-adjoint_iq.hdf5",
}
N_PROBE = 5          # frames sampled per file to estimate its level


def levels(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        subj = p.split(os.sep)[-4]
        with h5py.File(p, "r") as f:
            v = f["tracks/track_0/data/beamformed_data/values"]
            n = v.shape[0]
            idx = np.linspace(0, n - 1, N_PROBE).round().astype(int)
            iq = np.asarray(v[idx])
        e = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
        out[subj] = (float(e.max()), float(np.percentile(e, 99.9)))
    return out


res = {}
for name, pat in SETS.items():
    res[name] = levels(pat)
    mx = np.array([v[0] for v in res[name].values()])
    p999 = np.array([v[1] for v in res[name].values()])
    print(f"{name:18s} n={len(mx):2d}  max: median {np.median(mx):10.0f}  "
          f"spread {20*np.log10(mx.max()/mx.min()):5.1f} dB   "
          f"p99.9: median {np.median(p999):9.0f}")

print()
common = sorted(set(res["buffer3_standard"]) & set(res["buffer3_refocus"]))
ratio = np.array([res["buffer3_refocus"][s][1] / res["buffer3_standard"][s][1] for s in common])
print(f"REFoCUS / standard p99.9 gain ratio: median {np.median(ratio):.3f} "
      f"({20*np.log10(np.median(ratio)):+.1f} dB), spread "
      f"{20*np.log10(ratio.max()/ratio.min()):.1f} dB across subjects")
c1 = sorted(set(res["buffer1"]) & set(res["buffer3_standard"]))
r13 = np.array([res["buffer3_standard"][s][1] / res["buffer1"][s][1] for s in c1])
print(f"buffer3_standard / buffer1 p99.9 ratio: median {np.median(r13):.3f} "
      f"({20*np.log10(np.median(r13)):+.1f} dB)")
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "levels.npy"),
        res, allow_pickle=True)
print("\nsaved levels.npy")
