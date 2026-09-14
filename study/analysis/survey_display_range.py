"""What white point and dynamic range does each set actually need?

A display is two numbers: `ref` (white point) and `DR` (how many dB below it map to black).
Fixing both across sets whose distributions differ by ~25 dB clips one and flattens the other.

Derive both from the data instead, per set, over IN-SECTOR pixels only (outside the sector the
envelope is exactly 0, which would drag any low percentile to zero):

    p_hi   99.9th pct  - the bright structures; sets the white point
    p_lo    5th  pct   - the noise/background floor; should land near black
    DR = 20log10(p_hi / p_lo)

Reported per subject and aggregated, so a single (ref, DR) per set can be chosen that keeps
subjects mutually comparable while fitting that set's own signal span.
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
N_PROBE = 5


def stats(pattern):
    hi, lo = [], []
    for p in sorted(glob.glob(pattern)):
        with h5py.File(p, "r") as f:
            v = f["tracks/track_0/data/beamformed_data/values"]
            idx = np.linspace(0, v.shape[0] - 1, N_PROBE).round().astype(int)
            iq = np.asarray(v[idx])
        e = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
        ins = e[e > 0]                       # in-sector only
        hi.append(np.percentile(ins, 99.9))
        lo.append(np.percentile(ins, 5))
    return np.array(hi), np.array(lo)


print(f"{'set':18s} {'ref (p90 of hi)':>16s} {'floor (med lo)':>15s} {'DR needed':>10s} "
      f"{'subj spread':>12s}")
chosen = {}
for name, pat in SETS.items():
    hi, lo = stats(pat)
    ref = float(np.percentile(hi, 90))       # white point: only the top ~10% clip a little
    floor = float(np.median(lo))
    dr = 20 * np.log10(ref / floor)
    chosen[name] = (ref, dr)
    print(f"{name:18s} {ref:16.0f} {floor:15.1f} {dr:9.1f}dB "
          f"{20*np.log10(hi.max()/hi.min()):11.1f}dB")

print("\nrecommended per-set display settings:")
for k, (ref, dr) in chosen.items():
    print(f"  {k:18s} --reference {ref:.0f} --dynamic-range {dr:.0f}")
print("\nlevel offsets between sets (from the white points), for the record:")
b1 = chosen["buffer1"][0]
for k, (ref, _) in chosen.items():
    print(f"  {k:18s} {20*np.log10(ref/b1):+6.1f} dB vs buffer 1")
