"""Read the TX geometry per buffer from CombinedData.mat.

For buffer 3 (focused) the question is whether the 73 focused beams tile the sector without
gaps: beam spacing vs -6 dB beam width at depth. Also reports what buffer 2 (tracking) and
buffer 4 (passive) actually transmit, for the SNR comparison.
"""
import os
import sys

import h5py
import numpy as np

MAT = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\CombinedData.mat"

f = h5py.File(MAT, "r")


def refs(group, field):
    return np.atleast_1d(np.array(group[field]).squeeze())


def val(ref):
    return np.array(f[ref]).squeeze()


Trans = f["Trans"]
fc_MHz = float(np.array(Trans["frequency"]).squeeze())
c = float(np.array(f["Resource"]["Parameters"]["speedOfSound"]).squeeze())
lam_mm = c / (fc_MHz * 1e6) * 1e3
print(f"Trans.frequency = {fc_MHz:.4f} MHz   c = {c} m/s   lambda = {lam_mm:.4f} mm")
print(f"Trans.numelements = {int(np.array(Trans['numelements']).squeeze())}")
ep = np.array(Trans["ElementPos"])  # (4, n) or (n, 4)
if ep.shape[0] > ep.shape[1]:
    ep = ep.T
print(f"Trans.ElementPos shape {ep.shape}; x range [{ep[0].min():.2f}, {ep[0].max():.2f}] wl")
pitch = float(np.median(np.diff(np.sort(ep[0]))))
print(f"element pitch = {pitch:.4f} wl = {pitch * lam_mm:.4f} mm")

TX = f["TX"]
print(f"\nTX fields: {sorted(TX.keys())}")
n_tx = len(refs(TX, "Origin"))
print(f"TX entries: {n_tx}")

org = np.array([val(r).reshape(-1) for r in refs(TX, "Origin")])        # (n,3) wl
foc = np.array([float(val(r)) for r in refs(TX, "focus")])              # wl
steer = np.array([val(r).reshape(-1) for r in refs(TX, "Steer")]) if "Steer" in TX else None
apod = [val(r).reshape(-1) for r in refs(TX, "Apod")]

# Map TX index -> buffer via Receive.
R = f["Receive"]
bufnum = np.array([float(val(r)) for r in refs(R, "bufnum")])
# Event links TX and Receive; simpler: TX_index / Rcv_index arrays saved in the workspace.
print("\nper-buffer TX summary (grouped by unique (focus, n_active) signature):")
for name in ("Rcv_index", "Rcv_index_SW", "Rcv_index_BmodeSW", "TX_index"):
    if name in f:
        a = np.array(f[name]).reshape(-1)
        print(f"  {name}: shape {a.shape} range [{a.min():.0f}, {a.max():.0f}]")

n_active = np.array([int((np.abs(a) > 0).sum()) for a in apod])
print(f"\nTX focus (wl) unique: {np.unique(np.round(foc, 2))[:12]}")
print(f"TX n_active elements unique: {np.unique(n_active)[:12]}")

# Group TX by (focus, n_active) to identify the focused-B-mode family.
sig = {}
for i in range(n_tx):
    k = (round(float(foc[i]), 2), int(n_active[i]))
    sig.setdefault(k, []).append(i)
print("\n(focus_wl, n_active) -> count, origin-x span (wl)")
for k in sorted(sig, key=lambda k: -len(sig[k]))[:8]:
    idx = sig[k]
    xs = org[idx, 0]
    d = np.diff(np.sort(xs))
    d = d[d > 1e-9]
    print(f"  {k}: n={len(idx):4d}  x [{xs.min():8.3f},{xs.max():8.3f}]  "
          f"median dx={np.median(d) if d.size else float('nan'):.4f} wl "
          f"({(np.median(d) * lam_mm) if d.size else float('nan'):.3f} mm)")

f.close()
