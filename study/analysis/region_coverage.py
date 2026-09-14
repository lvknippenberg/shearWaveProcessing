"""Verasonics PData(3) reconstruction-region coverage: does it thin out with depth?

The user's hypothesis: the radial lines come from the number of lines and the WIDTH of each
reconstruction region, and get worse with depth. PData(3).Region holds 73 regions (one per
focused line) as explicit pixel-index lists (PixelsLA). Counting how many regions cover each
pixel gives the coverage map directly - no modelling assumptions.

If regions are fixed-width in mm (a parallelogram per line) rather than fixed-angle, their
angular extent shrinks as 1/r, so coverage falls with depth and eventually leaves seams.
"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAT = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\CombinedData.mat"
OUT = os.path.dirname(os.path.abspath(__file__))

f = h5py.File(MAT, "r")
reg_refs = np.atleast_1d(np.array(f["PData"]["Region"]).squeeze())
g = f[reg_refs[2]]                                  # PData(3)

size_refs = np.atleast_1d(np.array(f["PData"]["Size"]).squeeze())
n_z, n_x = [int(v) for v in np.array(f[size_refs[2]]).reshape(-1)[:2]]
org = np.array(f[np.atleast_1d(np.array(f["PData"]["Origin"]).squeeze())[2]]).reshape(-1)
pdl = np.array(f[np.atleast_1d(np.array(f["PData"]["PDelta"]).squeeze())[2]]).reshape(-1)
fc = float(np.array(f["Trans"]["frequency"]).squeeze())
c = float(np.array(f["Resource"]["Parameters"]["speedOfSound"]).squeeze())
lam_mm = c / (fc * 1e6) * 1e3
dx_wl, dz_wl = float(pdl[0]), float(pdl[2])
x0_wl, z0_wl = float(org[0]), float(org[2])
print(f"PData(3): {n_z} x {n_x} px, origin ({x0_wl:.2f},{z0_wl:.2f}) wl, "
      f"PDelta ({dx_wl:.3f},{dz_wl:.3f}) wl, lambda {lam_mm:.4f} mm")

# Region shapes (what geometry Verasonics used per line)
shape_refs = np.array(g["Shape"]).reshape(-1)
sh = f[shape_refs[0]]
print(f"\nRegion(1).Shape fields: {sorted(sh.keys())}")
for k in sh.keys():
    v = np.array(sh[k]).squeeze()
    if v.dtype.kind in "SU" or v.ndim == 0 or v.size < 6:
        try:
            if v.dtype.kind == "u" and v.size > 1:
                v = "".join(chr(int(ch)) for ch in v.reshape(-1))
        except Exception:
            pass
        print(f"   {k} = {v}")

# Coverage map from the explicit pixel lists
px_refs = np.array(g["PixelsLA"]).reshape(-1)
cover = np.zeros(n_z * n_x, np.int16)
for r in px_refs:
    idx = np.array(f[r]).reshape(-1).astype(np.int64) - 1      # MATLAB 1-based
    idx = idx[(idx >= 0) & (idx < cover.size)]
    cover[idx] += 1
cover = cover.reshape((n_x, n_z)).T if False else cover.reshape((n_z, n_x), order="F")
f.close()

x_mm = (x0_wl + np.arange(n_x) * dx_wl) * lam_mm
z_mm = (z0_wl + np.arange(n_z) * dz_wl) * lam_mm
print(f"\nx {x_mm.min():.1f}..{x_mm.max():.1f} mm, z {z_mm.min():.1f}..{z_mm.max():.1f} mm")

# Restrict to the insonified sector (+/-40 deg) to avoid counting the corners
X, Z = np.meshgrid(x_mm, z_mm)
ang = np.degrees(np.arctan2(X, np.maximum(Z, 1e-6)))
in_sector = (np.abs(ang) <= 39.0) & (Z > 5)

print("\ncoverage (regions per pixel) vs depth, inside the +/-39 deg sector:")
print(f"  {'depth':>8s} {'mean':>6s} {'min':>5s} {'%cov=0':>8s} {'%cov=1':>8s} {'%cov>=2':>8s}")
rows = []
for z_lo in range(10, 150, 10):
    band = in_sector & (Z >= z_lo) & (Z < z_lo + 10)
    if band.sum() == 0:
        continue
    cv = cover[band]
    rows.append((z_lo + 5, cv.mean(), cv.min(), (cv == 0).mean() * 100,
                 (cv == 1).mean() * 100, (cv >= 2).mean() * 100))
    print(f"  {z_lo:3d}-{z_lo+10:3d}mm {cv.mean():6.2f} {cv.min():5d} "
          f"{(cv == 0).mean()*100:7.2f}% {(cv == 1).mean()*100:7.2f}% {(cv >= 2).mean()*100:7.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
im = axes[0].imshow(cover, aspect="auto", cmap="viridis",
                    extent=[x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]], vmin=0, vmax=6)
axes[0].set_title("PData(3) reconstruction-region coverage\n(how many of the 73 regions include each pixel)")
axes[0].set_xlabel("x (mm)"); axes[0].set_ylabel("z (mm)")
plt.colorbar(im, ax=axes[0], label="regions per pixel")
r = np.array(rows)
axes[1].plot(r[:, 0], r[:, 1], "o-", label="mean coverage")
axes[1].plot(r[:, 0], r[:, 3], "s-", label="% pixels with NO region")
axes[1].plot(r[:, 0], r[:, 4], "^-", label="% pixels with exactly 1")
axes[1].set_xlabel("depth (mm)"); axes[1].legend(); axes[1].grid(alpha=.3)
axes[1].set_title("coverage vs depth")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "region_coverage.png"), dpi=95)
print("\nwrote region_coverage.png")
