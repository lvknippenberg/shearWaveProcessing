"""Buffer-3 focused beams: do they tile the sector, and is every pixel insonified?

Answers the 'is every pixel part of at least one transmit region' question quantitatively:
angular beam spacing vs the -6 dB beam width of the focused aperture.
"""
import h5py
import numpy as np

MAT = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\CombinedData.mat"
f = h5py.File(MAT, "r")


def refs(g, k):
    return np.atleast_1d(np.array(g[k]).squeeze())


def val(r):
    return np.array(f[r]).squeeze()


fc = float(np.array(f["Trans"]["frequency"]).squeeze())
c = float(np.array(f["Resource"]["Parameters"]["speedOfSound"]).squeeze())
lam = c / (fc * 1e6)                      # m
TX = f["TX"]
org = np.array([val(r).reshape(-1) for r in refs(TX, "Origin")])
foc = np.array([float(val(r)) for r in refs(TX, "focus")])
steer = np.array([val(r).reshape(-1) for r in refs(TX, "Steer")])
apod = [val(r).reshape(-1) for r in refs(TX, "Apod")]
n_act = np.array([int((np.abs(a) > 0).sum()) for a in apod])

sel = np.where((np.round(foc, 2) == 160.0) & (n_act == 79))[0]
az = steer[sel, 0]                                     # azimuth steer, radians
uniq_az = np.unique(np.round(az, 6))
print(f"buffer-3 family: {len(sel)} transmits, {len(uniq_az)} unique steering angles")
print(f"  azimuth range: {np.degrees(uniq_az).min():.2f} to {np.degrees(uniq_az).max():.2f} deg")
d_ang = np.degrees(np.median(np.diff(np.sort(uniq_az))))
print(f"  angular spacing: {d_ang:.4f} deg")

pitch = 0.2540 * lam
aperture = 79 * pitch
print(f"\n  aperture = {aperture * 1e3:.2f} mm, lambda = {lam * 1e3:.4f} mm")
# -6 dB two-way beam width of a uniformly weighted aperture ~ 0.886*lambda/D (one-way, rad)
bw_rad = 0.886 * lam / aperture
print(f"  one-way -6dB beam width = {np.degrees(bw_rad):.4f} deg")
print(f"  overlap factor (beamwidth / spacing) = {np.degrees(bw_rad) / d_ang:.2f}x")
if np.degrees(bw_rad) / d_ang >= 1.0:
    print("  -> beams OVERLAP: the sector is fully covered, no un-insonified gaps.")
else:
    print("  -> GAPS between beams.")

# Depth dependence: at the focus the beam is narrowest. Beam width in mm vs depth.
print("\n  depth   beam width   line spacing   ratio")
for z_mm in (20, 40, 60, 78.8, 100, 150):
    z = z_mm * 1e-3
    w = bw_rad * z
    s = np.radians(d_ang) * z
    print(f"  {z_mm:6.1f}mm {w * 1e3:8.2f}mm {s * 1e3:11.3f}mm {w / s:8.2f}")

print(f"\nTXPD present: {'TXPD' in TX}")
if "TXPD" in TX:
    t0 = val(refs(TX, "TXPD")[sel[0]])
    print(f"  TXPD[0] shape {np.shape(t0)} dtype {np.asarray(t0).dtype}")
    print("  (Verasonics' own per-pixel transmit-field weighting - the analogue of zea's pfield)")
f.close()
