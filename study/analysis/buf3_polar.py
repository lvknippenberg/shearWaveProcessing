"""Identify the buffer-3 radial streaks: resample to polar and find the angular period.

If the streaks sit at the 1.111 deg beam spacing they are beam/reconstruction artefacts.
Compares pfield off vs on, and against buffer 1 (widebeam, 21 tx) as a control that shows no
streaks. Also checks the focused buffer's per-transmit apodisation for a gap at the aperture
edges (steered beams lose aperture, which changes beam width with angle).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py

SCRATCH = os.path.dirname(os.path.abspath(__file__))
FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\output"

env_off = np.load(os.path.join(SCRATCH, "buf3_env_off.npy"))
env_on = np.load(os.path.join(SCRATCH, "buf3_env_on.npy"))

with h5py.File(os.path.join(FOLDER, "CombinedData_buffer3_iq.hdf5"), "r") as f:
    co3 = np.asarray(f["tracks/track_0/data/beamformed_data/coordinates"])
with h5py.File(os.path.join(FOLDER, "CombinedData_buffer1_iq.hdf5"), "r") as f:
    g = f["tracks/track_0/data/beamformed_data"]
    iq1 = np.asarray(g["values"][0])
    co1 = np.asarray(g["coordinates"])
env_b1 = np.sqrt(iq1[..., 0] ** 2 + iq1[..., 1] ** 2)

print(f"buffer3 env {env_off.shape}, coords {co3.shape}")


def to_polar(env, co, n_th=721, n_r=300, r_lo=0.05, r_hi=0.13):
    """Sample the Cartesian image on a polar grid about the array centre (x=0, z=0)."""
    x = co[..., 0]
    z = co[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape
    th = np.radians(np.linspace(-40, 40, n_th))
    r = np.linspace(r_lo, r_hi, n_r)
    R, TH = np.meshgrid(r, th, indexing="ij")
    X = R * np.sin(TH)
    Z = R * np.cos(TH)
    j = (X - x0) / (x1 - x0) * (nx - 1)
    i = (Z - z0) / (z1 - z0) * (nz - 1)
    ok = (i >= 0) & (i <= nz - 1) & (j >= 0) & (j <= nx - 1)
    out = np.full(R.shape, np.nan)
    ii = np.clip(i.astype(int), 0, nz - 1)
    jj = np.clip(j.astype(int), 0, nx - 1)
    out[ok] = env[ii[ok], jj[ok]]
    return th, r, out


def angular_spectrum(env, co, label):
    th, r, pol = to_polar(env, co)
    prof = np.nanmean(pol, axis=0)                      # mean over depth -> f(theta)
    prof = prof - np.nanmean(prof)
    prof = np.nan_to_num(prof)
    dth = np.degrees(th[1] - th[0])
    spec = np.abs(np.fft.rfft(prof * np.hanning(prof.size)))
    freq = np.fft.rfftfreq(prof.size, d=dth)            # cycles per degree
    k = np.argmax(spec[1:]) + 1
    period = 1.0 / freq[k] if freq[k] > 0 else np.inf
    print(f"  {label}: dominant angular period = {period:.3f} deg "
          f"(peak {spec[k]:.3g}); rel. modulation = "
          f"{np.nanstd(np.nan_to_num(np.nanmean(pol, axis=0))) / (np.nanmean(pol) + 1e-9):.4f}")
    return th, prof, freq, spec, period


print("\nangular structure (mean over 50-130 mm depth):")
r3 = angular_spectrum(env_off, co3, "buffer3 pfield=OFF")
r3on = angular_spectrum(env_on, co3, "buffer3 pfield=ON ")
r1 = angular_spectrum(env_b1, co1, "buffer1 widebeam  ")

print(f"\n  buffer-3 beam spacing (from TX.Steer) = 1.1111 deg")

fig, axes = plt.subplots(2, 1, figsize=(13, 8))
for (th, prof, freq, spec, per), lab in ((r3, "buffer3 pfield OFF"),
                                         (r3on, "buffer3 pfield ON"),
                                         (r1, "buffer1 widebeam")):
    axes[0].plot(np.degrees(th), prof, label=f"{lab} (period {per:.2f} deg)", lw=0.9)
    axes[1].plot(freq, spec, label=lab, lw=0.9)
axes[0].set_xlabel("azimuth (deg)"); axes[0].set_ylabel("depth-averaged envelope (demeaned)")
axes[0].legend(fontsize=8); axes[0].set_title("Angular profile: radial streaks appear as periodic ripple")
axes[1].axvline(1 / 1.1111, color="k", ls="--", lw=1, label="1/1.111 deg = beam spacing")
axes[1].set_xlabel("cycles per degree"); axes[1].set_ylabel("|FFT|"); axes[1].set_xlim(0, 2)
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(SCRATCH, "buf3_angular.png"), dpi=95)
print("\nwrote buf3_angular.png")
