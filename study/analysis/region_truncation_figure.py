"""Figure: simulated beam vs its region, and what that predicts against the measured image.

Three panels:
  a) beam width vs region width against depth - the user's "the beam is narrower than the region"
  b) the compounded transmit sensitivity, untruncated vs region-truncated
  c) angular spectra: the two predictions against the measured phantom image

Panel (c) is the one that matters. Any per-transmit AMPLITUDE weighting sits on a regular
1.111 deg lattice, so it can only produce structure at 1/1.111 cyc/deg and its harmonics. The
image peak is somewhere else entirely.
"""
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
LINE_SPACING_DEG = 1.1111
PHANTOM = (r"D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51\output"
           r"\CombinedData_buffer3_iq.hdf5")

maps = np.load(os.path.join(HERE, "region_truncation_maps.npy"), allow_pickle=True).item()
rows = np.load(os.path.join(HERE, "center_transmit_rows.npy"))
th_deg, r_mm = maps["th_deg"], maps["r_mm"]
dth = th_deg[1] - th_deg[0]


def ripple_of(pol, dth_deg, keep_deg, th_axis):
    p = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
    prof = p.mean(axis=0)
    trend = uniform_filter1d(prof, int(round(10.0 / dth_deg)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    keep = np.abs(th_axis) < keep_deg
    return rip[keep], th_axis[keep]


def spectrum(rip, dth_deg):
    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth_deg)
    ok = (freq > 0.2) & (freq < 2.2)
    s = spec[ok]
    return freq[ok], s / s.max()


# ---- measured phantom image, same polar treatment -------------------------------------
with h5py.File(PHANTOM, "r") as f:
    g = f["tracks/track_0/data/beamformed_data"]
    env = np.sqrt(np.asarray(g["values"][:])[..., 0] ** 2
                  + np.asarray(g["values"][:])[..., 1] ** 2).mean(axis=0)
    co = np.asarray(g["coordinates"])
x, z = co[..., 0], co[..., -1]
x0, x1, z0, z1 = np.nanmin(x), np.nanmax(x), np.nanmin(z), np.nanmax(z)
nz, nx = env.shape
th_i = np.radians(np.linspace(-25, 25, 3201))
r_i = np.linspace(0.045, 0.085, 200)
R, TH = np.meshgrid(r_i, th_i, indexing="ij")
jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
ii = np.clip(((R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
img_pol = env[ii, jj]
dth_i = np.degrees(th_i[1] - th_i[0])
rip_img, th_img = ripple_of(img_pol, dth_i, 25, np.degrees(th_i))

rip_un, th_keep = ripple_of(maps["untrunc"], dth, 25, th_deg)
rip_tr, _ = ripple_of(maps["trunc"], dth, 25, th_deg)

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

# (a) beam vs region
d, w6, w20, regw = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
ax[0].plot(d, w6, "o-", label="beam, half-max")
ax[0].plot(d, w20, "s-", label="beam, -20 dB")
ax[0].plot(d, regw, "k^-", lw=2, label="reconstruction region")
ax[0].fill_between(d, w6, regw, where=regw > w6, alpha=.18, color="crimson",
                   label="region wider than beam")
ax[0].set_xlabel("depth (mm)")
ax[0].set_ylabel("lateral width (mm)")
ax[0].set_title("(a) centre beam vs the region it reconstructs")
ax[0].legend(fontsize=8)
ax[0].grid(alpha=.3)

# (b) compounded sensitivity
ax[1].plot(th_keep, 100 * rip_un, lw=.9, label=f"untruncated  ({rip_un.std() * 100:.2f}% rms)")
ax[1].plot(th_keep, 100 * rip_tr, lw=.9,
           label=f"region-truncated  ({rip_tr.std() * 100:.2f}% rms)")
ax[1].plot(th_img, 100 * rip_img, lw=.9, color="k", alpha=.6,
           label=f"measured image  ({rip_img.std() * 100:.2f}% rms)")
ax[1].set_xlim(-10, 10)
ax[1].set_xlabel("angle (deg)")
ax[1].set_ylabel("ripple (%)")
ax[1].set_title("(b) compounded transmit sensitivity vs the image")
ax[1].legend(fontsize=8)
ax[1].grid(alpha=.3)

# (c) spectra
for rip, dd, lab, kw in ((rip_un, dth, "untruncated", dict(lw=1.4)),
                         (rip_tr, dth, "region-truncated", dict(lw=1.4)),
                         (rip_img, dth_i, "measured image", dict(lw=1.8, color="k"))):
    f, s = spectrum(rip, dd)
    ax[2].plot(1.0 / f, s, label=lab, **kw)
ax[2].axvline(LINE_SPACING_DEG, color="tab:blue", ls="--", lw=1)
ax[2].text(LINE_SPACING_DEG, 1.02, " 1.111 deg\n line spacing", color="tab:blue", fontsize=8)
ax[2].axvline(1.284, color="k", ls=":", lw=1)
ax[2].text(1.284, .72, "  1.284 deg\n  image peak", fontsize=8)
ax[2].set_xlim(0.6, 2.2)
ax[2].set_xlabel("angular period (deg)")
ax[2].set_ylabel("normalised power")
ax[2].set_title("(c) the image peak is NOT at the transmit pitch")
ax[2].legend(fontsize=8)
ax[2].grid(alpha=.3)

fig.tight_layout()
out = os.path.join(os.path.dirname(HERE), "montages", "region_vs_beam.png")
fig.savefig(out, dpi=135)
print("wrote", out)
for lab, rip, dd in (("untruncated", rip_un, dth), ("truncated", rip_tr, dth),
                     ("image", rip_img, dth_i)):
    f, s = spectrum(rip, dd)
    print(f"  {lab:16s} peak period {1.0 / f[np.argmax(s)]:.3f} deg   rms {rip.std() * 100:.2f}%")
