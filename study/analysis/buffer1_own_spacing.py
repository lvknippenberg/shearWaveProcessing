"""Does buffer 1 carry angular structure at ITS OWN transmit spacing (4 deg)?

The buffer-3 investigation used buffer 1 as a clean control, but only ever tested it at buffer
3's 1.111 deg line spacing. Buffer 1 has a different transmit periodicity - 21 widebeam
transmits steered +/-40 deg in 4.0 deg steps (Bmode_WB.rayDelta) - and by the coherent
plane-wave compounding criterion (step <= lambda/D = 1.39 deg) it is UNDER-sampled by 2.9x.

So the test the earlier work skipped: scan the whole frame-averaged angular ripple spectrum for
each buffer and see where the power actually sits, rather than probing one assumed frequency.
"""
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

F = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54\output"
OUT = os.path.dirname(os.path.abspath(__file__))


def angular_spectrum(gif_name, r_lo, r_hi, n_th=2001, th_max=36.0, n_r=200, detrend_deg=12.0):
    """Frame-averaged fractional angular ripple and its spectrum (cycles/deg)."""
    with h5py.File(os.path.join(F, gif_name), "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        n = g["values"].shape[0]
        co = np.asarray(g["coordinates"])
        x, z = co[..., 0], co[..., -1]
        x0, x1 = np.nanmin(x), np.nanmax(x)
        z0, z1 = np.nanmin(z), np.nanmax(z)
        nz, nx = co.shape[:2]
        th = np.radians(np.linspace(-th_max, th_max, n_th))
        r = np.linspace(r_lo, r_hi, n_r)
        R, TH = np.meshgrid(r, th, indexing="ij")
        jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
        ii = np.clip(((R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
        acc = np.zeros(n_th)
        for k in range(n):
            iq = np.asarray(g["values"][k])
            e = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
            pol = e[ii, jj]
            pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
            acc += pol.mean(axis=0)
    prof = acc / n
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(detrend_deg / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth)
    band = (freq > 0.05) & (freq < 3.0)
    return th, rip, freq, spec / spec[band].sum(), n


def power_at(freq, spec, period_deg, rel=0.12):
    f0 = 1.0 / period_deg
    m = (freq > f0 * (1 - rel)) & (freq < f0 * (1 + rel))
    return spec[m].sum() * 100


cases = [("CombinedData_buffer1_iq.hdf5", "buffer 1 widebeam (21 tx, 4.0 deg step)", 4.0),
         ("CombinedData_buffer3_iq.hdf5", "buffer 3 focused  (73 tx, 1.111 deg step)", 1.1111)]
bands = [(0.045, 0.085, "45-85 mm"), (0.105, 0.145, "105-145 mm")]

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
print(f"{'band':>11s}  {'buffer':42s} {'@4.0deg':>9s} {'@1.111deg':>10s} {'ripple':>8s}")
for row, (r_lo, r_hi, blab) in enumerate(bands):
    for name, lab, own in cases:
        th, rip, freq, spec, n = angular_spectrum(name, r_lo, r_hi)
        p4 = power_at(freq, spec, 4.0)
        p1 = power_at(freq, spec, 1.1111)
        print(f"{blab:>11s}  {lab:42s} {p4:8.2f}% {p1:9.2f}% {rip.std()*100:7.1f}%")
        ax = axes[row, 0 if "widebeam" in lab else 1]
        ax.plot(freq, spec * 100, lw=0.9, label=f"{blab} (N={n})")
        ax.axvline(1 / 4.0, color="tab:red", ls="--", lw=1, label="4.0 deg (buf1 step)")
        ax.axvline(1 / 1.1111, color="tab:green", ls=":", lw=1, label="1.111 deg (buf3 step)")
        ax.set_xlim(0, 1.6)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("cycles / deg")
        ax.set_ylabel("% of ripple power")
        if row == 0:
            ax.legend(fontsize=7)
plt.suptitle("Frame-averaged angular ripple spectrum: where does each buffer's power sit?",
             fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "buffer1_own_spacing.png"), dpi=95)
print("\nwrote buffer1_own_spacing.png")
