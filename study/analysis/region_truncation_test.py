"""Does truncating each beam to its own 6.667 deg region create the striations?

`CenterTransmit.mat` gives the simulated field of the centre beam and the region it reconstructs.
Region 37 is a SectorFT of 6.667 deg = exactly 6x the 1.111 deg line spacing, while the beam's own
half-max width near the focus is ~2.3 deg. So each transmit is used to reconstruct pixels out to
roughly its -20 dB contour, and every pixel is a sum of 6 transmits of very unequal strength.

That is testable with this one file. Resample the centre beam into polar coordinates about the
virtual apex, then replicate it at all 73 steering angles and sum, two ways:

* **untruncated** - every transmit contributes everywhere. This is the free-field compound, the
  quantity a pfield-style sensitivity map represents.
* **region-truncated** - each transmit contributes only inside its own +/-3.333 deg region, which
  is what the beamformer actually does.

If the region truncation is what produces the striations, the second will carry angular ripple
that the first does not, at the period we measure in the images (~1.26-1.28 deg). If both are
flat, the region/beam mismatch is not the mechanism, and the residual has to be cross-transmit
phase interference - which no magnitude-based normalisation can undo.

Note this compounds INTENSITIES (the profile is a magnitude map, it carries no phase). That is
deliberate: it isolates the amplitude-scalloping hypothesis specifically.
"""
import os

import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(os.path.dirname(HERE), "CenterTransmit.mat")

LINE_SPACING_DEG = 1.1111
N_TX = 73
REGION_HALF_DEG = 6.667 / 2
F_TRANS_MHZ, C = 3.125, 1540.0
LAM_MM = C / (F_TRANS_MHZ * 1e6) * 1e3

m = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
P, prof = m["TransmitPData"], m["TransmitProfile"].astype(np.float64)
nz, nx = int(P.Size[0]), int(P.Size[1])
dz, dx = float(P.PDelta[2]), float(P.PDelta[0])
x0, z0 = float(P.Origin[0]), float(P.Origin[2])
apex_mm = float(P.Region[36].Shape.Position[2]) * LAM_MM

x_mm = (x0 + np.arange(nx) * dx) * LAM_MM
z_mm = (z0 + np.arange(nz) * dz) * LAM_MM

# ---- resample the centre beam onto a polar grid about the virtual apex ----------------
R_LO, R_HI, N_R = 45.0, 105.0, 240           # mm from the apex, the band the ripple is measured in
TH_MAX, N_TH = 40.0, 3201                    # deg, fine enough to resolve 1.1 deg structure
r = np.linspace(R_LO, R_HI, N_R)
th = np.radians(np.linspace(-TH_MAX, TH_MAX, N_TH))
RR, TT = np.meshgrid(r, th, indexing="ij")
Xq = RR * np.sin(TT)
Zq = RR * np.cos(TT) + apex_mm               # apex sits at negative z

# bilinear sample of prof at (Zq, Xq)
ii = (Zq - z_mm[0]) / (z_mm[1] - z_mm[0])
jj = (Xq - x_mm[0]) / (x_mm[1] - x_mm[0])
beam = map_coordinates(prof, [ii.ravel(), jj.ravel()], order=1, mode="constant", cval=0.0)
beam = beam.reshape(RR.shape)
dth_deg = np.degrees(th[1] - th[0])
print(f"polar beam {beam.shape}, {dth_deg:.4f} deg/sample, r = {R_LO}-{R_HI} mm from apex")

pk_col = np.argmax(beam.mean(axis=0))
print(f"beam peak at theta = {np.degrees(th[pk_col]):+.3f} deg (should be ~0)")

# ---- compound over the 73 steering angles --------------------------------------------
shift_per_tx = LINE_SPACING_DEG / dth_deg
centre = (N_TX - 1) / 2.0
untrunc = np.zeros_like(beam)
trunc = np.zeros_like(beam)
half_px = REGION_HALF_DEG / dth_deg

col = np.arange(N_TH)
for k in range(N_TX):
    s = int(round((k - centre) * shift_per_tx))
    shifted = np.zeros_like(beam)
    if s >= 0:
        shifted[:, s:] = beam[:, : N_TH - s]
    else:
        shifted[:, :N_TH + s] = beam[:, -s:]
    untrunc += shifted
    # this transmit only reconstructs pixels inside its own region
    inside = np.abs(col - (pk_col + s)) <= half_px
    trunc += shifted * inside


def ripple(profile_2d, label):
    """Angular ripple of the depth-averaged, per-radius-normalised map."""
    pol = profile_2d / (np.median(profile_2d, axis=1, keepdims=True) + 1e-12)
    prof1 = pol.mean(axis=0)
    trend = uniform_filter1d(prof1, int(round(10.0 / dth_deg)) | 1, mode="nearest")
    rip = (prof1 - trend) / (trend + 1e-12)
    # restrict to the well-covered central angles, away from the sector edges
    keep = np.abs(np.degrees(th)) < 25
    rip = rip[keep]
    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth_deg)
    tot = spec[(freq > 0.05) & (freq < 5)].sum()
    fl = 1.0 / LINE_SPACING_DEG
    frac = spec[(freq > fl * 0.9) & (freq < fl * 1.1)].sum() / max(tot, 1e-30)
    sel = (freq > 0.3) & (freq < 2.0)
    peak = 1.0 / freq[sel][np.argmax(spec[sel])]
    amp = rip.std() * 100
    print(f"{label:34s} ripple {amp:7.3f}%   at line spacing {np.sqrt(frac) * amp:7.3f}%   "
          f"peak {peak:6.3f} deg")
    return amp


print()
ripple(untrunc, "untruncated compound (free field)")
ripple(trunc, "REGION-TRUNCATED compound")
print()

# How much signal does the truncation actually discard, and where?
kept = trunc.sum() / untrunc.sum()
print(f"region truncation keeps {kept:.1%} of the total compounded energy")
for zt in (50, 60, 70, 80, 90, 100):
    i = int(np.argmin(np.abs(r + apex_mm - zt)))
    print(f"  depth {r[i] + apex_mm:5.1f} mm: truncated/untruncated = "
          f"{trunc[i].sum() / untrunc[i].sum():.3f}")

np.save(os.path.join(HERE, "region_truncation_maps.npy"),
        {"untrunc": untrunc, "trunc": trunc, "th_deg": np.degrees(th), "r_mm": r},
        allow_pickle=True)
