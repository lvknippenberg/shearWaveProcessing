"""Does the simulated beam's own profile predict the nearest-1 ripple?

nearest-1 assigns each pixel to exactly ONE transmit, so there is no cross-transmit term of any
kind - coherent or incoherent. Whatever ripple remains must come from the beam being non-uniform
ACROSS ITS OWN STRIP: a pixel at the strip edge is 0.556 deg off the beam axis and is therefore
insonified more weakly than one on axis, and that pattern repeats every 1.111 deg.

That is a phase-free, purely geometric prediction, and `CenterTransmit.mat` is enough to make it:
mosaic the simulated beam profile in +/-half_width strips and measure the resulting angular
ripple. Because the strips do not overlap, summing and mosaicking are the same operation, so the
prediction does not depend on how the beams would have combined.

Sweeping the strip half-width from the nearest-1 case (0.556 deg) out to the Verasonics region
(3.333 deg) also shows how much of the artefact is single-beam non-uniformity and how much needs
compounding to explain.

Measured for comparison (apex-referenced, phantom): nearest-1 = 7.41%, nearest-73 = 14.97%.
"""
import os

import numpy as np
import scipy.io as sio
from scipy.ndimage import map_coordinates, uniform_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(os.path.dirname(HERE), "CenterTransmit.mat")
LINE_SPACING_DEG = 1.1111
N_TX = 73
F_TRANS_MHZ, C = 3.125, 1540.0
LAM_MM = C / (F_TRANS_MHZ * 1e6) * 1e3

m = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
P, prof = m["TransmitPData"], m["TransmitProfile"].astype(np.float64)
nz, nx = int(P.Size[0]), int(P.Size[1])
x_mm = (float(P.Origin[0]) + np.arange(nx) * float(P.PDelta[0])) * LAM_MM
z_mm = (float(P.Origin[2]) + np.arange(nz) * float(P.PDelta[2])) * LAM_MM
apex_mm = float(P.Region[36].Shape.Position[2]) * LAM_MM

R_LO, R_HI, N_R = 45.0, 105.0, 240
TH_MAX, N_TH = 40.0, 6401
r = np.linspace(R_LO, R_HI, N_R)
th = np.radians(np.linspace(-TH_MAX, TH_MAX, N_TH))
RR, TT = np.meshgrid(r, th, indexing="ij")
ii = (RR * np.cos(TT) + apex_mm - z_mm[0]) / (z_mm[1] - z_mm[0])
jj = (RR * np.sin(TT) - x_mm[0]) / (x_mm[1] - x_mm[0])
beam = map_coordinates(prof, [ii.ravel(), jj.ravel()], order=1, mode="constant",
                       cval=0.0).reshape(RR.shape)
dth = np.degrees(th[1] - th[0])
pk_col = int(np.argmax(beam.mean(axis=0)))
print(f"polar beam {beam.shape} about the apex, {dth:.4f} deg/sample")

# The profile is an INTENSITY-like map; the image envelope goes as amplitude. Report both so the
# conclusion cannot hinge on the convention.
for as_amp, lab_conv in ((False, "profile as amplitude"), (True, "profile as intensity -> sqrt")):
    b = np.sqrt(np.maximum(beam, 0)) if as_amp else beam
    print(f"\n--- {lab_conv} ---")
    print(f"{'strip half-width':26s} {'ripple @ pitch':>15s} {'peak':>9s} {'rms':>8s}")
    print("-" * 62)
    for half_deg in (LINE_SPACING_DEG / 2, 1.111, 1.667, 2.222, 3.333, None):
        acc = np.zeros_like(b)
        col = np.arange(N_TH)
        for k in range(N_TX):
            s = int(round((k - (N_TX - 1) / 2) * LINE_SPACING_DEG / dth))
            sh = np.zeros_like(b)
            if s >= 0:
                sh[:, s:] = b[:, : N_TH - s]
            else:
                sh[:, :N_TH + s] = b[:, -s:]
            if half_deg is not None:
                sh = sh * (np.abs(col - (pk_col + s)) <= half_deg / dth)
            acc += sh
        pol = acc / (np.median(acc, axis=1, keepdims=True) + 1e-12)
        p1 = pol.mean(axis=0)
        trend = uniform_filter1d(p1, int(round(10.0 / dth)) | 1, mode="nearest")
        rip = (p1 - trend) / (trend + 1e-12)
        keep = np.abs(np.degrees(th)) < 25
        rip = rip[keep]
        spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
        f = np.fft.rfftfreq(rip.size, d=dth)
        tot = spec[(f > 0.05) & (f < 5)].sum()
        fl = 1.0 / LINE_SPACING_DEG
        frac = spec[(f > fl * 0.9) & (f < fl * 1.1)].sum() / max(tot, 1e-30)
        sel = (f > 0.3) & (f < 2.5)
        rms = rip.std() * 100
        name = "untruncated (all overlap)" if half_deg is None else (
            f"+/-{half_deg:.3f} deg" + ("  (nearest-1)" if abs(half_deg - 0.5556) < 1e-3 else
                                        ("  (region rule)" if half_deg > 3 else "")))
        print(f"{name:26s} {np.sqrt(frac) * rms:14.2f}% {1.0 / f[sel][np.argmax(spec[sel])]:8.3f}d "
              f"{rms:7.2f}%")

print("\nmeasured on the phantom, apex-referenced: nearest-1 = 7.41%, nearest-73 = 14.97%")
