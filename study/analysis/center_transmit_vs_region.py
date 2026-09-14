"""Compare the simulated centre-beam transmit field with the pixel region it reconstructs.

The user's `CenterTransmit.mat` holds the Verasonics-simulated transmit field for the centre beam
(region 37 of 73) and the `TransmitPData` that defines which pixels that transmit is used to
reconstruct. The question it settles: near the focus, is the beam narrower than its own
reconstruction region?

That is a sharper version of the "too focused" hypothesis than the F-number argument, because it
compares the beam against the REGION rather than against the line spacing - and the region turns
out to be far wider than the spacing.
"""
import os

import numpy as np
import scipy.io as sio

HERE = os.path.dirname(os.path.abspath(__file__))
MAT = os.path.join(os.path.dirname(HERE), "CenterTransmit.mat")
LINE_SPACING_DEG = 1.1111

m = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)
P, prof = m["TransmitPData"], m["TransmitProfile"].astype(np.float64)
nz, nx = int(P.Size[0]), int(P.Size[1])
dz, dx = float(P.PDelta[2]), float(P.PDelta[0])
x0, z0 = float(P.Origin[0]), float(P.Origin[2])

# Verasonics S5-1: PData is in wavelengths of Trans.frequency.
F_TRANS_MHZ = 3.125
C = 1540.0
LAM_MM = C / (F_TRANS_MHZ * 1e6) * 1e3
print(f"grid {nz} x {nx}, PDelta {dz} lambda, lambda = {LAM_MM:.4f} mm "
      f"(Trans.frequency {F_TRANS_MHZ} MHz)")

x_lam = x0 + np.arange(nx) * dx
z_lam = z0 + np.arange(nz) * dz
x_mm, z_mm = x_lam * LAM_MM, z_lam * LAM_MM

reg = P.Region[36]
sh = reg.Shape
apex_z_lam = float(sh.Position[2])
half_angle = float(sh.angle) / 2.0
print(f"region 37: {sh.Name}, apex z = {apex_z_lam:.2f} lambda "
      f"({apex_z_lam * LAM_MM:.1f} mm), full angle = {np.degrees(sh.angle):.3f} deg "
      f"= {np.degrees(sh.angle) / LINE_SPACING_DEG:.2f} x the {LINE_SPACING_DEG} deg line spacing")

# MATLAB PixelsLA are 1-based, column-major linear indices into an (nz, nx) array.
mask = np.zeros(nz * nx, bool)
mask[np.asarray(reg.PixelsLA, int) - 1] = True
mask = mask.reshape((nx, nz)).T            # column-major -> (nz, nx)
print(f"region mask: {mask.sum()} px (numPixels says {reg.numPixels})")


def width_mm(row_vals, row_x_mm, drop_db):
    """-drop_db width of the profile in this row, in mm. None if it never rises that far."""
    pk = row_vals.max()
    if pk <= 0:
        return None, None
    thr = pk * 10 ** (-drop_db / 20.0)
    above = np.nonzero(row_vals >= thr)[0]
    if above.size < 2:
        return None, None
    return (row_x_mm[above[-1]] - row_x_mm[above[0]]), row_x_mm[int(np.argmax(row_vals))]


print()
print(f"{'depth':>7s} {'beam -6dB':>10s} {'beam -20dB':>11s} {'region':>9s} "
      f"{'region/beam':>12s} {'beam':>8s} {'region':>8s} {'in-beam':>9s}")
print(f"{'(mm)':>7s} {'(mm)':>10s} {'(mm)':>11s} {'(mm)':>9s} {'(-6dB)':>12s} "
      f"{'(deg)':>8s} {'(deg)':>8s} {'frac':>9s}")
print("-" * 84)

rows = []
for zt in (20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140):
    iz = int(np.argmin(np.abs(z_mm - zt)))
    if not mask[iz].any():
        continue
    w6, ctr = width_mm(prof[iz], x_mm, 6.0)
    w20, _ = width_mm(prof[iz], x_mm, 20.0)
    cols = np.nonzero(mask[iz])[0]
    reg_w = x_mm[cols[-1]] - x_mm[cols[0]]
    if w6 is None:
        continue
    # radius from the virtual apex, for converting mm -> deg
    r_mm = z_mm[iz] - apex_z_lam * LAM_MM
    beam_deg = np.degrees(2 * np.arctan(w6 / 2 / r_mm))
    reg_deg = np.degrees(2 * np.arctan(reg_w / 2 / r_mm))
    # what fraction of this row's region pixels sit inside the beam's -6 dB width
    lo, hi = ctr - w6 / 2, ctr + w6 / 2
    inb = np.mean((x_mm[cols] >= lo) & (x_mm[cols] <= hi))
    rows.append((z_mm[iz], w6, w20, reg_w, reg_w / w6, beam_deg, reg_deg, inb))
    print(f"{z_mm[iz]:7.1f} {w6:10.2f} {w20:11.2f} {reg_w:9.2f} {reg_w / w6:12.2f} "
          f"{beam_deg:8.3f} {reg_deg:8.3f} {inb:9.1%}")

print()
narrow = [r for r in rows if r[4] > 1]
if narrow:
    tight = min(rows, key=lambda r: r[1])
    print(f"narrowest beam: {tight[1]:.2f} mm at {tight[0]:.0f} mm depth, where the region is "
          f"{tight[3]:.2f} mm wide ({tight[4]:.1f}x)")
print(f"beam angular width is {np.median([r[5] for r in rows]):.3f} deg (median over depth) "
      f"vs {LINE_SPACING_DEG} deg line spacing")
np.save(os.path.join(HERE, "center_transmit_rows.npy"), np.array(rows))
