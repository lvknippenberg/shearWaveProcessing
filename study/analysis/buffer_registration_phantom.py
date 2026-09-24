"""Do the B-mode buffers put the same point at the same (x, z)? Wire targets in the resolution phantom.

Passive M-lines are drawn on buffer 1 (widebeam) or buffer 3 (focused) and then used to sample
buffer 4 (diverging waves). That is only valid if the three reconstructions map a point in the
tissue to the same image coordinates. In vivo this cannot be tested cleanly - the buffers are
acquired seconds apart, in different heartbeats. A static phantom removes all motion, so any
offset between buffers here is a reconstruction / geometry offset.

Method: frame-averaged envelope in dB per buffer; point targets = local maxima >= 12 dB above the
local median background, 20-120 mm deep; sub-pixel position by 3-point parabolic interpolation in
x and z. Targets are matched to the nearest buffer-4 target within 2 mm (all pairs are plotted;
the scattered +/-0.5-2 mm pairs are speckle/sidelobe mismatches). The statistics use the wires
only: both targets >= 20 dB above background and matched within 1 mm. Reported per buffer:
median offset (dx, dz) relative to buffer 4 with its spread, and linear fits of the offset
against position - a constant offset means a fixed shift, a slope means a scale error (e.g. sound
speed or apex position), a dx-vs-z slope a rotation/steering error.

    python study/analysis/buffer_registration_phantom.py
-> study/logs/buffer_registration_phantom.csv, study/montages/buffer_registration_phantom.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

F = Path(r"D:/swp_res/Resolution phantom/DefaultPatient_SW_data_18-June-2026_13-52-51/output")
BUFFERS = [("buffer 4 (diverging, passive SWE)", "CombinedData_buffer4_iq.hdf5"),
           ("buffer 1 (widebeam)", "CombinedData_buffer1_iq.hdf5"),
           ("buffer 3 (focused, standard)", "CombinedData_buffer3_iq.hdf5"),
           ("buffer 3 (focused, REFoCUS adjoint)", "CombinedData_buffer3_refocus-adjoint_iq.hdf5")]
PROMINENCE_DB, Z_RANGE, MATCH_MM = 12.0, (20.0, 120.0), 2.0
STRONG_DB, STRONG_MATCH_MM = 20.0, 1.0   # wires only: >= 20 dB above background, matched within 1 mm


def load_db(name):
    with h5py.File(F / name) as h:
        g = h["tracks/track_0/data/beamformed_data"]
        v = np.asarray(g["values"], np.float64)
        co = np.asarray(g["coordinates"])
    env = np.sqrt(v[..., 0] ** 2 + v[..., 1] ** 2).mean(axis=0)
    db = 20 * np.log10(env / env.max() + 1e-9)
    return db, co[0, :, 0] * 1e3, co[:, 0, -1] * 1e3


def targets(db, x, z):
    """(x, z) mm of point targets, sub-pixel."""
    dx, dz = np.median(np.diff(x)), np.median(np.diff(z))
    size = (int(round(3 / dz)) | 1, int(round(3 / dx)) | 1)
    peak = maximum_filter(db, size=size) == db
    bg = uniform_filter(db, size=(int(15 / dz), int(15 / dx)))
    zz = z[:, None] * np.ones_like(db)
    ok = peak & (db - bg > PROMINENCE_DB) & (zz > Z_RANGE[0]) & (zz < Z_RANGE[1]) & (db > -60)
    out = []
    for iz, ix in zip(*np.nonzero(ok)):
        if not (0 < iz < db.shape[0] - 1 and 0 < ix < db.shape[1] - 1):
            continue
        def para(a, b, c):
            den = a - 2 * b + c
            return 0.0 if den == 0 else 0.5 * (a - c) / den
        oz = para(db[iz - 1, ix], db[iz, ix], db[iz + 1, ix])
        ox = para(db[iz, ix - 1], db[iz, ix], db[iz, ix + 1])
        out.append((x[ix] + ox * dx, z[iz] + oz * dz, db[iz, ix] - bg[iz, ix]))
    return np.array(out)


def main():
    from swp.provenance import stamp_text
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = {lab: load_db(f) for lab, f in BUFFERS}
    tg = {lab: targets(*data[lab]) for lab in data}
    ref_lab = BUFFERS[0][0]
    ref = tg[ref_lab]
    print(f"targets found: " + ", ".join(f"{lab.split(' (')[0]}{' ' + lab.split('(')[1][:-1] if '(' in lab else ''}: {len(t)}"
                                         for lab, t in tg.items()))
    rows, pairs = [], {}
    for lab, t in list(tg.items())[1:]:
        m, strong = [], []
        for xr, zr, pr in ref:
            d = np.hypot(t[:, 0] - xr, t[:, 1] - zr)
            k = int(np.argmin(d))
            if d[k] <= MATCH_MM:
                m.append((xr, zr, t[k, 0] - xr, t[k, 1] - zr))
                if d[k] <= STRONG_MATCH_MM and pr >= STRONG_DB and t[k, 2] >= STRONG_DB:
                    strong.append((xr, zr, t[k, 0] - xr, t[k, 1] - zr))
        pairs[lab] = np.array(m)
        m = np.array(strong)                  # statistics on the wires only
        if len(m) < 3:
            rows.append(dict(buffer=lab, n=len(m)))
            continue
        ddx, ddz = m[:, 2] * 1e3, m[:, 3] * 1e3               # um
        sz = np.polyfit(m[:, 1], m[:, 3], 1)                  # dz vs z
        sx = np.polyfit(m[:, 0], m[:, 2], 1)                  # dx vs x
        rx = np.polyfit(m[:, 1], m[:, 2], 1)                  # dx vs z (rotation)
        rows.append(dict(buffer=lab, n=len(m),
                         dx_median_um=np.median(ddx), dx_iqr_um=np.subtract(*np.percentile(ddx, [75, 25])),
                         dz_median_um=np.median(ddz), dz_iqr_um=np.subtract(*np.percentile(ddz, [75, 25])),
                         dz_per_depth_pct=100 * sz[0], dx_per_lateral_pct=100 * sx[0],
                         dx_per_depth_um_per_cm=1e4 * rx[0]))
    out = _REPO / "study/logs/buffer_registration_phantom.csv"
    with open(out, "w", newline="") as fh:
        fh.write(stamp_text(config=dict(folder=str(F), prominence_db=PROMINENCE_DB, match_mm=MATCH_MM,
                                        strong_db=STRONG_DB, strong_match_mm=STRONG_MATCH_MM)))
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}, key=list(rows[-1]).index
                                                 if len(rows[-1]) > 2 else None))
        w.writeheader(); w.writerows(rows)
    print(f"\noffset of each buffer relative to {ref_lab} (wires >= {STRONG_DB:.0f} dB, matched within "
          f"{STRONG_MATCH_MM} mm; static phantom):")
    print(f"{'buffer':<38}{'n':>4}{'dx median [um]':>16}{'IQR':>7}{'dz median [um]':>16}{'IQR':>7}"
          f"{'dz/z [%]':>10}{'dx/x [%]':>10}{'dx/z [um/cm]':>14}")
    for r in rows:
        if "dx_median_um" not in r:
            print(f"{r['buffer']:<38}{r['n']:4d}  too few matches"); continue
        print(f"{r['buffer']:<38}{r['n']:4d}{r['dx_median_um']:16.0f}{r['dx_iqr_um']:7.0f}{r['dz_median_um']:16.0f}"
              f"{r['dz_iqr_um']:7.0f}{r['dz_per_depth_pct']:10.2f}{r['dx_per_lateral_pct']:10.2f}{r['dx_per_depth_um_per_cm']:14.0f}")

    # figure: buffer-4 image with every buffer's targets, and offsets vs depth
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
    db, x, z = data[ref_lab]
    ax[0].imshow(db, cmap="gray", vmin=-60, vmax=0, extent=(x[0], x[-1], z[-1], z[0]), aspect="auto")
    marks = ["o", "x", "+", "s"]
    for (lab, t), mk in zip(tg.items(), marks):
        ax[0].plot(t[:, 0], t[:, 1], mk, mfc="none", ms=6, label=lab)
    ax[0].set_xlim(-45, 45); ax[0].set_ylim(Z_RANGE[1] + 5, Z_RANGE[0] - 5)
    ax[0].legend(fontsize=6); ax[0].set_xlabel("x [mm]"); ax[0].set_ylabel("z [mm]")
    ax[0].set_title("targets per buffer on the buffer-4 image")
    for lab, m in pairs.items():
        if len(m):
            ax[1].plot(m[:, 1], m[:, 3] * 1e3, "o", ms=4, label=lab)
            ax[2].plot(m[:, 1], m[:, 2] * 1e3, "o", ms=4, label=lab)
    for a, t in ((ax[1], "axial offset dz vs buffer 4"), (ax[2], "lateral offset dx vs buffer 4")):
        a.axhline(0, color="k", lw=0.8); a.set_xlabel("depth z [mm]"); a.set_ylabel("offset [um]")
        a.axhspan(-197, 197, color="0.9", zorder=0, label="+/- half a pixel (197 um)")
        a.set_title(t); a.legend(fontsize=6); a.grid(alpha=0.3)
    fig.tight_layout()
    fp = _REPO / "study/montages/buffer_registration_phantom.png"
    fig.savefig(fp, dpi=120)
    print("wrote", out, "and", fp)


if __name__ == "__main__":
    main()
