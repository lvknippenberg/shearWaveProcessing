"""Map the buffer-3 M-lines onto buffer 4 (motion correction between beats) and check the mapping.

The passive M-lines of the 15 labelled windows were redrawn on the phase-matched focused-beam frame
(buffer 3, ``draw_labelled_mlines_b3.py``). That frame is 2-3 heartbeats away from the buffer-4
event, and the anatomy moved in between (median 3 mm, ``mline_difference_check.py``). Here each
line is moved onto buffer 4 with :func:`swp.mline.transfer.transfer_line`: a local translation
between the buffer-3 frame and the buffer-4 mean envelope around the event, estimated at anatomy
scale over an ensemble (box margins 8/12/16 mm x buffer-4 averaging windows +/-5/10/20 ms) and
applied as the ensemble median. Rotation is not modelled by default: in vivo the correlation is
nearly flat in the angle (a free rotation search ran to the +/-10 deg limit in half the windows and
dragged the translation with it); the regularised rotation variant is reported for comparison
(``rot_*`` columns). A mapping is ``reliable`` when >= 60 % of the ensemble puts the line within
1 mm of the consensus (``agree``) AND known shifts of the buffer-3 frame are recovered within 1 mm
(``known_err_mm``, computed by ``transfer_line`` itself).

Checks
  phantom     static resolution phantom: buffer 3 (standard, REFoCUS) -> buffer 4 for lines at
              several depths must give ~0 motion, and a known motion applied to the buffer-3 image
              must be recovered.
  known       in vivo, per window: the buffer-3 frame is moved by a known transform and the moved
              line mapped again - it must land where the unmoved line did (cross-modality precision
              on real cardiac images; ``known_err_mm``).
  agreement   the old buffer-1 line is mapped onto buffer 4 the same way. If both lines trace the
              same septum and their difference is motion, they come closer after mapping
              (``dist_b1_b3`` -> ``dist_mapped``).
  coherence   buffer-4 slow-time lag-1 coherence along each line (+/-1 mm): higher = more of the
              line sits in coherently moving tissue in the data that is actually processed.

Writes ``<folder>/output/mlines/passive_win<i>_mline_b3to4.npz`` (+ ``passive_mlines_b3to4.json``
with the transform and checks), ``study/logs/map_mlines_b3_to_b4.csv`` and the montage
``study/montages/map_mlines_b3_to_b4.png``.

    python study/analysis/map_mlines_b3_to_b4.py             phantom check + all windows
    python study/analysis/map_mlines_b3_to_b4.py --phantom   phantom check only
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402
from swp.mline.transfer import Transform, transfer_line, warp  # noqa: E402
from mline_difference_check import frame_env, line_distance    # noqa: E402

CONFIG = str(_REPO / "configs" / "passive.yaml")
B4_HALF_MS = (5.0, 10.0, 20.0)
COH_HALF_MS = 10.0
KNOWN = (Transform(dx=2.0, dz=-1.5), Transform(dx=-3.0, dz=2.0), Transform(dx=1.0, dz=1.0, angle=3.0))
ROT_KNOWN = KNOWN[2]         # in vivo: unmodelled-rotation tolerance (the shifts are checked by transfer_line)
ROT_ANGLES = np.arange(-10.0, 10.01, 1.0)
PHANTOM = Path(r"D:/swp_res/Resolution phantom/DefaultPatient_SW_data_18-June-2026_13-52-51/output")


def with_centre(t, c):
    return Transform(dx=t.dx, dz=t.dz, angle=t.angle, cx=float(c[0]), cz=float(c[1]))


def along(img, x, z, L):
    """Mean of img along line L (x, z mm) over a +/-1 mm band in depth."""
    f = RegularGridInterpolator((z, x), img, bounds_error=False, fill_value=np.nan)
    return float(np.nanmean([f(np.c_[L[:, 1] + d, L[:, 0]]) for d in (-1.0, 0.0, 1.0)]))


# ---------------------------------------------------------------------------------------------
def phantom_check():
    """Static phantom: buffer 3 -> buffer 4 must be ~identity; known motions must be recovered."""
    def mean_env(name):
        import h5py
        with h5py.File(PHANTOM / name) as h:
            g = h["tracks/track_0/data/beamformed_data"]
            v = np.asarray(g["values"], np.float64)
            co = np.asarray(g["coordinates"])
        return np.hypot(v[..., 0], v[..., 1]).mean(axis=0), co[0, :, 0] * 1e3, co[:, 0, -1] * 1e3
    b4 = mean_env("CombinedData_buffer4_iq.hdf5")
    rows = []
    for lab, fn in (("b3 standard", "CombinedData_buffer3_iq.hdf5"),
                    ("b3 REFoCUS", "CombinedData_buffer3_refocus-adjoint_iq.hdf5")):
        b3 = mean_env(fn)
        for zc in (45.0, 70.0, 95.0):
            L = np.c_[np.linspace(-12, 12, 40), zc + 0.25 * np.linspace(-12, 12, 40)]
            c = L.mean(axis=0)
            for kn in (Transform(),) + KNOWN:
                kt = with_centre(kn, c)
                src = (warp(*b3, kt), b3[1], b3[2]) if kn.dx or kn.dz or kn.angle else b3
                res = transfer_line(kt.apply(L), src, b4, angles=ROT_ANGLES if kn.angle else (0.0,), check=False)
                err = np.hypot(*(res.points - L).T)          # must come back to L (b4 = static b3)
                rows.append(dict(buffer=lab, depth_mm=zc, known=f"({kn.dx:+.1f},{kn.dz:+.1f},{kn.angle:+.0f}deg)",
                                 err_median_mm=float(np.median(err)), err_max_mm=float(err.max()),
                                 est_dx=res.transform.dx, est_dz=res.transform.dz, est_angle=res.transform.angle,
                                 corr=res.transform.corr, spread_mm=res.spread_mm))
                r = rows[-1]
                print(f"phantom {lab:12s} z={zc:.0f} applied {r['known']:>20s}: line error median {r['err_median_mm']:.2f} "
                      f"max {r['err_max_mm']:.2f} mm (est {res.transform.dx:+.2f},{res.transform.dz:+.2f},"
                      f"{res.transform.angle:+.2f}deg, corr {res.transform.corr:.2f}, spread {res.spread_mm:.2f})", flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
def invivo():
    import swp.passive as SP
    from swp.viz.mline import mline_from_points
    from passive_mline_split import split_line
    rows, figdata = [], []
    for c in sorted(json.load(open(_REPO / "study/logs/labelled_panels.json")), key=lambda c: c["subject"]):
        folder = f"{P.RAW_DATA}/{c['folder']}"
        cfg, p = SP._paths(folder, CONFIG)
        n = cfg["mline"].get("n_samples", 250)
        st, ws = SP.read_windows(p["windows_json"])
        win = int(c["window"]); w = ws[win]
        b3npz = os.path.join(p["mlines"], f"passive_win{win}_mline_b3.npz")
        if not os.path.exists(b3npz):
            continue
        old_full = SP._load_line(SP._window_npz(p["mlines"], win), n)
        old = old_full if c["part"] == "full" else mline_from_points(split_line(old_full, n)[c["part"]], n)
        new = SP._load_line(b3npz, n)
        A = np.c_[old.x, old.z] * 1e3                 # buffer-1 line (analysed segment)
        B = np.c_[new.x, new.z] * 1e3                 # buffer-3 line
        rec_path = os.path.join(p["mlines"], "passive_mlines_b3.json")
        rec = json.load(open(rec_path))[str(win)]
        f1 = SP.event_bmode_frames(folder, [w], buffer=1)[0]
        e1 = frame_env(os.path.join(p["output"], SP.bmode_file(1)), f1["frame"])
        e3 = frame_env(os.path.join(p["output"], SP.bmode_file(3)), rec["frame"])
        acq = SP.load_acq(folder, CONFIG)
        x4, z4 = acq.x * 1e3, acq.z * 1e3
        b4 = []
        for h in B4_HALF_MS:
            sel = np.abs(acq.t - w.t_peak) <= h * 1e-3
            b4.append((np.abs(acq.iq[sel]).mean(axis=0), x4, z4))
        sel = np.abs(acq.t - w.t_peak) <= COH_HALF_MS * 1e-3
        iq = acq.iq[sel]
        pc = np.conj(iq[:-1]) * iq[1:]
        coh = np.abs(pc.mean(axis=0)) / (np.abs(pc).mean(axis=0) + 1e-20)
        del acq, iq, pc

        r3 = transfer_line(B, e3, b4)                              # the mapping (translation)
        r3r = transfer_line(B, e3, b4, angles=ROT_ANGLES, check=False)   # + regularised rotation
        r1 = transfer_line(A, e1, b4, check=False)                 # old line, same treatment
        # known-motion test on the real images: move the b3 frame and its line, map again
        kt = with_centre(ROT_KNOWN, B.mean(axis=0))
        rk = transfer_line(kt.apply(B), (warp(*e3, kt), e3[1], e3[2]), b4, check=False)
        kerr_rot = float(np.median(np.hypot(*(rk.points - r3.points).T)))
        B4 = r3.points
        d_raw = line_distance(A, B)
        d_map = line_distance(r1.points, B4)
        t = r3.transform
        r = dict(subject=c["subject"], window=win, event=c["label"],
                 dx_mm=t.dx, dz_mm=t.dz, shift_mm=r3.shift_mm,
                 corr_before=t.corr0, corr_after=t.corr, spread_mm=r3.spread_mm, agree=r3.agree,
                 spread_rms_mm=r3.spread_rms_mm, reliable=r3.reliable(),
                 known_err_mm=r3.known_err_mm, known_err_rot_mm=kerr_rot,
                 rot_angle_deg=r3r.transform.angle, rot_corr=r3r.transform.corr, rot_spread_mm=r3r.spread_mm,
                 rot_effect_mm=float(np.median(np.hypot(*(r3.points - r3r.points).T))),
                 b1_shift_mm=r1.shift_mm, b1_spread_mm=r1.spread_mm,
                 dist_b1_b3_mm=d_raw[0], overlap_raw=d_raw[1], dist_mapped_mm=d_map[0], overlap_mapped=d_map[1],
                 coh_b1=along(coh, x4, z4, A), coh_b3=along(coh, x4, z4, B),
                 coh_b3to4=along(coh, x4, z4, B4), coh_b1to4=along(coh, x4, z4, r1.points))
        rows.append(r)
        dest = os.path.join(p["mlines"], f"passive_win{win}_mline_b3to4.npz")
        with open(dest + ".tmp", "wb") as fh:            # atomic: readers never see a partial file
            np.savez(fh, points=B4 * 1e-3, n_samples=n)
        os.replace(dest + ".tmp", dest)
        mp = os.path.join(p["mlines"], "passive_mlines_b3to4.json")
        allrec = json.load(open(mp)) if os.path.exists(mp) else {}
        allrec[str(win)] = dict(source="passive_win%d_mline_b3.npz" % win, b3_frame=rec["frame"],
                                b4_half_ms=list(B4_HALF_MS), transform=dict(dx_mm=t.dx, dz_mm=t.dz, angle_deg=t.angle,
                                                                           centre_mm=[t.cx, t.cz]),
                                corr_before=t.corr0, corr_after=t.corr, spread_mm=r3.spread_mm,
                                agree=r3.agree, reliable=r3.reliable(), known_err_mm=r["known_err_mm"])
        with open(mp, "w") as fh:
            json.dump(allrec, fh, indent=1)
        figdata.append((r, b4[1], coh, x4, z4, e3, A, B, B4, r1.points))
        print(f"{c['subject']} w{win} {c['label']}: b3->b4 ({t.dx:+.1f},{t.dz:+.1f}) mm "
              f"(corr {t.corr0:.2f}->{t.corr:.2f}, spread {r3.spread_mm:.2f} mm, agree {r3.agree:.0%}"
              f"{'' if r3.reliable() else ' UNRELIABLE'}, "
              f"known-shift err {r['known_err_mm']:.2f} mm (unmodelled +3 deg: {kerr_rot:.2f}); regularised rotation "
              f"{r3r.transform.angle:+.1f} deg moves the line {r['rot_effect_mm']:.1f} mm); "
              f"b1/b3 lines {d_raw[0]:.1f} -> {d_map[0]:.1f} mm apart; "
              f"coherence b1 {r['coh_b1']:.2f} b3 {r['coh_b3']:.2f} b3to4 {r['coh_b3to4']:.2f} b1to4 {r['coh_b1to4']:.2f}",
              flush=True)
    return rows, figdata


def montage(figdata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(figdata)
    fig, axes = plt.subplots(3, n, figsize=(2.3 * n, 7.4), squeeze=False)
    for j, (r, b4, coh, x4, z4, e3, A, B, B4, A4) in enumerate(figdata):
        allp = np.r_[A, B, B4]
        lo, hi = allp.min(0) - 10, allp.max(0) + 10
        panels = ((e3[0], e3[1], e3[2], "buffer 3 frame (drawn)", "gray"),
                  (b4[0], b4[1], b4[2], "buffer 4 mean +/-10 ms", "gray"),
                  (coh, x4, z4, "buffer 4 coherence", "viridis"))
        for i, (img, x, z, lab, cm) in enumerate(panels):
            ax = axes[i, j]
            show = 20 * np.log10(img / np.nanmax(img) + 1e-6) if cm == "gray" else img
            ax.imshow(show, cmap=cm, extent=(x[0], x[-1], z[-1], z[0]), aspect="auto",
                      vmin=(-45 if cm == "gray" else 0), vmax=(0 if cm == "gray" else 1))
            ax.plot(B[:, 0], B[:, 1], "--", color="yellow", lw=1.0)
            if i > 0:
                ax.plot(B4[:, 0], B4[:, 1], "-", color="red", lw=1.3)
                ax.plot(A[:, 0], A[:, 1], ":", color="cyan", lw=1.0)
            ax.set_xlim(lo[0], hi[0]); ax.set_ylim(hi[1], lo[1])
            ax.tick_params(labelsize=4)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6)
        axes[0, j].set_title(f"{r['subject'][-3:]} {r['event']}{'' if r['reliable'] else ' (UNRELIABLE)'}\n"
                             f"({r['dx_mm']:+.1f},{r['dz_mm']:+.1f}) mm, spread {r['spread_mm']:.1f}\ncorr {r['corr_before']:.2f}->{r['corr_after']:.2f}, "
                             f"coh {r['coh_b3']:.2f}->{r['coh_b3to4']:.2f}", fontsize=5.5)
    fig.suptitle("yellow dashed: line drawn on buffer 3; red: mapped onto buffer 4; cyan dotted: old buffer-1 line",
                 fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fp = _REPO / "study/montages/map_mlines_b3_to_b4.png"
    fig.savefig(fp, dpi=130)
    return fp


def main():
    from swp.provenance import stamp_text
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantom", action="store_true", help="phantom check only")
    a = ap.parse_args()
    cfgd = dict(b4_half_ms=B4_HALF_MS, coh_half_ms=COH_HALF_MS, margins_mm=(8, 12, 16), angles_deg="0 (rot variant -10..10, tol 0.01)",
                known=[(k.dx, k.dz, k.angle) for k in KNOWN])
    prow = phantom_check()
    with open(_REPO / "study/logs/map_mlines_b3_to_b4_phantom.csv", "w", newline="") as fh:
        fh.write(stamp_text(config=cfgd))
        wr = csv.DictWriter(fh, fieldnames=list(prow[0])); wr.writeheader(); wr.writerows(prow)
    print(f"phantom: line error median over all cases {np.median([r['err_median_mm'] for r in prow]):.2f} mm, "
          f"worst max {max(r['err_max_mm'] for r in prow):.2f} mm")
    if a.phantom:
        return
    rows, figdata = invivo()
    out = _REPO / "study/logs/map_mlines_b3_to_b4.csv"
    with open(out, "w", newline="") as fh:
        fh.write(stamp_text(config=cfgd))
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    med = lambda k: float(np.nanmedian([r[k] for r in rows]))
    print(f"\nreliable (>= 60 % of the ensemble within 1 mm): {sum(r['reliable'] for r in rows)}/{len(rows)}")
    print(f"medians over {len(rows)} windows: shift {med('shift_mm'):.1f} mm, regularised |angle| "
          f"{np.median([abs(r['rot_angle_deg']) for r in rows]):.1f} deg, corr {med('corr_before'):.2f}->{med('corr_after'):.2f}, "
          f"spread {med('spread_mm'):.2f} mm, known-motion error {med('known_err_mm'):.2f} mm; "
          f"b1/b3 line distance {med('dist_b1_b3_mm'):.1f} -> {med('dist_mapped_mm'):.1f} mm after mapping both; "
          f"coherence b3 {med('coh_b3'):.2f} -> b3to4 {med('coh_b3to4'):.2f} (higher in "
          f"{np.mean([r['coh_b3to4'] > r['coh_b3'] for r in rows]):.0%}), b1 {med('coh_b1'):.2f} -> b1to4 {med('coh_b1to4'):.2f}")
    print("wrote", out, "and", montage(figdata))


if __name__ == "__main__":
    main()
