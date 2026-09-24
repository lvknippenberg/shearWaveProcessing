"""Why do the buffer-3 M-lines differ from the buffer-1 ones: motion, or how the image is read?

For each of the 15 labelled passive windows, the old line (buffer-1 segment that was hand-labelled)
and the new one (``draw_labelled_mlines_b3.py``, buffer 3) are compared, and the difference is
split into its possible causes. Registration offsets between buffers were ruled out separately
(``buffer_registration_phantom.py``: < 35 um on phantom wires).

1. ``line_mm``      median perpendicular distance of the new line from the old one where they
                    overlap; ``overlap`` = fraction of the new line alongside the old one (the
                    rest covers another stretch of the septum); ``angle_deg`` between them.
2. ``shift_b1b3``   rigid shift of the anatomy between the two frames the lines were drawn on
                    (phase correlation of the log-envelope band-passed to 1-6 mm structure,
                    Hann-tapered, in a box around the lines; a known-shift self-test is run on
                    every window). ``line_after_mm`` = line distance after moving the old line by that
                    shift: what remains is interpretation / drawing, not anatomy motion.
3. ``shift_phase``  anatomy shift inside buffer 4 (one beat, one modality) between the event time
                    and the event time + the buffer-3 frame's phase offset: motion caused by the
                    phase mismatch alone.
4. ``shift_b1b4`` / ``shift_b3b4``  shift of each drawing frame relative to buffer 4 at the event
                    (different beats: buffer 1 is 1-2 beats earlier, buffer 3 2-3). Cross-modality,
                    so less precise; ``corr`` is the registration peak quality.
5. ``coh_b1`` / ``coh_b3``  buffer-4 slow-time lag-1 coherence around the event, averaged along
                    each line (+/- 1 mm): ~1 in moving tissue, lower in blood and noise. The line
                    with higher coherence sits more in myocardium in the data that is processed.

    python study/analysis/mline_difference_check.py
-> study/logs/mline_difference_check.csv, study/montages/mline_difference_check.png
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "study" / "analysis"))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402

CONFIG = str(_REPO / "configs" / "passive.yaml")
BOX_MM, SMOOTH_MM, B4_HALF_MS = 12.0, 1.0, 8.0


def frame_env(path, k):
    """(envelope (z, x), x mm, z mm) of frame k of a beamformed IQ file."""
    with h5py.File(path) as h:
        g = h["tracks/track_0/data/beamformed_data"]
        v = np.asarray(g["values"][int(k)], np.float64)
        co = np.asarray(g["coordinates"])
    return np.hypot(v[..., 0], v[..., 1]), co[0, :, 0] * 1e3, co[:, 0, -1] * 1e3


def on_grid(img, x, z, gx, gz):
    f = RegularGridInterpolator((z, x), img, bounds_error=False, fill_value=np.nan)
    Z, X = np.meshgrid(gz, gx, indexing="ij")
    return f(np.stack([Z, X], -1))


def prep(env):
    """log-envelope band-passed to anatomy scale (Gaussian 1 mm minus 6 mm: removes speckle AND the
    depth/gain trend that otherwise pins the registration at zero), standardised, Hann-tapered."""
    d = 20 * np.log10(env / np.nanmax(env) + 1e-6)
    d = np.where(np.isfinite(d), d, np.nanmedian(d))
    d = gaussian_filter(d, SMOOTH_MM / 0.394) - gaussian_filter(d, 6.0 / 0.394)
    d = (d - d.mean()) / (d.std() + 1e-12)
    return d * np.outer(np.hanning(d.shape[0]), np.hanning(d.shape[1]))


def register(ref, mov, pix):
    """(dx, dz) mm that moves `mov` onto `ref`, and the normalised correlation after the shift."""
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as nd_shift
    s, _, _ = phase_cross_correlation(ref, mov, upsample_factor=10, normalization=None)
    moved = nd_shift(mov, s, order=1, mode="nearest")
    corr = float(np.corrcoef(ref.ravel(), moved.ravel())[0, 1])
    return s[1] * pix, s[0] * pix, corr


def line_distance(a, b):
    """(median perpendicular distance [mm] of line b from line a where they overlap, overlap
    fraction of b, angle between them [deg]). A point of b overlaps when its nearest point on a is
    not one of a's ends - so a line drawn over a different stretch of the septum is not counted
    as a sideways offset."""
    d = np.hypot(b[:, None, 0] - a[None, :, 0], b[:, None, 1] - a[None, :, 1])
    k = d.argmin(axis=1)
    inner = (k > 0) & (k < len(a) - 1)
    perp = float(np.median(d.min(axis=1)[inner])) if inner.any() else float("nan")
    ang = lambda L: np.degrees(np.arctan2(L[-1, 1] - L[0, 1], L[-1, 0] - L[0, 0]))
    da = (ang(b) - ang(a) + 90) % 180 - 90
    return perp, float(inner.mean()), float(abs(da))


def sample(img, gx, gz, xs, zs):
    f = RegularGridInterpolator((gz, gx), img, bounds_error=False, fill_value=np.nan)
    return f(np.stack([zs, xs], -1))


def main():
    import swp.passive as SP
    from swp.viz.mline import mline_from_points
    from passive_mline_split import split_line
    from swp.provenance import stamp_text
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
        A = np.c_[old.x, old.z] * 1e3
        B = np.c_[new.x, new.z] * 1e3
        rec = json.load(open(os.path.join(p["mlines"], "passive_mlines_b3.json")))[str(win)]
        f1 = SP.event_bmode_frames(folder, [w], buffer=1)[0]
        # common grid: a box around both lines
        allp = np.r_[A, B]
        gx = np.arange(allp[:, 0].min() - BOX_MM, allp[:, 0].max() + BOX_MM, 0.394)
        gz = np.arange(allp[:, 1].min() - BOX_MM, allp[:, 1].max() + BOX_MM, 0.394)
        e1 = on_grid(*frame_env(os.path.join(p["output"], SP.bmode_file(1)), f1["frame"]), gx, gz)
        e3 = on_grid(*frame_env(os.path.join(p["output"], SP.bmode_file(3)), rec["frame"]), gx, gz)
        # buffer 4 around the event and around event + phase offset (same beat)
        acq = SP.load_acq(folder, CONFIG)
        off_s = (rec["frame_phase_ms"] - rec["event_phase_ms"]) * 1e-3
        def b4_mean(t_c):
            sel = np.abs(acq.t - t_c) <= B4_HALF_MS * 1e-3
            iq = acq.iq[sel]
            env = np.abs(iq).mean(axis=0)
            pcor = np.conj(iq[:-1]) * iq[1:]
            coh = np.abs(pcor.mean(axis=0)) / (np.abs(pcor).mean(axis=0) + 1e-20)
            x4, z4 = acq.x * 1e3, acq.z * 1e3
            return on_grid(env, x4, z4, gx, gz), on_grid(coh, x4, z4, gx, gz)
        e4, coh4 = b4_mean(w.t_peak)
        e4o, _ = b4_mean(w.t_peak + off_s)
        del acq
        P1, P3, P4, P4o = prep(e1), prep(e3), prep(e4), prep(e4o)
        s13 = register(P1, P3, 0.394)            # moves the b3 frame onto the b1 frame
        sph = register(P4, P4o, 0.394)           # phase-offset motion within buffer 4
        s14 = register(P4, P1, 0.394)
        s34 = register(P4, P3, 0.394)
        # old line moved like the anatomy: b1 -> b3 anatomy shift is minus s13
        A_moved = A - np.array([s13[0], s13[1]])
        # coherence along each line in buffer 4 (+/- 1 mm band via 3 offsets along z)
        def line_coh(L):
            vals = [sample(coh4, gx, gz, L[:, 0], L[:, 1] + dz) for dz in (-1.0, 0.0, 1.0)]
            return float(np.nanmean(vals))
        perp, overlap, angle = line_distance(A, B)
        perp_after, overlap_after, _ = line_distance(A_moved, B)
        # self-test: the registration must recover a known shift of the buffer-1 frame
        from scipy.ndimage import shift as nd_shift
        test = register(P1, prep(nd_shift(np.nan_to_num(e1, nan=np.nanmedian(e1)), (1.5 / 0.394, -2.0 / 0.394),
                                          order=1, mode="nearest")), 0.394)
        r = dict(subject=c["subject"], window=win, event=c["label"],
                 line_mm=perp, overlap=overlap, angle_deg=angle, line_after_mm=perp_after,
                 selftest_dx_mm=test[0], selftest_dz_mm=test[1],
                 shift_b1b3_mm=float(np.hypot(*s13[:2])), corr_b1b3=s13[2],
                 phase_offset_ms=rec["frame_phase_ms"] - rec["event_phase_ms"],
                 shift_phase_mm=float(np.hypot(*sph[:2])), corr_phase=sph[2],
                 shift_b1b4_mm=float(np.hypot(*s14[:2])), corr_b1b4=s14[2],
                 shift_b3b4_mm=float(np.hypot(*s34[:2])), corr_b3b4=s34[2],
                 coh_b1=line_coh(A), coh_b3=line_coh(B),
                 len_b1_mm=float(old.r[-1] * 1e3), len_b3_mm=float(new.r[-1] * 1e3))
        rows.append(r)
        figdata.append((r, gx, gz, e1, e3, coh4, A, B, A_moved))
        print(f"{c['subject']} w{win} {c['label']}: self-test ({test[0]:+.1f},{test[1]:+.1f}) for (+2.0,-1.5); overlap {overlap:.0%}, "
              f"angle {angle:.0f} deg; lines {r['line_mm']:.1f} mm apart -> {r['line_after_mm']:.1f} after anatomy "
              f"shift {r['shift_b1b3_mm']:.1f} mm (corr {s13[2]:.2f}); phase-only {r['shift_phase_mm']:.1f} mm; "
              f"b1->b4 {r['shift_b1b4_mm']:.1f}, b3->b4 {r['shift_b3b4_mm']:.1f} mm; coherence b1 {r['coh_b1']:.2f} b3 {r['coh_b3']:.2f}",
              flush=True)
    out = _REPO / "study/logs/mline_difference_check.csv"
    with open(out, "w", newline="") as fh:
        fh.write(stamp_text(config=dict(box_mm=BOX_MM, smooth_mm=SMOOTH_MM, b4_half_ms=B4_HALF_MS)))
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    med = lambda k: np.median([r[k] for r in rows])
    print(f"\nmedians over {len(rows)} windows: line distance {med('line_mm'):.1f} mm, after removing the anatomy "
          f"shift {med('line_after_mm'):.1f} mm; anatomy shift b1<->b3 {med('shift_b1b3_mm'):.1f} mm; "
          f"phase-only {med('shift_phase_mm'):.1f} mm; b1->b4 {med('shift_b1b4_mm'):.1f} mm, b3->b4 {med('shift_b3b4_mm'):.1f} mm; "
          f"b4 coherence along line: b1 {med('coh_b1'):.2f}, b3 {med('coh_b3'):.2f} "
          f"(b3 higher in {np.mean([r['coh_b3'] > r['coh_b1'] for r in rows]):.0%})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, len(figdata), figsize=(2.2 * len(figdata), 7.2), squeeze=False)
    for j, (r, gx, gz, e1, e3, coh4, A, B, Am) in enumerate(figdata):
        ext = (gx[0], gx[-1], gz[-1], gz[0])
        for i, (img, lab, cm) in enumerate(((e1, "buffer 1 frame", "gray"), (e3, "buffer 3 frame", "gray"),
                                            (coh4, "buffer 4 coherence @ event", "viridis"))):
            ax = axes[i, j]
            show = 20 * np.log10(img / np.nanmax(img) + 1e-6) if i < 2 else img
            ax.imshow(show, cmap=cm, extent=ext, aspect="auto", vmin=(-45 if i < 2 else 0), vmax=(0 if i < 2 else 1))
            ax.plot(A[:, 0], A[:, 1], "--", color="cyan", lw=1.1)
            ax.plot(B[:, 0], B[:, 1], "-", color="yellow", lw=1.3)
            if i == 0:
                ax.set_title(f"{r['subject'][-3:]} {r['event']}\n{r['line_mm']:.1f} -> {r['line_after_mm']:.1f} mm", fontsize=6)
            if j == 0:
                ax.set_ylabel(lab, fontsize=6)
            ax.tick_params(labelsize=4)
    fig.suptitle("cyan dashed: old line (buffer 1 segment); yellow: new line (buffer 3). "
                 "Title: line distance -> distance after removing the anatomy shift between the two frames", fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = _REPO / "study/montages/mline_difference_check.png"
    fig.savefig(fp, dpi=120)
    print("wrote", out, "and", fp)


if __name__ == "__main__":
    main()
