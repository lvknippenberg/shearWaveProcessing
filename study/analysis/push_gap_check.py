"""Measure the reference -> tracking interval from the data itself.

`sequence.assemble_tracking_frames` used to place the last reference frame one frame interval
before the first tracking frame, as if no push happened in between. The Verasonics sequence says
the real interval is PRI + the push burst rounded up to 100 us (`SWGeometry.push_gap_s`). This
checks which is true without trusting either: cardiac wall motion is smooth over ~2 ms, so the
axial displacement between the last reference frame and tracking frame 1 equals the local wall
velocity (measured frame-to-frame inside the reference block) times the elapsed time. A
least-squares slope of displacement against velocity, over pixels away from the push, is the
elapsed time.

Calibration: the same fit between two reference frames 6 frames apart must return 6 x PRI.
Tracking frame 0 is skipped (its pulse-inversion pair can be disturbed by the push).

    python study/analysis/push_gap_check.py            -> study/logs/push_gap_check.csv + stdout
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
from swp import paths as P                                   # noqa: E402
from swp.provenance import stamp_text                        # noqa: E402
from swp.viz.io import load_acquisition                      # noqa: E402
from swp.acquisition.sequence import SWGeometry              # noqa: E402

FOLDERS = (sorted(glob.glob(os.path.join(P.VOLTAGE_SWEEP, "Invivo", "Luuk*")))
           + sorted(glob.glob(os.path.join(P.INVIVO_0818, "Luuk_*"))))
KZ = 5                     # axial samples averaged in the complex correlation (~2 mm)
MIN_LATERAL_M = 8e-3       # ignore the push column: the ARF wave travels < 4 mm in 2 ms
MAX_V = 0.04               # m/s; above this the 1.7 ms displacement nears the 98.6 um wrap


def _phase_disp(a, b, c, f):
    """Axial displacement b relative to a (Kasai, complex average over KZ axial samples)."""
    from scipy.ndimage import uniform_filter1d
    p = np.conj(a) * b
    p = uniform_filter1d(p.real, KZ, axis=0) + 1j * uniform_filter1d(p.imag, KZ, axis=0)
    return c * np.angle(p) / (4 * np.pi * f), np.abs(p)


def gap_params(folder):
    """(PRI, push gap) in seconds - from the runtime .mat if present, else CombinedData.mat."""
    import scipy.io as sio
    runtime = os.path.join(folder, "AcquisitionParametersAndECG.mat")
    if not os.path.exists(runtime):
        from swp.acquisition.sequence import read_swi_meta
        g = read_swi_meta(os.path.join(folder, "CombinedData.mat")).sw
        return g.pri, g.push_gap_s()
    m = sio.loadmat(runtime, squeeze_me=True, struct_as_record=False)
    sw = m["SW"]
    g = SWGeometry(n_reference=int(sw.Ndetect_pre), n_tracking=int(sw.Ndetect), na=int(sw.na),
                   harmonic=bool(sw.HarmonicImaging), pri=float(sw.PRI_us) * 1e-6,
                   focus_x=0.0, focus_z=0.0, roi_xlims=(0, 0), roi_zlims=(0, 0),
                   push_cycles=float(sw.pushCycle), push_freq_hz=float(sw.PushFrequency) * 1e6,
                   switch_tpc=bool(getattr(sw, "SwitchTPCprofile", 0)))
    return g.pri, g.push_gap_s()


def slope(d, v, w):
    return float(np.sum(w * d * v) / np.sum(w * v * v))


K_FRAMES = range(1, 9)     # tracking frames used for the fixed-effects fit (wave not yet at 8 mm)


def centred_sums(d, v, w):
    """Weighted sums with this push's own mean removed (absorbs a global phase offset)."""
    sw = w.sum()
    if sw == 0:
        return 0.0, 0.0
    dm, vm = (w * d).sum() / sw, (w * v).sum() / sw
    return float((w * (d - dm) * (v - vm)).sum()), float((w * (v - vm) ** 2).sum())


def main():
    rows = []
    for folder in FOLDERS:
        pri, gap = gap_params(folder)
        pred_old, pred_new = 2 * pri, 2 * pri + gap
        for path in sorted(glob.glob(os.path.join(folder, "output", "CombinedData_buffer2_meas*_iq.hdf5")),
                           key=lambda p: int(p.split("meas")[-1].split("_")[0])):
            acq = load_acquisition(path)
            ref, iq, c, f = acq.ref_iq, acq.iq, acq.c, acq.f_demod
            # wall velocity at the end of the reference block (mean of the last 4 steps)
            steps = [_phase_disp(ref[i], ref[i + 1], c, f)[0] for i in range(ref.shape[0] - 5, ref.shape[0] - 1)]
            v = np.mean(steps, axis=0) / pri
            d_cross, mag = _phase_disp(ref[-1], iq[1], c, f)
            d_cal, _ = _phase_disp(ref[-7], ref[-1], c, f)
            v_cal = np.mean([_phase_disp(ref[i], ref[i + 1], c, f)[0]
                             for i in range(ref.shape[0] - 7, ref.shape[0] - 1)], axis=0) / pri
            far = np.abs(acq.x - (acq.push_x or 0.0))[None, :] > MIN_LATERAL_M
            ok = far & (np.abs(v) < MAX_V) & (np.abs(v) > 0.003) & (mag > np.percentile(mag, 50))
            w = ok.astype(float)
            ok_c = far & (np.abs(v_cal) < MAX_V) & (np.abs(v_cal) > 0.003) & (mag > np.percentile(mag, 50))
            fe = {}
            for k in K_FRAMES:
                d_k, _ = _phase_disp(ref[-1], iq[k], c, f)
                fe[f"fe_dv_{k}"], fe[f"fe_vv_{k}"] = centred_sums(d_k, v, w)
            rows.append(dict(folder=os.path.basename(folder)[:40], meas=path.split("meas")[-1].split("_")[0],
                             n_px=int(ok.sum()), dt_measured_us=1e6 * slope(d_cross, v, w),
                             dt_old_us=1e6 * pred_old, dt_new_us=1e6 * pred_new,
                             calib_measured_us=1e6 * slope(d_cal, v_cal, ok_c.astype(float)),
                             calib_true_us=1e6 * 6 * pri,
                             sdv=float(np.sum(w * d_cross * v)), svv=float(np.sum(w * v * v)),
                             sdv_cal=float(np.sum(ok_c * d_cal * v_cal)), svv_cal=float(np.sum(ok_c * v_cal * v_cal)),
                             pri_us=1e6 * pri, **fe))
    out = os.path.join(_REPO, "study", "logs", "push_gap_check.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(stamp_text(config=dict(KZ=KZ, MIN_LATERAL_M=MIN_LATERAL_M, MAX_V=MAX_V)))
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)
    print("\nPooled over all pushes (velocity-weighted; robust to slow-wall pushes):")
    print(f"{'acquisition':<42}{'pooled dt':>10}{'old':>7}{'new':>7}{'pooled calib/true':>19}")
    for fo in dict.fromkeys(r["folder"] for r in rows):
        r = [x for x in rows if x["folder"] == fo]
        pooled = 1e6 * sum(x["sdv"] for x in r) / sum(x["svv"] for x in r)
        pcal = 1e6 * sum(x["sdv_cal"] for x in r) / sum(x["svv_cal"] for x in r)
        print(f"{fo:<42}{pooled:7.0f} us{r[0]['dt_old_us']:7.0f}{r[0]['dt_new_us']:7.0f}"
              f"{pcal:12.0f}/{r[0]['calib_true_us']:.0f}")
    print("\nFixed effects (per-push intercept) for tracking frames 1..8, then dt(k) = dt(1) + (k-1)*slope."
          "\nThe slope must come out at the PRI; dt(1) is the measured reference -> frame-1 interval:")
    print(f"{'acquisition':<42}{'dt(1)':>9}{'slope':>9}{'PRI':>6}{'old dt(1)':>11}{'new dt(1)':>11}")
    for fo in dict.fromkeys(r["folder"] for r in rows):
        r = [x for x in rows if x["folder"] == fo]
        ks = np.array(list(K_FRAMES), float)
        dts = np.array([1e6 * sum(x[f"fe_dv_{k}"] for x in r) / sum(x[f"fe_vv_{k}"] for x in r)
                        for k in K_FRAMES])
        b, a = np.polyfit(ks - 1, dts, 1)
        print(f"{fo:<42}{a:6.0f} us{b:6.0f} us{r[0]['pri_us']:6.0f}{r[0]['dt_old_us']:8.0f} us"
              f"{r[0]['dt_new_us']:8.0f} us   per frame: {' '.join(f'{x:.0f}' for x in dts)}")
    print(f"\nPer push (median):\n{'acquisition':<42}{'n':>4}{'measured dt':>13}{'old':>7}{'new':>7}{'calib meas/true':>18}")
    for fo in dict.fromkeys(r["folder"] for r in rows):
        r = [x for x in rows if x["folder"] == fo]
        med = np.median([x["dt_measured_us"] for x in r]); q = np.percentile([x["dt_measured_us"] for x in r], [25, 75])
        cal = np.median([x["calib_measured_us"] for x in r])
        print(f"{fo:<42}{len(r):4d}{med:9.0f} us{r[0]['dt_old_us']:7.0f}{r[0]['dt_new_us']:7.0f}"
              f"{cal:11.0f}/{r[0]['calib_true_us']:.0f}   IQR {q[0]:.0f}-{q[1]:.0f}")


if __name__ == "__main__":
    main()
