"""In-vivo cardiac-motion-removal evaluation harness (no-push control + per-lobe speed-scan).

The reference-trained removal methods already exist in the repo; this harness gives them a trustworthy
test. For each candidate filter and push it builds TWO space-times with the SAME recipe:

  * TRACKING   : filter trained on the full pre-push reference, applied to the post-push frames.
  * NO-PUSH    : SPLIT reference - train the filter on the first half of the reference, apply it to the
                 second half (no push fired). A filter that truly removes cardiac motion makes this quiet.

Both are scored by the per-lobe speed-scan detector (each lobe its own best-fit c in 1-5 m/s; ranked by
the stronger lobe) + mirror-symmetry. The wanted signature is TRACKING keeps a physiological-speed
(~2-3 m/s) wavefront while NO-PUSH goes quiet. Pushes are ranked by pre-push reference wall-motion RMS
(a data-driven diastasis proxy, since ECG gating is unavailable) so we focus on the quiet-phase pushes.

    <zea-python> scripts/motion_removal.py [--pushes auto] [--methods all]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "swp_gui"), os.path.join(_ROOT, "scripts")):
    sys.path.insert(0, p)

import core                                              # noqa: E402
from swp.viz.estimators import loupas_displacement       # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
from swp.viz.core.geometry import robust_clim            # noqa: E402
import sweep_invivo40 as iv                               # noqa: E402  (roi_scan, SPEEDS)
from detect_v import symmetric_v_score                    # noqa: E402

FOLDER = iv.FOLDER
OUT = iv.OUTDIR
BP = ("temporal_bandpass", {"f_lo": 80, "f_hi": 500})     # common band tail (widget units)
SPAT = ("spatial_smooth", {"sigma_z_m": 900, "sigma_x_m": 1800})   # widget um -> registry scales
TAIL = dict(spatial_steps=[SPAT], temporal_steps=[("none", {})], directional="outward",
            offsets=7, offset_step_mm=1.0, mline_source="auto", quantity="displacement",
            mode="relative_to_reference")

# candidate motion-removal recipes: removal step BEFORE a common temporal band.
def recipe(iq_steps, pre_field, band=BP):
    return core.Recipe(iq_steps=iq_steps or [("none", {})],
                       motion_steps=(pre_field + [band]) if pre_field else [band], **TAIL)

# IQ-domain (before displacement estimation) vs DISPLACEMENT-domain (after), same technique both ways.
# The high-pass pair uses a low-cut-free band (f_lo=5) so the 60 Hz high-pass IS the low-cut being placed.
LOWBAND = ("temporal_bandpass", {"f_lo": 5, "f_hi": 500})
METHODS = {
    "baseline":    ("--",  recipe([], [])),                                             # band-pass only
    "svd_IQ":      ("IQ",  recipe([("svd_clutter", {"n_remove": 2})], [])),
    "svd_disp":    ("disp", recipe([], [("svd_clutter_field", {"n_remove": 2, "n_high_remove": 0})])),
    "bulk_IQ":     ("IQ",  recipe([("bulk_motion_compensation", {})], [])),
    "bulk_disp":   ("disp", recipe([], [("bulk_displacement_removal", {})])),
    "hp_IQ":       ("IQ",  recipe([("iq_slowtime_highpass", {"fc_hz": 60})], [], band=LOWBAND)),
    "hp_disp":     ("disp", recipe([], [("temporal_highpass", {"fc_hz": 60})], band=LOWBAND)),
    "oflow_IQ":    ("IQ",  recipe([("optical_flow_compensation", {})], [])),            # NEW non-rigid
    "refsub_disp": ("disp", recipe([], [("reference_subspace_projection", {"n_components": 2})])),
}
DOMAIN = {k: v[0] for k, v in METHODS.items()}
METHODS = {k: v[1] for k, v in METHODS.items()}


def split_ref_acq(acq):
    """No-push acquisition: 2nd half of the reference as 'tracking', 1st half as the 'reference'."""
    R = acq.ref_iq; n = R.shape[0]; h = n // 2
    tref = np.asarray(acq.t_ref, float) if acq.t_ref is not None else (np.arange(n) - n) / acq.prf
    return dataclasses.replace(acq, iq=R[h:], ref_iq=R[:h],
                               t=tref[h:] - tref[h], t_ref=tref[:h] - tref[h])


def wave_scores(st, r0, tmpl, b1):
    """(stronger-lobe ROI, its speed, symmetry) for one space-time."""
    from detect_v import _env
    E = _env(st)
    rL, cL, rR, cR = iv.roi_scan(E, st.t, st.r, r0, tmpl["t0_s"], b1,
                                 tmpl.get("d_min_m", 2e-3), tmpl.get("d_max_m", 16e-3))
    left = (rL if np.isfinite(rL) else -9) >= (rR if np.isfinite(rR) else -9)
    return (rL if left else rR), (cL if left else cR), symmetric_v_score(st, r0)[0]


def ref_activity(acq):
    """Pre-push wall-motion RMS (diastasis proxy): lower = quieter push."""
    est = loupas_displacement(acq.ref_iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod,
                              prf=acq.prf, mode="frame_to_frame")
    return float(np.sqrt(np.mean(est.velocity ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pushes", nargs="+", default=["auto"], help="'auto' = quietest 4 by ref RMS")
    ap.add_argument("--n-quiet", type=int, default=4)
    ap.add_argument("--methods", nargs="+", default=list(METHODS))
    a = ap.parse_args()
    tmpl = json.load(open(os.path.join(OUT, "v_roi_template.json"), encoding="utf-8"))
    b1 = tmpl["band_s"]
    rc = core.Recipe(mline_source="auto")

    # rank pushes by pre-push quietness
    acts = []
    for m in range(24):
        acq = core.load_acq(FOLDER, m, rc)
        acts.append((m, ref_activity(acq)))
    acts.sort(key=lambda t: t[1])
    print("push quietness (pre-push wall-motion RMS, low=quiet):")
    print("  " + "  ".join(f"m{m}:{v:.2e}" for m, v in acts))
    pushes = [m for m, _ in acts[:a.n_quiet]] if a.pushes == ["auto"] else [int(x) for x in a.pushes]
    print(f"\nevaluating pushes {pushes}  (quietest {a.n_quiet})\n")

    rows = []
    for m in pushes:
        acq = core.load_acq(FOLDER, m, rc); ml = core.load_mline_for(FOLDER, m, acq, rc)
        r0 = _r0_lateral_crossing(ml, float(acq.push_x))
        acq_np = split_ref_acq(acq)
        print(f"=== push {m} ===")
        print(f"  {'method':>13} | {'TRACK roi/c/sym':>22} | {'NOPUSH roi/c/sym':>22} | contrast")
        for name in a.methods:
            cfg = core.to_config(METHODS[name], acq)
            st_t = core.run_recipe(acq, ml, cfg).st
            rt, ct, syt = wave_scores(st_t, r0, tmpl, b1)
            cfgn = core.to_config(METHODS[name], acq_np)
            st_n = core.run_recipe(acq_np, ml, cfgn).st
            rn, cn, syn = wave_scores(st_n, r0, tmpl, b1)
            contrast = rt - rn
            rows.append(dict(push=m, method=name, track_roi=round(rt, 3), track_c=ct, track_sym=round(syt, 2),
                             nopush_roi=round(rn, 3), nopush_c=cn, nopush_sym=round(syn, 2),
                             contrast=round(contrast, 3)))
            print(f"  {name:>13} | {rt:5.3f} @{ct:3.1f} sym{syt:4.2f}  | "
                  f"{rn:5.3f} @{cn:3.1f} sym{syn:4.2f}  | {contrast:+.3f}")
    with open(os.path.join(OUT, "motion_removal.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # summary: mean contrast + mean nopush (want high contrast, low nopush) per method
    print("\n=== IQ-domain vs DISPLACEMENT-domain summary (mean over pushes) ===")
    print("  want: high trackROI, LOW nopushROI (quiet control), high contrast, physiological track_c")
    print(f"  {'method':>13} {'dom':>4} {'trackROI':>9} {'nopushROI':>10} {'contrast':>9} {'track_c(med)':>13}")
    for name in a.methods:
        mr = [r for r in rows if r["method"] == name]
        tc = np.median([r["track_c"] for r in mr])
        print(f"  {name:>13} {DOMAIN.get(name, '?'):>4} {np.mean([r['track_roi'] for r in mr]):9.3f} "
              f"{np.mean([r['nopush_roi'] for r in mr]):10.3f} "
              f"{np.mean([r['contrast'] for r in mr]):9.3f} {tc:13.1f}")
    print(f"\nwrote {os.path.join(OUT, 'motion_removal.csv')}")


if __name__ == "__main__":
    main()
