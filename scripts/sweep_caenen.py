"""Apply the phantom extraction recipes to the CAENEN pig-heart ARF-SWE data (Cartesian grid), with the
same per-lobe wavefront-tilt (speed) scan used for the in-vivo sweep.

Caenen data = polar/Cartesian IQ exported from MATLAB (export_push.m -> _h5_tmp/push_<p>.h5), loaded via
swp_bridge. Cart grid is primary (polar r-axis is compressed). The ARF push is on-axis (x=0, z=25.3 mm),
so r0 = the M-line crossing of x=0; the anatomical M-line was drawn once per push (SWE_results/push_<p>/
mline.npz). Same 700 recipes as scripts/sweep_extract.py (seed 0); Loupas cached per (push, IQ-config).

As for the in-vivo septum, the pig LV wall is oblique + inhomogeneous, so each lobe is scored
independently with its own best-fit speed (c in 1.0..5.0 m/s), ranked by the stronger lobe; mirror-
symmetry is reported but does not gate.

    <zea-python> scripts/sweep_caenen.py [--push 1-52] [--n 700 --seed 0]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWE = r"D:/Luuk van Knippenberg/Claude/Data Caenen/SWE_results"
H5DIR = os.path.join(SWE, "_h5_tmp")
for p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "swp_gui"),
          os.path.join(_ROOT, "scripts"), SWE):
    sys.path.insert(0, p)

from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
from swp.viz.mline.mline import mline_from_points        # noqa: E402
from swp_bridge import load_push, build_acq              # noqa: E402
import sweep_extract as sw                                # noqa: E402
from sweep_invivo40 import roi_scan                       # noqa: E402  (per-lobe speed scan)
from detect_v import _env, symmetric_v_score             # noqa: E402

OUTDIR = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"


def crop_to_mline(acq, ml, margin=10e-3):
    """Crop the full-sector Cartesian volume to a box around the M-line (+ margin for the offset lines
    and filter support). The M-line region is interior, so filtering/estimation is unchanged there but
    ~5-6x cheaper than on the whole 70x86 mm sector."""
    xlo, xhi = ml.x.min() - margin, ml.x.max() + margin
    zlo, zhi = ml.z.min() - margin, ml.z.max() + margin
    ix = np.where((acq.x >= xlo) & (acq.x <= xhi))[0]
    iz = np.where((acq.z >= zlo) & (acq.z <= zhi))[0]
    return dataclasses.replace(acq, x=acq.x[ix], z=acq.z[iz],
                               iq=acq.iq[:, iz][:, :, ix], ref_iq=acq.ref_iq[:, iz][:, :, ix])


def load_caenen(pushid):
    Fp, Fc, zp, xp, zc, xc, taxis, params = load_push(os.path.join(H5DIR, f"push_{pushid}.h5"))
    acq = build_acq(Fc, zc, xc, taxis, params, "cart")            # Cartesian grid
    pts = np.load(os.path.join(SWE, f"push_{pushid}", "mline.npz"))["points"]
    ml = mline_from_points(np.asarray(pts, float), n_samples=250)
    r0 = _r0_lateral_crossing(ml, 0.0)                            # push on-axis: x=0
    return crop_to_mline(acq, ml), ml, r0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", nargs="+", type=int, default=list(range(1, 53)))
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "sweep_caenen.csv"))
    a = ap.parse_args()

    tmpl = json.load(open(os.path.join(OUTDIR, "v_roi_template.json"), encoding="utf-8"))
    t0 = tmpl["t0_s"]; b1 = tmpl["band_s"]; b2 = tmpl["band_s"] * 2.0
    dmin = tmpl.get("d_min_m", 2e-3); dmax = tmpl.get("d_max_m", 16e-3)
    rng = np.random.default_rng(a.seed)
    recipes = [sw.sample_recipe(rng) for _ in range(a.n)]
    pushes = [p for p in a.push if os.path.exists(os.path.join(H5DIR, f"push_{p}.h5"))]

    fields = ["push", "recipe_id", "iq", "sm", "tm", "sz_um", "f_lo", "f_hi", "offsets",
              "roi1_max", "roi1_side", "roi1_speed", "roi1_L", "roi1_R", "cL", "cR", "roi1_q",
              "roi2_max", "sym_best", "sym_q"]
    # RESUME: keep only fully-completed pushes (== a.n rows) from any existing CSV, then append the rest.
    complete = set()
    if os.path.exists(a.out):
        counts, keep = {}, []
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            counts[int(r["push"])] = counts.get(int(r["push"]), 0) + 1
        complete = {p for p, c in counts.items() if c >= a.n}
        keep = [r for r in csv.DictReader(open(a.out, encoding="utf-8")) if int(r["push"]) in complete]
        with open(a.out, "w", newline="", encoding="utf-8") as f:   # rewrite, dropping partial pushes
            wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(keep)
    pushes = [p for p in pushes if p not in complete]
    print(f"resume: {len(complete)} pushes already complete, {len(pushes)} remaining")
    fout = open(a.out, "a" if complete else "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields)
    if not complete:
        w.writeheader()

    tstart = time.time()
    for p in pushes:
        acq, ml, r0 = load_caenen(p)
        est_cache = {}
        for rid, rec in enumerate(recipes):
            if rec["iq"] not in est_cache:
                est_cache[rec["iq"]] = sw.estimator_for_iq(acq, rec["iq"])
            est = est_cache[rec["iq"]]
            per = {}
            for q in sw.QUANTS:
                st = sw.spacetime_for(est, acq, ml, r0, rec, q)
                E = _env(st)
                rL1, cL1, rR1, cR1 = roi_scan(E, st.t, st.r, r0, t0, b1, dmin, dmax)
                rL2, _, rR2, _ = roi_scan(E, st.t, st.r, r0, t0, b2, dmin, dmax)
                per[q] = (rL1, cL1, rR1, cR1, rL2, rR2, symmetric_v_score(st, r0)[0])

            def side_max(q):
                v = [x for x in (per[q][0], per[q][2]) if np.isfinite(x)]
                return max(v) if v else -9
            bq1 = max(sw.QUANTS, key=side_max)
            bqs = max(sw.QUANTS, key=lambda q: per[q][6])
            rL1, cL1, rR1, cR1, rL2, rR2, _ = per[bq1]
            left = (rL1 if np.isfinite(rL1) else -9) >= (rR1 if np.isfinite(rR1) else -9)
            rmax = rL1 if left else rR1
            w.writerow({"push": p, "recipe_id": rid, "iq": rec["iq"], "sm": rec["sm"], "tm": rec["tm"],
                        "sz_um": round(rec["sz_um"], 0), "f_lo": rec["f_lo"], "f_hi": rec["f_hi"],
                        "offsets": rec["offsets"],
                        "roi1_max": round(rmax, 3) if np.isfinite(rmax) else "",
                        "roi1_side": "L" if left else "R", "roi1_speed": cL1 if left else cR1,
                        "roi1_L": round(rL1, 3) if np.isfinite(rL1) else "",
                        "roi1_R": round(rR1, 3) if np.isfinite(rR1) else "",
                        "cL": cL1, "cR": cR1, "roi1_q": bq1[:3],
                        "roi2_max": round(np.nanmax([rL2, rR2]), 3) if np.isfinite(np.nanmax([rL2, rR2])) else "",
                        "sym_best": round(per[bqs][6], 3), "sym_q": bqs[:3]})
        fout.flush()
        print(f"  push{p:2d} done  ({time.time()-tstart:.0f}s)")
    fout.close()
    print(f"\nwrote {a.out}  ({len(pushes)} pushes x {a.n} recipes, {time.time()-tstart:.0f}s)")


if __name__ == "__main__":
    main()
