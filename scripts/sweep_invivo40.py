"""Apply the phantom extraction recipes to the 40 V IN-VIVO data, scanning the wavefront TILT (speed).

Assumption (user): the in-vivo shear wave has the same frequency content and lobe width as the phantom,
but a different speed (tissue stiffness differs), so the ROI template's *shape* is reused and only its
tilt c is allowed to float, at c in {1.0, 1.5, ..., 5.0} m/s (0.5 steps).

The septum is oblique to the beam and inhomogeneous, so an in-vivo wave is generally NOT mirror-
symmetric about r0 and the two sides may differ in speed/attenuation (user's point). We therefore score
**each lobe independently** (its own best-fit speed) and rank by the STRONGER lobe (roi1_max), instead
of averaging the two sides (which would dilute a one-sided wave). Mirror-symmetry is still computed and
reported, but it is NOT used to gate candidates (it is biased low for genuine asymmetric in-vivo waves).

Runs the SAME 700 recipes as the phantom sweep (scripts/sweep_extract.py, seed 0) on all 24 pushes of
Luuk40V (anatomical per-push M-lines), coarse-IQ Loupas cached per (push, IQ-config).

CAVEAT: the in-vivo active plots are cardiac-motion-dominated (see docs/HANDOFF sec 0); a speed scan can
make the ROI ride a sloped cardiac band, so a high ROI-contrast alone is NOT proof of an ARF shear wave.
Mirror-symmetry (a symmetric V about r0) is the stronger corroborator; report both.

    <zea-python> scripts/sweep_invivo40.py [--meas 0-23] [--n 700 --seed 0]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import core                                              # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
import sweep_extract as sw                                # noqa: E402  (reuse sampler + spacetime_for)
from detect_v import _env, symmetric_v_score             # noqa: E402

FOLDER = (r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo/"
          r"Luuk40V_SW_data_04-August-2026_13-36-07")
OUTDIR = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
SPEEDS = np.arange(1.0, 5.01, 0.5)                        # 1.0 .. 5.0 m/s, 0.5 steps
REC = core.Recipe(mline_source="auto")                   # in-vivo: use the saved anatomical M-line


def _side_best(E, t, r, r0, mask, t0, band, dmin, dmax, colmean):
    """Best ROI-contrast on ONE lobe over the speed grid, with that lobe's own best-fit speed.

    In vivo the septum is oblique to the beam and inhomogeneous, so the wave is generally NOT mirror-
    symmetric about r0 and the two sides can differ in speed/attenuation. We therefore score each lobe
    independently (its own c) and rank by the stronger lobe; mirror-symmetry is reported, not gated."""
    dt = float(t[1] - t[0]); d = np.abs(r - r0); nb = max(1, int(round(band / dt)))
    cols = np.where(mask & (d >= dmin) & (d <= dmax))[0]
    if cols.size < 3:
        return np.nan, np.nan
    best_v, best_c = -np.inf, np.nan
    for c in SPEEDS:
        roi, bg = [], []
        for j in cols:
            ia = int(round((t0 + d[j] / c - t[0]) / dt))
            if 0 <= ia < len(t):
                roi.append(float(E[max(0, ia - nb):ia + nb + 1, j].mean())); bg.append(float(colmean[j]))
        if roi:
            rm, bm = np.mean(roi), np.mean(bg); v = (rm - bm) / (rm + bm + 1e-12)
            if v > best_v:
                best_v, best_c = float(v), float(c)
    return (best_v if np.isfinite(best_v) else np.nan), best_c


def roi_scan(E, t, r, r0, t0, band, dmin, dmax):
    """Per-lobe ROI-contrast (each lobe its own best-fit speed). Returns
    (roiL, cL, roiR, cR): left = r<r0, right = r>=r0."""
    colmean = E.mean(axis=0)
    rL, cL = _side_best(E, t, r, r0, r < r0, t0, band, dmin, dmax, colmean)
    rR, cR = _side_best(E, t, r, r0, r >= r0, t0, band, dmin, dmax, colmean)
    return rL, cL, rR, cR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", nargs="+", type=int, default=list(range(24)))
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "sweep_invivo40.csv"))
    a = ap.parse_args()

    tmpl = json.load(open(os.path.join(OUTDIR, "v_roi_template.json"), encoding="utf-8"))
    t0 = tmpl["t0_s"]; b1 = tmpl["band_s"]; b2 = tmpl["band_s"] * 2.0
    dmin = tmpl.get("d_min_m", 2e-3); dmax = tmpl.get("d_max_m", 16e-3)
    rng = np.random.default_rng(a.seed)
    recipes = [sw.sample_recipe(rng) for _ in range(a.n)]

    # per-lobe ROI (each lobe its own best-fit speed); ranked by the STRONGER lobe (roi1_max).
    # mirror-symmetry is reported (sym_best) but NOT used to gate (in vivo the septum is oblique /
    # inhomogeneous -> genuine waves are often one-sided / asymmetric).
    fields = ["meas", "recipe_id", "iq", "sm", "tm", "sz_um", "f_lo", "f_hi", "offsets",
              "roi1_max", "roi1_side", "roi1_speed", "roi1_L", "roi1_R", "cL", "cR", "roi1_q",
              "roi2_max", "sym_best", "sym_q"]
    fout = open(a.out, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields); w.writeheader()

    tstart = time.time()
    for m in a.meas:
        acq = core.load_acq(FOLDER, m, REC)
        ml = core.load_mline_for(FOLDER, m, acq, REC)
        r0 = _r0_lateral_crossing(ml, float(acq.push_x))
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
                sy = symmetric_v_score(st, r0)[0]
                per[q] = (rL1, cL1, rR1, cR1, rL2, rR2, sy)

            def side_max(q):
                v = [x for x in (per[q][0], per[q][2]) if np.isfinite(x)]
                return max(v) if v else -9
            bq1 = max(sw.QUANTS, key=side_max)
            bqs = max(sw.QUANTS, key=lambda q: per[q][6])
            rL1, cL1, rR1, cR1, rL2, rR2, _ = per[bq1]
            left_wins = (rL1 if np.isfinite(rL1) else -9) >= (rR1 if np.isfinite(rR1) else -9)
            roi_max = rL1 if left_wins else rR1
            w.writerow({"meas": m, "recipe_id": rid, "iq": rec["iq"], "sm": rec["sm"], "tm": rec["tm"],
                        "sz_um": round(rec["sz_um"], 0), "f_lo": rec["f_lo"], "f_hi": rec["f_hi"],
                        "offsets": rec["offsets"],
                        "roi1_max": round(roi_max, 3) if np.isfinite(roi_max) else "",
                        "roi1_side": "L" if left_wins else "R",
                        "roi1_speed": cL1 if left_wins else cR1,
                        "roi1_L": round(rL1, 3) if np.isfinite(rL1) else "",
                        "roi1_R": round(rR1, 3) if np.isfinite(rR1) else "",
                        "cL": cL1, "cR": cR1, "roi1_q": bq1[:3],
                        "roi2_max": round(np.nanmax([rL2, rR2]), 3) if np.isfinite(np.nanmax([rL2, rR2])) else "",
                        "sym_best": round(per[bqs][6], 3), "sym_q": bqs[:3]})
        fout.flush()
        print(f"  meas{m:2d} done  ({time.time()-tstart:.0f}s)")
    fout.close()
    print(f"\nwrote {a.out}  ({len(a.meas)} pushes x {a.n} recipes, {time.time()-tstart:.0f}s)")


if __name__ == "__main__":
    main()
