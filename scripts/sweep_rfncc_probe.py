"""LIMITED RF-NCC probe: a handful of near-optimal recipes run with the fine-grid RF-NCC estimator
(re-beamformed fine axial grid) instead of coarse-IQ Loupas, "just to see what happens" at low SNR.

RF-NCC is the displacement-precision benchmark but needs a fine re-beamform (~10 s/voltage) + a slower
estimator, so this is deliberately a SMALL set: a few near-optimal field-filter recipes x the low-SNR
voltages, scored by the SAME detectors as the main sweep (ROI-contrast band1/band2 + mirror-symmetry).
Each voltage's RF-NCC estimator is cached once and reused across recipes/quantities. The printout puts
the RF-NCC best next to the coarse-Loupas sweep best at the same voltage, so it is clear whether the
extra beamform buys a cleaner V. If it does, we widen the RF-NCC recipe set later.

    <zea-python> scripts/sweep_rfncc_probe.py [--volts 50V 30V 25V 20V 15V]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import core                                              # noqa: E402
from swp.viz.estimators import rf_ncc_displacement       # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
import sweep_extract as sw                                # noqa: E402  (reuse eval_one + folders)

# a few near-optimal field-filter recipes, light -> heavy smoothing (SI-unit params; eval_one applies
# FIELD_FILTERS directly). Loupas/rel-ref/outward are fixed exactly as in the main sweep.
def _rec(sm_name, sp, mo, tm_name, tw, flo, fhi, off, iq="none"):
    spatial = {"gauss": [("spatial_smooth", {"sigma_z_m": sp[0], "sigma_x_m": sp[1]})],
               "median": [("spatial_median", {"size_z_m": sp[0], "size_x_m": sp[1]})]}[sm_name]
    temporal = ([("temporal_moving_mean", {"window": tw})] if tm_name == "mean"
                else [("temporal_moving_median", {"window": tw})])
    return {"iq": iq, "motion": [("temporal_bandpass", {"f_lo": flo, "f_hi": fhi, "order": 2})] + mo,
            "spatial": spatial, "temporal": temporal, "offsets": off, "step_m": 0.8e-3,
            "sm": sm_name, "tm": tm_name, "f_lo": flo, "f_hi": fhi}


RECIPES = {
    "A_light":  _rec("gauss", (800e-6, 1600e-6), [], "mean", 5, 80, 600, 9),
    "B_heavy":  _rec("gauss", (1400e-6, 2800e-6), [], "mean", 7, 40, 500, 9),
    "C_median": _rec("median", (900e-6, 1800e-6), [], "median", 5, 40, 600, 9),
    "D_svd":    _rec("gauss", (1100e-6, 2200e-6), [], "mean", 9, 20, 500, 9, iq="svd1"),
}


def coarse_best(v):
    """Best coarse-Loupas ROI-contrast at this voltage from the finished main sweep (if present)."""
    p = os.path.join(sw.BASE, "sweep_results.csv")
    if not os.path.exists(p):
        return None
    rr = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["voltage"] == v]
    if not rr:
        return None
    b = max(rr, key=lambda r: float(r["roi1_best"]))
    return (float(b["roi1_best"]), float(b["sym_best"]), b["roi1_q"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volts", nargs="+", default=["50V", "30V", "25V", "20V", "15V"])
    ap.add_argument("--out", default=os.path.join(sw.BASE, "rfncc_probe.csv"))
    a = ap.parse_args()
    tmpl1 = json.load(open(os.path.join(sw.BASE, "v_roi_template.json"), encoding="utf-8"))
    tmpl2 = {**tmpl1, "band_s": tmpl1["band_s"] * 2.0}

    rrec = core.Recipe(mline_source="horizontal_push", estimator="rf_ncc")
    fout = open(a.out, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=["voltage", "recipe", "roi1", "roi2", "sym", "q"])
    w.writeheader()
    print(f"{'volt':>5} {'recipe':>9} {'roi1':>6} {'roi2':>6} {'sym':>6} {'q':>4}   vs coarse-Loupas best")
    t0 = time.time()
    for v in a.volts:
        folder = os.path.join(sw.PH, [d for d in os.listdir(sw.PH) if sw.VOLTS[v] in d][0])
        pts = core.mline_points_for(folder, 0)
        acq = core.load_acq(folder, 0, rrec, mline_points=pts)        # fine re-beamform + as_rf (RF)
        ml = core.load_mline_for(folder, 0, acq, rrec)
        r0 = _r0_lateral_crossing(ml, float(acq.push_x))
        est = rf_ncc_displacement(acq.iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod,
                                  prf=acq.prf, mode="relative_to_reference", reference=acq.ref_iq)
        cb = coarse_best(v)
        best = None
        for name, rec in RECIPES.items():
            per = {q: sw.eval_one(est, acq, ml, r0, rec, q, tmpl1, tmpl2) for q in sw.QUANTS}
            bq = max(sw.QUANTS, key=lambda q: per[q][0])
            roi1, roi2, sym = per[bq]
            w.writerow({"voltage": v, "recipe": name, "roi1": round(roi1, 3), "roi2": round(roi2, 3),
                        "sym": round(sym, 3), "q": bq[:3]})
            if best is None or roi1 > best[1]:
                best = (name, roi1, roi2, sym, bq[:3])
            tail = f"   coarse best roi1={cb[0]:.3f} ({cb[2]})" if cb else ""
            print(f"{v:>5} {name:>9} {roi1:>6.3f} {roi2:>6.3f} {sym:>6.3f} {bq[:3]:>4}{tail}")
        print(f"      -> RF-NCC best: {best[0]} roi1={best[1]:.3f} sym={best[3]:.3f} ({best[4]})  "
              f"[{time.time()-t0:.0f}s]")
        fout.flush()
    fout.close()
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
