"""Extensive low-SNR extraction sweep across all phantom voltages, scored by BOTH detectors.

Fixes the settled winners (Loupas + relative-to-reference + outward directional) and sweeps the
SNR-relevant levers: IQ clutter (SVD), band-pass corners, spatial smoothing (Gaussian sigma / median
size / NLM), temporal smoothing (moving mean/median / Savitzky-Golay), and M-line offsets. Every
recipe is evaluated on every voltage x 3 quantities, and scored by:
  - ROI-contrast on the LOCKED 50 V V-template, at band1 (drawn) AND band2 (2x wide), and
  - the mirror-symmetry V-detector.
The loupas estimator is cached per (voltage, IQ-config) so 16 k evaluations stay tractable.

    <zea-python> scripts/sweep_extract.py --n 700 --seed 0 [--out sweep_results.csv]

Output: one CSV row per (recipe, voltage) with the best-quantity scores + which quantity, for later
per-voltage ranking (scripts/sweep_analyze.py).
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
from swp.viz.estimators import loupas_displacement       # noqa: E402
from swp.viz.filters import FIELD_FILTERS, svd_clutter    # noqa: E402
from swp.viz.filters.context import FilterCtx            # noqa: E402
from swp.viz.speed.spacetime import build_spacetime, SpaceTime  # noqa: E402
from swp.viz.filters.directional import outward_spacetime  # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
from detect_v import roi_contrast, symmetric_v_score      # noqa: E402

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
PH = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom"
# voltage -> folder (acquisition order high->low)
VOLTS = {"50V": "13-21-52", "45V": "13-23-29", "40V": "13-25-06", "35V": "13-26-43",
         "30V": "13-27-38", "25V": "13-28-32", "20V": "13-29-31", "15V": "13-30-35"}
QUANTS = ["displacement", "velocity", "acceleration"]


def sample_recipe(rng):
    iq = str(rng.choice(["none", "svd1", "svd2"], p=[0.6, 0.25, 0.15]))
    flo = int(rng.choice([10, 20, 40, 80, 120])); fhi = int(rng.choice([350, 500, 650, 800]))
    motion = [("temporal_bandpass", {"f_lo": flo, "f_hi": fhi, "order": 2})]
    if rng.random() < 0.2:
        motion = [("polynomial_drift", {"order": int(rng.choice([2, 3])), "fit_frac": 1.0})] + motion
    if rng.random() < 0.12:
        motion = motion + [("svd_clutter_field", {"n_remove": 1, "n_high_remove": 0})]
    sm = str(rng.choice(["gauss", "median", "none", "nlm"], p=[0.5, 0.3, 0.12, 0.08]))
    if sm == "gauss":
        spatial = [("spatial_smooth", {"sigma_z_m": float(rng.uniform(200e-6, 2500e-6)),
                                       "sigma_x_m": float(rng.uniform(400e-6, 4000e-6))})]
    elif sm == "median":
        spatial = [("spatial_median", {"size_z_m": float(rng.uniform(300e-6, 1500e-6)),
                                       "size_x_m": float(rng.uniform(600e-6, 3000e-6))})]
    elif sm == "nlm":
        spatial = [("nlm_denoise", {"h_um": float(rng.uniform(2, 15)), "patch_size": 5,
                                    "patch_distance": 6})]
    else:
        spatial = []
    tm = str(rng.choice(["mean", "median", "savgol", "none"], p=[0.45, 0.2, 0.2, 0.15]))
    if tm == "mean":
        temporal = [("temporal_moving_mean", {"window": int(rng.choice([1, 3, 5, 7, 9]))})]
    elif tm == "median":
        temporal = [("temporal_moving_median", {"window": int(rng.choice([3, 5, 7]))})]
    elif tm == "savgol":
        temporal = [("savgol_temporal", {"window": int(rng.choice([5, 7, 9, 11])),
                                         "polyorder": int(rng.choice([2, 3]))})]
    else:
        temporal = []
    return {"iq": iq, "motion": motion, "spatial": spatial, "temporal": temporal,
            "offsets": int(rng.choice([1, 3, 5, 7, 9])), "step_m": float(round(rng.uniform(0.4e-3, 1.2e-3), 5)),
            "sm": sm, "tm": tm, "f_lo": flo, "f_hi": fhi,
            "sz_um": (spatial[0][1].get("sigma_z_m", spatial[0][1].get("size_z_m", 0)) * 1e6) if spatial else 0.0}


def load_voltage(v):
    folder = [d for d in os.listdir(PH) if VOLTS[v] in d][0]
    acq = core.load_acq(os.path.join(PH, folder), 0, core.Recipe(mline_source="horizontal_push"))
    ml = core.load_mline_for(os.path.join(PH, folder), 0, acq, core.Recipe(mline_source="horizontal_push"))
    r0 = _r0_lateral_crossing(ml, float(acq.push_x))
    return acq, ml, r0


def estimator_for_iq(acq, iq_cfg):
    iq = acq.iq
    if iq_cfg == "svd1":
        iq = svd_clutter(iq, n_remove=1)
    elif iq_cfg == "svd2":
        iq = svd_clutter(iq, n_remove=2)
    return loupas_displacement(iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod, prf=acq.prf,
                               mode="relative_to_reference", reference=acq.ref_iq)


def spacetime_for(est, acq, ml, r0, recipe, quantity):
    """Post-estimator pipeline (quantity -> field filters -> M-line -> outward directional) -> SpaceTime.
    Shared by eval_one and the montage renderer so both use the exact same processing."""
    prf = acq.prf; t = acq.t
    if quantity == "velocity":
        f_all, t_all = est.velocity, 0.5 * (t[:-1] + t[1:])
    elif quantity == "acceleration":
        f_all, t_all = np.diff(est.velocity, axis=0) * prf, t[1:-1]
    else:
        f_all, t_all = est.displacement, t
    fld, times = f_all[1:], t_all[1:]
    ctx = FilterCtx(dz=acq.dz, dx=acq.dx, prf=prf, t=times, x=acq.x, z=acq.z, f_demod=acq.f_demod, c=acq.c)
    for name, params in recipe["motion"] + recipe["spatial"] + recipe["temporal"]:
        fld = FIELD_FILTERS[name](fld, ctx, **params)
    st = build_spacetime(fld, acq.z, acq.x, ml, times, quantity=quantity,
                         n_offsets=recipe["offsets"], offset_step_m=recipe["step_m"], agg="mean")
    return SpaceTime(outward_spacetime(st.data, st.r, r0), st.r, st.t, st.quantity)


def eval_one(est, acq, ml, r0, recipe, quantity, tmpl1, tmpl2):
    st = spacetime_for(est, acq, ml, r0, recipe, quantity)
    return (roi_contrast(st, tmpl1, r0=r0), roi_contrast(st, tmpl2, r0=r0),
            symmetric_v_score(st, r0)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--volts", nargs="+", default=list(VOLTS))
    ap.add_argument("--out", default=os.path.join(BASE, "sweep_results.csv"))
    a = ap.parse_args()

    tmpl1 = json.load(open(os.path.join(BASE, "v_roi_template.json"), encoding="utf-8"))
    tmpl2 = {**tmpl1, "band_s": tmpl1["band_s"] * 2.0}   # second band, 2x wide
    rng = np.random.default_rng(a.seed)
    recipes = [sample_recipe(rng) for _ in range(a.n)]

    fields = ["recipe_id", "voltage", "iq", "sm", "tm", "sz_um", "f_lo", "f_hi", "offsets",
              "roi1_best", "roi2_best", "sym_best", "roi1_q", "roi2_q", "sym_q",
              "roi1_disp", "roi1_vel", "roi1_acc"]
    fout = open(a.out, "w", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=fields); w.writeheader()
    json.dump({i: {k: recipes[i][k] for k in ("iq", "motion", "spatial", "temporal", "offsets", "step_m")}
               for i in range(len(recipes))},
              open(a.out.replace(".csv", "_recipes.json"), "w", encoding="utf-8"))

    t0 = time.time()
    for v in a.volts:
        acq, ml, r0 = load_voltage(v)
        est_cache = {}
        for rid, rec in enumerate(recipes):
            if rec["iq"] not in est_cache:
                est_cache[rec["iq"]] = estimator_for_iq(acq, rec["iq"])
            est = est_cache[rec["iq"]]
            per = {q: eval_one(est, acq, ml, r0, rec, q, tmpl1, tmpl2) for q in QUANTS}
            r1 = {q: per[q][0] for q in QUANTS}; r2 = {q: per[q][1] for q in QUANTS}
            sy = {q: per[q][2] for q in QUANTS}
            w.writerow({"recipe_id": rid, "voltage": v, "iq": rec["iq"], "sm": rec["sm"], "tm": rec["tm"],
                        "sz_um": round(rec["sz_um"], 0), "f_lo": rec["f_lo"], "f_hi": rec["f_hi"],
                        "offsets": rec["offsets"],
                        "roi1_best": round(max(r1.values()), 3), "roi2_best": round(max(r2.values()), 3),
                        "sym_best": round(max(sy.values()), 3),
                        "roi1_q": max(QUANTS, key=lambda q: r1[q])[:3],
                        "roi2_q": max(QUANTS, key=lambda q: r2[q])[:3],
                        "sym_q": max(QUANTS, key=lambda q: sy[q])[:3],
                        "roi1_disp": round(r1["displacement"], 3), "roi1_vel": round(r1["velocity"], 3),
                        "roi1_acc": round(r1["acceleration"], 3)})
        fout.flush()
        print(f"  {v} done  ({time.time()-t0:.0f}s, {len(recipes)} recipes)")
    fout.close()
    print(f"\nwrote {a.out}  ({a.n} recipes x {len(a.volts)} volts, {time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
