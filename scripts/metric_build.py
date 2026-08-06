"""Build a better wave-clarity metric from the human labels. Re-runs each scored recipe for all three
quantities, extracts candidate image features from the space-time (origin_coherence [symmetric,
origin-aware], wavefront_coherence [in-band/smoothness], a broadband-noise ratio, and a symmetry
term), aggregates ACROSS quantities (mean/min/max), and reports which candidate / simple combo best
tracks the human score. Fits on --fit round (default round3, the cleanest criterion) and cross-checks
on the others by rank (scales differ between rounds).

    <zea-python> scripts/metric_build.py [--fit round3] [--check round1 round2]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))

import core                                              # noqa: E402
from swp.viz.metrics import origin_coherence, wavefront_coherence  # noqa: E402

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
PHANTOM_25V = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom/DefaultPatient_SW_data_04-August-2026_13-28-32"
Q = ["displacement", "velocity", "acceleration"]


def noise_ratio(st):
    """Fraction of (temporally-demeaned) energy in the broadband high-freq corner of the 2-D FFT -
    high for grainy/noisy plots, low for smooth ones. So 1-noise ~ smoothness."""
    d = st.data - st.data.mean(axis=0, keepdims=True)
    nt, nr = d.shape
    P = np.abs(np.fft.fft2(d)) ** 2
    ft = np.abs(np.fft.fftfreq(nt))[:, None]
    fr = np.abs(np.fft.fftfreq(nr))[None, :]
    hi = (ft > 0.25) & (fr > 0.25)                       # outer quarter of both axes = fine noise
    tot = P.sum() - P[0, 0]
    return float(P[hi].sum() / (tot + 1e-20))


def feats_for_st(st, r0):
    oc, left, right = origin_coherence(st, r0, return_sides=True)
    sym = float(min(left, right) / (max(left, right) + 1e-9))     # side balance in [0,1]
    return {"oc": oc, "wfc": float(wavefront_coherence(st)),
            "smooth": 1.0 - noise_ratio(st), "sym": sym}


def extract_round(rn, acq, ml, cache):
    man = json.load(open(os.path.join(BASE, rn, "manifest.json"), encoding="utf-8"))
    sc = {int(r["id"]): int(r["score"])
          for r in csv.DictReader(open(os.path.join(BASE, rn, "scores.csv"), encoding="utf-8"))}
    rows = []
    for it in man["items"]:
        if it["id"] not in sc:
            continue
        rec = core.Recipe(**it["recipe"])
        per = {}
        for q in Q:
            res = core.run_recipe(acq, ml, core.to_config(dataclasses.replace(rec, quantity=q), acq))
            per[q] = feats_for_st(res.st, res.r0)
        # aggregate each feature across quantities (mean and min)
        agg = {}
        for f in ("oc", "wfc", "smooth", "sym"):
            vals = [per[q][f] for q in Q]
            agg[f + "_mean"] = float(np.mean(vals)); agg[f + "_min"] = float(min(vals))
            agg[f + "_disp"] = float(per["displacement"][f])
        rows.append((sc[it["id"]], agg))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="round3")
    ap.add_argument("--check", nargs="+", default=["round1", "round2"])
    args = ap.parse_args()

    r0 = core.Recipe(mline_source="horizontal_push")
    acq = core.load_acq(PHANTOM_25V, 0, r0)
    ml = core.load_mline_for(PHANTOM_25V, 0, acq, r0)

    sets = {rn: extract_round(rn, acq, ml, {}) for rn in [args.fit] + args.check}
    for rn in sets:
        print(f"  extracted {len(sets[rn])} from {rn}")

    def corr_on(rows, score_fn):
        y = np.array([s for s, _ in rows], float)
        x = np.array([score_fn(a) for _, a in rows], float)
        return spearmanr(x, y)[0]

    # candidate metrics (functions of the aggregated features)
    cands = {
        "oc_mean": lambda a: a["oc_mean"],
        "oc_max": lambda a: max(a["oc_disp"], a["oc_mean"]),   # ~ existing behaviour proxy
        "wfc_mean": lambda a: a["wfc_mean"],
        "smooth_mean": lambda a: a["smooth_mean"],
        "sym_mean": lambda a: a["sym_mean"],
        "oc*sym (mean)": lambda a: a["oc_mean"] * a["sym_mean"],
        "oc*smooth (mean)": lambda a: a["oc_mean"] * a["smooth_mean"],
        "wfc*sym (mean)": lambda a: a["wfc_mean"] * a["sym_mean"],
        "oc*wfc (mean)": lambda a: a["oc_mean"] * a["wfc_mean"],
        "oc*smooth*sym (mean)": lambda a: a["oc_mean"] * a["smooth_mean"] * a["sym_mean"],
        "oc_min": lambda a: a["oc_min"],
        "oc_mean+0.5 smooth": lambda a: a["oc_mean"] + 0.5 * a["smooth_mean"],
    }
    fit = sets[args.fit]
    print(f"\n=== candidate metrics vs human, FIT on {args.fit} (n={len(fit)}) ===")
    ranked = sorted(((corr_on(fit, fn), name) for name, fn in cands.items()), reverse=True)
    for rho, name in ranked:
        checks = "  ".join(f"{rn}:{corr_on(sets[rn], cands[name]):+.2f}" for rn in args.check)
        print(f"  {name:24s} {args.fit}:{rho:+.3f}   [{checks}]")

    # simple linear fit (least squares) of oc_mean, wfc_mean, smooth_mean, sym_mean on fit round
    X = np.array([[a["oc_mean"], a["wfc_mean"], a["smooth_mean"], a["sym_mean"], 1.0]
                  for _, a in fit])
    y = np.array([s for s, _ in fit], float)
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    names = ["oc_mean", "wfc_mean", "smooth_mean", "sym_mean", "bias"]
    print("\nleast-squares weights (fit round):", {n: round(float(v), 2) for n, v in zip(names, w)})

    def linfit(a):
        return (w[0]*a["oc_mean"] + w[1]*a["wfc_mean"] + w[2]*a["smooth_mean"] + w[3]*a["sym_mean"] + w[4])
    print(f"linear combo   {args.fit}:{corr_on(fit, linfit):+.3f}   [" +
          "  ".join(f"{rn}:{corr_on(sets[rn], linfit):+.2f}" for rn in args.check) + "]")

    # dump features for later
    with open(os.path.join(BASE, "features.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        keys = sorted(fit[0][1].keys())
        wtr.writerow(["round", "score"] + keys)
        for rn in sets:
            for s, a in sets[rn]:
                wtr.writerow([rn, s] + [round(a[k], 4) for k in keys])
    print(f"\nwrote {os.path.join(BASE, 'features.csv')}")


if __name__ == "__main__":
    main()
