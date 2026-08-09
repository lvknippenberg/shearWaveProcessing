"""Analyze the 40 V in-vivo per-lobe speed-scan sweep: top contenders, per-push best, best-fit speed
distribution, and a montage of the strongest candidates with each lobe's fitted tilt overlaid.

Ranked by ROI-contrast on the STRONGER lobe (roi1_max), each lobe scored with its own best-fit speed
(the septum is oblique + inhomogeneous, so genuine waves are often one-sided / asymmetric). Mirror-
symmetry is REPORTED but never gates (it is biased low for real asymmetric in-vivo waves). "Credible"
= strong lobe with a physiological best-fit speed (not pinned at the 1.0/5.0 boundary); real waves
should also recur at a similar speed across neighbouring (same cardiac-phase) pushes.

    <zea-python> scripts/invivo40_analyze.py [--k 20] [--montage 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import core                                              # noqa: E402
import sweep_extract as sw                                # noqa: E402
import sweep_invivo40 as iv                               # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing        # noqa: E402
from swp.viz.core.geometry import robust_clim            # noqa: E402

OUTDIR = iv.OUTDIR
QFULL = {"dis": "displacement", "vel": "velocity", "acc": "acceleration"}
UNIT = {"displacement": 1e6, "velocity": 1e3, "acceleration": 1.0}
PH_FLOOR = 0.05          # phantom no-wave ROI floor (reference)
BOUNDARY = (1.0, 5.0)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        for k in ("roi1_max", "roi1_L", "roi1_R", "roi2_max", "sym_best", "roi1_speed", "cL", "cR"):
            r[k] = _f(r[k])
        for k in ("meas", "recipe_id", "sz_um", "f_lo", "f_hi", "offsets"):
            r[k] = int(_f(r[k]))
    return [r for r in rows if np.isfinite(r["roi1_max"])]


def row_str(r):
    b = "*" if r["roi1_speed"] in BOUNDARY else " "
    return (f"  m{r['meas']:>2} id{r['recipe_id']:>3}  roi={r['roi1_max']:.3f}({r['roi1_side']}) @ "
            f"{r['roi1_speed']:.1f}{b} [L{r['roi1_L']:.2f}/R{r['roi1_R']:.2f}]  sym={r['sym_best']:.2f}"
            f"  {r['roi1_q']}  {r['sm']:>7} {r['sz_um']:>5}um {r['f_lo']}-{r['f_hi']}Hz o{r['offsets']} {r['iq']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--montage", type=int, default=8)
    ap.add_argument("--csv", default=os.path.join(OUTDIR, "sweep_invivo40.csv"))
    a = ap.parse_args()
    rows = load(a.csv)
    pushes = sorted({r["meas"] for r in rows})
    L = []

    by_roi = sorted(rows, key=lambda r: r["roi1_max"], reverse=True)
    L.append(f"=== TOP {a.k} by ROI-contrast on the stronger lobe;  * = speed pinned at 1.0/5.0 boundary ===")
    L += [row_str(r) for r in by_roi[:a.k]]

    cred = [r for r in by_roi if r["roi1_speed"] not in BOUNDARY]
    L.append(f"\n=== TOP {a.k} CREDIBLE (physiological best-fit speed, not boundary-pinned) ===")
    L += [row_str(r) for r in cred[:a.k]] or ["  (none)"]

    L.append("\n=== PER-PUSH best (by stronger-lobe ROI-contrast) ===")
    L.append(f"  {'push':>4} {'roi':>6} {'side':>4} {'spd':>4} {'symOfBest':>9}  {'q':>4}  best-recipe")
    for m in pushes:
        rm = [r for r in rows if r["meas"] == m]
        b = max(rm, key=lambda r: r["roi1_max"])
        L.append(f"  {m:>4} {b['roi1_max']:>6.3f} {b['roi1_side']:>4} {b['roi1_speed']:>4.1f} "
                 f"{b['sym_best']:>9.2f}  {b['roi1_q']:>4}  {b['sm']}/{b['f_lo']}-{b['f_hi']}Hz/o{b['offsets']}")

    top = by_roi[:200]
    spd = [r["roi1_speed"] for r in top]
    L.append("\n=== best-fit speed of the top-200 lobe contenders ===")
    for c in sorted(set(spd)):
        n = spd.count(c); L.append(f"  {c:>4.1f} m/s : {'#'*n} {n}")
    L.append(f"  median top-200 speed = {np.median(spd):.2f} m/s; boundary-pinned "
             f"{100*sum(1 for c in spd if c in BOUNDARY)/len(spd):.0f}%; "
             f"left-lobe wins {100*sum(1 for r in top if r['roi1_side']=='L')/len(top):.0f}%")
    L.append(f"\n  reference: phantom no-wave ROI floor ~= {PH_FLOOR}; phantom 20V best ~=0.25, 30V ~=0.35, 50V ~=0.52.")
    report = "\n".join(L)
    print(report)
    open(os.path.join(OUTDIR, "invivo40_leaderboard.txt"), "w", encoding="utf-8").write(report)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(spd, bins=iv.SPEEDS.tolist() + [5.5], color="tab:blue")
    ax[0].set_title("best-fit tilt speed, top-200 lobe contenders"); ax[0].set_xlabel("c [m/s]")
    ax[1].scatter([r["sym_best"] for r in rows], [r["roi1_max"] for r in rows], s=3, alpha=0.2)
    ax[1].axhline(PH_FLOOR, color="0.5", ls=":")
    ax[1].set_xlabel("mirror-symmetry (descriptive)"); ax[1].set_ylabel("ROI-contrast, stronger lobe")
    ax[1].set_title("ROI vs symmetry (symmetry does NOT gate)")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "invivo40_speed_sym.png"), dpi=140); plt.close(fig)

    # montage: strongest CREDIBLE candidates (fall back to top-ROI), each lobe drawn at its own speed
    show = (cred or by_roi)[:a.montage]
    rng = np.random.default_rng(0)
    recipes = [sw.sample_recipe(rng) for _ in range(700)]
    tmpl = json.load(open(os.path.join(OUTDIR, "v_roi_template.json"), encoding="utf-8"))
    ncol = min(4, a.montage); nrow = int(np.ceil(len(show) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    cache = {}
    for ax, r in zip(axs.ravel(), show):
        m = r["meas"]
        if m not in cache:
            acq = core.load_acq(iv.FOLDER, m, iv.REC); ml = core.load_mline_for(iv.FOLDER, m, acq, iv.REC)
            cache[m] = (acq, ml, _r0_lateral_crossing(ml, float(acq.push_x)), {})
        acq, ml, r0, ec = cache[m]
        rec = recipes[r["recipe_id"]]
        if rec["iq"] not in ec:
            ec[rec["iq"]] = sw.estimator_for_iq(acq, rec["iq"])
        quant = QFULL[r["roi1_q"]]
        st = sw.spacetime_for(ec[rec["iq"]], acq, ml, r0, rec, quant)
        u = UNIT[quant]; rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        cl = (robust_clim(st.data, rc, 97) * u) or 1.0
        rr = st.r * 1e3; tt = st.t * 1e3
        ax.imshow(st.data * u, extent=(rr[0], rr[-1], tt[-1], tt[0]), cmap="RdBu_r",
                  vmin=-cl, vmax=cl, aspect="auto", origin="upper")
        left = st.r < r0; right = st.r >= r0
        for cc, mask, hi in ((r["cL"], left, r["roi1_side"] == "L"), (r["cR"], right, r["roi1_side"] == "R")):
            if np.isfinite(cc):
                tl = (tmpl["t0_s"] + np.abs(st.r - r0) / cc) * 1e3
                ax.plot(st.r[mask] * 1e3, tl[mask], "k-", lw=1.4 if hi else 0.8, alpha=1.0 if hi else 0.5)
        ax.axvline(r0 * 1e3, color="0.3", ls="--", lw=0.8)
        ax.set_title(f"m{m} id{r['recipe_id']} {r['roi1_side']} c={r['roi1_speed']:.1f} ({quant[:3]})\n"
                     f"roi={r['roi1_max']:.2f} [L{r['roi1_L']:.2f}/R{r['roi1_R']:.2f}] sym={r['sym_best']:.2f}",
                     fontsize=8)
        ax.set_xlabel("r [mm]", fontsize=8); ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("40 V in-vivo: strongest speed-scanned lobe contenders (each lobe at its own tilt)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUTDIR, "invivo40_montage.png"), dpi=140); plt.close(fig)
    print("\nwrote invivo40_leaderboard.txt, invivo40_speed_sym.png, invivo40_montage.png")


if __name__ == "__main__":
    main()
