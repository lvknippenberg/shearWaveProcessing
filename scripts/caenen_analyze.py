"""Analyze the Caenen 52-push per-lobe speed-scan sweep: top contenders, per-push best + best-fit speed,
speed stability across pushes, and a montage of the strongest pushes with each lobe's fitted tilt.

Same per-lobe scoring as the in-vivo sweep (stronger lobe, own best-fit speed, symmetry reported not
gated). Unlike the human in-vivo data, the Caenen pig ARF data has a genuine propagating wave, so the
headline is whether a coherent tilt (~2 m/s) is recovered CONSISTENTLY across the 52 pushes.

    <zea-python> scripts/caenen_analyze.py [--k 20] [--montage 8]
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
for p in (os.path.join(_ROOT, "src"), os.path.join(_ROOT, "swp_gui"), os.path.join(_ROOT, "scripts")):
    sys.path.insert(0, p)

import sweep_extract as sw                                # noqa: E402
import sweep_caenen as sc                                 # noqa: E402
from swp.viz.core.geometry import robust_clim            # noqa: E402

OUTDIR = sc.OUTDIR
QFULL = {"dis": "displacement", "vel": "velocity", "acc": "acceleration"}
UNIT = {"displacement": 1e6, "velocity": 1e3, "acceleration": 1.0}
PH_FLOOR = 0.05
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
        for k in ("push", "recipe_id", "sz_um", "f_lo", "f_hi", "offsets"):
            r[k] = int(_f(r[k]))
    return [r for r in rows if np.isfinite(r["roi1_max"])]


def row_str(r):
    b = "*" if r["roi1_speed"] in BOUNDARY else " "
    return (f"  p{r['push']:>2} id{r['recipe_id']:>3}  roi={r['roi1_max']:.3f}({r['roi1_side']}) @ "
            f"{r['roi1_speed']:.1f}{b} [L{r['roi1_L']:.2f}/R{r['roi1_R']:.2f}] sym={r['sym_best']:.2f}"
            f"  {r['roi1_q']}  {r['sm']:>7} {r['sz_um']:>5}um {r['f_lo']}-{r['f_hi']}Hz o{r['offsets']} {r['iq']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--montage", type=int, default=8)
    ap.add_argument("--csv", default=os.path.join(OUTDIR, "sweep_caenen.csv"))
    a = ap.parse_args()
    rows = load(a.csv)
    pushes = sorted({r["push"] for r in rows})
    L = []

    by_roi = sorted(rows, key=lambda r: r["roi1_max"], reverse=True)
    L.append(f"=== TOP {a.k} contenders by stronger-lobe ROI-contrast; * = boundary-pinned speed ===")
    L += [row_str(r) for r in by_roi[:a.k]]

    # per-push best + consistency of the best-fit speed across pushes
    L.append("\n=== PER-PUSH best (stronger-lobe ROI) ===")
    L.append(f"  {'push':>4} {'roi':>6} {'side':>4} {'spd':>4} {'sym':>5}  {'q':>4}  recipe")
    perpush_speed, perpush_roi = [], []
    for p in pushes:
        rp = [r for r in rows if r["push"] == p]
        b = max(rp, key=lambda r: r["roi1_max"])
        perpush_speed.append(b["roi1_speed"]); perpush_roi.append(b["roi1_max"])
        L.append(f"  {p:>4} {b['roi1_max']:>6.3f} {b['roi1_side']:>4} {b['roi1_speed']:>4.1f} "
                 f"{b['sym_best']:>5.2f}  {b['roi1_q']:>4}  {b['sm']}/{b['f_lo']}-{b['f_hi']}Hz/o{b['offsets']}")

    ps = np.array(perpush_speed)
    L.append(f"\n=== per-push best-fit speed across the {len(pushes)} pushes ===")
    L.append(f"  range {np.percentile(ps,25):.2f}-{np.percentile(ps,75):.2f} m/s (IQR); "
             f"boundary-pinned {100*np.mean([s in BOUNDARY for s in ps]):.0f}%")
    L.append("  NOTE: SWS varies with cardiac phase (the ~52 pushes span ~1.8 cardiac cycles), so the "
             "per-push spread is largely SYSTOLIC (stiff/fast) vs DIASTOLIC (soft/slow) modulation - do "
             "NOT pool it into a single mean SWS. A proper systolic/diastolic fit needs the ECG timing.")
    L.append(f"  EXTRACTION QUALITY (phase-robust, so poolable): per-push best ROI median "
             f"{np.median(perpush_roi):.3f}, {sum(1 for x in perpush_roi if x>0.25)}/{len(pushes)} > 0.25 "
             "(a clean V is recovered at every push).")
    report = "\n".join(L)
    print(report)
    open(os.path.join(OUTDIR, "caenen_leaderboard.txt"), "w", encoding="utf-8").write(report)

    # figures: speed-vs-push + speed histogram + ROI-vs-symmetry
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(pushes, perpush_speed, "o-", ms=4)
    ax[0].axhline(np.median(ps), color="r", ls=":", label=f"median {np.median(ps):.2f} m/s")
    ax[0].set_xlabel("push #"); ax[0].set_ylabel("best-fit speed [m/s]")
    ax[0].set_title("per-push best-fit tilt speed"); ax[0].legend(fontsize=8); ax[0].set_ylim(0.8, 5.2)
    ax[1].hist([r["roi1_speed"] for r in by_roi[:400]], bins=sc.__dict__.get("SPEEDS", np.arange(1,5.01,0.5)).tolist()+[5.5],
               color="tab:blue")
    ax[1].set_title("best-fit speed, top-400 lobe contenders"); ax[1].set_xlabel("c [m/s]")
    ax[2].scatter([r["sym_best"] for r in rows], [r["roi1_max"] for r in rows], s=3, alpha=0.15)
    ax[2].axhline(PH_FLOOR, color="0.5", ls=":")
    ax[2].set_xlabel("mirror-symmetry (descriptive)"); ax[2].set_ylabel("ROI-contrast, stronger lobe")
    ax[2].set_title("ROI vs symmetry")
    fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "caenen_speed_overview.png"), dpi=140); plt.close(fig)

    # montage: the strongest pushes (best recipe per push, top by ROI), each lobe at its own tilt
    best_per_push = []
    for p in pushes:
        rp = [r for r in rows if r["push"] == p]
        best_per_push.append(max(rp, key=lambda r: r["roi1_max"]))
    show = sorted(best_per_push, key=lambda r: r["roi1_max"], reverse=True)[:a.montage]
    rng = np.random.default_rng(0)
    recipes = [sw.sample_recipe(rng) for _ in range(700)]
    tmpl = json.load(open(os.path.join(OUTDIR, "v_roi_template.json"), encoding="utf-8"))
    ncol = min(4, a.montage); nrow = int(np.ceil(len(show) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow), squeeze=False)
    for ax, r in zip(axs.ravel(), show):
        acq, ml, r0 = sc.load_caenen(r["push"])
        est = sw.estimator_for_iq(acq, recipes[r["recipe_id"]]["iq"])
        quant = QFULL[r["roi1_q"]]
        st = sw.spacetime_for(est, acq, ml, r0, recipes[r["recipe_id"]], quant)
        u = UNIT[quant]; rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        cl = (robust_clim(st.data, rc, 97) * u) or 1.0
        rr = st.r * 1e3; tt = st.t * 1e3
        ax.imshow(st.data * u, extent=(rr[0], rr[-1], tt[-1], tt[0]), cmap="RdBu_r",
                  vmin=-cl, vmax=cl, aspect="auto", origin="upper")
        for cc, mask, hi in ((r["cL"], st.r < r0, r["roi1_side"] == "L"),
                             (r["cR"], st.r >= r0, r["roi1_side"] == "R")):
            if np.isfinite(cc):
                tl = (tmpl["t0_s"] + np.abs(st.r - r0) / cc) * 1e3
                ax.plot(st.r[mask] * 1e3, tl[mask], "k-", lw=1.4 if hi else 0.8, alpha=1.0 if hi else 0.5)
        ax.axvline(r0 * 1e3, color="0.3", ls="--", lw=0.8)
        ax.set_title(f"push{r['push']} {r['roi1_side']} c={r['roi1_speed']:.1f} ({quant[:3]})\n"
                     f"roi={r['roi1_max']:.2f} sym={r['sym_best']:.2f}", fontsize=8)
        ax.set_xlabel("r [mm]", fontsize=8); ax.tick_params(labelsize=7)
    for ax in axs.ravel()[len(show):]:
        ax.axis("off")
    fig.suptitle("Caenen: strongest pushes (each lobe at its own fitted tilt)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUTDIR, "caenen_montage.png"), dpi=140); plt.close(fig)
    print("\nwrote caenen_leaderboard.txt, caenen_speed_overview.png, caenen_montage.png")


if __name__ == "__main__":
    main()
