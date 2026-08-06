"""Symmetric-V detector for ARF space-time plots, and its validation.

Detects whether the plot contains the symmetric shear-wave "V": two outward wavefronts sharing ONE
apex at (r0, t0 ~ 0). It searches apex-time t0 and slowness 1/c *internally* only to localize the V,
and outputs a DETECTION SCORE (0..1) - NOT a speed. Both lobes must peak at the same apex (the score
is the geometric mean of the two aligned envelope stacks), so a one-sided or non-propagating plot
scores low. Robust to smoothing (smoothing raises the envelope coherence).

Validation: rank each finetune dataset by your pairwise Bradley-Terry (fallback: absolute score), then
show the V-score of the TOP vs BOTTOM contenders + overlay the detected V on the top ones. If the
detector fires on the top (clear-V) plots and not the bottom, it's trustworthy for the 20 V task.

    <zea-python> scripts/detect_v.py [--rounds finetune_50V finetune_30V] [--k 6]
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys

import numpy as np
from scipy.signal import hilbert
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import core                                              # noqa: E402
from swp.viz.core.geometry import robust_clim           # noqa: E402
from pairwise_analyze import bradley_terry              # noqa: E402

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
Q = ["displacement", "velocity", "acceleration"]


def symmetric_v_score(st, r0, d_min=2e-3, d_max=16e-3, nd=40,
                      cmin=1.0, cmax=6.0, n_c=60, t0_max=3.0e-3):
    """MIRROR-SYMMETRY V-detector. Score = Pearson correlation of the LEFT-lobe and RIGHT-lobe envelope
    images (as functions of distance-from-r0 and time): a symmetric shear wave makes the two lobes
    mirror images (score -> 1); noise / cardiac motion / a one-sided pattern does not (score -> 0). No
    speed is fitted or reported. Returns (score, c_localised, t0_localised) - c/t0 only localise the V
    (via an envelope slant-stack) for the *overlay line*, they are NOT a measurement.

    Robust to smoothing (smoothing sharpens the mirror correlation); cleanly rejects the no-wave case
    (a no-push reference scores ~0); grades with SNR (50V~0.97, 30V~0.6, 20V~0.45)."""
    D = st.data - st.data.mean(axis=0, keepdims=True)
    D = D - D.mean(axis=1, keepdims=True)              # kill any uniform bulk band
    E = np.abs(hilbert(D, axis=0))
    d = st.r - r0
    L = np.where(d < 0)[0]
    R = np.where(d >= 0)[0]
    if L.size < 3 or R.size < 3:
        return 0.0, np.nan, np.nan
    dd = np.linspace(d_min, d_max, nd)
    EL = np.stack([np.interp(dd, -d[L][::-1], E[ti, L][::-1]) for ti in range(E.shape[0])])
    ER = np.stack([np.interp(dd, d[R], E[ti, R]) for ti in range(E.shape[0])])
    a = EL.ravel() - EL.mean()
    b = ER.ravel() - ER.mean()
    score = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # localise the V (best shared slope + apex) only to draw the overlay
    nt = E.shape[0]; dt = float(st.t[1] - st.t[0]); base = np.arange(nt)
    ad = np.abs(d); cols = np.where((ad >= d_min) & (ad <= d_max))[0]
    i_early = max(1, int(round(t0_max / dt)))
    bc, bt, bestpk = np.nan, np.nan, -1.0
    for c in np.linspace(cmin, cmax, n_c):
        acc = np.zeros(nt)
        for j, s in zip(cols, ad[cols] / c / dt):
            acc += np.interp(base + s, base, E[:, j])
        pk = float(acc[:i_early].max())
        if pk > bestpk:
            bestpk, bc, bt = pk, float(c), float(base[:i_early][int(np.argmax(acc[:i_early]))] * dt)
    return score, bc, bt


def _env(st):
    D = st.data - st.data.mean(axis=0, keepdims=True)
    D = D - D.mean(axis=1, keepdims=True)
    return np.abs(hilbert(D, axis=0))


def roi_contrast(st, template, r0=None):
    """Wavefront energy contrast inside the LOCKED V-ROI template (from 50 V). For each column on the
    two lobes, compare the envelope peak on the V band ``t = t0 + |r-r0|/c`` (±band) to that column's
    time-median background: contrast (mean_ROI - mean_bg)/(mean_ROI + mean_bg) in [-1,1], >0 = a
    wavefront rides the template. In-window control (median over time), so no NaN and a no-wave plot
    scores ~0. Symmetric (mean of the two lobes). ``template`` = dict saved by draw_v_roi.py."""
    E = _env(st); t = st.t; dt = float(t[1] - t[0])
    r0 = template["r0_m"] if r0 is None else r0
    c = template["c_mps"]; t0 = template["t0_s"]; band = template["band_s"]
    dmin = template.get("d_min_m", 2e-3); dmax = template.get("d_max_m", 16e-3)
    d = np.abs(st.r - r0); nb = max(1, int(round(band / dt)))

    def side(mask):
        cols = np.where(mask & (d >= dmin) & (d <= dmax))[0]
        if cols.size < 3:
            return np.nan
        roi, bg = [], []
        for j in cols:
            ia = int(round((t0 + d[j] / c - t[0]) / dt))
            if 0 <= ia < len(t):
                # unbiased: MEAN envelope on the V band vs MEAN over the whole column (background).
                # For noise band-mean ~ column-mean -> contrast ~0; for a wavefront band-mean > bg -> >0.
                roi.append(float(E[max(0, ia - nb):ia + nb + 1, j].mean()))
                bg.append(float(E[:, j].mean()))
        if not roi:
            return np.nan
        rm, bm = np.mean(roi), np.mean(bg)
        return (rm - bm) / (rm + bm + 1e-12)

    return float(np.nanmean([side(st.r < r0), side(st.r >= r0)]))


def best_over_quantities(acq, ml, rec):
    """V-score per quantity + the max (the V may be clearest in different quantities at different SNR)."""
    out = {}
    for q in Q:
        res = core.run_recipe(acq, ml, core.to_config(dataclasses.replace(rec, quantity=q), acq))
        out[q] = (res, symmetric_v_score(res.st, res.r0))
    vscores = {q: out[q][1][0] for q in Q}
    bq = max(Q, key=lambda q: vscores[q])
    return out, vscores, bq


def ranking(rn):
    """Ordered ids best->worst: Bradley-Terry from pairs.csv if present, else absolute scores."""
    man = json.load(open(os.path.join(BASE, rn, "manifest.json"), encoding="utf-8"))
    items = {it["id"]: it for it in man["items"]}
    pp = os.path.join(BASE, rn, "pairs.csv")
    sp = os.path.join(BASE, rn, "scores.csv")
    if os.path.exists(pp):
        comps = [(int(r["a"]), int(r["b"]), r["winner"]) for r in csv.DictReader(open(pp, encoding="utf-8"))]
        present = sorted({c[0] for c in comps} | {c[1] for c in comps})
        bt = bradley_terry(present, comps)
        order = sorted(present, key=lambda i: bt[i], reverse=True)
        return items, order, man
    if os.path.exists(sp):
        sc = {int(r["id"]): int(r["score"]) for r in csv.DictReader(open(sp, encoding="utf-8"))}
        order = sorted(sc, key=lambda i: sc[i], reverse=True)
        return items, order, man
    raise SystemExit(f"{rn}: neither pairs.csv nor scores.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", default=["finetune_50V", "finetune_30V"])
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args()

    for rn in a.rounds:
        items, order, man = ranking(rn)
        acq = core.load_acq(man["folder"], man.get("meas", 0),
                            core.Recipe(mline_source=man.get("mline", "horizontal_push")))
        ml = core.load_mline_for(man["folder"], man.get("meas", 0), acq,
                                 core.Recipe(mline_source=man.get("mline", "horizontal_push")))
        top, bot = order[:a.k], order[-a.k:]
        print(f"\n=== {rn}: V-detector on TOP {a.k} vs BOTTOM {a.k} (by your ranking) ===")
        cache = {}
        for label, ids in (("TOP", top), ("BOTTOM", bot)):
            for i in ids:
                rec = core.Recipe(**items[i]["recipe"])
                out, vs, bq = best_over_quantities(acq, ml, rec)
                cache[i] = (out, vs, bq)
                print(f"  {label:6s} id{i:3d}  V(disp/vel/acc)={vs['displacement']:.2f}/"
                      f"{vs['velocity']:.2f}/{vs['acceleration']:.2f}  max={max(vs.values()):.2f}({bq[:3]})")
        tv = np.mean([max(cache[i][1].values()) for i in top])
        bv = np.mean([max(cache[i][1].values()) for i in bot])
        print(f"  --> mean V-score  TOP={tv:.2f}  BOTTOM={bv:.2f}  (want TOP >> BOTTOM)")

        # overlay the detected V on the top-k (best quantity)
        fig, axs = plt.subplots(1, a.k, figsize=(3.2 * a.k, 3.6), squeeze=False)
        for ax, i in zip(axs[0], top):
            out, vs, bq = cache[i]
            res, (score, c, t0) = out[bq]
            st = res.st
            unit = 1e6 if bq == "displacement" else (1e3 if bq == "velocity" else 1.0)
            rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
            cl = (robust_clim(st.data, rc, 97) * unit) or 1.0
            r = st.r * 1e3; t = st.t * 1e3
            ax.imshow(st.data * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                      vmin=-cl, vmax=cl, aspect="auto", origin="upper")
            # detected symmetric V lines
            if np.isfinite(c):
                rr = st.r
                tline = (t0 + np.abs(rr - res.r0) / c) * 1e3
                ax.plot(rr[rr >= res.r0] * 1e3, tline[rr >= res.r0], "k-", lw=1.2)
                ax.plot(rr[rr < res.r0] * 1e3, tline[rr < res.r0], "k-", lw=1.2)
            ax.axvline(res.r0 * 1e3, color="0.3", ls="--", lw=0.8)
            ax.set_title(f"id{i} V={score:.2f} ({bq[:3]})", fontsize=8)
            ax.set_xlabel("r [mm]", fontsize=7); ax.tick_params(labelsize=6)
        axs[0][0].set_ylabel("t [ms]", fontsize=8)
        fig.suptitle(f"{rn}: detected symmetric V overlaid on your TOP {a.k} (best quantity)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        out_png = os.path.join(BASE, f"vdetect_{rn}.png")
        fig.savefig(out_png, dpi=140); plt.close(fig)
        print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
