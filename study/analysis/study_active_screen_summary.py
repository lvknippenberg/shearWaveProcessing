"""Summarise ``study/logs/study_active_screen.csv`` against the Caenen positive control.

Reference distribution: Caenen's pig pushes scored by the same harness and recipe
(``study/logs/invivo_recipe_contrast.csv``). A study push counts as a *candidate* ARF wave when
its push/control RMS ratio exceeds the 5th percentile of the Caenen pushes AND its origin-coherence
difference is positive - i.e. it looks like the weakest real waves we know of.

    python study/analysis/study_active_screen_summary.py
-> stdout + study/montages/study_active_screen.png
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
RECIPE = "caenen_dir"


def read(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(l for l in fh if not l.startswith("#")))


def main():
    from scipy.stats import wilcoxon
    rows = [r for r in read(_REPO / "study/logs/study_active_screen.csv")
            if r["recipe"] == RECIPE and not r["error"]]
    ref = [r for r in read(_REPO / "study/logs/invivo_recipe_contrast.csv")
           if r["dataset"] == "caenen" and r["recipe"] == RECIPE]
    la = np.array([float(r["log2_amp"]) for r in rows]); doc = np.array([float(r["d_oc"]) for r in rows])
    la_ref = np.array([float(r["log2_amp"]) for r in ref])
    thr = np.percentile(la_ref, 5)
    cand = (la > thr) & (doc > 0)
    subj = sorted({r["subject"] for r in rows})
    print(f"{len(rows)} pushes, {len({r['folder'] for r in rows})} acquisitions, {len(subj)} subjects ({RECIPE})")
    print(f"push/control RMS: median {2 ** np.median(la):.2f}x, above 1 in {np.mean(la > 0):.0%}, "
          f"Wilcoxon p = {wilcoxon(la).pvalue:.1e}")
    print(f"Caenen reference: median {2 ** np.median(la_ref):.1f}x, 5th percentile {2 ** thr:.2f}x")
    print(f"candidates (ratio > Caenen p5 and dOC > 0): {cand.sum()} of {len(rows)} "
          f"({cand.mean():.1%}); in {len({r['subject'] for r, c in zip(rows, cand) if c})} subject(s)")
    for r, c in zip(rows, cand):
        if c:
            print(f"   {r['subject']} {r['folder'][:32]} m{r['meas']}  t_after_R {r['t_after_R_ms']} ms  "
                  f"ratio {2 ** float(r['log2_amp']):.2f}  dOC {float(r['d_oc']):+.2f}  "
                  f"{r['el']} el / {r['cycles']} cyc / {r['V']} V")
    print("\nby push settings:")
    keys = sorted({(r["el"], r["cycles"], r["V"]) for r in rows})
    for k in keys:
        s = np.array([float(r["log2_amp"]) for r in rows if (r["el"], r["cycles"], r["V"]) == k])
        print(f"   {k[0]:>3} el {k[1]:>5} cyc {k[2]:>5} V   n={len(s):4d}  median {2 ** np.median(s):.2f}x  above 1 {np.mean(s > 0):.0%}")

    t = np.array([float(r["t_after_R_ms"]) for r in rows])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.linspace(min(la.min(), la_ref.min()) - 0.2, max(la.max(), la_ref.max()) + 0.2, 50)
    ax[0].hist(la, bins, alpha=0.7, label=f"study ARF pushes (n={len(la)})", density=True)
    ax[0].hist(la_ref, bins, alpha=0.7, label=f"Caenen pig pushes (n={len(la_ref)})", density=True)
    ax[0].axvline(0, color="k", ls="--"); ax[0].axvline(thr, color="C1", ls=":")
    ax[0].set_xlabel("log2(push / no-push RMS)"); ax[0].legend(fontsize=8)
    ax[0].set_title("push effect: study vs a dataset with real ARF waves")
    ok = np.isfinite(t)
    ax[1].plot(t[ok], la[ok], ".", alpha=0.35, ms=4)
    edges = np.arange(0, 1250, 100)
    med = [np.median(la[ok][(t[ok] >= a) & (t[ok] < a + 100)]) if np.any((t[ok] >= a) & (t[ok] < a + 100)) else np.nan
           for a in edges]
    ax[1].plot(edges + 50, med, "-o", color="C3", label="median per 100 ms")
    ax[1].axhline(0, color="k", ls="--"); ax[1].set_xlabel("push time after R-peak [ms] (20 Hz pushes)")
    ax[1].set_ylabel("log2(push / no-push RMS)"); ax[1].legend(fontsize=8)
    ax[1].set_title("by cardiac phase (R-peak gated acquisitions)")
    per = {}
    for r in rows:
        per.setdefault(r["folder"], []).append(float(r["log2_amp"]))
    fm = np.sort([np.median(v) for v in per.values()])
    ax[2].plot(2 ** fm, np.arange(len(fm)), "o", ms=4)
    ax[2].axvline(1, color="k", ls="--"); ax[2].axvline(2 ** np.median(la_ref), color="C1", ls=":", label="Caenen median")
    ax[2].set_xscale("log"); ax[2].set_xlabel("median push/control per acquisition"); ax[2].set_ylabel("acquisition (sorted)")
    ax[2].legend(fontsize=8); ax[2].set_title(f"{len(fm)} acquisitions")
    fig.suptitle(f"In-vivo study ARF screen - literature recipe ({RECIPE}), each push vs its own no-push control, "
                 f"auto septal M-line", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = _REPO / "study/montages/study_active_screen.png"
    fig.savefig(out, dpi=130)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
