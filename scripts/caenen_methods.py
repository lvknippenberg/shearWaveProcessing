"""Caenen method-combination analysis: which recipe levers best EXTRACT the wave, aggregated over pushes.

Unlike the per-push speed (which varies with cardiac phase and must NOT be pooled), the extraction
QUALITY of a method should hold across phases, so we median the stronger-lobe ROI-contrast over the 52
pushes per recipe and rank recipes + levers. Compared here against the phantom low-SNR sweep and the
in-vivo 40 V sweep to see whether the winning family is the same.

    <zea-python> scripts/caenen_methods.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        for k in ("roi1_max", "sym_best"):
            try:
                r[k] = float(r[k])
            except ValueError:
                r[k] = np.nan
        for k in ("recipe_id", "sz_um", "f_lo", "f_hi", "offsets"):
            r[k] = int(float(r[k]))
    return [r for r in rows if np.isfinite(r["roi1_max"])]


def main():
    rows = load(os.path.join(OUT, "sweep_caenen.csv"))
    n_push = len({r["push"] for r in rows})
    by_rec = defaultdict(list)
    for r in rows:
        by_rec[r["recipe_id"]].append(r)

    # per-recipe: median ROI / sym over pushes + its (constant) lever values
    rec = {}
    for rid, rr in by_rec.items():
        r0 = rr[0]
        rec[rid] = dict(roi=float(np.nanmedian([x["roi1_max"] for x in rr])),
                        sym=float(np.nanmedian([x["sym_best"] for x in rr])),
                        q=Counter(x["roi1_q"] for x in rr).most_common(1)[0][0],
                        iq=r0["iq"], sm=r0["sm"], tm=r0["tm"], sz=r0["sz_um"],
                        f_lo=r0["f_lo"], f_hi=r0["f_hi"], off=r0["offsets"])
    ranked = sorted(rec.items(), key=lambda kv: kv[1]["roi"], reverse=True)

    L = [f"=== Caenen method sweep: {len(rec)} recipes x {n_push} pushes; recipe ROI = median over pushes ==="]
    L.append(f"\n--- TOP 15 recipes (by median stronger-lobe ROI) ---")
    L.append(f"  {'id':>4} {'ROI':>6} {'sym':>5} {'q':>4} {'iq':>6} {'spatial':>7} {'sz_um':>6} "
             f"{'temporal':>8} {'band':>10} {'off':>4}")
    for rid, v in ranked[:15]:
        L.append(f"  {rid:>4} {v['roi']:>6.3f} {v['sym']:>5.2f} {v['q']:>4} {v['iq']:>6} {v['sm']:>7} "
                 f"{v['sz']:>6} {v['tm']:>8} {v['f_lo']}-{v['f_hi']:<6} {v['off']:>4}")

    def lever_table(name, keyfn, fmt=str):
        g = defaultdict(list)
        for v in rec.values():
            g[keyfn(v)].append(v["roi"])
        L.append(f"\n--- lever: {name}  (mean of per-recipe median ROI; n recipes) ---")
        for k in sorted(g, key=lambda k: -np.mean(g[k])):
            L.append(f"    {fmt(k):>14}: {np.mean(g[k]):.3f}   (n={len(g[k])})")

    lever_table("spatial smoother", lambda v: v["sm"])
    lever_table("IQ pre-filter", lambda v: v["iq"])
    lever_table("temporal", lambda v: v["tm"])
    lever_table("best quantity", lambda v: v["q"])
    lever_table("band f_lo [Hz]", lambda v: v["f_lo"], fmt=lambda k: f"{k}")
    lever_table("band f_hi [Hz]", lambda v: v["f_hi"], fmt=lambda k: f"{k}")
    lever_table("offsets", lambda v: v["off"], fmt=lambda k: f"{k}")
    # spatial sigma binned (gauss/median only, sz>0)
    g = defaultdict(list)
    for v in rec.values():
        if v["sz"] > 0:
            g[int(round(v["sz"] / 400) * 400)].append(v["roi"])
    L.append("\n--- spatial size [um] (binned; gauss/median) ---")
    for k in sorted(g):
        L.append(f"    {k:>6}: {np.mean(g[k]):.3f}  (n={len(g[k])})")

    report = "\n".join(L)
    print(report)
    open(os.path.join(OUT, "caenen_methods.txt"), "w", encoding="utf-8").write(report)

    # figure: lever effects
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6))
    def barlever(ax, name, keyfn, ttl):
        g = defaultdict(list)
        for v in rec.values():
            g[keyfn(v)].append(v["roi"])
        ks = sorted(g, key=lambda k: -np.mean(g[k]))
        ax.bar([str(k) for k in ks], [np.mean(g[k]) for k in ks], color="tab:blue")
        ax.set_title(ttl, fontsize=10); ax.tick_params(labelsize=8); ax.set_ylabel("median ROI")
        ax.tick_params(axis="x", rotation=35)
    barlever(axs[0], "sm", lambda v: v["sm"], "spatial smoother")
    barlever(axs[1], "q", lambda v: v["q"], "best quantity")
    barlever(axs[2], "band", lambda v: f"{v['f_lo']}-{v['f_hi']}", "band")
    barlever(axs[3], "iq", lambda v: v["iq"], "IQ pre-filter")
    for a in axs[2:3]:
        a.tick_params(axis="x", rotation=60, labelsize=6)
    fig.suptitle(f"Caenen — which recipe levers best extract the wave (median ROI over {n_push} pushes)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "caenen_methods.png"), dpi=140); plt.close(fig)
    print(f"\nwrote caenen_methods.txt + caenen_methods.png")


if __name__ == "__main__":
    main()
