"""Does the shear wave weaken across the 10 pushes of a burst? (transmit-supply sag check)

The hydrophone repeats analysis (SafetyTableAll) found the peak-optimised single capture reading
1.1-1.9x higher than the mean of three consecutive firings, and Verasonics warns that the HV
transmit load (~0.6 A) exceeds the supply (0.5 A). If the rail sags within a burst, the later
pushes of each 10-push phantom measurement should image a weaker wave than the first.

This checks that directly: mirror symmetry and origin coherence as a function of push index,
per (elements, cycles, voltage) cell of the 2026-08-17 sweep.

    python scripts/push_index_trend.py --root D:/swp_ph17 --outdir <dir>
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import swe_lib as L                                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quantity", default="velocity")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for folder in L.measurement_folders(a.root):
        p = L.acq_params(folder)
        for k in range(L.n_pushes(folder)):
            try:
                st, r0 = L.st_for(folder, k, L.REC_PHANTOM, quantity=a.quantity, phantom=True)
            except Exception as exc:                              # noqa: BLE001
                print(f"  {p['el']}el {p['cyc']}c {p['V']:.0f}V push{k}: {exc}")
                continue
            oc, sym = L.scores(st, r0)
            rows.append((p["el"], p["cyc"], int(round(p["V"])), k, oc, sym))
            L._ACQ_CACHE.pop((folder, k, True), None)
        print(f"  {p['el']:2d}el {p['cyc']}c {p['V']:.0f}V done", flush=True)

    arr = np.array([(r[0], r[2], r[3], r[5]) for r in rows], float)   # el, V, push, sym
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    ax = axes[0]
    for el, col in ((41, "tab:blue"), (61, "tab:orange"), (79, "tab:green")):
        for V, alpha in ((30, 0.55), (40, 1.0)):
            m = (arr[:, 0] == el) & (arr[:, 1] == V)
            if not m.any():
                continue
            ks = np.arange(10)
            med = [np.median(arr[m & (arr[:, 2] == k), 3]) for k in ks]
            ax.plot(ks, med, "-o", color=col, alpha=alpha, ms=4,
                    label=f"{el} el, {V} V")
    ax.set_xlabel("push index within the 10-push burst")
    ax.set_ylabel("mirror symmetry (median over the 2 pulse lengths)")
    ax.set_title("wave quality vs push index")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    # normalise each cell to its own first push, then pool
    norm = {}
    for el, cyc, V, k, oc, sym in rows:
        norm.setdefault((el, cyc, V), {})[k] = sym
    rel = np.full((len(norm), 10), np.nan)
    for i, (_, d) in enumerate(sorted(norm.items())):
        if 0 not in d or d[0] <= 0.05:
            continue
        for k, v in d.items():
            rel[i, k] = v / d[0]
    ax.plot(np.arange(10), np.nanmedian(rel, axis=0), "-o", color="0.2", ms=5)
    ax.fill_between(np.arange(10), np.nanpercentile(rel, 25, axis=0),
                    np.nanpercentile(rel, 75, axis=0), color="0.6", alpha=0.25)
    ax.axhline(1.0, color="firebrick", ls="--", lw=1)
    ax.set_xlabel("push index within the burst")
    ax.set_ylabel("mirror symmetry relative to push 0")
    ax.set_title("pooled over all 30 cells (median, IQR)")
    ax.grid(alpha=0.3)

    fig.suptitle("Push-to-push trend within a burst - is the transmit supply sagging?",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(a.outdir, "push_index_trend.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)

    with open(os.path.join(a.outdir, "push_index_trend.csv"), "w", encoding="utf-8",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["elements", "cycles", "V", "push", "oc", "sym"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.3f}", f"{r[5]:.3f}"])
    med = np.nanmedian(rel, axis=0)
    print(f"pooled relative symmetry, push 0 -> 9: {np.round(med, 3)}")
    print(f"push 9 vs push 0: {100*(med[-1]-1):+.1f} %")


if __name__ == "__main__":
    main()
