"""Cross-dataset metric check. For every scored dataset, correlate the human 1-5 score with several
quantity-aggregations of origin_coherence (disp / vel / acc / mean-of-all / mean-of-vel+acc /
min-of-vel+acc / max), to (a) find which aggregation matches the human's stated criterion
("velocity+acceleration clarity drives the score, displacement-only is downweighted"), and (b) see
whether origin_coherence holds up across voltage and breaks on the in-vivo (cardiac) case.

Uses stored per-quantity oc in each manifest (no re-running).

    <zea-python> scripts/metric_crossdataset.py
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"
DATASETS = ["phantom_15V", "phantom_30V", "phantom_50V", "invivo_30V", "invivo_40V"]

CANDS = {
    "oc disp": lambda oc: oc["displacement"],
    "oc vel": lambda oc: oc["velocity"],
    "oc acc": lambda oc: oc["acceleration"],
    "oc mean(all 3)": lambda oc: np.mean(list(oc.values())),
    "oc mean(vel,acc)": lambda oc: 0.5 * (oc["velocity"] + oc["acceleration"]),
    "oc min(vel,acc)": lambda oc: min(oc["velocity"], oc["acceleration"]),
    "oc max(all 3)": lambda oc: max(oc.values()),
}


def load(ds):
    md = os.path.join(BASE, ds)
    man = json.load(open(os.path.join(md, "manifest.json"), encoding="utf-8"))
    sp = os.path.join(md, "scores.csv")
    if not os.path.exists(sp):
        return None
    sc = {int(r["id"]): int(r["score"]) for r in csv.DictReader(open(sp, encoding="utf-8"))}
    return [(sc[it["id"]], it["oc"]) for it in man["items"] if it["id"] in sc]


def main():
    results = {}                                          # dataset -> {cand: rho}
    dists = {}
    for ds in DATASETS:
        rows = load(ds)
        if not rows:
            print(f"  {ds}: not scored yet — skipping")
            continue
        y = np.array([s for s, _ in rows], float)
        dists[ds] = np.bincount(y.astype(int), minlength=6)[1:]
        results[ds] = {name: spearmanr([fn(oc) for _, oc in rows], y)[0] for name, fn in CANDS.items()}

    if not results:
        print("no scored datasets found."); return

    names = list(CANDS)
    print(f"\n{'candidate':20s}" + "".join(f"{ds.replace('phantom_','ph').replace('invivo_','iv'):>10}"
                                            for ds in results) + f"{'MEAN':>8}")
    print("-" * (20 + 10 * len(results) + 8))
    means = {}
    for name in names:
        rhos = [results[ds][name] for ds in results]
        means[name] = float(np.nanmean(rhos))
        print(f"{name:20s}" + "".join(f"{results[ds][name]:>+10.2f}" for ds in results)
              + f"{means[name]:>+8.2f}")
    best = max(means, key=lambda k: means[k])
    print(f"\nbest aggregation (mean rho across datasets): {best}  ({means[best]:+.2f})")

    print("\nhuman score distribution (1..5) per dataset:")
    for ds in results:
        d = dists[ds]
        print(f"  {ds:14s} {d}  (mean {np.average(range(1,6), weights=d):.2f})")

    # figure
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names)); w = 0.8 / max(1, len(results))
    for i, ds in enumerate(results):
        ax.bar(x + i * w, [results[ds][n] for n in names], w, label=ds)
    ax.plot(x + 0.4 - w / 2, [means[n] for n in names], "k*-", ms=12, label="mean")
    ax.set_xticks(x + 0.4 - w / 2); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Spearman rho vs human score"); ax.axhline(0, color="k", lw=0.5)
    ax.set_title("origin_coherence quantity-aggregation vs human, per dataset")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(BASE, "crossdataset.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
