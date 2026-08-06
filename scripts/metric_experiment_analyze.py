"""Analyse the metric-validation experiment: how well does push_specificity S agree with the human
1-5 scores? Compares S (and, for reference, the old origin_coherence) against the manual scores with
Spearman + Pearson correlation, a scatter, and S-by-score box plots.

    KERAS_BACKEND=torch  <zea-python>  scripts/metric_experiment_analyze.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None, help="round name (default: the active current_round.txt)")
    args = ap.parse_args()
    round_name = args.round
    if round_name is None:
        cr = os.path.join(BASE, "current_round.txt")
        round_name = open(cr, encoding="utf-8").read().strip() if os.path.exists(cr) else "round1"
    OUT_ROOT = os.path.join(BASE, round_name)
    print(f"analysing {round_name}\n")

    manifest = json.load(open(os.path.join(OUT_ROOT, "manifest.json"), encoding="utf-8"))
    items = {it["id"]: it for it in manifest["items"]}
    scores = {}
    with open(os.path.join(OUT_ROOT, "scores.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores[int(row["id"])] = int(row["score"])

    ids = sorted(scores)
    manual = np.array([scores[i] for i in ids], float)
    s_max = np.array([items[i]["S_max"] for i in ids], float)
    oc_max = np.array([max(items[i]["oc"].values()) for i in ids], float)
    # per-quantity S at the human's implied best column (we don't know which they used -> use max)
    print(f"{len(ids)} plots scored.\n")

    def corr(name, x):
        sr, sp = spearmanr(manual, x)
        pr, pp = pearsonr(manual, x)
        print(f"  {name:28s}  Spearman rho={sr:+.3f} (p={sp:.1e})   Pearson r={pr:+.3f} (p={pp:.1e})")
        return sr

    print("Correlation of manual 1-5 score vs:")
    r_s = corr("push_specificity S_max", s_max)
    r_oc = corr("origin_coherence (old) max", oc_max)
    for q in ("displacement", "velocity", "acceleration"):
        corr(f"S ({q})", np.array([items[i]["S"][q] for i in ids], float))

    hp = [i for i in ids if items[i].get("handpicked")]
    if hp:
        print("\nHand-picked anchors (expectation | your score | S_max | label):")
        for i in sorted(hp, key=lambda i: (items[i].get("expectation", ""), -items[i]["S_max"])):
            it = items[i]
            print(f"  [{it.get('expectation',''):4s}] score={scores[i]}  S_max={it['S_max']:.2f}  "
                  f"{it.get('label','')}")

    # ---- figure ----
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.12, 0.12, manual.size)
    is_hp = np.array([bool(items[i].get("handpicked")) for i in ids])
    axs[0].scatter((manual + jit)[~is_hp], s_max[~is_hp], s=26, alpha=0.6, color="tab:blue",
                   label="random")
    exp_color = {"good": "tab:green", "mid": "goldenrod", "bad": "tab:red"}
    for exp, col in exp_color.items():
        m = np.array([items[i].get("expectation") == exp for i in ids]) & is_hp
        if m.any():
            axs[0].scatter((manual + jit)[m], s_max[m], s=110, marker="*", color=col,
                           edgecolor="k", label=f"anchor: {exp}", zorder=5)
    axs[0].set_xlabel("manual score (1-5)"); axs[0].set_ylabel("push_specificity S_max")
    axs[0].set_title(f"NEW metric vs human\nSpearman rho = {r_s:+.2f}")
    axs[0].set_xticks([1, 2, 3, 4, 5]); axs[0].grid(alpha=0.3); axs[0].legend(fontsize=7)

    axs[1].scatter(manual + jit, oc_max, s=28, alpha=0.7, color="tab:red")
    axs[1].set_xlabel("manual score (1-5)"); axs[1].set_ylabel("origin_coherence (old) max")
    axs[1].set_title(f"OLD metric vs human\nSpearman rho = {r_oc:+.2f}")
    axs[1].set_xticks([1, 2, 3, 4, 5]); axs[1].grid(alpha=0.3)

    # S_max distribution grouped by manual score (should rise monotonically if aligned)
    data_by = [s_max[manual == k] for k in range(1, 6)]
    bp = axs[2].boxplot(data_by, positions=range(1, 6), widths=0.6, patch_artist=True,
                        showmeans=True)
    for b in bp["boxes"]:
        b.set(facecolor="tab:blue", alpha=0.5)
    for k in range(1, 6):
        v = s_max[manual == k]
        if v.size:
            axs[2].plot(np.full(v.size, k) + rng.uniform(-0.1, 0.1, v.size), v, ".", color="0.2",
                        ms=4, alpha=0.6)
    axs[2].set_xlabel("manual score (1-5)"); axs[2].set_ylabel("push_specificity S_max")
    axs[2].set_title("S_max distribution per human score\n(monotone rise = good agreement)")
    axs[2].grid(alpha=0.3, axis="y")

    fig.suptitle(f"Metric validation on the 25 V phantom ({len(ids)} blind-scored randomized recipes)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT_ROOT, "metric_validation.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")

    # also dump a merged table for inspection
    with open(os.path.join(OUT_ROOT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "manual", "S_max", "S_best_quantity", "oc_max",
                    "S_disp", "S_vel", "S_acc"])
        for i in ids:
            it = items[i]
            w.writerow([i, scores[i], round(it["S_max"], 3), it["S_best_quantity"],
                        round(max(it["oc"].values()), 3),
                        round(it["S"]["displacement"], 3), round(it["S"]["velocity"], 3),
                        round(it["S"]["acceleration"], 3)])
    print(f"wrote {os.path.join(OUT_ROOT, 'results.csv')}")


if __name__ == "__main__":
    main()
