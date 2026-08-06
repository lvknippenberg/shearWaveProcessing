"""Pool all scored rounds and find (a) which recipe methods/settings your high scores favour, and
(b) whether any combination of the stored quantities (origin_coherence, C_push, ...) tracks your
scores better than the current metrics. Uses the saved manifests + scores (no re-running).

    <zea-python> scripts/metric_experiment_patterns.py [--rounds round1 round2]
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
from scipy.stats import spearmanr

BASE = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/metric_experiment"


def load_round(rn):
    man = json.load(open(os.path.join(BASE, rn, "manifest.json"), encoding="utf-8"))
    scores = {}
    sp = os.path.join(BASE, rn, "scores.csv")
    if os.path.exists(sp):
        for row in csv.DictReader(open(sp, encoding="utf-8")):
            scores[int(row["id"])] = int(row["score"])
    out = []
    for it in man["items"]:
        if it["id"] in scores:
            out.append((scores[it["id"]], it))
    return out


def recipe_features(rec):
    """Binary / categorical features of a recipe (for association with the human score)."""
    f = {}
    f["est=" + rec["estimator"]] = 1
    f["mode=" + rec["mode"].split("_")[0]] = 1
    f["dir=" + rec["directional"]] = 1
    f["agg=" + rec["mline_agg"]] = 1
    f["offsets=" + str(rec["offsets"])] = 1
    for stage in ("iq_steps", "motion_steps", "spatial_steps", "temporal_steps"):
        names = [s[0] for s in rec[stage]]
        for nm in names:
            f["has:" + nm] = 1
    # coarse "amount of processing" features
    f["#motion"] = len(rec["motion_steps"])
    f["#iq"] = len(rec["iq_steps"])
    f["any_spatial"] = int(len(rec["spatial_steps"]) > 0)
    f["any_temporal"] = int(len(rec["temporal_steps"]) > 0)
    f["any_bandpass"] = int(any(s[0] in ("temporal_bandpass", "temporal_highpass")
                                for s in rec["motion_steps"]))
    f["any_directional"] = int(rec["directional"] != "none")
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", nargs="+", default=["round1", "round2"])
    args = ap.parse_args()

    data = []
    for rn in args.rounds:
        data += load_round(rn)
    print(f"pooled {len(data)} scored plots from {args.rounds}\n")
    y = np.array([s for s, _ in data], float)

    # ---- (a) recipe-feature -> mean human score ----
    all_feats = set()
    feat_rows = []
    for s, it in data:
        f = recipe_features(it["recipe"])
        feat_rows.append(f); all_feats.update(f)

    print("=== recipe features vs mean human score (binary present/absent, n>=8) ===")
    rows = []
    for feat in sorted(all_feats):
        present = np.array([feat in f for f in feat_rows])
        if feat.startswith(("#",)):    # numeric feature -> correlation
            vals = np.array([f.get(feat, 0) for f in feat_rows], float)
            rho, p = spearmanr(vals, y)
            rows.append((feat, np.nan, np.nan, int(present.sum()), rho))
            continue
        if present.sum() < 8 or (~present).sum() < 8:
            continue
        m_yes, m_no = y[present].mean(), y[~present].mean()
        rows.append((feat, m_yes, m_no, int(present.sum()), m_yes - m_no))
    rows.sort(key=lambda r: (np.nan_to_num(r[4], nan=-9)), reverse=True)
    print(f"{'feature':34s}{'mean(with)':>11}{'mean(without)':>14}{'n':>5}{'delta/rho':>11}")
    for feat, my, mn, n, d in rows:
        my_s = f"{my:.2f}" if np.isfinite(my) else "  -"
        mn_s = f"{mn:.2f}" if np.isfinite(mn) else "  -"
        print(f"  {feat:32s}{my_s:>11}{mn_s:>14}{n:>5}{d:>+11.2f}")

    # ---- (b) do stored metric combinations beat origin_coherence? ----
    def col(key_fn):
        return np.array([key_fn(it) for _, it in data], float)
    oc_max = col(lambda it: max(it["oc"].values()))
    oc_disp = col(lambda it: it["oc"]["displacement"])
    oc_mean = col(lambda it: np.mean(list(it["oc"].values())))
    s_max = col(lambda it: it["S_max"])
    cpush_max = col(lambda it: max(it["C_push"].values()))
    cnopush_max = col(lambda it: max(it["C_nopush"].values()))
    print("\n=== candidate scores vs human (Spearman rho) ===")
    cands = {
        "origin_coherence max": oc_max, "origin_coherence disp": oc_disp,
        "origin_coherence mean": oc_mean, "push_specificity S_max": s_max,
        "C_push max": cpush_max, "C_nopush max": cnopush_max,
        "oc_max * (1 - C_nopush_max)": oc_max * (1 - cnopush_max),
        "oc_max - C_nopush_max": oc_max - cnopush_max,
    }
    ranked = sorted(((spearmanr(v, y)[0], spearmanr(v, y)[1], k) for k, v in cands.items()),
                    reverse=True)
    for rho, p, k in ranked:
        print(f"  {k:34s} rho={rho:+.3f} (p={p:.1e})")

    # ---- figure: top feature deltas ----
    barf = [(feat, d) for feat, my, mn, n, d in rows if np.isfinite(my)]
    barf.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, max(5, 0.32 * len(barf))))
    names = [b[0] for b in barf]; deltas = [b[1] for b in barf]
    colors = ["tab:green" if d > 0 else "tab:red" for d in deltas]
    ax.barh(range(len(names)), deltas, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("mean human score WITH − WITHOUT the feature")
    ax.set_title(f"What your high scores favour ({len(data)} plots, {'+'.join(args.rounds)})")
    fig.tight_layout()
    out = os.path.join(BASE, "patterns.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
