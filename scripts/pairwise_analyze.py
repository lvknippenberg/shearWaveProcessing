"""Rank a dataset's plots from pairwise comparisons (Bradley-Terry) and correlate the ranking with the
recipe parameters to pin the optimum. Reads <round>/pairs.csv + manifest.json.

    <zea-python> scripts/pairwise_analyze.py [--round finetune_50V]
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


def bradley_terry(items, comps, iters=500):
    idx = {it: i for i, it in enumerate(items)}
    n = len(items)
    W = np.zeros(n)
    N = np.zeros((n, n))
    for a, b, win in comps:
        i, j = idx[a], idx[b]
        N[i, j] += 1; N[j, i] += 1
        if win == "a":
            W[i] += 1
        elif win == "b":
            W[j] += 1
        else:
            W[i] += 0.5; W[j] += 0.5
    p = np.ones(n)
    for _ in range(iters):
        newp = np.empty(n)
        for i in range(n):
            denom = np.sum(N[i] / (p[i] + p))
            newp[i] = W[i] / denom if denom > 0 else p[i]
        newp = np.clip(newp, 1e-9, None)
        newp /= newp.mean()
        if np.max(np.abs(newp - p)) < 1e-9:
            p = newp; break
        p = newp
    return {it: float(p[idx[it]]) for it in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None)
    a = ap.parse_args()
    rn = a.round
    if rn is None:
        cr = os.path.join(BASE, "current_round.txt")
        rn = open(cr, encoding="utf-8").read().strip() if os.path.exists(cr) else "finetune_50V"
    OUT = os.path.join(BASE, rn)
    items_man = {it["id"]: it for it in json.load(open(os.path.join(OUT, "manifest.json"),
                                                       encoding="utf-8"))["items"]}
    pairs_path = os.path.join(OUT, "pairs.csv")
    if not os.path.exists(pairs_path):
        raise SystemExit(f"no pairs.csv in {OUT} — run the pairwise app first")
    comps = [(int(r["a"]), int(r["b"]), r["winner"])
             for r in csv.DictReader(open(pairs_path, encoding="utf-8"))]
    present = sorted({c[0] for c in comps} | {c[1] for c in comps})
    print(f"[{rn}] {len(comps)} comparisons over {len(present)} plots "
          f"(~{2*len(comps)/max(1,len(present)):.1f} per plot)")

    strength = bradley_terry(present, comps)
    score = {i: float(np.log(strength[i])) for i in present}   # log-strength = BT ability

    # rank + params
    rows = sorted(present, key=lambda i: score[i], reverse=True)
    print("\ntop 5 by pairwise ranking (recipe):")
    for i in rows[:5]:
        rec = items_man[i]["recipe"]
        sp = rec["spatial_steps"][0][0] if rec["spatial_steps"] else "none"
        print(f"  id{i:3d} BT={score[i]:+.2f}  dir={rec['directional']} off{rec['offsets']} "
              f"sp={sp} tp={(rec['temporal_steps'][0][0] if rec['temporal_steps'] else 'none')} "
              f"band={_band(rec)}")

    # feature -> mean BT strength (binary) and numeric correlations
    y = np.array([score[i] for i in present])
    def feat_present(name):
        return np.array([_has(items_man[i]["recipe"], name) for i in present])
    print("\n=== which parameters the pairwise ranking favours ===")
    cats = {"dir=outward": lambda r: r["directional"] == "outward",
            "dir=none": lambda r: r["directional"] == "none",
            "offsets=5": lambda r: r["offsets"] == 5, "offsets=7": lambda r: r["offsets"] == 7,
            "offsets=9": lambda r: r["offsets"] == 9,
            "spatial=gauss": lambda r: _first(r, "spatial_steps") == "spatial_smooth",
            "spatial=median": lambda r: _first(r, "spatial_steps") == "spatial_median",
            "spatial=nlm": lambda r: _first(r, "spatial_steps") == "nlm_denoise",
            "temporal=mean": lambda r: _first(r, "temporal_steps") == "temporal_moving_mean",
            "temporal=savgol": lambda r: _first(r, "temporal_steps") == "savgol_temporal"}
    res = []
    for name, fn in cats.items():
        m = np.array([fn(items_man[i]["recipe"]) for i in present])
        if 3 <= m.sum() <= len(present) - 3:
            res.append((name, float(y[m].mean() - y[~m].mean()), int(m.sum())))
    res.sort(key=lambda r: r[1], reverse=True)
    for name, d, n in res:
        print(f"  {name:16s} deltaBT={d:+.2f}  (n={n})")
    # numeric params
    print("\nnumeric params (Spearman of value vs BT strength):")
    for pname, getter in [("offsets", lambda r: r["offsets"]),
                          ("gauss sigma_z", lambda r: _param(r, "spatial_smooth", "sigma_z_m")),
                          ("band f_lo", lambda r: _mp(r, "temporal_bandpass", "f_lo")),
                          ("band f_hi", lambda r: _mp(r, "temporal_bandpass", "f_hi"))]:
        vals = np.array([getter(items_man[i]["recipe"]) for i in present], float)
        ok = np.isfinite(vals)
        if ok.sum() > 8:
            print(f"  {pname:14s} rho={spearmanr(vals[ok], y[ok])[0]:+.2f}  (n={int(ok.sum())})")

    # figure: BT strength vs offsets and directional
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    offs = np.array([items_man[i]["recipe"]["offsets"] for i in present])
    for o in sorted(set(offs)):
        ax[0].scatter([o] * (offs == o).sum(), y[offs == o], alpha=0.6)
    ax[0].set_xlabel("offsets"); ax[0].set_ylabel("BT log-strength"); ax[0].set_title("offsets")
    ax[0].grid(alpha=0.3)
    dirs = [items_man[i]["recipe"]["directional"] for i in present]
    for k, d in enumerate(sorted(set(dirs))):
        yy = y[np.array([x == d for x in dirs])]
        ax[1].scatter([k] * len(yy), yy, alpha=0.6); ax[1].text(k, yy.mean(), f"  mean {yy.mean():.2f}")
    ax[1].set_xticks(range(len(set(dirs)))); ax[1].set_xticklabels(sorted(set(dirs)))
    ax[1].set_ylabel("BT log-strength"); ax[1].set_title("directional"); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Pairwise ranking vs parameters ({rn})")
    fig.tight_layout(); out = os.path.join(OUT, "pairwise.png")
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


def _first(rec, stage):
    return rec[stage][0][0] if rec[stage] else "none"


def _has(rec, name):
    return any(s[0] == name for st in ("iq_steps", "motion_steps", "spatial_steps", "temporal_steps")
              for s in rec[st])


def _band(rec):
    for s in rec["motion_steps"]:
        if s[0] == "temporal_bandpass":
            return f"{int(s[1].get('f_lo', 0))}-{int(s[1].get('f_hi', 0))}"
    return "-"


def _mp(rec, method, key):
    for s in rec["motion_steps"]:
        if s[0] == method:
            return float(s[1].get(key, np.nan))
    return np.nan


def _param(rec, method, key):
    for s in rec["spatial_steps"]:
        if s[0] == method:
            return float(s[1].get(key, np.nan))
    return np.nan


if __name__ == "__main__":
    main()
