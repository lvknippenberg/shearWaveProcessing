"""Compare per-folder EchoPrime calls with a protocol block prior (PLAX block, then PSAX block).

For each subject, in acquisition order, find the single change point k that maximises
sum(log p_PLAX[:k]) + sum(log p_PSAX[k:]) (k = 0..n, so all-PSAX / all-PLAX are allowed).
Folders whose per-folder call disagrees with the block fit are flagged for review. Also
writes the combined decision column `suggested` = per-folder call when confident, block
fit otherwise, and `review` = True when an eye check is advised.

Usage: python block_prior.py views.csv out.csv [--conf 0.8]
"""
import argparse

import numpy as np
import pandas as pd

EPS = 1e-4


def block_fit(p_plax, p_psax):
    a, b = np.log(np.asarray(p_plax) + EPS), np.log(np.asarray(p_psax) + EPS)
    scores = [a[:k].sum() + b[k:].sum() for k in range(len(a) + 1)]
    k = int(np.argmax(scores))
    return np.array(["PLAX"] * k + ["PSAX"] * (len(a) - k)), k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("out")
    ap.add_argument("--conf", type=float, default=0.8)
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    parts = []
    for subj, g in d.groupby("subject", sort=True):
        g = g.copy()
        g["block"], k = block_fit(g.p_PLAX, g.p_PSAX)
        confident = (g.p_view >= a.conf) & (g.frame_agreement >= 0.9) & g.view.isin(["PLAX", "PSAX", "Apical"])
        g["confident"] = confident
        g["suggested"] = np.where(confident, g.view, g.block)
        g["review"] = ~confident | (g.view != g.block)
        parts.append(g)
    r = pd.concat(parts)
    r.to_csv(a.out, index=False)
    n = len(r)
    print(f"{n} folders: confident {r.confident.sum()} ({r.confident.mean():.0%}); "
          f"per-folder vs block disagree {(r.view != r.block).sum()}; "
          f"confident AND disagree {(r.confident & (r.view != r.block)).sum()}; review {r.review.sum()}")
    print(r[r.confident & (r.view != r.block)][["subject", "folder", "view", "p_view", "block"]].to_string())


if __name__ == "__main__":
    main()
