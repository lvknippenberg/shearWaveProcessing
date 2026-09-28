"""Accuracy vs confidence margin, and safe coverage, of the buffer-3 head estimators (2026-09-28).

Reads study/logs/unwrap_explore_eval_20260928.csv. For each method: accuracy on the trigger-count
ground truth per margin quintile, the margin threshold that keeps >= 98 % exact on ground truth
(highest-margin-first), and how many of the folders WITHOUT a trigger count it would then resolve.
Also: which folder properties go with the current method's errors, and why continuity fails
(wrap phase jump = how far apart in cardiac phase the frames on either side of the wrap are).

    python study/analysis/unwrap_explore_coverage.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
df = pd.read_csv(_REPO / "study" / "logs" / "unwrap_explore_eval_20260928.csv")
gt = df[(df.truth >= 0) & df.ecg_ok]
rest = df[(df.truth < 0) & df.ecg_ok]
pd.set_option("display.width", 220)
print(f"ground truth (ECG ok) {len(gt)}; to resolve without trigger count (ECG ok) {len(rest)}; "
      f"no trigger count AND no valid ECG {int(((df.truth < 0) & ~df.ecg_ok).sum())}\n")

METHODS = ["b1_A", "b1z_A", "b1z_L", "b1zc+ss_A", "b1+ss_A", "b1zc+ss+cont_A", "ss_Ad", "vote_Ad", "cont_Ld"]
rows = []
for m in METHODS:
    g = gt.dropna(subset=[m + "_margin"]).sort_values(m + "_margin", ascending=False)
    ok = (g[m + "_err"] == 0).to_numpy()
    prec = np.cumsum(ok) / np.arange(1, len(ok) + 1)
    k98 = int(np.max(np.flatnonzero(prec >= 0.98)) + 1) if (prec >= 0.98).any() else 0
    thr = g[m + "_margin"].iloc[k98 - 1] if k98 else np.inf
    q = pd.qcut(g[m + "_margin"].rank(method="first"), 5, labels=False)
    by_q = g.groupby(q)[m + "_err"].apply(lambda e: (e == 0).mean()).round(2).tolist()
    rows.append(dict(method=m, exact=round(ok.mean(), 3), exact_by_margin_quintile_low_to_high=by_q,
                     thr98=round(thr, 4), gt_accepted=f"{k98}/{len(g)}",
                     rest_resolved=f"{int((rest[m + '_margin'] >= thr).sum())}/{len(rest)}"))
print(pd.DataFrame(rows).to_string(index=False))

cur = gt.assign(err=gt.b1_A_err, lowm=gt.b1_A_margin < 0.006)
print("\ncurrent method (b1_A) at the 0.006 threshold on ground truth:")
for lab, g in cur.groupby("lowm"):
    print(f"  margin {'<' if lab else '>='} 0.006: n={len(g)}, exact {(g.err == 0).mean():.3f}, within 1 {(g.err <= 1).mean():.3f}")
print("  exact by frame count:", cur.groupby("n").err.apply(lambda e: round((e == 0).mean(), 3)).to_dict())
print("  margin median by frame count:", cur.groupby("n").b1_A_margin.median().round(4).to_dict())
print("  margin median, folders w/o trigger count, by frame count:",
      rest.groupby("n").b1_A_margin.median().round(4).to_dict())

print("\ncontinuity (cont_Ld) exact vs wrap phase jump (ms between the cardiac phases either side of the wrap):")
c = gt.assign(bin=pd.cut(gt.wrap_jump_ms, [0, 50, 100, 200, 300, 600]))
print(c.groupby("bin", observed=True).cont_Ld_err.agg(n="size", exact=lambda e: round((e == 0).mean(), 2)).to_string())
print("wrap jump median (ms): ground truth", round(gt.wrap_jump_ms.median()), "| rest", round(rest.wrap_jump_ms.median()),
      "| by n:", df.groupby("n").wrap_jump_ms.median().round(0).to_dict())
