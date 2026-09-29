"""Round 2: general-line semblance / burst AUC and skip thresholds (after passive_general_screen_eval.py).

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
G = pd.read_csv(os.path.join(D, "general_screen_windows.csv"))
Sn = pd.read_csv(os.path.join(D, "general_screen_scan.csv"))
pd.set_option("display.width", 250)
g = G[G.confidence.notna()].copy()
y = (g.confidence >= 2).astype(int)
print("scored windows:", len(g))
for c in ["sem", "burst", "sem_velocity gauss", "vel_burst"]:
    ok = g[c].notna()
    print(f"  AUC {c:22s} {roc_auc_score(y[ok], g.loc[ok, c]):.2f}  ({'general line' if c in ('sem','burst') else 'event line'})")
g["comb"] = g["sem"].rank(pct=True) + g.burst.rank(pct=True)
print(f"  AUC sem+burst (rank sum, general) {roc_auc_score(y, g.comb):.2f}")
for thr in (0.3, 0.4, 0.5, 0.6):
    drop = g["sem"] < thr
    print(f"  general sem < {thr}: skips {drop.sum():2d}/{len(g)}  conf0 {(drop & (g.confidence == 0)).sum()}/30  "
          f"conf1 {(drop & (g.confidence == 1)).sum()}  usable lost {(drop & (y == 1)).sum()}/57 (clear {(drop & (g.confidence == 3)).sum()})")
print("\nmedian general sem by confidence:", g.groupby("confidence")["sem"].median().round(2).to_dict())
print("median general burst by confidence:", g.groupby("confidence").burst.median().round(2).to_dict())
print("general vs event-line semblance, Spearman: %.2f" % g[["sem", "sem_velocity gauss"]].corr("spearman").iloc[0, 1])
