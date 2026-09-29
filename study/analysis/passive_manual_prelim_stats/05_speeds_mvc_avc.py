"""Q4: hand speeds by label, hand vs auto, AVC timing vs QS2, MVC vs AVC.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
from scipy import stats
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
W = pd.read_csv(os.path.join(D, "windows.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
s = W[W.confidence.notna()].copy()
s["lab"] = s.label.fillna("?")
g = s[s.confidence >= 1].copy()
g["abs_speed"] = g.speed_m_s.abs()
g["abs_disp"] = g.speed_disp_m_s.abs()
print("hand speed |c| by label and confidence (m/s):")
print(g.groupby(["lab", "confidence"]).abs_speed.describe()[["count", "25%", "50%", "75%", "min", "max"]].round(2))
print("\nsign of hand speed (+ = away from r=0) by label:\n", pd.crosstab(g.lab, np.sign(g.speed_m_s)))
print("\nunlinked displacement:", g.unlinked.sum(), " of", len(g))
u = g[g.unlinked == True]
print(u[["subject", "window", "lab", "confidence", "speed_m_s", "speed_disp_m_s"]].to_string())
print("\nanchor view:", g.anchor_view.value_counts().to_dict())
# hand vs auto (velocity gauss), clear windows
c = g[g.confidence >= 2]
for v in ["velocity gauss", "displacement gauss", "velocity median", "Keijzer velocity", "acceleration"]:
    a = c[f"auto_{v}"]
    same_sign = (np.sign(a) == np.sign(c.speed_m_s)).mean()
    ratio = (a.abs() / c.abs_speed)
    print(f"  auto {v:20s} same direction {same_sign:.0%}  |auto|/|hand| median {ratio.median():.2f} IQR {ratio.quantile(.25):.2f}-{ratio.quantile(.75):.2f}  railed(>=19.9) {(a.abs() >= 19.9).sum()}")
# AVC phase vs qs2
a = s[s.expect == "AVC"].copy()
a["d_qs2"] = a.phase_ms - a.qs2_ms
print("\nAVC windows: phase - QS2 (ms) by confidence:\n", a.groupby("confidence").d_qs2.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(0))
m = s[s.expect == "MVC"]
print("\nMVC windows: phase (ms) by confidence:\n", m.groupby("confidence").phase_ms.describe()[["count", "min", "25%", "50%", "75%", "max"]].round(0))
# flat puzzle: clear windows -> flat fraction vs speed
print("\nvel_flat vs |speed| (conf>=2): rho %.2f" % stats.spearmanr(c.vel_flat, c.abs_speed)[0])
print(c.groupby(pd.cut(c.abs_speed, [0, 2, 3, 4, 6, 30])).agg(n=("vel_flat", "size"), flat=("vel_flat", "median"), track=("vel_track", "median")).round(2))
# travel time along the line at hand speed
c = c.assign(travel_ms=c.mline_length_mm / c.abs_speed)
print("\ntravel time over the line at the hand speed (ms): median %.1f, IQR %.1f-%.1f" % (c.travel_ms.median(), c.travel_ms.quantile(.25), c.travel_ms.quantile(.75)))
# per subject paired MVC vs AVC (clear only)
pp = c.groupby(["subject", "lab"]).abs_speed.median().unstack()
pp = pp.dropna(subset=["MVC", "AVC"]) if {"MVC", "AVC"} <= set(pp.columns) else pp.iloc[0:0]
print("\nsubjects with both MVC and AVC conf>=2:", len(pp))
if len(pp) >= 3:
    print(pp[["MVC", "AVC"]].round(2).to_string())
    print("Wilcoxon AVC vs MVC: p %.3f; median ratio AVC/MVC %.2f" % (stats.wilcoxon(pp.AVC, pp.MVC).pvalue, (pp.AVC / pp.MVC).median()))
mv, av = c[c.lab == "MVC"].abs_speed, c[c.lab == "AVC"].abs_speed
print("unpaired MVC n=%d median %.2f | AVC n=%d median %.2f | MWU p %.3f" % (len(mv), mv.median(), len(av), av.median(), stats.mannwhitneyu(mv, av).pvalue))
