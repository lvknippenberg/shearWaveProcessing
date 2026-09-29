"""Q3: confidence by label / phase search / edge peak / ECG; objective measures vs confidence.

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
print("scored windows:", len(s), " line skipped:", W.line_skipped.sum(), " pending:", W.confidence.isna().sum() - W.line_skipped.sum())
print("\nconfidence overall:", s.confidence.value_counts().sort_index().to_dict())
print("\nconfidence by label:\n", pd.crosstab(s.label.fillna("?"), s.confidence, margins=True))
print("\nconfidence by expect (phase search vs energy top-up):\n", pd.crosstab(s.expect.fillna("top-up"), s.confidence, margins=True))
print("\nECG:\n", pd.crosstab(s.ecg_status.fillna("?"), s.confidence))
print("\npeak position in its phase search window (0/1 = edge):")
s["edge"] = (s.pos_in_search < 0.05) | (s.pos_in_search > 0.95)
print(pd.crosstab(s.edge, s.confidence, margins=True))
print(s.groupby("confidence")[["pos_in_search"]].describe())
cols = ["score", "vel_burst", "disp_burst", "vel_flat", "disp_flat", "vel_track_best", "disp_track_best",
        "vel_track", "disp_track", "sem_velocity gauss", "sem_displacement gauss", "mline_length_mm", "hr_bpm", "phase_ms"]
print("\nmedian per confidence:\n", s.groupby("confidence")[cols].median().T.round(3))
print("\nSpearman vs confidence:")
for c in cols:
    ok = s[c].notna()
    if ok.sum() > 5:
        r, pv = stats.spearmanr(s.loc[ok, c], s.loc[ok, "confidence"])
        print(f"  {c:28s} rho {r:+.2f}  p {pv:.3f}  n {ok.sum()}")
# subject effect: fraction of windows with conf>=2 per subject
g = s.groupby("subject").confidence.apply(lambda x: (x >= 2).mean())
print("\nper subject fraction conf>=2:\n", g.round(2).to_string())
print("\nsubjects with >=1 conf>=2 window:", (g > 0).sum(), "of", len(g))
# within-subject consistency: ICC-ish - chi2 of conf>=2 by subject
tab = pd.crosstab(s.subject, s.confidence >= 2)
chi2, pv, *_ = stats.chi2_contingency(tab)
print(f"subject x (conf>=2) chi2 {chi2:.1f} p {pv:.3f}")
