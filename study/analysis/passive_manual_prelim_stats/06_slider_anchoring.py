"""Caveat: slopes accepted at the slider's starting (automatic) speed; seconds per prompt.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
W = pd.read_csv(os.path.join(D, "windows.csv"))
s = W[W.confidence >= 1].copy()
a = s["auto_velocity gauss"]
start = np.where((a.abs() >= 1.0) & (a.abs() < 19.9), a.abs(), 3.0)     # slider start rule
s["start"] = start
s["moved"] = (s.speed_m_s.abs() - s.start).abs() > 0.026
print("slopes accepted at the slider's starting speed (not tilted):")
print(pd.crosstab(s.confidence, s.moved, margins=True))
print(pd.crosstab(s.label.fillna("?"), s.moved))
t = s[s.moved]
r = (t["auto_velocity gauss"].abs() / t.speed_m_s.abs())
print("\nmoved slopes: |auto|/|hand| median %.2f IQR %.2f-%.2f" % (r.median(), r.quantile(.25), r.quantile(.75)))
print("hand |c| of moved vs unmoved (conf>=2):", s[(s.confidence >= 2) & s.moved].speed_m_s.abs().median().round(2),
      s[(s.confidence >= 2) & ~s.moved].speed_m_s.abs().median().round(2))
# time spent per slope prompt
P = pd.read_csv(os.path.join(D, "prompts.csv"), parse_dates=["time"])
P = P.sort_values("time")
P["dt_s"] = P.time.diff().dt.total_seconds()
P.loc[P.dt_s > 600, "dt_s"] = np.nan
print("\nseconds per prompt (gaps >10 min dropped):\n", P.groupby("task").dt_s.describe()[["count", "50%", "75%"]].round(0))
print("session span:", P.time.min(), "->", P.time.max())
