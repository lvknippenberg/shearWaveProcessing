"""Q3/Q4: zero-scored windows by detection category; MVC/AVC sensitivity; repeatability. Writes windows_scored.csv.

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
s = W[W.confidence.notna()].copy()
edge = (s.pos_in_search < 0.05) | (s.pos_in_search > 0.95)
cat = np.select([s.expect.isna(), edge], ["energy top-up (no phase match)", "peak on search-window edge"],
                "interior peak in phase window")
s["cat"] = cat
print(pd.crosstab(s.cat, s.confidence, margins=True))
print("\nburst ratio (vel RMS window/pads) median by cat x conf:\n", s.pivot_table(index="cat", columns="confidence", values="vel_burst", aggfunc="median").round(2))
# MVC vs AVC sensitivity
c = s[s.confidence >= 2].copy(); c["v"] = c.speed_m_s.abs()
a = c["auto_velocity gauss"].abs()
start = np.where((a >= 1) & (a < 19.9), a, 3.0)
c["moved"] = (c.v - start).abs() > 0.026
c["travel_frames"] = c.mline_length_mm / c.v / 1.08
def cmp(d, name):
    mv, av = d[d.label == "MVC"].v, d[d.label == "AVC"].v
    p = stats.mannwhitneyu(mv, av).pvalue if len(mv) > 2 and len(av) > 2 else np.nan
    pp = d.groupby(["subject", "label"]).v.median().unstack()
    pp = pp.dropna(subset=["MVC", "AVC"]) if {"MVC", "AVC"} <= set(pp.columns) else pp.iloc[0:0]
    pw = stats.wilcoxon(pp.AVC, pp.MVC).pvalue if len(pp) >= 5 else np.nan
    print(f"{name:34s} MVC n={len(mv):2d} {mv.median():.2f} [{mv.quantile(.25):.2f}-{mv.quantile(.75):.2f}] | "
          f"AVC n={len(av):2d} {av.median():.2f} [{av.quantile(.25):.2f}-{av.quantile(.75):.2f}] | MWU p={p:.2f} | "
          f"paired n={len(pp)} AVC/MVC {np.median(pp.AVC / pp.MVC) if len(pp) else np.nan:.2f} p={pw:.2f}")
cmp(c, "conf>=2")
cmp(c[c.confidence == 3], "conf 3")
cmp(c[c.moved], "conf>=2, slope tilted by hand")
cmp(c[c.travel_frames >= 5], "conf>=2, travel >= 5 frames")
cmp(c[(c.ecg_status != "unusable")], "conf>=2, ECG usable")
print("\nconf>=2 windows with travel < 5 frames (speed poorly resolved):", (c.travel_frames < 5).sum(), "of", len(c))
print("frame period 1.08 ms; line length median %.0f mm" % c.mline_length_mm.median())
# usable per acquisition
u = s.groupby("folder").agg(n=("confidence", "size"), n2=("confidence", lambda x: (x >= 2).sum()),
                             mvc2=("label", lambda x: 0))
u["mvc2"] = s[(s.label == "MVC") & (s.confidence >= 2)].groupby("folder").size().reindex(u.index).fillna(0)
u["avc2"] = s[(s.label == "AVC") & (s.confidence >= 2)].groupby("folder").size().reindex(u.index).fillna(0)
print("\nper acquisition: >=1 usable window %d/%d, >=1 usable MVC %d, >=1 usable AVC %d, both %d" % (
    (u.n2 > 0).sum(), len(u), (u.mvc2 > 0).sum(), (u.avc2 > 0).sum(), ((u.mvc2 > 0) & (u.avc2 > 0)).sum()))
# repeatability: two MVC windows in the same acquisition, both conf>=2
r = c[c.label == "MVC"].groupby("folder").v.agg(list)
pairs = [(x[0], x[1]) for x in r if len(x) >= 2]
d = np.array([abs(a - b) / ((a + b) / 2) for a, b in pairs])
print("MVC beat-to-beat pairs (same acq, both conf>=2): n=%d, |diff|/mean median %.0f%% IQR %.0f-%.0f%%" % (
    len(d), np.median(d) * 100, np.percentile(d, 25) * 100, np.percentile(d, 75) * 100))
r = c[c.label == "AVC"].groupby("folder").v.agg(list)
pairs = [(x[0], x[1]) for x in r if len(x) >= 2]
if pairs:
    d = np.array([abs(a - b) / ((a + b) / 2) for a, b in pairs])
    print("AVC pairs: n=%d, median %.0f%%" % (len(d), np.median(d) * 100))
s.to_csv(os.path.join(D, "windows_scored.csv"), index=False)
