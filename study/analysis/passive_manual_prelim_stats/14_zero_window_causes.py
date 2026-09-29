"""Round 2: one cause per zero-scored window. Writes window_causes.csv.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np, glob
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
C0 = os.path.join(REPO, "study", "analysis", "general_screen_cache")
G = pd.read_csv(os.path.join(D, "general_screen_windows.csv"))
Cd = pd.read_csv(os.path.join(D, "general_screen_candidates.csv"))
unc = Cd[~Cd.covered]
pd.set_option("display.width", 250)

inside = {}
for f in [x for x in glob.glob(os.path.join(C0, "*.npz")) if not x.endswith("_track.npz")]:
    z = np.load(f, allow_pickle=True)
    folder = str(z["folder"])
    t, e = z["ov_t"], z["e_masked"]
    tv0, tv1 = t[e > 0][0], t[e > 0][-1]
    rp, rr = z["r_peaks_s"], float(z["rr_ms"]) * 1e-3
    qs2 = (546 - 2.1 * 60 / rr) * 1e-3 if rp.size else np.nan
    for _, w in G[G.folder == folder].iterrows():
        tp = w.t_peak_ms * 1e-3
        if w.expect == "AVC":
            c = [(r + qs2 - .12, r + qs2 + .12) for r in rp if r + qs2 - .12 - 1e-4 <= tp <= r + qs2 + .12 + 1e-4]
            lo, hi = c[0]
            inside[(folder, w.window)] = (np.clip(hi, tv0, tv1) - np.clip(lo, tv0, tv1)) / (hi - lo)
G["search_inside"] = [inside.get((f, w), 1.0) for f, w in zip(G.folder, G.window)]


def cause(r):
    if r.expect == "AVC" and r.search_inside < 0.5:
        return "1 AVC search window beyond the recording"
    if not isinstance(r.expect, str):
        return "2 energy ranking (top-up / no ECG)"
    if len(unc[(unc.folder == r.folder) & ((unc.t_ms - r.t_peak_ms).abs() <= 150)]):
        return "3 wrong time: event-like stretch <=150 ms away"
    if r.pos_in_search < 0.05 or r.pos_in_search > 0.95:
        return "4 peak on search edge (no burst inside)"
    return "5 interior peak, nothing visible"


g = G[G.confidence.notna()].copy()
g["cause"] = g.apply(cause, axis=1)
print(pd.crosstab(g.cause, g.confidence, margins=True))
a = g[(g.expect == "AVC") & (g.search_inside >= 0.5)]
print("\nAVC phase windows inside the recording: usable %d of %d" % ((a.confidence >= 2).sum(), len(a)))
m = g[g.expect == "MVC"]
print("MVC phase windows: usable %d of %d" % ((m.confidence >= 2).sum(), len(m)))
print("\ncause 5 windows:\n", g[g.cause.str.startswith("5") & (g.confidence == 0)][
    ["subject", "window", "label", "phase_ms", "qs2_ms", "sem", "burst", "vel_burst"]].round(2).to_string())
g[["folder", "subject", "window", "label", "confidence", "search_inside", "cause"]].to_csv(os.path.join(D, "window_causes.csv"), index=False)
