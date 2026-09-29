"""Round 2: search windows beyond the recording / on masked zero energy.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np, glob, os
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
C0 = os.path.join(REPO, "study", "analysis", "general_screen_cache")
G = pd.read_csv(os.path.join(D, "general_screen_windows.csv"))
rows = []
for f in [x for x in glob.glob(os.path.join(C0, "*.npz")) if not x.endswith("_track.npz")]:
    z = np.load(f, allow_pickle=True)
    folder = str(z["folder"])
    t, e = z["ov_t"], z["e_masked"]
    valid = e > 0
    tv0, tv1 = t[valid][0], t[valid][-1]
    rp, rr = z["r_peaks_s"], float(z["rr_ms"]) * 1e-3
    qs2 = (546 - 2.1 * 60 / rr) * 1e-3 if rp.size else np.nan
    for _, w in G[G.folder == folder].iterrows():
        tp = w.t_peak_ms * 1e-3
        k = np.argmin(np.abs(t - tp))
        # the search window this peak came from
        lo = hi = np.nan
        if w.expect == "MVC":
            r = rp[rp <= tp + 1e-4].max(); lo, hi = r, r + 0.150
        elif w.expect == "AVC":
            cand = [(r + qs2 - 0.12, r + qs2 + 0.12) for r in rp if r + qs2 - 0.12 - 1e-4 <= tp <= r + qs2 + 0.12 + 1e-4]
            if cand:
                lo, hi = cand[0]
        inside = (np.clip(hi, tv0, tv1) - np.clip(lo, tv0, tv1)) / (hi - lo) if np.isfinite(lo) else np.nan
        rows.append(dict(subject=w.subject, window=w.window, label=w.label, expect=w.expect, conf=w.confidence,
                         t_ms=w.t_peak_ms, energy_zero=bool(e[k] == 0), search_inside=inside,
                         valid_ms=(tv0 * 1e3, tv1 * 1e3), rec_end_ms=t[-1] * 1e3))
X = pd.DataFrame(rows)
X["truncated"] = X.search_inside < 0.999
print("windows whose peak sits on zero (masked) energy:", X.energy_zero.sum())
print(pd.crosstab([X.expect.fillna("top-up"), X.truncated], X.conf.fillna(-1)))
print("\ntruncated search windows:\n", X[X.truncated].sort_values(["subject", "window"])[
    ["subject", "window", "label", "conf", "t_ms", "search_inside", "energy_zero", "valid_ms"]].round(2).to_string())
print("\nrecording length (ms):", sorted(set(np.round(X.rec_end_ms, -1))))
