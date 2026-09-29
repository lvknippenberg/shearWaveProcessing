"""Round 2: detected vs uncovered windows in the folders with uncovered stretches.

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
C = pd.read_csv(os.path.join(D, "general_screen_candidates.csv"))
pd.set_option("display.width", 250)
for subj in ["C000000021", "C000000022", "C000000023", "C000000026", "C000000027"]:
    g = G[G.subject == subj]
    print(f"\n{subj}: RR {g.rr_ms.iloc[0]:.0f} ms, QS2 {g.qs2_ms.iloc[0]:.0f} ms")
    print("  detected:", ", ".join(f"w{int(r.window)} {r.label} @{r.t_peak_ms:.0f} (R+{r.phase_ms:.0f}) conf {r.confidence:.0f} gen-sem {r['sem']:.2f}"
                                  for _, r in g.iterrows()))
    c = C[C.subject == subj]
    print("  event-like:", ", ".join(f"@{r.t_ms:.0f} (R+{r.phase_ms:.0f}) sem {r['sem']:.2f} burst {r.burst:.1f}{'' if r.covered else ' UNCOVERED'}"
                                    for _, r in c.iterrows()))
    # detector energy vs velocity semblance at the AVC candidates: why was the other time chosen?
    z = np.load(glob.glob(os.path.join(C0, f"{subj}__*.npz"))[0])
    t, e = z["ov_t"] * 1e3, z["e_masked"]
    for _, r in c[~c.covered].iterrows():
        k = np.argmin(np.abs(t - r.t_ms))
        print(f"    energy at uncovered {r.t_ms:.0f} ms: {e[k]:.2e};", end="")
    for _, r in g.iterrows():
        k = np.argmin(np.abs(t - r.t_peak_ms))
        print(f" w{int(r.window)}: {e[k]:.2e};", end="")
    print()
