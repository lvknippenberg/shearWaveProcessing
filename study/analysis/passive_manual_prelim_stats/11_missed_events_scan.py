"""Round 2: event-like stretches over whole recordings, covered or not. Writes general_screen_candidates.csv.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
G = pd.read_csv(os.path.join(D, "general_screen_windows.csv"))
Sn = pd.read_csv(os.path.join(D, "general_screen_scan.csv"))
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
# thresholds typical of windows the reader scored clear (general-line medians): sem 0.82, burst 2.2
# use the lower quartile of usable windows as the "event-like" bar
u = G[G.confidence >= 2]
S_T, B_T = u["sem"].quantile(0.25), u.burst.quantile(0.25)
print(f"event-like bar (lower quartile of usable windows on the general line): sem >= {S_T:.2f}, burst >= {B_T:.2f}")
Sn["hit"] = (Sn["sem"] >= S_T) & (Sn.burst >= B_T)
print("scan positions that are event-like: %d of %d" % (Sn.hit.sum(), len(Sn)))
# group contiguous hits into candidate events
cands = []
for f, g in Sn.groupby("folder"):
    g = g.sort_values("t_ms").reset_index(drop=True)
    run = []
    for k, row in g.iterrows():
        if row.hit:
            run.append(row)
        if (not row.hit or k == len(g) - 1) and run:
            best = max(run, key=lambda x: x["sem"] * x.burst)
            cands.append(dict(folder=f, subject=f.split("\\")[2], t_ms=best.t_ms, phase_ms=best.phase_ms,
                              rr_ms=best.rr_ms, sem=best["sem"], burst=best.burst, c=best.c,
                              span_ms=(run[-1].t_ms - run[0].t_ms) + 10,
                              covered=any(r.detected >= 0 for r in run)))
            run = []
C = pd.DataFrame(cands)
C["hr"] = 60000 / C.rr_ms
C["qs2"] = 546 - 2.1 * C.hr
C["phase_class"] = np.select(
    [C.phase_ms <= 150, (C.phase_ms - C.qs2).between(-80, 70), C.phase_ms > C.rr_ms - 150],
    ["MVC-like (R+0-150)", "AVC-like (QS2-80..+70)", "late diastole (atrial)"], "other")
print(f"\ncandidate events: {len(C)} in {C.folder.nunique()} folders; covered by a detected window: {C.covered.sum()}")
print(pd.crosstab(C.phase_class, C.covered, margins=True))
m = C[~C.covered].sort_values(["subject", "t_ms"])
print("\nUNCOVERED event-like stretches:\n", m[["subject", "t_ms", "phase_ms", "qs2", "phase_class", "sem", "burst", "c", "span_ms"]].round(2).to_string())
# for each detected window scored 0: is there an uncovered event-like stretch in the same beat-phase class?
Z = G[G.confidence == 0][["folder", "window", "label", "t_peak_ms", "phase_ms", "qs2_ms"]]
print("\nzero-scored windows (%d) with an uncovered candidate within 150 ms: " % len(Z), end="")
n = 0
for _, z in Z.iterrows():
    mm = m[(m.folder == z.folder) & ((m.t_ms - z.t_peak_ms).abs() <= 150)]
    if len(mm):
        n += 1
        print(f"\n  {z.folder.split(chr(92))[2]} w{z.window} {z.label} @{z.t_peak_ms:.0f} ms -> candidate(s) at "
              + ", ".join(f"{t:.0f} ms (sem {s:.2f})" for t, s in zip(mm.t_ms, mm['sem'])), end="")
print(f"\n  total {n}")
C.to_csv(os.path.join(D, "general_screen_candidates.csv"), index=False)
