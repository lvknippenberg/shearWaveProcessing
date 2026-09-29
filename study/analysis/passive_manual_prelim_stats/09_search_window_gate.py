"""The AVC search-window gate that was considered (in-sample).

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
s = pd.read_csv(os.path.join(D, "windows_scored.csv"))
edge = (s.pos_in_search < 0.05) | (s.pos_in_search > 0.95)
d = s.phase_ms - s.qs2_ms
for lo, hi in ((-80, 30), (-80, 70), (-90, 70), (-120, 120)):
    out_avc = (s.expect == "AVC") & ((d < lo) | (d > hi))
    drop = edge | out_avc
    print(f"AVC QS2{lo:+d}..{hi:+d} + edge gate: drops {drop.sum():2d} windows: "
          f"conf0 {(drop & (s.confidence == 0)).sum()}, conf1 {(drop & (s.confidence == 1)).sum()}, "
          f"usable lost {(drop & (s.confidence >= 2)).sum()} (clear {(drop & (s.confidence == 3)).sum()})")
print("\nAVC phase-QS2 of usable windows:", sorted(d[(s.expect == 'AVC') & (s.confidence >= 2)].round(0)))
print("AVC phase-QS2 of zero windows:", sorted(d[(s.expect == 'AVC') & (s.confidence == 0)].round(0)))
