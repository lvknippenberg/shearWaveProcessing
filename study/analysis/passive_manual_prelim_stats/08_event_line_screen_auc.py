"""Event-line semblance / burst as a screen (AUC, skip thresholds).

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
s = pd.read_csv(os.path.join(D, "windows_scored.csv"))
y = (s.confidence >= 2).astype(int)
for c in ["sem_velocity gauss", "vel_burst", "vel_flat", "vel_track_best", "sem_displacement gauss", "disp_burst", "score"]:
    print(f"{c:24s} AUC {roc_auc_score(y, s[c]):.2f}")
x = s["vel_burst"]
for thr in (1.1, 1.2, 1.3, 1.4, 1.5):
    drop = x < thr
    print(f"skip prompts with velocity burst ratio < {thr}: skips {drop.sum():2d}/{len(s)} "
          f"(of which conf 0: {(drop & (s.confidence == 0)).sum()}, conf>=2 lost: {(drop & (y == 1)).sum()})")
x = s["sem_velocity gauss"]
for thr in (0.4, 0.5, 0.6):
    drop = x < thr
    print(f"skip prompts with semblance < {thr}: skips {drop.sum():2d} (conf 0: {(drop & (s.confidence == 0)).sum()}, conf>=2 lost: {(drop & (y == 1)).sum()})")
