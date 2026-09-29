"""Q2: line source buffer vs pre-load, motion-corrected saves, registration verdicts.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
L = pd.read_csv(os.path.join(D, "lines.csv"))
L = L[~L.skipped].copy()
L["pre_buf"] = L.preload.str.extract(r"buffer (\d)").astype(float)
pd.set_option("display.width", 250)
print(pd.crosstab([L.kind, L.pre_buf.fillna(-1)], [L.source_buffer, L.motion_correction], margins=True))
mc = L[L.motion_correction == True]
print("\nmotion-corrected lines (saved buffer-4 line != drawn coordinates):", len(mc))
print(mc[["subject", "kind", "window", "label", "preload", "source_buffer", "nudge_mm", "mapping_reliable",
          "mapping_shift_mm", "src_to_b4_shift_mm"]].to_string())
print("\nshift of these lines (mm): median %.2f  max %.2f" % (mc.src_to_b4_shift_mm.median(), mc.src_to_b4_shift_mm.max()))
print("\nregistration verdicts shown on the mirrored buffers (all lines):")
for b in (1, 3):
    print(f"  buffer {b}: reliable {L[f'b{b}_reliable'].value_counts(dropna=False).to_dict()}  shift median {L[f'b{b}_shift_mm'].median():.2f} mm, 90% {L[f'b{b}_shift_mm'].quantile(.9):.2f}")
print("\nnudged lines:", (L.nudge_mm > 0).sum())
# timing of the session: time per line with preload on buffer 1 vs 4
