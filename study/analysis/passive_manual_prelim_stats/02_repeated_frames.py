"""Q1: event prompts that show identical buffer-1/3 frames; pre-load sources; label patterns.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd, numpy as np
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
L = pd.read_csv(os.path.join(D, "lines.csv"))
W = pd.read_csv(os.path.join(D, "windows.csv"))
ev = L[(L.kind == "event") & (~L.skipped)]
gen = L[L.kind == "general"].set_index("folder")
n_pairs = same13 = same1 = same3 = 0
same_gen = 0
rows = []
for f, g in ev.groupby("folder"):
    g = g.sort_values("window")
    for i in range(len(g)):
        a = g.iloc[i]
        if (a.b1_frame == gen.loc[f, "b1_frame"]) and (a.b3_frame == gen.loc[f, "b3_frame"]):
            same_gen += 1
        for j in range(i + 1, len(g)):
            b = g.iloc[j]
            n_pairs += 1
            s1, s3 = a.b1_frame == b.b1_frame, a.b3_frame == b.b3_frame
            same1 += s1; same3 += s3; same13 += s1 and s3
            if s1 and s3:
                rows.append((a.subject, a.window, b.window, a.label, b.label, int(a.b1_frame), int(a.b3_frame),
                             int(a.b4_frame), int(b.b4_frame)))
print(f"event-line prompts: {len(ev)};  pairs within a folder: {n_pairs}")
print(f"  pairs with the SAME buffer-1 frame: {same1}, same buffer-3 frame: {same3}, both: {same13}")
print(f"  event prompts showing the same b1+b3 frames as the folder's general line: {same_gen}")
print(pd.DataFrame(rows, columns=["subj", "w_a", "w_b", "lab_a", "lab_b", "b1", "b3", "b4_a", "b4_b"]).to_string())
print("\npreload of event lines:\n", ev.preload.str.replace(r"\(buffer \d\)", "", regex=True).value_counts())
print("\nlabel per window:\n", W.label.value_counts(dropna=False))
print("\nwindows per folder label pattern:\n", W.groupby("folder").label.apply(lambda s: "-".join(s.fillna("?"))).value_counts())
