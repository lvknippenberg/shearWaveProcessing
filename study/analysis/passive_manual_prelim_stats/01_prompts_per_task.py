"""Q1: is any (folder, task, window) answered twice? prompts per folder and task.

Preliminary evaluation of the manual passive study, 2026-09-29 (docs/passive_manual_prelim_2026-09-29.md). Reads study/logs/passive_manual_prelim/.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(REPO, "src"))
import pandas as pd
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
p = pd.read_csv(os.path.join(D, "prompts.csv"))
a = pd.read_csv(os.path.join(D, "archives.csv"))
print(p.groupby(["task", "action"]).size())
print("\nfolders:", p.folder.nunique())
dup = p[p.n_same > 1]
print("\nrecords answered more than once:", len(dup))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(dup[["subject", "folder", "seq", "time", "task", "window", "action", "hash", "confidence", "n_same"]].to_string())
print("\narchives:\n", a.to_string())
# per folder: event accepts vs windows vs slopes
g = p.groupby(["subject", "folder", "task"]).size().unstack(fill_value=0)
print("\n", g.to_string())
