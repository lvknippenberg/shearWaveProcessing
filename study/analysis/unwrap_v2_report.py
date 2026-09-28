"""Summarise the buffer-3 unwrap VERSION 2 re-run of all SW folders (2026-09-28).

Merges study/logs/buffer3_unwrap_v2_w*.csv -> buffer3_unwrap_v2_20260928.csv and reports: result per
method, per campaign (26 / 32 frames) and per acquisition index within a subject; the running
cross-checks where the trigger count exists (combined and continuity agreement); and, for the
folders unwrapped by VERSION 1 (applied head from the unwrap_explore cache), how the head changed.

    python study/analysis/unwrap_v2_report.py
"""
import glob
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
LOGS = _REPO / "study" / "logs"
_TS = re.compile(r"_(\d{1,2}-[A-Za-z]+-\d{4}_\d{2}-\d{2}-\d{2})$")

df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(LOGS / "buffer3_unwrap_v2_w*.csv")))], ignore_index=True)
df = df.sort_values("folder").reset_index(drop=True)
df.to_csv(LOGS / "buffer3_unwrap_v2_20260928.csv", index=False)
df["subject"] = df.folder.str.split("/").str[0]
df["t"] = df.folder.str.split("/").str[1].map(lambda s: datetime.strptime(_TS.search(s).group(1), "%d-%B-%Y_%H-%M-%S"))
df["acq"] = df.groupby("subject").t.rank(method="first").astype(int)
df["result"] = df.method.where(df.status.isin(["unwrapped", "chronological"]), "ambiguous")
pd.set_option("display.width", 200)
print(f"{len(df)} folders; failed: {int((df.status == 'FAILED').sum())}\n")
print("result:", df.result.value_counts().to_dict())
print("\nby campaign (frames):")
print(pd.crosstab(df.n_frames, df.result, margins=True).to_string())
print("\nby acquisition index (1 = first of the subject):")
print(pd.crosstab(df.acq.clip(upper=5).map(lambda a: str(a) if a < 5 else "5+"), df.result, margins=True).to_string())
tc = df[df.method == "trigger-count"]
for c in ("combined_agrees", "continuity_agrees"):
    if c in tc:
        x = tc[c].dropna().astype(bool)
        print(f"\ntrigger-count folders: {c} {int(x.sum())}/{len(x)} ({x.mean():.1%})")
amb = df[df.result == "ambiguous"]
print("\nambiguous, reason:", amb.reason.value_counts().to_dict() if "reason" in amb else {})

ev = pd.read_csv(LOGS / "unwrap_explore_eval_20260928.csv")[["folder", "applied"]]
ev["folder"] = ev.folder.str.replace("__", "/", n=1)
m = df.merge(ev, on="folder", how="left")
v1 = m[m.applied >= 0]
v2first = v1["first"].where(v1.result != "ambiguous", -1)
d = ((v2first - v1.applied + v1.n_frames // 2) % v1.n_frames) - v1.n_frames // 2
print(f"\nfolders unwrapped by VERSION 1: {len(v1)}; v2 head - v1 head:",
      d.where(v2first >= 0).value_counts(dropna=False).sort_index().to_dict(), "(NaN = now ambiguous)")
