"""Does the VERSION-2 unwrap give continuous playback? (2026-09-28)

Per folder: the weakest neighbour link INSIDE the playback sequence (frame k -> k+1, k < n-1; the
GIF loop point is excluded) relative to the median link, in stored slot order and in the VERSION-2
order. A wrap jump inside the sequence shows as a low minimum in stored order; after a correct
unwrap it moves to the loop point and the inside minimum rises to the ordinary-motion level.
Features: low-pass de-meaned log envelope from the unwrap_explore cache (stored order).

    python study/analysis/unwrap_v2_playback_check.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unwrap_explore_eval import CACHE, _REPO, feats  # noqa: E402

v2 = pd.read_csv(_REPO / "study" / "logs" / "buffer3_unwrap_v2_20260928.csv")
rows = []
for _, r in v2.iterrows():
    p = CACHE / (r.folder.replace("/", "__", 1) + ".npz")
    if not p.exists():
        continue
    d = np.load(p)
    _, X = feats(d, "L", True)
    n = len(X)
    link = np.array([X[i] @ X[(i + 1) % n] for i in range(n)])     # stored slot i -> i+1 (cyclic)
    med = np.median(link)
    resolved = r.status in ("unwrapped", "chronological")
    head = int(r["first"]) if resolved else 0
    inside_v2 = [link[(head + k) % n] for k in range(n - 1)]         # playback k -> k+1, k < n-1
    rows.append(dict(folder=r.folder, method=r.method if resolved else "ambiguous",
                     stored_min=float(link[:n - 1].min() / med), v2_min=float(min(inside_v2) / med)))
df = pd.DataFrame(rows)
df.to_csv(_REPO / "study" / "logs" / "unwrap_v2_playback_check_20260928.csv", index=False)
pd.set_option("display.width", 200)
print(f"{len(df)} folders: weakest link inside the playback, relative to the median link (1 = ordinary)")
print(df.groupby("method")[["stored_min", "v2_min"]].median().round(2).to_string())
for lab, g in df.groupby(df.method != "ambiguous"):
    worse = (g.v2_min < g.stored_min - 0.05).sum()
    better = (g.v2_min > g.stored_min + 0.05).sum()
    print(f"{'resolved' if lab else 'ambiguous'} ({len(g)}): v2 better {better}, worse {worse}, "
          f"inside minimum < 0 (a jump): stored {(g.stored_min < 0).sum()} -> v2 {(g.v2_min < 0).sum()}")
