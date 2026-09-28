"""Ideas from Claude/Buffer3_circular/Plan.txt, tested on the unwrap feature cache (2026-09-28).

1. SYNTHETIC GROUND TRUTH (plan sec. 15): buffer 1 is chronological by construction. Subsample it
   to the buffer-3 cadence (39.4 ms, nearest frame), rotate by a random known shift and recover it
   with the image-only methods. This is a true ground truth, independent of the trigger log.
2. TRAJECTORY PREDICTION (plan methods B/D): the wrap is where linear extrapolation
   f_{t+1} ~ 2 f_t - f_{t-1} fails, forwards and backwards - vs plain neighbour similarity.
3. IDENTIFIABILITY (plan phase 4): the phase jump across the wrap (n T mod RR, from the trigger log
   alone) predicts beforehand whether an image-only method can find the wrap.

    python study/analysis/unwrap_explore_plan.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unwrap_explore_eval import CACHE, _REPO, best_margin, cdist, feats, s_cont  # noqa: E402


def s_traj(X, n):
    """Per candidate head (higher = more likely): linear-extrapolation error across the boundary,
    forward (predict slot b from b-2, b-1) + backward (predict b-1 from b+1, b), / median error."""
    fwd = np.array([np.linalg.norm(X[t] - (2 * X[t - 1] - X[t - 2])) for t in range(n)])       # predicts t
    bwd = np.array([np.linalg.norm(X[t] - (2 * X[(t + 1) % n] - X[(t + 2) % n])) for t in range(n)])
    ef, eb = fwd / np.median(fwd), bwd / np.median(bwd)
    return np.array([ef[f] + eb[(f - 1) % n] for f in range(n)])           # higher = more likely wrap


def main(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in sorted(glob.glob(str(CACHE / "*.npz"))):
        d = np.load(p)
        if not bool(d["n1_ok"]):
            continue
        fm1, fm3 = float(d["frame_ms1"]), float(d["frame_ms3"])
        r = np.asarray(d["r"], float)
        rr = float(np.median(np.diff(r))) if r.size > 2 else np.nan
        t1 = d["t1"] - d["t1"][0]
        n = int(np.floor(t1[-1] / fm3)) + 1                    # frames at the loop cadence that fit
        idx = np.clip(np.round(np.arange(n) * fm3 / fm1).astype(int), 0, len(t1) - 1)
        for key, dm in (("L", True), ("A", False)):
            X1, _ = feats(d, key, dm)
            S = X1[idx]                                        # chronological synthetic "buffer 3"
            k = int(rng.integers(n))
            Y = S[(np.arange(n) - k) % n]                      # stored slot q holds chronological (q - k)
            hc, mc = best_margin(-s_cont(Y, n))
            ht, mt = best_margin(s_traj(Y, n))
            wrap = (n * fm3) % rr if np.isfinite(rr) else np.nan
            rows.append(dict(folder=Path(p).stem, feat=key + ("d" if dm else ""), n=n, truth=k,
                             wrap_jump_ms=min(wrap, rr - wrap) if np.isfinite(rr) else np.nan,
                             cont_err=cdist(hc, k, n), cont_margin=mc, traj_err=cdist(ht, k, n), traj_margin=mt))
    df = pd.DataFrame(rows)
    df.to_csv(_REPO / "study" / "logs" / "unwrap_explore_plan_20260928.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"synthetic: {df.folder.nunique()} buffer-1 clips, subsampled to {df.n.min()}-{df.n.max()} frames, random known shift")
    for feat, g in df.groupby("feat"):
        print(f"\n[{feat}] exact: continuity {(g.cont_err == 0).mean():.3f}, trajectory {(g.traj_err == 0).mean():.3f}")
        b = g.assign(bin=pd.cut(g.wrap_jump_ms, [0, 50, 100, 200, 300, 600]))
        print(b.groupby("bin", observed=True).agg(n=("cont_err", "size"),
                                                  continuity=("cont_err", lambda e: round((e == 0).mean(), 2)),
                                                  trajectory=("traj_err", lambda e: round((e == 0).mean(), 2))).to_string())
        for m in ("cont", "traj"):
            q = pd.qcut(g[m + "_margin"].rank(method="first"), 5, labels=False)
            print(f"  {m} exact by margin quintile (low->high):",
                  g.groupby(q)[m + "_err"].apply(lambda e: round((e == 0).mean(), 2)).tolist())


if __name__ == "__main__":
    main()
