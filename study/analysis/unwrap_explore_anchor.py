"""Single-anchor accuracy as a function of the anchor's cardiac phase (2026-09-28). Read-only.

The user's proposal generalised: take ONE buffer-1 frame at phase p (time since R-peak, as a
fraction of RR), find the most similar stored buffer-3 slot q*, and place it at the chronological
buffer-3 frame with the same phase (known from the trigger log) -> head = q* - k. When the buffer-3
span holds that phase twice (two beats), each gives a candidate; a candidate counts as correct if
EITHER is right ('oracle') and the reported accuracy uses the higher-correlation one ('pick').
Evaluated against the trigger count, per 0.05-RR phase bin, plus the diagnostics that explain it:
  motion(p)  1 - corr between buffer-1 frames 40 ms apart at that phase (how fast the image changes)
  peak(p)    sharpness: best buffer-3 corr minus the best corr outside +-1 slot of it
Also per folder: the time gap between the end of buffer 3 and the start of buffer 1.

    python study/analysis/unwrap_explore_anchor.py -> study/logs/unwrap_explore_anchor_20260928.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from unwrap_explore_eval import CACHE, SHIFT, _REPO, cdist, feats, phase  # noqa: E402


def main(key="L", demean=True):
    rows, gaps = [], []
    for p in sorted(glob.glob(str(CACHE / "*.npz"))):
        d = np.load(p)
        truth = int(d["tc_first"])
        if truth < 0 or not bool(d["ecg_ok"]) or not (bool(d["n1_ok"]) and bool(d["n3_ok"])):
            continue
        n = len(d["A3"])
        fm3, fm1 = float(d["frame_ms3"]), float(d["frame_ms1"])
        r = np.asarray(d["r"], float)
        rr = float(np.median(np.diff(r)))
        ph1 = phase(d["t1"] + fm1 / 2, r)
        ph3 = phase(d["t3"] + fm3 / 2 + SHIFT * fm3, r)
        gaps.append(dict(folder=Path(p).stem, n=n, gap_b3_to_b1_ms=float(d["t1"][0] - d["t3"][-1]),
                         b1_span_rr=float((d["t1"][-1] - d["t1"][0] + fm1) / rr),
                         b3_span_rr=float(n * fm3 / rr)))
        X1, X3 = feats(d, key, demean)
        M = X3 @ X1.T
        step = max(1, int(round(40 / fm1)))
        for j in np.flatnonzero(np.isfinite(ph1)):
            ks = np.flatnonzero(np.abs(ph3 - ph1[j]) < fm3 / 2)
            if not ks.size:
                continue
            col = M[:, j]
            qs = int(np.argmax(col))
            far = [col[q] for q in range(n) if cdist(q, qs, n) >= 2]
            heads = [(qs - k) % n for k in ks]
            j2 = min(j + step, len(X1) - 1)
            rows.append(dict(folder=Path(p).stem, n=n, phase_rr=ph1[j] / rr,
                             motion=1 - float(X1[j] @ X1[j2]), peak=float(col[qs] - max(far)),
                             n_cand=len(ks), ok_oracle=any(h == truth for h in heads),
                             ok_first=heads[0] == truth, err_first=cdist(heads[0], truth, n)))
    df = pd.DataFrame(rows)
    df.to_csv(_REPO / "study" / "logs" / "unwrap_explore_anchor_20260928.csv", index=False)
    df["bin"] = (df.phase_rr.clip(0, 0.999) // 0.05) * 0.05
    g = df.groupby("bin").agg(n=("ok_first", "size"), exact_single=("ok_first", "mean"),
                              exact_if_2beats_resolved=("ok_oracle", "mean"), two_beats=("n_cand", lambda s: (s > 1).mean()),
                              motion=("motion", "median"), peak=("peak", "median"))
    pd.set_option("display.width", 200)
    print(f"feature {key}{'d' if demean else ''}: single-anchor accuracy by anchor phase (fraction of RR)")
    print(g.round(3).to_string())
    print("\ncorr(motion, exact) per bin:", round(float(np.corrcoef(g.motion, g.exact_single)[0, 1]), 2))
    G = pd.DataFrame(gaps)
    print("\nbuffer 3 -> buffer 1 gap (ms):", G.gap_b3_to_b1_ms.describe().round(0).to_dict())
    print("buffer-1 span (RR):", G.b1_span_rr.describe().round(2).to_dict())
    print("buffer-3 span (RR):", G.b3_span_rr.describe().round(2).to_dict())


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["L"]), demean=(len(sys.argv) < 3 or sys.argv[2] != "raw"))
