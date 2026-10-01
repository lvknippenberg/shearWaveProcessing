"""Per-folder score tracks along the general line, for the automatic MVC / AVC window study.

For every cached folder (study/analysis/general_screen_cache), on a 2.5 ms grid of window
centres over the whole recording, for window lengths 100 and 120 ms (clipped at the record ends):

  energy     mean squared velocity (mean over the line)
  prop       the same after removing the per-time spatial mean (the flat, in-phase band)
  sem, c     slant-stack semblance and signed speed of the window (swp.passive_screen.window_scores,
             20 ms pads), i.e. the propagation score the screen uses
  burst      RMS inside / RMS in the pads

-> study/analysis/roi_auto_cache/<subject>__<folder>.npz (regenerable). Analysis:
passive_roi_auto.py.

    python study/analysis/passive_roi_auto_features.py [--workers 6]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
SRC = os.path.join(REPO, "study", "analysis", "general_screen_cache")
OUT = os.path.join(REPO, "study", "analysis", "roi_auto_cache")
STEP = 0.0025
WINDOWS = (0.100, 0.120)


def features(npz):
    from swp.passive_screen import window_scores
    z = np.load(npz, allow_pickle=True)
    v, t, r = np.asarray(z["v_data"], float), np.asarray(z["v_t"], float), np.asarray(z["v_r"], float)
    tc = np.arange(t[0], t[-1] + 1e-9, STEP)
    e = np.mean(v ** 2, axis=1)
    vp = v - v.mean(axis=1, keepdims=True)
    ep = np.mean(vp ** 2, axis=1)
    out = dict(folder=z["folder"], general_hash=z["general_hash"], tc=tc)
    for w in WINDOWS:
        k = f"{int(round(w * 1e3))}"
        ins = [(t >= c - w / 2) & (t <= c + w / 2) for c in tc]
        out[f"energy{k}"] = np.array([e[m].mean() if m.any() else np.nan for m in ins])
        out[f"prop{k}"] = np.array([ep[m].mean() if m.any() else np.nan for m in ins])
        rows = [window_scores(v, r, t, c, half=w / 2) for c in tc]
        for q in ("sem", "c", "burst"):
            out[f"{q}{k}"] = np.array([x[q] for x in rows])
    dst = os.path.join(OUT, os.path.basename(npz))
    np.savez_compressed(dst, **out)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(SRC, "*.npz"))) if not f.endswith("_track.npz")
             and (a.force or not os.path.exists(os.path.join(OUT, os.path.basename(f))))]
    print(f"{len(files)} folder(s)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for dst in ex.map(features, files):
            print("  ", os.path.basename(dst), flush=True)


if __name__ == "__main__":
    main()
