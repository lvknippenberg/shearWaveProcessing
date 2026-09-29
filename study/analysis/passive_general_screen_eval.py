"""Evaluate the general-line screen (after passive_general_screen.py).

Q1  Can a score computed on the GENERAL line - available right after detection, before any event
    line is drawn - predict the reader's confidence for each detected window? (AUC vs conf >= 2,
    next to the same score on the event line, which is only available after drawing.)
Q2  Does the detector miss events? The validated score is slid over the whole recording; windows
    that score high but overlap no detected window are candidate missed events, reported with their
    cardiac phase.

Scores per 100 ms window (+-20 ms pads, as the worker processes it), on the "velocity gauss" view:
  sem    slant-stack semblance (swp.viz.metrics.slant_stack_speed, as processed.json 'auto');
  burst  RMS inside the window / RMS in the pads.

Output: study/logs/passive_manual_prelim/general_screen_{windows,scan}.csv
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.manual import store as S                        # noqa: E402
from swp.viz.metrics import slant_stack_speed            # noqa: E402
from swp.viz.speed.spacetime import SpaceTime            # noqa: E402

CACHE = os.path.join(REPO, "study", "analysis", "general_screen_cache")
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
HALF, PAD = 0.050, 0.020
EDGE = 0.030                                            # filter transients at the record ends
STEP = 0.010


def window_scores(v, r, t, tc):
    """Scores of the 100 ms window centred on tc (s)."""
    m = (t >= tc - HALF - PAD) & (t <= tc + HALF + PAD)
    inside = (t >= tc - HALF) & (t <= tc + HALF)
    if m.sum() < 20:
        return dict(sem=np.nan, c=np.nan, burst=np.nan)
    st = SpaceTime(v[m], r, t[m], "velocity")
    sem, c = slant_stack_speed(st, None, cmin=1.0, cmax=20.0, remove_flat=False)
    pads = m & ~inside
    burst = np.sqrt(np.mean(v[inside] ** 2)) / (np.sqrt(np.mean(v[pads] ** 2)) + 1e-30)
    return dict(sem=float(sem), c=float(c), burst=float(burst))


def phase_of(tc, rp):
    prev = rp[rp <= tc]
    return (tc - prev.max()) * 1e3 if prev.size else np.nan


def main():
    W = pd.read_csv(os.path.join(D, "windows_scored.csv"))
    rows, scan = [], []
    for f in sorted(glob.glob(os.path.join(CACHE, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        folder = str(z["folder"])
        v, r, t = np.asarray(z["v_data"], float), np.asarray(z["v_r"], float), np.asarray(z["v_t"], float)
        rp = np.asarray(z["r_peaks_s"], float)
        win = S.read_json(S.Paths(folder).windows_json)
        wins = win["windows"]
        for i, w in enumerate(wins):
            rows.append(dict(folder=folder, window=i, **window_scores(v, r, t, w["t_peak"])))
        # scan the whole recording
        for tc in np.arange(t[0] + EDGE + HALF, t[-1] - EDGE - HALF, STEP):
            sc = window_scores(v, r, t, tc)
            near = [i for i, w in enumerate(wins) if abs(w["t_peak"] - tc) <= HALF]
            scan.append(dict(folder=folder, t_ms=tc * 1e3, phase_ms=phase_of(tc, rp),
                             rr_ms=float(z["rr_ms"]), detected=near[0] if near else -1, **sc))
    G = pd.DataFrame(rows).merge(W, on=["folder", "window"], how="left", suffixes=("_gen", ""))
    Sn = pd.DataFrame(scan)
    G.to_csv(os.path.join(D, "general_screen_windows.csv"), index=False)
    Sn.to_csv(os.path.join(D, "general_screen_scan.csv"), index=False)
    print(f"{G.folder.nunique()} folders, {len(G)} windows ({G.confidence.notna().sum()} scored), "
          f"{len(Sn)} scan positions")


if __name__ == "__main__":
    main()
