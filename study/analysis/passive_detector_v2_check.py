"""Dry run of the detector changes on the folders read with the old energy detector (2026-09-29).

Uses the general-line cache of passive_general_screen.py (nothing is re-read from Z: and nothing is
written to the folders). Two detectors are compared with the windows the reader scored:

  adopted   energy picking in the phase windows, search windows < 50 % inside the recording
            dropped (min_inside), every window screened by the general-line semblance (< 0.3 ->
            no line asked)                                   = configs/passive_manual.yaml
  rejected  semblance picking inside the phase windows (swp.passive_screen.pick_windows)

A detected window is matched to an old one within 50 ms. Caveat: the reader's scores were given on
EVENT lines at the OLD times; a window that matches no old one has no score.

Output: study/logs/passive_manual_prelim/detector_v2_check_{adopted,rejected}.csv + summary.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.mline.select import detect_line_bursts, detect_phase_windows      # noqa: E402
from swp.passive_screen import pick_windows, screen_windows               # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from passive_general_spacetime_plots import track_for                     # noqa: E402

CACHE = os.path.join(REPO, "study", "analysis", "general_screen_cache")
D = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
MATCH_S = 0.050
MIN_INSIDE, SCREEN_MIN = 0.5, 0.3


def adopted(z, tr, rp, rr):
    w = None
    if rp is not None:
        span = (float(z["ov_t"][0]) - 1.5, float(z["ov_t"][-1]) + 1.5)
        rp = rp[(rp >= span[0]) & (rp <= span[1])]
        w, _ = detect_phase_windows(z["ov_D"], z["ov_t"], rp, rr, min_inside=MIN_INSIDE)
    if not w:
        w, _ = detect_line_bursts(z["ov_D"], z["ov_t"], window_ms=100.0, max_events=4)
    return screen_windows(w, tr, SCREEN_MIN)


def rejected(z, tr, rp, rr):
    return pick_windows(tr, rp, rr, energy=(z["ov_t"], z["e_masked"]), min_inside=MIN_INSIDE,
                        screen_min=SCREEN_MIN)


def compare(detector, W):
    rows = []
    for f in sorted(x for x in glob.glob(os.path.join(CACHE, "*.npz")) if not x.endswith("_track.npz")):
        z = np.load(f, allow_pickle=True)
        folder = str(z["folder"])
        tr = track_for(f)                                   # cached with the figures
        rp = np.asarray(z["r_peaks_s"], float)
        rr = float(z["rr_ms"]) * 1e-3
        new = detector(z, tr, rp if rp.size else None, rr if rp.size else None)
        old = W[W.folder == folder]
        used = set()
        for w in new:
            d = (old.t_peak_ms * 1e-3 - w.t_peak).abs()
            j = d.idxmin() if len(d) else None
            hit = j is not None and d[j] <= MATCH_S
            if hit:
                used.add(j)
            rows.append(dict(folder=folder, subject=os.path.basename(os.path.dirname(folder)),
                             side="new", t_ms=w.t_peak * 1e3, expect=w.expect or "top-up",
                             screen=w.screen, screened=bool(w.screened),
                             c_auto=float(tr["c"][int(np.argmin(np.abs(tr["t"] - w.t_peak)))]),
                             old_window=int(old.window[j]) if hit else None,
                             old_conf=old.confidence[j] if hit else np.nan))
        for j, o in old.iterrows():
            if j not in used:
                rows.append(dict(folder=folder, subject=o.subject, side="old only", t_ms=o.t_peak_ms,
                                 expect=o.expect if isinstance(o.expect, str) else "top-up",
                                 screened=False, old_window=int(o.window), old_conf=o.confidence))
    return pd.DataFrame(rows)


def summary(X, W, name):
    n = X[X.side == "new"].copy()
    n["screened"] = n.screened.astype(bool)
    a = n[~n.screened]
    print(f"\n=== {name}: {len(n)} windows in {n.folder.nunique()} folders (old {len(W)}), "
          f"screened {int(n.screened.sum())}, asked {len(a)}")
    print(pd.crosstab(n.old_conf.fillna(-1).map({-1: "no old window", 0: "0", 1: "1", 2: "2", 3: "3"}),
                      n.screened.map({True: "screened", False: "asked"}), margins=True))
    o = X[X.side == "old only"]
    print("old windows not picked at all, by confidence:", o.old_conf.value_counts().sort_index().to_dict())
    print(f"old usable (>= 2) still asked: {int((a.old_conf >= 2).sum())} of {int((W.confidence >= 2).sum())};  "
          f"old zero-scored still asked: {int((a.old_conf == 0).sum())} of {int((W.confidence == 0).sum())};  "
          f"asked without an old score: {int(a.old_conf.isna().sum())}")
    fast = a.c_auto.abs() >= 6
    print(f"asked windows at an automatic speed >= 6 m/s: {int(fast.sum())} of {len(a)}")


def main():
    W = pd.read_csv(os.path.join(D, "windows_scored.csv"))
    for name, det in (("adopted", adopted), ("rejected", rejected)):
        X = compare(det, W)
        X.to_csv(os.path.join(D, f"detector_v2_check_{name}.csv"), index=False)
        summary(X, W, name)


if __name__ == "__main__":
    main()
