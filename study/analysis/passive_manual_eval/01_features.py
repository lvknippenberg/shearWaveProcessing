"""Objective space-time features of every event window of a snapshot -> results/<snap>/features.csv

Per window, for each of the five views (prefix = short view name):
  sem        automatic slant-stack semblance (from processed.json = the CURRENT metric)
  autoc      its signed speed
  burst      RMS inside the window / RMS in the +-20 ms pads
  flat       energy share of the per-time spatial mean (in-phase band) inside the window
  best_*     best straight line with mean |signal| objective (track score, speed, coherence)
  bests_*    best straight line with |mean signal| objective (one polarity = one wavefront)
  hand_*     the reader's line: track score and polarity coherence (only where a slope was drawn)
and from the GENERAL line's whole-recording space-time (general_st.npz) at the window:
  gen_sem, gen_c, gen_burst   the detection screen (swp.passive_screen.window_scores)

    python 01_features.py [--snapshot <stamp>] [--jobs 6]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common as C
import stlib as L

SHORT = {"displacement gauss": "disp", "velocity median": "vmed", "velocity gauss": "vg",
         "Keijzer velocity": "keij", "acceleration": "acc"}


def features(args):
    snap, row = args
    out = dict(subject=row["subject"], folder=row["folder"], window=int(row["window"]))
    p = C.st_path(snap, row["subject"], row["folder"], row["window"])
    if not p.exists():
        return out
    st = C.load_st(p)
    t0, t1 = row["t0_ms"] * 1e-3, row["t1_ms"] * 1e-3
    has_line = row.get("state") == "slope" and np.isfinite(row.get("anchor_t_ms", np.nan)) \
        and np.isfinite(row.get("speed", np.nan))
    for v in st["views"]:
        k = SHORT[v["name"]]
        d, t, r = v["data"], v["t"], v["r"]
        rms = L.panel_rms(d, t, t0, t1)
        out[f"{k}_burst"] = L.burst_ratio(d, t, t0, t1)
        out[f"{k}_flat"] = L.flat_fraction(d, t, t0, t1)
        b = L.best_line(d, t, r, t0, t1, objective="abs")
        out.update({f"{k}_best_score": b["score"], f"{k}_best_c": b["c"], f"{k}_best_coh": b["coherence"],
                    f"{k}_best_t": b["t_a"]})
        b = L.best_line(d, t, r, t0, t1, objective="signed")
        out.update({f"{k}_bests_score": b["score"], f"{k}_bests_c": b["c"], f"{k}_bests_coh": b["coherence"],
                    f"{k}_bests_t": b["t_a"]})
        if has_line:
            c = row["speed"] if abs(row["speed"]) >= 0.05 else 0.05
            tr, coh, cov = L.line_score(d, t, r, row["anchor_t_ms"] * 1e-3, row["anchor_r_mm"] * 1e-3, c,
                                        rms=rms, min_cover=0.3)
            out.update({f"{k}_hand_track": tr, f"{k}_hand_coh": coh, f"{k}_hand_cover": cov})
        out["mline_mm"] = float(r[-1] * 1e3)
        out["n_t"] = len(t)
    # the detection screen on the general line at this window
    g = snap / "data" / row["subject"] / row["folder"] / "general_st.npz"
    if g.exists():
        from swp.passive_screen import window_scores
        z = np.load(g)
        v, r, t = np.asarray(z["v"], float), np.asarray(z["r"], float), np.asarray(z["t"], float)
        tc, half = 0.5 * (t0 + t1), 0.5 * (t1 - t0)
        s = window_scores(v, r, t, tc, half=half)
        out.update(gen_sem=s["sem"], gen_c=s["c"], gen_burst=s["burst"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    w = C.tables(snap)["windows"]
    w = w[w.state.isin(["slope", "no-slope", "slope-skipped"])]
    jobs = [(snap, r) for r in w.to_dict("records")]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(features, jobs, chunksize=4))
    f = pd.DataFrame(rows)
    out = C.out_dir(snap, "") / "features.csv"
    f.to_csv(out, index=False)
    print(f"{len(f)} windows -> {out}")


if __name__ == "__main__":
    main()
