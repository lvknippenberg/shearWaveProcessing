"""Benchmark the normalised-Radon speed (Vos 2017 / Keijzer 2019) against the hand-drawn panels.

The 30 hand-drawn passive panels (15 windows x displacement/velocity, cached by
``score_panels.py --prepare``) carry the manual speed, the automatic slant-stack speed that was
reported, and a visibility score (clear / plausible / guess / none). This adds
``swp.viz.metrics.normalized_radon_speed`` - the line on which the signal is strongest - and
compares all three.

Hand speeds are not ground truth (+/-25 % drawing precision, docs/passive_speed_estimation.md);
the comparison is most meaningful on the ``clear`` panels.

    python study/analysis/radon_benchmark.py  -> study/logs/radon_benchmark.csv + stdout
"""
from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
from swp.provenance import stamp_text                         # noqa: E402
from swp.viz.metrics import normalized_radon_speed, line_tracking  # noqa: E402
from swp.viz.speed.spacetime import SpaceTime                 # noqa: E402

CACHE = _REPO / "study" / "analysis" / "panel_cache"
CONF = _REPO / "study" / "logs" / "panel_confidence.csv"
CMIN, CMAX = 1.0, 20.0


def confidence():
    if not CONF.exists():
        return {}
    with open(CONF) as fh:
        return {(r["subject"], r["window"], r["part"], r["quantity"]): r["meaning"]
                for r in csv.DictReader(fh)}


def hand_line(points, r):
    """(t0 [s] at the M-line centre, speed [m/s]) of the hand line; points are (t ms, r mm)."""
    (t1, r1), (t2, r2) = points
    if abs(t2 - t1) < 1e-9:
        return float("nan"), float("nan")
    c = (r2 - r1) / (t2 - t1)                                  # mm/ms = m/s
    r_mid = 0.5 * (r[0] + r[-1]) * 1e3
    return (t1 + (r_mid - r1) / c) * 1e-3, c


def main():
    conf = confidence()
    rows = []
    for f in sorted(glob.glob(str(CACHE / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        subj, win, part, q = Path(f).stem.split("_")
        st = SpaceTime(z["data"], z["r"], z["t"], str(z["quantity"]))
        hand, auto = float(z["hand"]), float(z["auto"])
        if not np.isfinite(hand):
            continue
        t0h, _ = hand_line(np.asarray(z["points"], float), st.r)
        nr = normalized_radon_speed(st, cmin=CMIN, cmax=CMAX)
        rows.append(dict(subject=subj, window=win[3:], part=part, quantity=q,
                         confidence=conf.get((subj, win[3:], part, q), "?"),
                         hand=hand, slant=auto, radon=nr["speed"], radon_polarity=nr["polarity"],
                         track_hand=line_tracking(st, t0h, hand), track_radon=nr["tracking"]))
    out = _REPO / "study" / "logs" / "radon_benchmark.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write(stamp_text(config=dict(cmin=CMIN, cmax=CMAX, estimator="normalized_radon_speed")))
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out, "\n")

    def summary(sel, name):
        if not sel:
            return
        h = np.array([x["hand"] for x in sel])
        print(f"{name:<11} n={len(sel):2d}", end="")
        for key in ("slant", "radon"):
            e = np.array([x[key] for x in sel])
            same = np.sign(e) == np.sign(h)
            ratio = np.abs(e) / np.abs(h)
            railed = (np.abs(e) >= CMAX * 0.99) | (np.abs(e) <= CMIN * 1.01)
            print(f" | {key}: dir ok {same.mean():4.0%}, median |c|/hand {np.median(ratio):4.2f}, "
                  f"within 25% {np.mean(same & (np.abs(ratio - 1) <= 0.25)):4.0%}, railed {railed.mean():3.0%}",
                  end="")
        th = np.nanmedian([x["track_hand"] for x in sel]); tr = np.nanmedian([x["track_radon"] for x in sel])
        print(f" | tracking hand {th:.2f} radon {tr:.2f}")

    summary(rows, "all")
    for lev in ("clear", "plausible", "guess", "none"):
        summary([x for x in rows if x["confidence"] == lev], lev)
    for q in ("displacement", "velocity"):
        summary([x for x in rows if x["quantity"] == q], q)


if __name__ == "__main__":
    main()
