"""Append hand-picked good / bad / middling recipes to the metric-validation experiment as blind
anchors, then shuffle the plot order so they are indistinguishable during scoring. Run AFTER
metric_experiment_generate.py (it appends to the existing manifest.json / plots/).

    KERAS_BACKEND=torch  <zea-python>  scripts/metric_experiment_handpicked.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))

import core                                              # noqa: E402
from swp.viz.metrics import push_specificity, origin_coherence  # noqa: E402
from metric_experiment_generate import (PHANTOM_25V, QUANTITIES, render_plot, round_dir,
                                        item_record, evaluate)  # noqa: E402


def R(**kw):
    return core.Recipe(mline_source="horizontal_push", **kw)


# (expectation, label, recipe) — expectation is only for OUR analysis, never shown while scoring.
HANDPICKED = [
    # ---- expected GOOD (clean outward wave should be visible) ----
    ("good", "consensus bp120-700 gauss mean3 outward off7",
     R(motion_steps=[("temporal_bandpass", {"f_lo": 120, "f_hi": 700})],
       spatial_steps=[("spatial_smooth", {"sigma_z_m": 600, "sigma_x_m": 1200})],
       temporal_steps=[("temporal_moving_mean", {"window": 3})], offsets=7, directional="outward")),
    ("good", "bulk-motion + bp120-700 gauss mean3 outward",
     R(iq_steps=[("bulk_motion_compensation", {})],
       motion_steps=[("temporal_bandpass", {"f_lo": 120, "f_hi": 700})],
       spatial_steps=[("spatial_smooth", {"sigma_z_m": 600, "sigma_x_m": 1200})],
       temporal_steps=[("temporal_moving_mean", {"window": 3})], offsets=7, directional="outward")),
    ("good", "bp80-400 median mean3 outward off5",
     R(motion_steps=[("temporal_bandpass", {"f_lo": 80, "f_hi": 400})],
       spatial_steps=[("spatial_median", {})],
       temporal_steps=[("temporal_moving_mean", {"window": 3})], offsets=5, directional="outward")),
    ("good", "minimal bp100-600 outward off1",
     R(motion_steps=[("temporal_bandpass", {"f_lo": 100, "f_hi": 600})], spatial_steps=[],
       temporal_steps=[], offsets=1, directional="outward")),
    # ---- expected MIDDLING ----
    ("mid", "poly3 gauss mean3 outward",
     R(motion_steps=[("polynomial_drift", {"order": 3})],
       spatial_steps=[("spatial_smooth", {})], temporal_steps=[("temporal_moving_mean", {})],
       directional="outward")),
    ("mid", "kasai bp120-700 median mean3 off3",
     R(estimator="kasai", motion_steps=[("temporal_bandpass", {"f_lo": 120, "f_hi": 700})],
       spatial_steps=[("spatial_median", {})], temporal_steps=[("temporal_moving_mean", {})],
       offsets=3, directional="outward")),
    # ---- expected BAD (no / destroyed wave) ----
    ("bad", "raw: no motion/spatial/temporal, no directional",
     R(motion_steps=[], spatial_steps=[], temporal_steps=[], offsets=1, directional="none")),
    ("bad", "wrong band bp400-900 (removes the wave)",
     R(motion_steps=[("temporal_bandpass", {"f_lo": 400, "f_hi": 900})],
       spatial_steps=[("spatial_smooth", {})], temporal_steps=[("temporal_moving_mean", {})],
       directional="outward")),
    ("bad", "over-smoothed: strong NLM + long savgol",
     R(motion_steps=[("temporal_bandpass", {"f_lo": 120, "f_hi": 700})],
       spatial_steps=[("nlm_denoise", {"h_um": 15})],
       temporal_steps=[("savgol_temporal", {"window": 15, "polyorder": 2})], offsets=9,
       directional="outward")),
    ("bad", "frame-to-frame, no motion removal",
     R(mode="frame_to_frame", motion_steps=[], spatial_steps=[], temporal_steps=[],
       directional="none")),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="round2")
    args = ap.parse_args()
    rdir = round_dir(args.round)
    man_path = os.path.join(rdir, "manifest.json")
    manifest = json.load(open(man_path, encoding="utf-8"))
    items = manifest["items"]
    for it in items:
        it.setdefault("handpicked", False)
        it.setdefault("expectation", "")
        it.setdefault("label", "random")
    next_id = max(it["id"] for it in items) + 1

    r0cfg = R()
    acq = core.load_acq(PHANTOM_25V, 0, r0cfg)
    ml = core.load_mline_for(PHANTOM_25V, 0, acq, r0cfg)
    plots_dir = os.path.join(rdir, "plots")

    print(f"[{args.round}] appending {len(HANDPICKED)} hand-picked anchors (ids from {next_id}):")
    for k, (exp, label, rec) in enumerate(HANDPICKED):
        cols = evaluate(acq, ml, rec)
        cid = next_id + k
        png = os.path.join(plots_dir, f"plot_{cid:03d}.png")
        render_plot({q: (rp, cols[q][1]["S"]) for q, (rp, _) in cols.items()}, png)
        items.append(item_record(cid, os.path.relpath(png, rdir), rec, cols,
                                 handpicked=True, expectation=exp, label=label))
        print(f"  [{exp:4s}] {label:48s} S_max={items[-1]['S_max']:.2f}")

    # shuffle the display order so anchors are interspersed (ids stay stable; analyzer keys by id)
    rng = np.random.default_rng(123)
    rng.shuffle(items)
    manifest["items"] = items
    manifest["n"] = len(items)
    manifest["n_handpicked"] = len(HANDPICKED)
    json.dump(manifest, open(man_path, "w", encoding="utf-8"), indent=2)
    print(f"\nmanifest now has {len(items)} plots ({len(HANDPICKED)} anchors), shuffled. "
          "Score them blind with swp_gui/score_app.py")


if __name__ == "__main__":
    main()
