"""One-at-a-time (OAT) ablation: which processing steps matter most, scored by push-specificity.

From a baseline recipe, vary ONE stage at a time (swap in each alternative method / "none"), holding
the rest fixed, and measure the change in the **push_specificity** score S = C_push·(1 − C_nopush/C_push)
(clarity of the outward wave in the post-push window, penalised by the same recipe's clarity in the
pre-push *no-push* reference — the built-in negative control that origin_coherence lacked). Evaluated on
a small set: the **phantom** (positive control, a real wave with no cardiac motion) + a few **in-vivo**
pushes. Output: an importance bar chart (how much each stage's choice moves S) + the best method per
stage + a per-case breakdown.

    KERAS_BACKEND=torch  <zea-python>  scripts/ablation.py  [--out ablation.png]

Coarse path only (no fine-grid RF re-beamform), so it runs in a couple of minutes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "swp_gui"))

import core                                              # noqa: E402
from swp.viz.metrics import push_specificity             # noqa: E402

PH = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Phantom"
IV = r"D:/Luuk van Knippenberg/Claude/2026_08_04 voltage sweep/Invivo"

# evaluation set: (label, folder, meas, mline_source, weight). Phantom = positive control.
EVAL_SET = [
    ("phantom 50V", PH + "/DefaultPatient_SW_data_04-August-2026_13-21-52", 0, "horizontal_push", 1.0),
    ("phantom 30V", PH + "/DefaultPatient_SW_data_04-August-2026_13-27-38", 0, "horizontal_push", 1.0),
    ("invivo 40V m5", IV + "/Luuk40V_SW_data_04-August-2026_13-36-07", 5, "auto", 1.0),
    ("invivo 30V m6", IV + "/Luuk30V_SW_data_04-August-2026_13-35-06", 6, "auto", 1.0),
    ("invivo 30V m12", IV + "/Luuk30V_SW_data_04-August-2026_13-35-06", 12, "auto", 1.0),
]

BASE = dict(iq_steps=[], estimator="loupas", mode="relative_to_reference", quantity="displacement",
            motion_steps=[("temporal_bandpass", {})], spatial_steps=[("spatial_smooth", {})],
            temporal_steps=[("temporal_moving_mean", {})], offsets=7, offset_step_mm=0.8,
            mline_agg="mean", directional="outward", speed="ttp_ransac")

# each axis: list of (label, override-dict applied on top of BASE)
AXES = {
    "1 IQ pre-filter": [("none", {"iq_steps": []}),
                        ("slowtime-hp", {"iq_steps": [("iq_slowtime_highpass", {})]}),
                        ("SVD-clutter", {"iq_steps": [("svd_clutter", {})]}),
                        ("bulk-motion", {"iq_steps": [("bulk_motion_compensation", {})]})],
    "2 estimator": [("loupas", {"estimator": "loupas"}), ("kasai", {"estimator": "kasai"}),
                    ("xcorr", {"estimator": "xcorr"})],
    "2b mode": [("rel-reference", {"mode": "relative_to_reference"}),
                ("frame-to-frame", {"mode": "frame_to_frame"})],
    "2c quantity": [("displacement", {"quantity": "displacement"}),
                    ("velocity", {"quantity": "velocity"})],
    "3 motion removal": [("none", {"motion_steps": []}),
                         ("bandpass", {"motion_steps": [("temporal_bandpass", {})]}),
                         ("poly3", {"motion_steps": [("polynomial_drift", {"order": 3})]}),
                         ("SVD-field", {"motion_steps": [("svd_clutter_field", {})]}),
                         ("ref-poly", {"motion_steps": [("reference_motion_comp", {})]}),
                         ("ref-subspace", {"motion_steps": [("reference_subspace_projection", {})]}),
                         ("adaptive-hp", {"motion_steps": [("adaptive_highpass", {})]}),
                         ("unwrap+bp", {"motion_steps": [("phase_unwrap_temporal", {}),
                                                         ("temporal_bandpass", {})]})],
    "4 spatial": [("none", {"spatial_steps": []}),
                  ("gaussian", {"spatial_steps": [("spatial_smooth", {})]}),
                  ("median", {"spatial_steps": [("spatial_median", {})]}),
                  ("bilateral", {"spatial_steps": [("bilateral_denoise", {})]}),
                  ("nlm", {"spatial_steps": [("nlm_denoise", {})]}),
                  ("aniso-diff", {"spatial_steps": [("aniso_diffusion", {})]}),
                  ("coherence-diff", {"spatial_steps": [("coherence_diffusion", {})]})],
    "5 temporal": [("none", {"temporal_steps": []}),
                   ("moving-mean", {"temporal_steps": [("temporal_moving_mean", {})]}),
                   ("moving-median", {"temporal_steps": [("temporal_moving_median", {})]}),
                   ("savgol", {"temporal_steps": [("savgol_temporal", {})]})],
    "6 M-line offsets": [("1 line", {"offsets": 1}), ("7 lines", {"offsets": 7})],
    "7 directional": [("off", {"directional": "none"}), ("outward", {"directional": "outward"})],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(PH, "..", "ablation.png"))
    args = ap.parse_args()

    # cache each case's acquisition + M-line once
    cases = []
    for label, folder, meas, src, w in EVAL_SET:
        r0 = core.Recipe(mline_source=src, offsets=7)
        pts = core.mline_points_for(folder, meas)
        acq = core.load_acq(folder, meas, r0, mline_points=pts)
        ml = core.load_mline_for(folder, meas, acq, r0)
        cases.append((label, folder, meas, src, w, acq, ml))
        print(f"  loaded {label}")

    def score(override):
        """mean push-specificity S over the eval set for a recipe = BASE + override."""
        ss = []
        for label, folder, meas, src, w, acq, ml in cases:
            rc = core.Recipe(mline_source=src, **{**BASE, **override})
            cfg = core.to_config(rc, acq)
            try:
                rp = core.run_recipe(acq, ml, cfg)
                rn = core.run_recipe(acq, ml, cfg, nopush=True)
                ss.append((label, push_specificity(rp, rn)["S"], w))
            except Exception as exc:  # noqa: BLE001
                ss.append((label, np.nan, w))
        return ss

    base_cases = score({})
    base_mean = np.nanmean([s for _, s, _ in base_cases])
    print(f"\nBASELINE mean S = {base_mean:.3f}   " +
          "  ".join(f"{l}={s:.2f}" for l, s, _ in base_cases))

    results = {}                                          # axis -> list of (alt_label, mean_S, per_case)
    t0 = time.time()
    for axis, alts in AXES.items():
        rows = []
        for alt_label, override in alts:
            sc = score(override)
            rows.append((alt_label, float(np.nanmean([s for _, s, _ in sc])), sc))
        results[axis] = rows
        best = max(rows, key=lambda r: r[1])
        spread = max(r[1] for r in rows) - min(r[1] for r in rows)
        print(f"  [{axis:18s}] importance(spread)={spread:.3f}  best={best[0]}({best[1]:.3f})  "
              + " ".join(f"{a}:{m:.2f}" for a, m, _ in rows))
    print(f"(ablation ran in {time.time()-t0:.0f}s)")

    # ---- figure: importance per axis (bar) + best method annotation ----
    axes_sorted = sorted(results.items(), key=lambda kv: max(r[1] for r in kv[1]) - min(r[1] for r in kv[1]),
                         reverse=True)
    names = [a for a, _ in axes_sorted]
    spreads = [max(r[1] for r in rows) - min(r[1] for r in rows) for _, rows in axes_sorted]
    best_lbl = [max(rows, key=lambda r: r[1])[0] for _, rows in axes_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 1.4]})
    y = np.arange(len(names))
    ax1.barh(y, spreads, color="tab:purple", alpha=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=9); ax1.invert_yaxis()
    for i, (s, b) in enumerate(zip(spreads, best_lbl)):
        ax1.text(s + 0.005, i, f"best: {b}", va="center", fontsize=8)
    ax1.set_xlabel("importance = spread of mean push-specificity S across the stage's methods")
    ax1.set_title("Which stages matter most", fontsize=11)
    ax1.axvline(0, color="k", lw=0.5)

    # per-alternative mean-S heat-ish table for the top axes
    ax2.axis("off")
    lines = [f"BASELINE mean S = {base_mean:.3f}", ""]
    for axis, rows in axes_sorted:
        rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
        lines.append(f"{axis}:")
        lines.append("   " + "   ".join(f"{a}={m:.2f}" for a, m, _ in rows_sorted))
    ax2.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5,
             transform=ax2.transAxes)
    ax2.set_title("Mean S per method (higher = better; per stage, best first)", fontsize=11)

    fig.suptitle("OAT ablation of the ARF-SWE pipeline — scored by push-specificity S "
                 "(phantom positive control + in-vivo pushes)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.abspath(args.out)
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
