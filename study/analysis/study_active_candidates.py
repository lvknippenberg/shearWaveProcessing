"""Render the study ARF pushes flagged by ``study_active_screen_summary.py`` for visual review.

Rows = candidate pushes; columns = buffer-5 B-mode with the automatic M-line, push space-time, its
no-push control (same colour scale). A real ARF wave is a V opening from r0 in the push panel and
absent from the control.

    python study/analysis/study_active_candidates.py  -> study/montages/study_active_candidates.png
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parents[2]
for p in ("src", "swp_gui", "scripts"):
    sys.path.insert(0, str(_REPO / p))
os.environ.setdefault("KERAS_BACKEND", "torch")

from swp import paths as P                                     # noqa: E402

RECIPE = "caenen_dir"


def candidates():
    with open(_REPO / "study/logs/study_active_screen.csv", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(l for l in fh if not l.startswith("#"))
                if r["recipe"] == RECIPE and not r["error"]]
    with open(_REPO / "study/logs/invivo_recipe_contrast.csv", encoding="utf-8") as fh:
        ref = [float(r["log2_amp"]) for r in csv.DictReader(l for l in fh if not l.startswith("#"))
               if r["dataset"] == "caenen" and r["recipe"] == RECIPE]
    thr = np.percentile(ref, 5)
    return [r for r in rows if float(r["log2_amp"]) > thr and float(r["d_oc"]) > 0]


def main():
    import core
    import auto_mline
    import swe_lib as L
    import invivo_recipe_contrast as H
    from swp.viz.io import load_acquisition
    from swp.viz.mline.mline import mline_from_points
    from swp.viz.pipeline import _r0_lateral_crossing
    cs = candidates()
    fig, axes = plt.subplots(len(cs), 3, figsize=(11, 2.9 * len(cs)), squeeze=False)
    for i, r in enumerate(cs):
        folder = os.path.join(P.RAW_DATA, r["subject"], r["folder"])
        m = int(r["meas"])
        acq = load_acquisition(os.path.join(folder, "output", f"CombinedData_buffer2_meas{m}_iq.hdf5"))
        px, pz = float(acq.push_x or 0.0), float(acq.push_z)
        pts = auto_mline.propose(folder, m, px * 1e3, pz * 1e3, order=1, x_span_mm=34.0)
        ml = mline_from_points(pts, 250)
        r0 = _r0_lateral_crossing(ml, px)
        img, ext = core._bmode_h5py(os.path.join(folder, "output", "CombinedData_buffer5_iq.hdf5"), m)
        ax = axes[i, 0]
        ax.imshow(img, cmap="gray", extent=ext, aspect="auto")
        ax.plot(ml.x * 1e3, ml.z * 1e3, "y-", lw=1.2)
        ax.plot(px * 1e3, pz * 1e3, "r+", ms=10)
        ax.set_xlim(px * 1e3 - 30, px * 1e3 + 30); ax.set_ylim(pz * 1e3 + 25, pz * 1e3 - 25)
        ax.set_title(f"{r['subject']} m{m}, {r['t_after_R_ms']} ms after R", fontsize=8)
        stp = H.spacetime(acq, ml, r0, H.RECIPES[RECIPE], nopush=False)
        stn = H.spacetime(acq, ml, r0, H.RECIPES[RECIPE], nopush=True)
        clim = L.draw(axes[i, 1], stp, r0)
        L.draw(axes[i, 2], stn, r0, clim=clim)
        axes[i, 1].set_title(f"push (ratio {2 ** float(r['log2_amp']):.1f}x, dOC {float(r['d_oc']):+.2f})", fontsize=8)
        axes[i, 2].set_title("no-push control, same scale", fontsize=8)
    for ax in axes[-1, 1:]:
        ax.set_xlabel("r along M-line [mm]")
    fig.suptitle("Study ARF pushes that exceed the weakest 5 % of Caenen's real waves - candidates, "
                 "not detections (7 of 1552)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = _REPO / "study/montages/study_active_candidates.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
