"""Validate the automatic septal M-line against the hand-drawn ones.

The 2026-08-04 in-vivo 40 V acquisition has a manually drawn M-line saved for every push
(`output/mlines/active_meas{m}_mline.npz`). This script rebuilds the same space-times with the
automatic proposal (`scripts/auto_mline.propose`) and compares the two, both geometrically
(distance between the curves) and in outcome (origin coherence / mirror symmetry per push). It is
the check that the automatic line used for the 2026-08-18 in-vivo data -- which has no manual
lines -- does not change the conclusions.

    python scripts/validate_auto_mline.py --folder <old 40V in-vivo folder> --out <fig.png>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in ("src", "swp_gui"):
    if os.path.join(_ROOT, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_ROOT, _p))

import swe_lib as L                                              # noqa: E402
import core                                                     # noqa: E402
from auto_mline import propose                                  # noqa: E402
from swp.viz.mline import mline_from_points                     # noqa: E402
from swp.viz.pipeline import _r0_lateral_crossing               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantity", default="displacement")
    a = ap.parse_args()

    rec = L.REC_INVIVO
    rows = []
    for m in range(L.n_pushes(a.folder)):
        npz = os.path.join(a.folder, "output", "mlines", f"active_meas{m}_mline.npz")
        if not os.path.isfile(npz):
            continue
        acq, ml_manual, _ = L.load_push(a.folder, m, phantom=False)
        pts_auto = propose(a.folder, m, float(acq.push_x) * 1e3, float(acq.push_z) * 1e3,
                           x_span_mm=34, order=1)
        ml_auto = mline_from_points(pts_auto, 250)
        est = L.estimator(acq)
        out = {}
        for tag, ml in (("manual", ml_manual), ("auto", ml_auto)):
            r0 = _r0_lateral_crossing(ml, float(acq.push_x))
            st = L.spacetime(est, acq, ml, r0, rec, a.quantity)
            out[tag] = L.scores(st, r0)
        # geometric distance: mean |z_auto - z_manual| where the lateral spans overlap
        za = np.interp(ml_manual.x, pts_auto[:, 0], pts_auto[:, 1], left=np.nan, right=np.nan)
        d_mm = float(np.nanmean(np.abs(za - ml_manual.z))) * 1e3
        rows.append((m, out["manual"], out["auto"], d_mm))
        print(f"  m{m:2d}  manual oc={out['manual'][0]:.2f} sym={out['manual'][1]:.2f}   "
              f"auto oc={out['auto'][0]:.2f} sym={out['auto'][1]:.2f}   "
              f"mean |dz| = {d_mm:.1f} mm", flush=True)
        L._ACQ_CACHE.pop((a.folder, m, False), None)

    ms = [r[0] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, i, name in ((axes[0], 0, "origin coherence"), (axes[1], 1, "mirror symmetry")):
        ax.plot(ms, [r[1][i] for r in rows], "-o", label="manual M-line", ms=4)
        ax.plot(ms, [r[2][i] for r in rows], "--s", label="automatic M-line", ms=4)
        ax.set_title(name)
        ax.set_xlabel("push")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[2].plot(ms, [r[3] for r in rows], "-o", color="0.3", ms=4)
    axes[2].set_title("mean |depth difference| between the two lines")
    axes[2].set_xlabel("push")
    axes[2].set_ylabel("mm")
    axes[2].grid(alpha=0.3)
    oc_d = np.mean([r[2][0] - r[1][0] for r in rows])
    sym_d = np.mean([r[2][1] - r[1][1] for r in rows])
    fig.suptitle(f"Automatic vs hand-drawn septal M-line, 2026-08-04 in-vivo 40 V "
                 f"({a.quantity}, {rec['tag']})\n"
                 f"mean auto-minus-manual: origin coherence {oc_d:+.3f}, mirror symmetry "
                 f"{sym_d:+.3f}; mean line separation "
                 f"{np.mean([r[3] for r in rows]):.1f} mm", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(a.out, dpi=130)
    print("wrote", a.out)
    print(f"mean auto-manual: oc {oc_d:+.3f}, sym {sym_d:+.3f}")


if __name__ == "__main__":
    main()
