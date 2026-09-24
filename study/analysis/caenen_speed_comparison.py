"""Our Caenen pig-data speeds against the speeds Caenen's group measured on the same pushes.

Reference: the per-push velocity estimates received from Caenen (2026-09-24, a MATLAB scatter plot:
x = push number 1-52, y = propagation speed, one colour per observer / M-line). They could only
estimate a speed for 11 of the 52 pushes. The plot was digitised (ring-shaped matched filter per
MATLAB default colour on the axes frame); the values are in ``study/logs/caenen_reported_speeds.csv``.
A few overlapping markers are hidden behind others, which barely moves a per-push median.
Assumption: their push number is our ``push_<p>`` (= the ``..._ARF_<p>.mat`` file index).

Ours, per push, on our single drawn M-line (``SWE_results/push_<p>/mline.npz``), per side of r0
with the one-sided slant stack of ``SWE_results/analyze_push.py`` (0.8-8 m/s):

* ``stored`` - the 2026-08-04 batch (``summary.csv``, 120-700 Hz displacement consensus band);
* ``literature`` - Caenen's own recipe run by us: lag-1 velocity, Gaussian 1.9 x 2.0 mm on the
  autocorrelation, 6th-order 75-750 Hz band-pass, outward directional, full tracking window.
  **Its automatic side speeds rail at whatever lower bound is set (0.5-0.8 m/s) - the slant stack
  fails on these velocity panels, as Caenen et al. report for their Radon fit. Not a measurement;
  kept to show the failure. A like-for-like comparison needs hand-drawn slopes on these pushes.**

    python study/analysis/caenen_speed_comparison.py
-> study/logs/caenen_speed_comparison.csv, study/montages/caenen_speed_comparison.png
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

# (push, colour, speed m/s) - digitised from Caenen's plot
REPORTED = [
    (1, "blue", 1.64), (1, "orange", 1.70), (1, "yellow", 1.82), (1, "green", 1.95), (1, "purple", 2.10),
    (5, "orange", 4.45), (5, "blue", 5.06), (5, "yellow", 5.59),
    (15, "green", 1.21), (15, "purple", 1.33), (15, "yellow", 1.40),
    (18, "green", 1.74), (18, "blue", 1.89), (18, "purple", 1.93),
    (20, "orange", 1.37), (20, "purple", 1.52), (20, "green", 1.64), (20, "yellow", 1.67),
    (21, "green", 1.17), (21, "purple", 1.23), (21, "blue", 1.42),
    (22, "blue", 1.25), (22, "green", 1.34), (22, "orange", 1.45),
    (23, "green", 1.49), (23, "yellow", 1.66),
    (24, "green", 1.08), (24, "yellow", 1.43), (24, "blue", 1.80),
    (25, "green", 1.10), (25, "yellow", 1.25), (25, "purple", 1.36),
    (27, "purple", 1.80), (27, "yellow", 2.40),
]


def side_speeds(st, r0, cmin=0.8, cmax=8.0):
    """(|c| left, |c| right) by one-sided slant stack, as SWE_results/analyze_push.py."""
    from swp.viz.metrics import slant_stack_speed
    from swp.viz.speed.spacetime import SpaceTime
    out = []
    for mask in (st.r < r0, st.r >= r0):
        if mask.sum() < 10:
            out.append(np.nan); continue
        sub = SpaceTime(st.data[:, mask], st.r[mask], st.t, st.quantity)
        _, c = slant_stack_speed(sub, None, cmin=cmin, cmax=cmax)
        out.append(abs(c) if np.isfinite(c) else np.nan)
    return tuple(out)


def literature_spacetime(acq, ml, r0):
    from swp.viz.estimators import loupas_displacement
    from swp.viz.filters import FIELD_FILTERS
    from swp.viz.filters.context import FilterCtx
    from swp.viz.filters.directional import outward_spacetime
    from swp.viz.speed.spacetime import build_spacetime, SpaceTime
    est = loupas_displacement(acq.iq, dz=acq.dz, dx=acq.dx, c=acq.c, f_demod=acq.f_demod, prf=acq.prf,
                              kernel_z_m=1.9e-3, kernel_x_m=2.0e-3, kernel_shape="gaussian",
                              mode="frame_to_frame")
    fld = est.velocity[1:]
    t = np.arange(fld.shape[0]) / acq.prf
    ctx = FilterCtx(dz=acq.dz, dx=acq.dx, prf=acq.prf, t=t, x=acq.x, z=acq.z)
    fld = FIELD_FILTERS["temporal_bandpass"](fld, ctx, f_lo=75, f_hi=750, order=3)
    st = build_spacetime(fld, acq.z, acq.x, ml, t, quantity="velocity", n_offsets=7,
                         offset_step_m=0.8e-3, agg="mean")
    return SpaceTime(outward_spacetime(st.data, st.r, r0), st.r, st.t, st.quantity)


def main():
    import caenen
    from swp.viz.pipeline import _r0_lateral_crossing
    with open(_REPO / "study/logs/caenen_reported_speeds.csv", "w", newline="") as fh:
        fh.write("# digitised from the scatter plot received from Caenen on 2026-09-24\n")
        w = csv.writer(fh); w.writerow(["push", "colour", "speed_m_s"]); w.writerows(REPORTED)
    stored = {}
    with open(os.path.join(P.CAENEN_SWE, "summary.csv")) as fh:
        for r in csv.DictReader(fh):
            v = [float(r[k]) for k in ("cart_120_700_cL", "cart_120_700_cR") if r[k] not in ("", None)]
            stored[int(r["push"])] = (float(r["cart_120_700_cL"]) if r["cart_120_700_cL"] else np.nan,
                                      float(r["cart_120_700_cR"]) if r["cart_120_700_cR"] else np.nan)
    pushes = sorted({p for p, _, _ in REPORTED})
    rows = []
    for p in pushes:
        theirs = [s for q, _, s in REPORTED if q == p]
        acq, ml = caenen.load(p)
        r0 = _r0_lateral_crossing(ml, float(acq.push_x))
        lit = side_speeds(literature_spacetime(acq, ml, r0), r0)
        rows.append(dict(push=p, caenen_median=float(np.median(theirs)), caenen_min=min(theirs),
                         caenen_max=max(theirs), n_caenen=len(theirs),
                         stored_L=stored[p][0], stored_R=stored[p][1], lit_L=lit[0], lit_R=lit[1]))
        print(f"push {p:2d}: Caenen {np.median(theirs):.2f} [{min(theirs):.2f}-{max(theirs):.2f}]  "
              f"stored L/R {stored[p][0]:.2f}/{stored[p][1]:.2f}  literature L/R {lit[0]:.2f}/{lit[1]:.2f}",
              flush=True)
    from swp.provenance import stamp_text
    out = _REPO / "study/logs/caenen_speed_comparison.csv"
    with open(out, "w", newline="") as fh:
        fh.write(stamp_text(config=dict(side_speed="slant_stack 0.8-8 m/s per side", recipe="caenen literature")))
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    their = np.array([r["caenen_median"] for r in rows])
    print("\nratio ours / Caenen per push (median over pushes; push 5 excluded as their outlier):")
    keep = np.array([r["push"] != 5 for r in rows])
    for key in ("stored_L", "stored_R", "lit_L", "lit_R"):
        v = np.array([r[key] for r in rows])
        ok = keep & np.isfinite(v)
        print(f"  {key:<9} median {np.median(v[ok] / their[ok]):.2f}  "
              f"within 25% {np.mean(np.abs(v[ok] / their[ok] - 1) <= 0.25):.0%}  (n={ok.sum()})")
    for key in ("stored", "lit"):
        v = np.nanmean([[r[f"{key}_L"], r[f"{key}_R"]] for r in rows], axis=1)
        ok = keep & np.isfinite(v)
        rho = np.corrcoef(v[ok], their[ok])[0, 1]
        print(f"  {key:<9} mean of sides: median ratio {np.median(v[ok] / their[ok]):.2f}, "
              f"Pearson r {rho:.2f} across pushes")

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for p, col, s in REPORTED:
        ax[0].plot(p, s, "o", mfc="none", color="0.45", ms=7)
    ax[0].plot([], [], "o", mfc="none", color="0.45", label="Caenen (each observer / line)")
    x = np.array(pushes)
    ax[0].plot(x - 0.25, [r["stored_L"] for r in rows], "v", color="C0", label="ours, stored 120-700 disp, left")
    ax[0].plot(x - 0.25, [r["stored_R"] for r in rows], "^", color="C0", label="ours, stored, right")
    ax[0].plot(x + 0.25, [r["lit_L"] for r in rows], "v", color="C3", label="ours, literature recipe, left (auto fit RAILS at bound)")
    ax[0].plot(x + 0.25, [r["lit_R"] for r in rows], "^", color="C3", label="ours, literature recipe, right")
    ax[0].set_xlabel("# push"); ax[0].set_ylabel("propagation speed [m/s]"); ax[0].set_ylim(0, 8)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    ax[0].set_title("the 11 pushes Caenen could measure")
    for key, c, lab in (("stored", "C0", "stored"), ("lit", "C3", "literature recipe - railed, not a measurement")):
        v = np.nanmean([[r[f"{key}_L"], r[f"{key}_R"]] for r in rows], axis=1)
        ax[1].plot(their, v, "o", color=c, label=f"ours ({lab}), mean of sides")
    lim = [0, 6]
    ax[1].plot(lim, lim, "k--", lw=1); ax[1].fill_between(lim, [0.75 * l for l in lim], [1.25 * l for l in lim],
                                                          color="0.85", label="+/-25 %")
    ax[1].set_xlim(lim); ax[1].set_ylim(0, 8)
    ax[1].set_xlabel("Caenen median [m/s]"); ax[1].set_ylabel("ours [m/s]"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Caenen pig ARF-SWE: our speeds vs the speeds reported by Caenen's group", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fp = _REPO / "study/montages/caenen_speed_comparison.png"
    fig.savefig(fp, dpi=130)
    print("wrote", fp)


if __name__ == "__main__":
    main()
