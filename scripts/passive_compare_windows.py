"""Compare win0, win1, win2 with the SAME optimal passive recipe (config default) to test the
MVC/AVC labelling: hypothesis is win0 ~ win2 (both MVC) vs win1 (AVC).

Optimal recipe: displacement, no IQ pre-filter, no SVD, bp10-150, gaussian 0.6 mm, mean3, NO directional
filter, 5 M-lines (mean). Radon (signed slant-stack) speed reported per window, reference M-mode
orientation. Uses the saved anatomical septal M-lines (passive_win{i}_mline.npz).
"""
from __future__ import annotations
import os, sys
from dataclasses import replace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swp.viz import runconfig as rc
from swp.viz.io import load_acquisition
from swp.viz.estimators import ESTIMATORS
from swp.viz.pipeline import Step, run_pipeline
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
import search_passive2 as S2

RECIPE = dict(quantity="displacement", prefilter="none", svd=0, motion="bp10_150",
              spatial="gauss0.6", temporal="mean3", direction="none", N=5, agg="mean")
WINDOWS = [0, 1, 2, 3]


def main():
    folder = r"D:\Luuk van Knippenberg\Claude\invivo_sw"
    cfg = rc.load_config(os.path.join(_ROOT, "configs", "passive.yaml"))
    out = os.path.join(folder, "output"); cfg["data"]["root"] = out
    iq_path = rc.hdf5_path(cfg, 0); bmode = os.path.join(out, os.path.basename(iq_path))
    mldir = os.path.join(out, "mlines"); outdir = os.path.join(rc.outdir(cfg, out), "search2")

    acq = load_acquisition(iq_path); prf, dz, dx = acq.prf, acq.dz, acq.dx
    base_cfg = rc.build_pipeline_config(cfg, acq=acq)
    gen = _ensure_mline(os.path.join(mldir, "passive_general_mline.npz"), bmode,
                        _middle_frame(bmode), "gen", n_samples=cfg["mline"]["n_samples"])
    b = cfg["detect"]["band"]
    ov = run_pipeline(_stride_acq(acq, 2), gen, replace(base_cfg, directional=False,
                      field_filters=[Step("temporal_bandpass", dict(f_lo=b[0], f_hi=b[1]))]
                      + [s for s in base_cfg.field_filters if s.name != "temporal_bandpass"]), focus=None)
    windows, _ = detect_line_bursts(np.asarray(ov.st.data).T, np.asarray(ov.st.t),
                                    window_ms=100.0, max_events=4)

    fig, axs = plt.subplots(1, len(WINDOWS), figsize=(5.0 * len(WINDOWS), 4.4))
    speeds = {}
    for ax, wi in zip(axs, WINDOWS):
        w = windows[wi]
        ml = _ensure_mline(os.path.join(mldir, f"passive_win{wi}_mline.npz"), bmode,
                           _frame_at_time(acq.t, w.t_peak), f"win{wi}", n_samples=cfg["mline"]["n_samples"])
        i0 = _frame_at_time(acq.t, w.t0 - 0.02); i1 = _frame_at_time(acq.t, w.t1 + 0.02) + 1
        acq_w = replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1]); tf = np.asarray(acq_w.t, float)
        res = ESTIMATORS["loupas"](acq_w.iq, dz=dz, dx=dx, c=acq_w.c, f_demod=acq_w.f_demod,
                                   prf=prf, mode="frame_to_frame")
        base = {("none", 0): res}
        st = S2.build_st(base, acq_w, ml, prf, dz, dx, tf, RECIPE)
        score, semb, c = S2.prop_fit(st.data, st.r, float(st.t[1] - st.t[0]))
        speeds[wi] = c
        S2.draw(ax, st, f"win{wi} @ {w.t_peak*1e3:.0f} ms")
        ax.set_xlabel("t [ms]"); ax.set_ylabel("r [mm]")
        print(f"win{wi} @ {w.t_peak*1e3:.0f}ms: |c|={abs(c):.2f} m/s  score={score:.3f}")
    fig.suptitle("Passive SWE -- same optimal recipe [disp/no-dir/bp10-150/gauss0.6/mean3/5-line] on "
                 + ", ".join(f"win{w}" for w in WINDOWS), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(outdir, "compare_windows.png"); fig.savefig(p, dpi=150); plt.close(fig)
    print("wrote", p)
    print("speeds (m/s):", {k: round(abs(v), 2) for k, v in speeds.items()})


if __name__ == "__main__":
    main()
