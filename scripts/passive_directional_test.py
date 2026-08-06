"""Quick test: does the directional (k-omega) filter flatten the passive wavefront?

The MATLAB passive pipeline (ProcessDW=true) applies NO directional filter -- it Radon-transforms
the raw band-passed M-mode. Our pipeline applies a leftward directional filter. This renders win2
(AVC) velocity bp10-150 as: no-directional (like MATLAB) vs leftward vs rightward, in the reference
M-mode orientation, with a signed slant-stack (Radon-like) speed fit overlaid.
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
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swp.viz import runconfig as rc
from swp.viz.io import load_acquisition
from swp.viz.core.geometry import robust_clim
from swp.viz.mline.mline import sample_along_mline
from swp.viz.metrics import slant_stack_speed
from swp.viz.filters.directional import directional_spacetime
from swp.viz.speed.spacetime import SpaceTime
from swp.viz.pipeline import Step, run_pipeline
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
import search_passive as S
from passive_best_montage import base_for_rank

WINDOW, LABEL = 2, "AVC"
BAND = (10, 150)


def st_no_dir(acq_w, mline, prf, dz, dx, tf, base):
    """velocity, bp10-150, gauss0.6, off5x0.5 -- WITHOUT any directional filter."""
    rec = dict(quantity="velocity", motion=f"bp{BAND[0]}_{BAND[1]}", spatial="gauss0.6",
               temporal="none", offsets=5, step=0.5)
    fld, times = S.quantity_field(base[0], "velocity", tf)
    fld = S.apply_spatial(fld, S._spatial_step("gauss0.6"), dz, dx)
    st_raw = sample_along_mline(fld, acq_w.z, acq_w.x, mline, n_offsets=5, offset_step_m=0.5e-3)
    st_m = S.apply_motion_st(st_raw, S._motion_step(f"bp{BAND[0]}_{BAND[1]}"), prf)
    return SpaceTime(st_m, mline.r, times, "velocity")


def draw(ax, st, title):
    unit = 1e3
    rmask = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    clim = robust_clim(st.data, rmask, pct=97) * unit
    ext = [st.t[0] * 1e3, st.t[-1] * 1e3, st.r[-1] * 1e3, st.r[0] * 1e3]
    ax.imshow(st.data.T * unit, extent=ext, cmap="RdBu_r", vmin=-clim, vmax=clim,
              aspect="auto", origin="upper")
    # Radon-like signed fit WITHOUT flat removal (mimics MATLAB EstimateSWSradon on signed m-mode)
    sem, c, line = slant_stack_speed(st, cmin=1.0, cmax=6.0, remove_flat=False, return_line=True)
    if line is not None:
        rr, tt = line
        m = (tt >= st.t[0]) & (tt <= st.t[-1])
        ax.plot(tt[m] * 1e3, rr[m] * 1e3, "k", lw=1.7, alpha=0.9)
    ax.set_ylim(st.r[-1] * 1e3, st.r[0] * 1e3)
    ax.set_title(f"{title}\nRadon(signed) |c|={abs(c):.1f} m/s sem={sem:.2f}", fontsize=9)
    ax.set_xlabel("t [ms]"); ax.set_ylabel("r [mm]")


def main():
    folder = r"D:\Luuk van Knippenberg\Claude\invivo_sw"
    cfg = rc.load_config(os.path.join(_ROOT, "configs", "passive.yaml"))
    out = os.path.join(folder, "output"); cfg["data"]["root"] = out
    iq_path = rc.hdf5_path(cfg, 0)
    bmode = os.path.join(out, os.path.basename(iq_path))
    mldir = os.path.join(out, "mlines")
    acq = load_acquisition(iq_path)
    prf, dz, dx = acq.prf, acq.dz, acq.dx
    base_cfg = rc.build_pipeline_config(cfg, acq=acq)
    gen = _ensure_mline(os.path.join(mldir, "passive_general_mline.npz"), bmode,
                        _middle_frame(bmode), "gen", n_samples=cfg["mline"]["n_samples"])
    b = cfg["detect"]["band"]
    ov = run_pipeline(_stride_acq(acq, 2), gen, replace(base_cfg, directional=False,
                      field_filters=[Step("temporal_bandpass", dict(f_lo=b[0], f_hi=b[1]))]
                      + [s for s in base_cfg.field_filters if s.name != "temporal_bandpass"]),
                      focus=None)
    windows, _ = detect_line_bursts(np.asarray(ov.st.data).T, np.asarray(ov.st.t),
                                    window_ms=100.0, max_events=4)
    w = windows[WINDOW]
    ml = _ensure_mline(os.path.join(mldir, f"passive_win{WINDOW}_mline.npz"), bmode,
                       _frame_at_time(acq.t, w.t_peak), f"win{WINDOW}", n_samples=cfg["mline"]["n_samples"])
    i0 = _frame_at_time(acq.t, w.t0 - 0.02); i1 = _frame_at_time(acq.t, w.t1 + 0.02) + 1
    acq_w = replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
    tf = np.asarray(acq_w.t, float)
    base = base_for_rank(acq_w, {0})

    st = st_no_dir(acq_w, ml, prf, dz, dx, tf, base)
    variants = [
        ("no directional (like MATLAB)", st),
        ("directional leftward (our pipeline)", SpaceTime(directional_spacetime(st.data, "neg"), st.r, st.t, "velocity")),
        ("directional rightward", SpaceTime(directional_spacetime(st.data, "pos"), st.r, st.t, "velocity")),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (title, s) in zip(axs, variants):
        draw(ax, s, title)
    fig.suptitle(f"win{WINDOW} ({LABEL}) velocity bp{BAND[0]}-{BAND[1]} -- effect of the directional filter",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(rc.outdir(cfg, out), "search", f"directional_test_win{WINDOW}_{LABEL}.png")
    fig.savefig(p, dpi=140); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
