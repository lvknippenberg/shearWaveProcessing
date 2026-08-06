"""Filter-setting variety montages for a single passive burst window, with physiological
reference slopes (2 & 3 m/s) overlaid and coherence recomputed in a physiological speed band.

Motivated by the domain review:
  * ignore win1 (noise); win0/win3 = mitral-valve-closure (MVC), win2 = aortic-valve-closure (AVC);
  * the shear wave should travel ~2-3 m/s -- the earlier pc~0.9 fits (c~6-8 m/s) were near-flat
    bands, NOT real propagation;
  * in PLAX both MVC and AVC waves travel right->left in the image (basal->apical), so the
    directional filter is FIXED (leftward, origin r0 at the high-r / basal / right end);
  * MVC and AVC may have different optimal filters (AVC stronger/clearer).

For the chosen window this renders a grid over quantity x band (light Gaussian + temporal held
fixed, direction leftward), each panel showing:
  * the space-time (RdBu),
  * dashed guide lines at c = 2 and 3 m/s from the origin onset (so the wavefront tilt can be
    compared to the expected speed by eye),
  * pc_phys = passive_coherence restricted to c in [1.5, 4.0] m/s, and the best in-band speed.

Usage:
  python scripts/passive_filter_variety.py --window 0 --label MVC
  python scripts/passive_filter_variety.py --window 2 --label AVC
"""
from __future__ import annotations
import argparse, os, sys, math
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
from swp.viz.speed.spacetime import SpaceTime
from swp.viz.pipeline import Step, run_pipeline
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
import search_passive as S
from passive_best_montage import base_for_rank

# --- filter variety: quantity x band. Spatial/temporal held light + fixed; direction LEFTWARD. ---
QUANTITIES = ["displacement", "velocity"]
BANDS = [(5, 150), (10, 150), (20, 200), (30, 250), (40, 300), (60, 300)]
SPATIAL = "gauss0.6"        # light Gaussian (search: median never wins; 0.6-1.0 mm best)
TEMPORAL = "mean3"          # light slow-time smoothing
OFFSETS, STEP = 5, 0.5      # tight M-line offset averaging (search-best)
REF_SPEEDS = (2.0, 3.0)     # m/s guide lines


def motion_label(band):
    return f"bp{band[0]}_{band[1]}"


def draw_panel(ax, st, r0, title, ref_speeds=REF_SPEEDS):
    unit = 1e3 if st.quantity == "velocity" else 1e6
    img = st.data * unit
    rmask = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    clim = robust_clim(st.data, rmask, pct=97) * unit
    ax.imshow(img, extent=st.extent_ms_mm(), cmap="RdBu_r", vmin=-clim, vmax=clim,
              aspect="auto", origin="upper")
    # origin onset: time of peak |envelope| at the origin (r0) column
    from scipy.signal import hilbert
    d = st.data - st.data.mean(axis=0, keepdims=True)
    env = np.abs(hilbert(d, axis=0))
    j0 = int(np.argmin(np.abs(st.r - r0)))
    t_anchor = st.t[int(np.argmax(env[:, j0]))]
    # wave travels toward decreasing r (right->left); t = t_anchor + (r0 - r)/c
    rr = st.r[st.r <= r0]
    for c, ls in zip(ref_speeds, ("--", ":")):
        tt = t_anchor + (r0 - rr) / c
        m = (tt >= st.t[0]) & (tt <= st.t[-1])
        ax.plot(rr[m] * 1e3, tt[m] * 1e3, "k", ls=ls, lw=1.1, alpha=0.8,
                label=f"{c:.0f} m/s")
    ax.axvline(r0 * 1e3, color="0.2", lw=0.7, alpha=0.5)
    ax.set_ylim(st.t[-1] * 1e3, st.t[0] * 1e3)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left", framealpha=0.6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=r"D:\Luuk van Knippenberg\Claude\invivo_sw")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    cfg = rc.load_config(os.path.join(_ROOT, "configs", "passive.yaml"))
    output_dir = os.path.join(a.folder, "output")
    cfg["data"]["root"] = output_dir
    iq_path = rc.hdf5_path(cfg, 0)
    bmode_path = os.path.join(output_dir, os.path.basename(iq_path))
    mlines_dir = os.path.join(output_dir, "mlines")
    outdir = os.path.join(rc.outdir(cfg, output_dir), "search")

    acq = load_acquisition(iq_path)
    prf, dz, dx = acq.prf, acq.dz, acq.dx
    base_cfg = rc.build_pipeline_config(cfg, acq=acq)
    gen = _ensure_mline(os.path.join(mlines_dir, "passive_general_mline.npz"), bmode_path,
                        _middle_frame(bmode_path), "general", n_samples=cfg["mline"]["n_samples"])
    band = cfg["detect"]["band"]
    ov_cfg = replace(base_cfg, directional=False,
                     field_filters=[Step("temporal_bandpass", dict(f_lo=band[0], f_hi=band[1]))]
                     + [s for s in base_cfg.field_filters if s.name != "temporal_bandpass"])
    ov = run_pipeline(_stride_acq(acq, 2), gen, ov_cfg, focus=None)
    windows, _ = detect_line_bursts(np.asarray(ov.st.data).T, np.asarray(ov.st.t),
                                    window_ms=100.0, max_events=4)
    w = windows[a.window]
    frame = _frame_at_time(acq.t, w.t_peak)
    mline = _ensure_mline(os.path.join(mlines_dir, f"passive_win{a.window}_mline.npz"), bmode_path,
                          frame, f"win{a.window}", n_samples=cfg["mline"]["n_samples"])
    i0 = _frame_at_time(acq.t, w.t0 - 0.02); i1 = _frame_at_time(acq.t, w.t1 + 0.02) + 1
    acq_w = replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
    tf = np.asarray(acq_w.t, float)
    base = base_for_rank(acq_w, {0})

    recipes = [dict(quantity=q, motion=motion_label(b), spatial=SPATIAL, temporal=TEMPORAL,
                    direction="leftward", offsets=OFFSETS, step=STEP, band=b)
               for b in BANDS for q in QUANTITIES]

    n = len(recipes); ncols = 4; nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows), squeeze=False)
    print(f"win{a.window} {a.label}: quantity x band grid (direction=leftward, {SPATIAL}, {TEMPORAL}, "
          f"off{OFFSETS}x{STEP}mm); pc restricted to c in [1.5,4] m/s")
    for k, rec in enumerate(recipes):
        st, r0 = S.build_spacetime_for(acq_w, mline, prf, dz, dx, tf, base, rec)
        dt = float(st.t[1] - st.t[0])
        pc_phys, c_phys = S.pc_fast(st.data, mline.r, dt, r0, cmin=1.5, cmax=4.0, n_speeds=51)
        pc_wide, c_wide = S.pc_fast(st.data, mline.r, dt, r0, cmin=0.5, cmax=8.0, n_speeds=61)
        ax = axs[k // ncols][k % ncols]
        draw_panel(ax, st, r0,
                   f"{rec['quantity'][:4]} / bp{rec['band'][0]}-{rec['band'][1]}\n"
                   f"pc(1.5-4)={pc_phys:.2f} @ {c_phys:.1f} m/s\n"
                   f"pc(wide)={pc_wide:.2f} @ {c_wide:.1f} m/s")
        if k % ncols == 0:
            ax.set_ylabel("t [ms]", fontsize=7)
        if k // ncols == nrows - 1:
            ax.set_xlabel("r [mm]", fontsize=7)
        print(f"  {rec['quantity'][:4]:4} bp{rec['band'][0]:>2}-{rec['band'][1]:<3}: "
              f"pc_phys={pc_phys:.3f} c_phys={c_phys:.1f} | pc_wide={pc_wide:.3f} c_wide={c_wide:.1f}")
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Passive SWE filter variety -- win{a.window} @ {w.t_peak*1e3:.0f} ms"
                 f"{'  (' + a.label + ')' if a.label else ''}  |  direction leftward (basal->apical), "
                 f"dashed = 2 & 3 m/s reference slopes", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, f"filtervariety_win{a.window}{('_' + a.label) if a.label else ''}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
