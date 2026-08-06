"""Refined low-band filter grid for one passive burst window, with a SIGNED directional
slant-stack (semblance) speed fitted and its moveout line overlaid.

Follows the domain review: AVC (win2) carries the stronger/clearer wave; it shows up in VELOCITY
at a LOW band; the wave travels right->left (basal->apical, directional=leftward, origin r0 at the
high-r/basal end); the expected speed is ~2-3 m/s. `passive_coherence` (envelope) is fooled by the
flat band, so speed is read here with `metrics.slant_stack_speed` (signed, semblance-based).

Grid: quantity/temporal (disp/none, velo/none, velo/mean3) x low bands. gauss0.6, off5x0.5, leftward.
Each panel overlays the fitted slant-stack moveout (solid) + 2 & 3 m/s references (grey), and the
title reports semblance and best speed in [0.8, 6] m/s.

Usage:
  python scripts/passive_slantstack.py --window 2 --label AVC
  python scripts/passive_slantstack.py --window 0 --label MVC
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
from swp.viz.metrics import slant_stack_speed
from swp.viz.pipeline import Step, run_pipeline
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
import search_passive as S
from passive_best_montage import base_for_rank

LOW_BANDS = [(5, 80), (5, 120), (5, 150), (10, 100), (10, 150), (20, 120)]
COLS = [("displacement", "none"), ("velocity", "none"), ("velocity", "mean3")]
SPATIAL, OFFSETS, STEP = "gauss0.6", 5, 0.5


def leading_edge_fit(st, frac=0.4, cmin=1.0, cmax=6.0):
    """Track the wavefront LEADING EDGE (first up-crossing of frac*peak per along-line column)
    and robustly (Theil-Sen) fit t vs r -> signed speed. Matches the MATLAB wavefront-line fit,
    which reads faster (~2-3 m/s) than the whole-band slant-stack."""
    from scipy.signal import hilbert
    data = st.data - st.data.mean(0, keepdims=True)
    data = data - data.mean(1, keepdims=True)                 # remove flat bulk band
    env = np.abs(hilbert(data, axis=0))
    ns = env.shape[1]
    peak = env.max(0)
    good = peak > 0.5 * np.median(peak)
    r, tt = st.r, np.full(ns, np.nan)
    for j in range(ns):
        if not good[j]:
            continue
        ip = int(np.argmax(env[:, j]))                        # this column's peak (arrival)
        below = np.where(env[:ip + 1, j] < frac * peak[j])[0]  # walk back to the rising edge
        tt[j] = st.t[below[-1] if below.size else 0]
    m = np.isfinite(tt)
    if m.sum() < 6:
        return float("nan"), None
    rr, ts = r[m], tt[m]
    sl = []
    for i in range(len(rr)):
        dr = rr - rr[i]
        ok = np.abs(dr) > 1e-4
        sl.append((ts[ok] - ts[i]) / dr[ok])
    slope = float(np.median(np.concatenate(sl)))              # s/m
    if slope == 0 or not np.isfinite(slope):
        return float("nan"), None
    c = 1.0 / slope
    inter = float(np.median(ts - slope * rr))
    return c, (r, inter + slope * r)


def _plot_line(ax, st, line, color, lw):
    rr, tt = line
    m = (tt >= st.t[0]) & (tt <= st.t[-1])
    ax.plot(tt[m] * 1e3, rr[m] * 1e3, color=color, lw=lw, alpha=0.9)


def draw(ax, st, title, band_line, edge_line):
    """Reference orientation: x = Time (ms), y = along-line r (mm), r=0 at top.
    black = whole-band slant-stack; green = leading-edge fit; grey dashed = 2 & 3 m/s refs."""
    unit = 1e3 if st.quantity == "velocity" else 1e6
    rmask = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    clim = robust_clim(st.data, rmask, pct=97) * unit
    extent = [st.t[0] * 1e3, st.t[-1] * 1e3, st.r[-1] * 1e3, st.r[0] * 1e3]
    ax.imshow(st.data.T * unit, extent=extent, cmap="RdBu_r",
              vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    anchor = edge_line if edge_line is not None else band_line
    if anchor is not None:                             # 2 & 3 m/s refs through the wavefront
        rr, tt = anchor
        t_mid, r_mid = tt[len(tt) // 2], rr[len(rr) // 2]
        sgn = np.sign(tt[-1] - tt[0]) or 1.0
        for c, ls in ((2.0, "--"), (3.0, ":")):
            tl = t_mid + sgn * (rr - r_mid) / c
            mm = (tl >= st.t[0]) & (tl <= st.t[-1])
            ax.plot(tl[mm] * 1e3, rr[mm] * 1e3, color="0.35", ls=ls, lw=0.9, alpha=0.85)
    if edge_line is not None:
        _plot_line(ax, st, edge_line, "limegreen", 1.9)
    if band_line is not None:
        _plot_line(ax, st, band_line, "k", 1.5)
    ax.set_ylim(st.r[-1] * 1e3, st.r[0] * 1e3)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=r"D:\Luuk van Knippenberg\Claude\invivo_sw")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    cfg = rc.load_config(os.path.join(_ROOT, "configs", "passive.yaml"))
    output_dir = os.path.join(a.folder, "output"); cfg["data"]["root"] = output_dir
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

    nrows, ncols = len(LOW_BANDS), len(COLS)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.7 * ncols, 3.0 * nrows), squeeze=False)
    print(f"win{a.window} {a.label}: slant-stack speed (signed, semblance, c in [0.8,6] m/s), "
          f"direction leftward, {SPATIAL}, off{OFFSETS}x{STEP}mm")
    for ri, b in enumerate(LOW_BANDS):
        for cti, (q, temporal) in enumerate(COLS):
            rec = dict(quantity=q, motion=f"bp{b[0]}_{b[1]}", spatial=SPATIAL, temporal=temporal,
                       direction="leftward", offsets=OFFSETS, step=STEP)
            st, r0 = S.build_spacetime_for(acq_w, mline, prf, dz, dx, tf, base, rec)
            sem, c, band_line = slant_stack_speed(st, cmin=1.0, cmax=6.0, return_line=True)
            c_edge, edge_line = leading_edge_fit(st)
            if not (1.0 <= abs(c_edge) <= 5.0):   # drop unstable edge fits (report n/a)
                edge_line, c_edge = None, float("nan")
            etxt = f"edge |c|={abs(c_edge):.1f}" if edge_line is not None else "edge n/a"
            ax = axs[ri][cti]
            draw(ax, st, f"{q[:4]}/{temporal} bp{b[0]}-{b[1]}\n"
                         f"band |c|={abs(c):.1f} m/s  ({etxt})", band_line, edge_line)
            if cti == 0:
                ax.set_ylabel("r [mm]", fontsize=7)
            if ri == nrows - 1:
                ax.set_xlabel("t [ms]", fontsize=7)
            print(f"  {q[:4]}/{temporal:5} bp{b[0]:>2}-{b[1]:<3}: "
                  f"band c={c:+.2f}  edge c={c_edge:+.2f} m/s")
    fig.suptitle(f"Passive SWE slant-stack -- win{a.window} @ {w.t_peak*1e3:.0f} ms"
                 f"{'  (' + a.label + ')' if a.label else ''}  |  solid = fitted moveout, "
                 f"grey = 2 & 3 m/s; direction leftward (basal->apical)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = os.path.join(outdir, f"slantstack_win{a.window}{('_' + a.label) if a.label else ''}.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
