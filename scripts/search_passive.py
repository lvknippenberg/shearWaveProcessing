"""Exhaustive parameter search for PASSIVE SWE, scored by ``metrics.passive_coherence``.

Mirrors the active-SWE search (iq2sws ``archive/search.py``) but for the natural (valve-closure)
shear waves in the buffer-4 stream.  Differences that matter for passive:

  * no ARF push / reference -> ``mode: frame_to_frame`` displacement (cumulative vs window start);
  * the analysis unit is a ~100 ms **burst window** around each detected valve closure, not the
    whole cine, so the sweep is run **per window** (like per-measurement on the active side) and a
    cross-window **consensus** recipe is reported;
  * the wave is **one-sided** (originates at one M-line end) -> ``passive_coherence`` (one-sided
    origin coherence, free origin time, full propagation range) is the score, and the **direction**
    (leftward/rightward, which also fixes r0) is a search axis;
  * the shear-wave band is much lower (tens of Hz) than the ARF band.

Swept axes (per window):
  quantity x motion(band/highpass/poly/svd) x spatial x temporal x direction x M-line(offsets,step)

Efficiency: the frame_to_frame estimate is cached per SVD rank; the (linear, separable) temporal
motion/smoothing filters and the directional filter are applied on the **sampled space-time**
(exact for the Gaussian-spatial branch, which commutes with linear temporal filters and the linear
M-line interpolation; the median-spatial branch is a close approximation).  ``passive_coherence`` is
vectorised over candidate speeds.  ~6.7k combos/window run in a few minutes.

Outputs (``<folder>/output/swp_passive/search/``):
  results_win{i}.csv, marginals_win{i}.png, top_win{i}.png (per window),
  consensus_montage.png, baseline_montage.png, leaderboard.md.

Usage:
  python scripts/search_passive.py "D:\\Luuk van Knippenberg\\Claude\\invivo_sw" \
      --config configs/passive.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace

import numpy as np
from scipy.signal import hilbert
from scipy.ndimage import uniform_filter1d, median_filter, gaussian_filter

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swp.viz import runconfig as rc
from swp.viz.io import load_acquisition, load_mline
from swp.viz.mline import mline_from_points
from swp.viz.mline.mline import sample_along_mline
from swp.viz.estimators import ESTIMATORS
from swp.viz.filters import svd_clutter
from swp.viz.filters.directional import directional_spacetime
from swp.viz.speed.spacetime import SpaceTime
from swp.viz.viz import spacetime_montage
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
from swp.viz.pipeline import Step

# ----------------------------------------------------------------------------- parameter grid
QUANTITIES = ["displacement", "velocity"]                 # acceleration: post-hoc on the top-K
# motion axis -- (label, svd_rank, spacetime temporal step or None)
MOTIONS = [("none", 0, None)]
MOTIONS += [(f"hp{fc}", 0, ("hp", fc)) for fc in (10, 20, 30)]
MOTIONS += [(f"poly{o}", 0, ("poly", o)) for o in (1, 2, 3)]
MOTIONS += [(f"bp{lo}_{hi}", 0, ("bp", (lo, hi)))
            for lo, hi in ((5, 100), (5, 150), (10, 150), (20, 200), (30, 250), (40, 300))]
MOTIONS += [(f"svd{n}", n, None) for n in (1, 2)]
# spatial axis -- (label, ("gauss"|"median", sigma_or_size_mm) or None). x-extent = 2 * z-extent.
SPATIALS = [("none", None)]
SPATIALS += [(f"gauss{s}", ("gauss", s)) for s in (0.3, 0.6, 1.0, 1.5)]
SPATIALS += [(f"median{s}", ("median", s)) for s in (0.6, 1.2)]
# temporal smoothing axis -- (label, ("mean"|"median", window) or None)
TEMPORALS = [("none", None), ("mean3", ("mean", 3)), ("mean5", ("mean", 5)), ("median3", ("median", 3))]
DIRECTIONS = ["leftward", "rightward"]                    # also fixes r0 (origin end)
MLINES = [(5, 0.5), (5, 1.0), (9, 0.5), (9, 1.0)]         # (n_offsets, offset_step_mm)
RANKS = sorted({r for _, r, _ in MOTIONS})

N_GRID = len(QUANTITIES) * len(MOTIONS) * len(SPATIALS) * len(TEMPORALS) * len(DIRECTIONS) * len(MLINES)


# ----------------------------------------------------------------------------- fast metric
def pc_fast(data, r, dt, r0, cmin=0.5, cmax=8.0, n_speeds=41, d_min=2e-3, d_max=None, demean=True):
    """Vectorised :func:`swp.viz.metrics.passive_coherence` (one-sided, free t0, full range).

    Score = (best finite-speed slant-stack peak - no-moveout flat peak) / (per-column peak - flat):
    a genuinely propagating wave stacks far better when aligned to its slope (-> ~1); a flat
    non-propagating band or noise gains little from alignment (-> ~0).  Returns (coherence, speed).
    """
    if demean:
        data = data - data.mean(axis=0, keepdims=True)
    env = np.abs(hilbert(data, axis=0))                   # (nt, ncols)
    nt = env.shape[0]
    d = np.abs(r - r0)
    if d_max is None:
        d_max = float(d.max())
    cols = np.where((d >= d_min) & (d <= d_max))[0]
    if cols.size < 4:
        return 0.0, float("nan")
    e = env[:, cols]                                      # (nt, nc)
    dc = d[cols]
    ci = np.arange(e.shape[1])
    colpeak = e.max(axis=0).mean() + 1e-12
    flat = float(e.mean(axis=1).max())                    # c -> inf (no moveout) baseline
    base_i = np.arange(nt)[:, None]
    speeds = np.linspace(cmin, cmax, n_speeds)
    best, best_c = 0.0, float("nan")
    for c in speeds:
        shift = dc / c / dt                               # (nc,) per-column outward moveout
        valid = shift <= (nt - 1)
        if int(valid.sum()) < 4:
            continue
        idx = base_i + shift[None, :]                     # (nt, nc) fractional time for each column
        i0 = np.floor(idx).astype(np.intp)
        frac = idx - i0
        in_range = (idx >= 0) & (idx <= nt - 1)
        i0c = np.clip(i0, 0, nt - 1)
        i1c = np.clip(i0 + 1, 0, nt - 1)
        vals = e[i0c, ci] * (1 - frac) + e[i1c, ci] * frac
        vals = np.where(in_range, vals, 0.0)
        stacked = vals[:, valid].mean(axis=1)             # align outward moveout, average columns
        peak = float(stacked.max())
        if peak > best:
            best, best_c = peak, float(c)
    headroom = colpeak - flat
    coh = float(np.clip((best - flat) / headroom, 0.0, 1.0)) if headroom > 1e-6 * colpeak else 0.0
    return coh, best_c


# ----------------------------------------------------------------------------- filter helpers
def apply_spatial(field, sstep, dz, dx):
    """Full-field (F, nz, nx) spatial filter."""
    if sstep is None:
        return field
    kind, s = sstep
    if kind == "gauss":
        return gaussian_filter(field, sigma=(0.0, s * 1e-3 / dz, 2 * s * 1e-3 / dx), mode="nearest")
    # median (odd kernels)
    def odd(v):
        v = max(1, int(round(v)))
        return v if v % 2 else v + 1
    return median_filter(field, size=(1, odd(s * 1e-3 / dz), odd(2 * s * 1e-3 / dx)), mode="nearest")


def apply_motion_st(st, mstep, prf):
    """Temporal motion filter on a (nt, ns) space-time (band/highpass/poly)."""
    if mstep is None:
        return st
    from scipy.signal import butter, filtfilt
    kind, p = mstep
    n = st.shape[0]
    if kind == "poly":
        tt = np.arange(n, dtype=float)
        V = np.vander(tt, p + 1)
        coef, *_ = np.linalg.lstsq(V, st, rcond=None)
        return st - V @ coef
    ny = 0.5 * prf
    if kind == "hp":
        wn = min(max(p / ny, 1e-3), 0.99)
        b, a = butter(2, wn, btype="highpass")
    else:  # bp
        lo, hi = p
        lo, hi = lo / ny, min(hi / ny, 0.99)
        b, a = butter(2, [lo, hi], btype="bandpass")
    if n <= 3 * max(len(a), len(b)):
        return st - st.mean(axis=0, keepdims=True)
    return filtfilt(b, a, st, axis=0)


def apply_temporal_st(st, tstep):
    """Temporal smoothing on a (nt, ns) space-time (moving mean/median)."""
    if tstep is None:
        return st
    kind, w = tstep
    if kind == "mean":
        return uniform_filter1d(st, size=max(1, int(w)), axis=0, mode="nearest")
    w = max(1, int(w))
    return median_filter(st, size=(w if w % 2 else w + 1, 1), mode="nearest")


def quantity_field(res, quantity, t_frames):
    """(field, times) after drop_first=1, mirroring pipeline.run_pipeline."""
    if quantity == "velocity":
        f_all, t_all = res.velocity, 0.5 * (t_frames[:-1] + t_frames[1:])
    elif quantity == "acceleration":
        f_all, t_all = np.diff(res.velocity, axis=0), t_frames[1:-1]
    else:
        f_all, t_all = res.displacement, t_frames
    return f_all[1:], t_all[1:]


# ----------------------------------------------------------------------------- core search
def search_window(acq_w, mline, prf, dz, dx, t_frames):
    """Return (rows, base) for one burst window.  rows: list of dict with recipe + coherence."""
    est = ESTIMATORS["loupas"]
    ekw = dict(dz=dz, dx=dx, c=acq_w.c, f_demod=acq_w.f_demod, prf=prf, mode="frame_to_frame")
    base = {}
    for rank in RANKS:
        iqf = svd_clutter(acq_w.iq, n_remove=rank) if rank > 0 else acq_w.iq
        base[rank] = est(iqf, **ekw)

    r = mline.r
    rows = []
    for quantity in QUANTITIES:
        # field per rank (drop_first applied); all share the same time axis per quantity
        fields, times = {}, None
        for rank in RANKS:
            fld, times = quantity_field(base[rank], quantity, t_frames)
            fields[rank] = fld
        dt = float(times[1] - times[0])
        for slabel, sstep in SPATIALS:
            sfields = {rank: apply_spatial(fields[rank], sstep, dz, dx) for rank in RANKS}
            for noff, step in MLINES:
                st_raw = {rank: sample_along_mline(sfields[rank], acq_w.z, acq_w.x, mline,
                                                   n_offsets=noff, offset_step_m=step * 1e-3)
                          for rank in RANKS}
                for mlabel, mrank, mstep in MOTIONS:
                    st_m = apply_motion_st(st_raw[mrank], mstep, prf)
                    for tlabel, tstep in TEMPORALS:
                        st_mt = apply_temporal_st(st_m, tstep)
                        for direction in DIRECTIONS:
                            if direction == "leftward":
                                data_d, r0 = directional_spacetime(st_mt, "neg"), float(r[-1])
                            else:
                                data_d, r0 = directional_spacetime(st_mt, "pos"), float(r[0])
                            coh, c = pc_fast(data_d, r, dt, r0)
                            rows.append(dict(quantity=quantity, motion=mlabel, spatial=slabel,
                                             temporal=tlabel, direction=direction,
                                             offsets=noff, step=step, coherence=coh, speed=c))
    return rows, base


# ----------------------------------------------------------------------------- montage builders
class _Speed:
    """Minimal stand-in for SpeedResult so spacetime_montage can draw a panel."""
    def __init__(self, label):
        self._label, self.t_pred_pos, self.t_pred_neg = label, np.array([np.nan]), np.array([np.nan])
    def label(self):
        return self._label


class _Panel:
    def __init__(self, st, r0, label):
        self.st, self.r0, self.speed = st, r0, _Speed(label)
        self.config = type("C", (), {"label": lambda s: ""})()


def build_spacetime_for(acq_w, mline, prf, dz, dx, t_frames, base, s):
    """Rebuild the (directional) space-time for one recipe dict ``s`` for plotting."""
    rank = int(s["motion"][3:]) if s["motion"].startswith("svd") else 0
    fld, times = quantity_field(base[rank], s["quantity"], t_frames)
    sstep = _spatial_step(s["spatial"])
    fld = apply_spatial(fld, sstep, dz, dx)
    st_raw = sample_along_mline(fld, acq_w.z, acq_w.x, mline,
                                n_offsets=int(s["offsets"]), offset_step_m=s["step"] * 1e-3)
    st_m = apply_motion_st(st_raw, _motion_step(s["motion"]), prf)
    st_mt = apply_temporal_st(st_m, _temporal_step(s["temporal"]))
    r = mline.r
    if s["direction"] == "leftward":
        data_d, r0 = directional_spacetime(st_mt, "neg"), float(r[-1])
    else:
        data_d, r0 = directional_spacetime(st_mt, "pos"), float(r[0])
    return SpaceTime(data_d, r, times, s["quantity"]), r0


def _spatial_step(label):
    for lab, st in SPATIALS:
        if lab == label:
            return st
    return None


def _motion_step(label):
    for lab, _, st in MOTIONS:
        if lab == label:
            return st
    return None


def _temporal_step(label):
    for lab, st in TEMPORALS:
        if lab == label:
            return st
    return None


def recipe_str(s):
    return (f"{s['quantity'][:4]}/{s['motion']}/{s['spatial']}/{s['temporal']}/"
            f"{s['direction'][:1]}/off{s['offsets']}·{s['step']}mm")


def top_montage(acq_w, mline, prf, dz, dx, t_frames, base, top, path, suptitle):
    panels, titles = [], []
    for s in top:
        st, r0 = build_spacetime_for(acq_w, mline, prf, dz, dx, t_frames, base, s)
        panels.append(_Panel(st, r0, f"c={s['speed']:.1f} m/s"))
        titles.append(f"{recipe_str(s)}\npc={s['coherence']:.3f}")
    spacetime_montage(panels, path, ncols=min(4, len(panels)), suptitle=suptitle, panel_titles=titles)


def marginals_figure(rows, path, title):
    params = ["quantity", "motion", "spatial", "temporal", "direction", "offsets"]
    fig, axs = plt.subplots(2, 3, figsize=(18, 9))
    for ax, p in zip(axs.ravel(), params):
        vals = sorted(set(r[p] for r in rows), key=lambda v: (isinstance(v, str), v))
        data = [[r["coherence"] for r in rows if r[p] == v] for v in vals]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels([str(v) for v in vals], rotation=45, ha="right", fontsize=8)
        ax.set_title(f"passive_coherence vs {p}"); ax.set_ylabel("coherence"); ax.grid(alpha=0.3)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(path, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", help="measurement folder (uses <folder>/output)")
    p.add_argument("--config", default="configs/passive.yaml")
    p.add_argument("--window-ms", type=float, default=100.0)
    p.add_argument("--max-events", type=int, default=4)
    p.add_argument("--pad-ms", type=float, default=20.0)
    p.add_argument("--overview-stride", type=int, default=2)
    p.add_argument("--n-consensus-montage", type=int, default=1)
    a = p.parse_args()

    cfg = rc.load_config(a.config if os.path.isabs(a.config) else os.path.join(_ROOT, a.config))
    output_dir = os.path.join(a.folder, "output")
    cfg["data"]["root"] = output_dir
    iq_path = rc.hdf5_path(cfg, 0)
    bmode_path = os.path.join(output_dir, cfg["data"].get("bmode") or os.path.basename(iq_path))
    mlines_dir = os.path.join(output_dir, "mlines")
    outdir = os.path.join(rc.outdir(cfg, output_dir), "search")
    os.makedirs(outdir, exist_ok=True)

    print(f"passive search grid = {N_GRID} combos/window "
          f"({len(QUANTITIES)}q x {len(MOTIONS)}motion x {len(SPATIALS)}spatial x "
          f"{len(TEMPORALS)}temporal x {len(DIRECTIONS)}dir x {len(MLINES)}mline)")
    t_load = time.time()
    acq = load_acquisition(iq_path)
    prf, dz, dx = acq.prf, acq.dz, acq.dx
    print(f"  loaded {acq.summary()} in {time.time()-t_load:.1f}s")

    base_cfg = rc.build_pipeline_config(cfg, acq=acq)

    # --- burst-window detection (same as the passive workflow) ---
    gen_npz = os.path.join(mlines_dir, "passive_general_mline.npz")
    gen_mline = _ensure_mline(gen_npz, bmode_path, _middle_frame(bmode_path),
                              "GENERAL passive M-line (buffer 4)",
                              n_samples=cfg["mline"].get("n_samples", 250))
    detect_band = cfg.get("detect", {}).get("band", [5.0, 150.0])
    from swp.viz.pipeline import run_pipeline
    smoothing = [s for s in base_cfg.field_filters if s.name != "temporal_bandpass"]
    ov_filters = [Step("temporal_bandpass", dict(f_lo=detect_band[0], f_hi=detect_band[1]))] + smoothing
    ov_cfg = replace(base_cfg, directional=False, field_filters=ov_filters)
    ov = run_pipeline(_stride_acq(acq, a.overview_stride), gen_mline, ov_cfg, focus=None)
    D_st = np.asarray(ov.st.data).T
    windows, _ = detect_line_bursts(D_st, np.asarray(ov.st.t), window_ms=a.window_ms,
                                    max_events=a.max_events)
    print(f"  detected {len(windows)} burst window(s): "
          + ", ".join(f"#{i}@{w.t_peak*1e3:.0f}ms" for i, w in enumerate(windows)))

    pad_s = a.pad_ms * 1e-3
    per_window_best, all_rows = [], defaultdict(list)   # recipe-key -> list of coherence
    lead = ["# Passive SWE -- exhaustive parameter search (scored by passive_coherence)\n",
            f"Grid = **{N_GRID} combos/window**. Metric = one-sided origin coherence in [0,1]; "
            "higher = a clearer propagating wavefront. Best-fit shear-wave speed `c` (m/s) reported "
            "alongside.\n"]

    key = lambda s: (s["quantity"], s["motion"], s["spatial"], s["temporal"], s["direction"],
                     s["offsets"], s["step"])

    for i, w in enumerate(windows):
        frame = _frame_at_time(acq.t, w.t_peak)
        npz = os.path.join(mlines_dir, f"passive_win{i}_mline.npz")
        mline = _ensure_mline(npz, bmode_path, frame,
                              f"passive M-line -- window {i} @ {w.t_peak*1e3:.0f} ms",
                              n_samples=cfg["mline"].get("n_samples", 250))
        i0 = _frame_at_time(acq.t, w.t0 - pad_s)
        i1 = _frame_at_time(acq.t, w.t1 + pad_s) + 1
        acq_w = replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        t_frames = np.asarray(acq_w.t, float)

        t0 = time.time()
        rows, base = search_window(acq_w, mline, prf, dz, dx, t_frames)
        rows.sort(key=lambda r: -r["coherence"])
        dt = time.time() - t0
        print(f"  win{i} @ {w.t_peak*1e3:.0f}ms: {len(rows)} combos in {dt:.0f}s; "
              f"best pc={rows[0]['coherence']:.3f} c={rows[0]['speed']:.1f} m/s "
              f"[{recipe_str(rows[0])}]")

        with open(os.path.join(outdir, f"results_win{i}.csv"), "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)
        marginals_figure(rows, os.path.join(outdir, f"marginals_win{i}.png"),
                         f"win{i} @ {w.t_peak*1e3:.0f}ms: passive_coherence marginals ({len(rows)} combos)")
        top_montage(acq_w, mline, prf, dz, dx, t_frames, base, rows[:8],
                    os.path.join(outdir, f"top_win{i}.png"),
                    f"win{i} @ {w.t_peak*1e3:.0f}ms -- top 8 by passive_coherence")

        per_window_best.append((i, w, mline, acq_w, t_frames, base, rows))
        for s in rows:
            all_rows[key(s)].append(s["coherence"])

        lead.append(f"\n## win{i} @ {w.t_peak*1e3:.0f} ms  (window [{w.t0*1e3:.0f}, {w.t1*1e3:.0f}] ms; "
                    f"top 15 of {len(rows)})\n")
        lead.append("| rank | quantity | motion | spatial | temporal | dir | off·step | pc | c (m/s) |")
        lead.append("|---|---|---|---|---|---|---|---|---|")
        for rk, s in enumerate(rows[:15], 1):
            lead.append(f"| {rk} | {s['quantity']} | {s['motion']} | {s['spatial']} | {s['temporal']} "
                        f"| {s['direction']} | {s['offsets']}·{s['step']}mm | {s['coherence']:.3f} "
                        f"| {s['speed']:.1f} |")

    # --- cross-window consensus: recipe with the highest MEAN coherence across windows ---
    n_win = len(windows)
    consensus = []
    for k, vals in all_rows.items():
        if len(vals) == n_win:
            consensus.append((k, float(np.mean(vals)), float(np.min(vals))))
    consensus.sort(key=lambda kv: -kv[1])
    lead.append("\n## Consensus across all windows (mean passive_coherence, recipes present in every window)\n")
    lead.append("| rank | quantity | motion | spatial | temporal | dir | off·step | mean pc | min pc |")
    lead.append("|---|---|---|---|---|---|---|---|---|")
    for rk, (k, mean_pc, min_pc) in enumerate(consensus[:20], 1):
        q, mo, sp, tp, di, off, stp = k
        lead.append(f"| {rk} | {q} | {mo} | {sp} | {tp} | {di} | {off}·{stp}mm "
                    f"| {mean_pc:.3f} | {min_pc:.3f} |")

    # --- consensus montage(s): apply the best consensus recipe(s) to every window ---
    for ci in range(min(a.n_consensus_montage, len(consensus))):
        k, mean_pc, _ = consensus[ci]
        q, mo, sp, tp, di, off, stp = k
        srec = dict(quantity=q, motion=mo, spatial=sp, temporal=tp, direction=di, offsets=off, step=stp)
        panels, titles = [], []
        for (i, w, mline, acq_w, t_frames, base, rows) in per_window_best:
            st, r0 = build_spacetime_for(acq_w, mline, prf, dz, dx, t_frames, base, srec)
            pc, c = pc_fast(st.data, mline.r, float(st.t[1]-st.t[0]), r0)
            panels.append(_Panel(st, r0, f"c={c:.1f} m/s"))
            titles.append(f"win{i} @ {w.t_peak*1e3:.0f}ms\npc={pc:.3f}")
        name = "consensus_montage.png" if ci == 0 else f"consensus_montage_{ci}.png"
        spacetime_montage(panels, os.path.join(outdir, name), ncols=min(4, len(panels)),
                          suptitle=f"CONSENSUS recipe [{recipe_str(srec)}]  mean pc={mean_pc:.3f}",
                          panel_titles=titles)

    # --- baseline (the starting configs/passive.yaml recipe) across windows, for comparison ---
    baseline = dict(quantity="displacement", motion="bp5_150", spatial="gauss0.6",
                    temporal="mean3", direction="leftward", offsets=7, step=0.8)
    panels, titles, base_pcs = [], [], []
    for (i, w, mline, acq_w, t_frames, base, rows) in per_window_best:
        st, r0 = build_spacetime_for(acq_w, mline, prf, dz, dx, t_frames, base, baseline)
        pc, c = pc_fast(st.data, mline.r, float(st.t[1]-st.t[0]), r0)
        base_pcs.append(pc)
        panels.append(_Panel(st, r0, f"c={c:.1f} m/s"))
        titles.append(f"win{i} @ {w.t_peak*1e3:.0f}ms\npc={pc:.3f}")
    spacetime_montage(panels, os.path.join(outdir, "baseline_montage.png"), ncols=min(4, len(panels)),
                      suptitle=f"BASELINE (starting passive.yaml recipe) [{recipe_str(baseline)}]",
                      panel_titles=titles)
    lead.append(f"\n## Baseline vs best\nStarting recipe `{recipe_str(baseline)}` gives per-window pc = "
                + ", ".join(f"{v:.3f}" for v in base_pcs) + f" (mean {np.mean(base_pcs):.3f}).\n")

    with open(os.path.join(outdir, "leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lead))
    print(f"done -> {outdir}")
    if consensus:
        k, mean_pc, min_pc = consensus[0]
        print(f"CONSENSUS best: {k} mean_pc={mean_pc:.3f} min_pc={min_pc:.3f}")


if __name__ == "__main__":
    main()
