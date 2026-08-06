"""Passive SWE exhaustive search v2 -- incorporates the MATLAB-comparison findings.

Adds, over `search_passive.py`:
  1. directional (k-omega) filter: none | leftward           (finding: 'none' is MATLAB-like)
  2. N shifted M-lines, N in {1,5,11}, aggregated by mean | median
  3. IQ pre-filter before displacement estimation: none | low-pass {150, 250} Hz (slow-time)
  4. band-pass / polynomial / SVD-eigenvalue motion removal (expanded)
  5. spatial: none | Gaussian(sigma) | median(size)
  6. temporal: none | moving mean/median, N in 1..7
  7. quantity: displacement | velocity | acceleration

Scored by a signed **propagation-clarity** metric = (best-slope tau-p semblance) - (flat semblance),
over physiological speeds c in [1.5, 6] m/s and BOTH travel directions -- rewards a coherent *tilted*
wavefront over a flat band (a proxy for 'is the shear wave clearly propagating?'). The best speed is
reported as a signed Radon speed. Because the metric is only a proxy, the deliverable is the PLOTS:

Outputs (`<folder>/output/swp_passive/search2/`), per window:
  top_win{i}.png            -- top-16 space-times by clarity score (reference M-mode orientation)
  marg_{axis}_win{i}.png    -- vary ONE axis, hold the rest at the best combo (the key visual)
  results_win{i}.csv, leaderboard.md

Usage:  python scripts/search_passive2.py --window 2 --label AVC
        python scripts/search_passive2.py --window 0 --label MVC
"""
from __future__ import annotations
import argparse, csv, os, sys, time, math
from dataclasses import replace
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.ndimage import map_coordinates
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
from swp.viz.estimators import ESTIMATORS
from swp.viz.filters import svd_clutter
from swp.viz.filters.directional import directional_spacetime
from swp.viz.speed.spacetime import SpaceTime
from swp.viz.pipeline import Step, run_pipeline
from swp.mline.select import detect_line_bursts
from swp.passive import _stride_acq, _frame_at_time, _middle_frame, _ensure_mline
import search_passive as S

# --------------------------------------------------------------------------- axes (label, value)
QUANTITIES = ["displacement", "velocity", "acceleration"]
IQ_PREFILTER = [("none", None), ("lp250", 250.0)]
SVD_RANKS = [0, 1]
MOTIONS = ([("none", None)]
           + [(f"bp{lo}_{hi}", ("bp", (lo, hi))) for lo, hi in ((5, 150), (10, 150), (10, 80))]
           + [("poly1", ("poly", 1))])
SPATIALS = [("none", None), ("gauss0.6", ("gauss", 0.6)), ("gauss1.0", ("gauss", 1.0)),
            ("median1.0", ("median", 1.0))]
TEMPORALS = [("none", None), ("mean3", ("mean", 3)), ("mean5", ("mean", 5)), ("median3", ("median", 3))]
DIRECTIONS = ["none", "leftward"]
MLINE_N = [1, 5, 11]
MLINE_AGG = ["mean", "median"]
STEP_MM = 0.5

MOT = dict(MOTIONS); SPA = dict(SPATIALS); TMP = dict(TEMPORALS); PRE = dict(IQ_PREFILTER)
AXES = ["quantity", "prefilter", "svd", "motion", "spatial", "temporal", "direction", "N", "agg"]
LEVELS = {"quantity": QUANTITIES, "prefilter": [l for l, _ in IQ_PREFILTER], "svd": SVD_RANKS,
          "motion": [l for l, _ in MOTIONS], "spatial": [l for l, _ in SPATIALS],
          "temporal": [l for l, _ in TEMPORALS], "direction": DIRECTIONS, "N": MLINE_N, "agg": MLINE_AGG}


# --------------------------------------------------------------------------- signal helpers
def lowpass_iq(iq, fc, prf):
    if fc is None:
        return iq
    wn = min(fc / (0.5 * prf), 0.99)
    b, a = butter(3, wn, "low")
    if iq.shape[0] <= 3 * max(len(a), len(b)):
        return iq
    return filtfilt(b, a, iq.real, axis=0) + 1j * filtfilt(b, a, iq.imag, axis=0)


def sample_stack(field, z, x, mline, N, step_m):
    """Return (nt, N, ns): the field sampled along N parallel shifted M-lines."""
    if N <= 1:
        offs = np.array([0.0])
    else:
        offs = (np.arange(N) - (N - 1) / 2.0) * step_m
    zq = np.concatenate([mline.z + o * mline.nx_hat[:, 1] for o in offs])
    xq = np.concatenate([mline.x + o * mline.nx_hat[:, 0] for o in offs])
    iz = np.interp(zq, z, np.arange(z.size))
    ix = np.interp(xq, x, np.arange(x.size))
    nt = field.shape[0]; p = iz.size
    coords = np.vstack([np.repeat(np.arange(nt), p), np.tile(iz, nt), np.tile(ix, nt)])
    v = map_coordinates(field, coords, order=1, mode="nearest")
    return v.reshape(nt, offs.size, mline.n_samples)


def prop_fit(data, r, dt, cmin=1.5, cmax=6.0, n_speeds=25, return_line=False):
    """Signed tau-p propagation clarity: (best tilted-slope semblance) - (flat semblance), over BOTH
    travel directions and physiological speeds. Higher score = a clearer tilted (propagating) wavefront
    relative to a flat band. Returns (score, best_semblance, c_signed[, (slowness, t0)]).
    Columns are decimated to ~64 for the fit (a slope needs no more)."""
    step = max(1, data.shape[1] // 64)
    d = data[:, ::step]
    d = d - d.mean(axis=0, keepdims=True)
    nt, ns = d.shape
    lever = (r[::step] - float(r[::step].mean())) / dt
    ci = np.arange(ns); base = np.arange(nt)[:, None]

    def semb(shift, want=False):
        lo = int(np.ceil(max(0.0, -shift.min()))); hi = nt - int(np.ceil(max(0.0, shift.max())))
        if hi - lo < 8:
            return (0.0, None, lo) if want else 0.0
        idx = base[lo:hi] + shift[None, :]
        i0 = np.floor(idx).astype(np.intp); frac = idx - i0
        al = d[i0, ci] * (1 - frac) + d[np.clip(i0 + 1, 0, nt - 1), ci] * frac
        num = al.sum(1); den = (al ** 2).sum(1)
        s = float((num ** 2).sum() / (ns * den.sum() + 1e-20))
        return (s, num, lo) if want else s

    flat = semb(np.zeros(ns))
    p_pos = np.linspace(1.0 / cmax, 1.0 / cmin, n_speeds)
    slow = np.concatenate([-p_pos[::-1], p_pos])
    best, best_p = -1.0, 0.0
    for p in slow:
        s = semb(p * lever)
        if s > best:
            best, best_p = s, float(p)
    score = best - flat
    c_signed = 1.0 / best_p if best_p != 0.0 else float("inf")
    if not return_line:
        return score, best, c_signed
    from scipy.signal import hilbert
    _, stack, lo = semb(best_p * lever, want=True)
    t0 = lo + (int(np.argmax(np.abs(hilbert(stack)))) if stack is not None else 0)
    return score, best, c_signed, (best_p, t0)


# --------------------------------------------------------------------------- core search
def search_window(acq_w, mline, prf, dz, dx, tf):
    r = mline.r
    # base estimations cached per (prefilter, svd)
    base = {}
    est = ESTIMATORS["loupas"]
    for plabel, fc in IQ_PREFILTER:
        iq_lp = lowpass_iq(acq_w.iq, fc, prf)
        for rank in SVD_RANKS:
            iqf = svd_clutter(iq_lp, n_remove=rank) if rank > 0 else iq_lp
            base[(plabel, rank)] = est(iqf, dz=dz, dx=dx, c=acq_w.c, f_demod=acq_w.f_demod,
                                       prf=prf, mode="frame_to_frame")
    rows = []
    for (plabel, rank), res in base.items():
        for q in QUANTITIES:
            fld, times = S.quantity_field(res, q, tf)
            dt = float(times[1] - times[0])
            for slabel in LEVELS["spatial"]:
                sf = S.apply_spatial(fld, SPA[slabel], dz, dx)
                for N in MLINE_N:
                    stack = sample_stack(sf, acq_w.z, acq_w.x, mline, N, STEP_MM * 1e-3)
                    for agg in MLINE_AGG:
                        st_raw = stack.mean(1) if agg == "mean" else np.median(stack, axis=1)
                        for mlabel in LEVELS["motion"]:
                            st_m = S.apply_motion_st(st_raw, MOT[mlabel], prf)
                            for tlabel in LEVELS["temporal"]:
                                st_mt = S.apply_temporal_st(st_m, TMP[tlabel])
                                for direction in DIRECTIONS:
                                    data = (st_mt if direction == "none"
                                            else directional_spacetime(st_mt, "neg"))
                                    score, semb, c = prop_fit(data, r, dt)
                                    rows.append(dict(quantity=q, prefilter=plabel, svd=rank,
                                                     motion=mlabel, spatial=slabel, temporal=tlabel,
                                                     direction=direction, N=N, agg=agg,
                                                     score=score, semblance=semb, speed=c))
    return rows, base


def build_st(base, acq_w, mline, prf, dz, dx, tf, rec):
    res = base[(rec["prefilter"], int(rec["svd"]))]
    fld, times = S.quantity_field(res, rec["quantity"], tf)
    sf = S.apply_spatial(fld, SPA[rec["spatial"]], dz, dx)
    stack = sample_stack(sf, acq_w.z, acq_w.x, mline, int(rec["N"]), STEP_MM * 1e-3)
    st_raw = stack.mean(1) if rec["agg"] == "mean" else np.median(stack, axis=1)
    st_m = S.apply_motion_st(st_raw, MOT[rec["motion"]], prf)
    st_mt = S.apply_temporal_st(st_m, TMP[rec["temporal"]])
    data = st_mt if rec["direction"] == "none" else directional_spacetime(st_mt, "neg")
    return SpaceTime(data, mline.r, times, rec["quantity"])


# --------------------------------------------------------------------------- plotting
def draw(ax, st, title):
    unit = 1e3 if st.quantity == "velocity" else (1e4 if st.quantity == "acceleration" else 1e6)
    rmask = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    clim = robust_clim(st.data, rmask, pct=97) * unit
    ext = [st.t[0] * 1e3, st.t[-1] * 1e3, st.r[-1] * 1e3, st.r[0] * 1e3]
    ax.imshow(st.data.T * unit, extent=ext, cmap="RdBu_r", vmin=-clim, vmax=clim,
              aspect="auto", origin="upper")
    dt = float(st.t[1] - st.t[0])
    out = prop_fit(st.data, st.r, dt, return_line=True)
    score, semb, c = out[0], out[1], out[2]
    line = out[3] if len(out) > 3 else None
    if line is not None:
        best_p, t0 = line
        t_line = st.t[t0] + best_p * (st.r - st.r.mean())
        m = (t_line >= st.t[0]) & (t_line <= st.t[-1])
        ax.plot(t_line[m] * 1e3, st.r[m] * 1e3, "k", lw=1.5, alpha=0.9)
    ax.set_ylim(st.r[-1] * 1e3, st.r[0] * 1e3)
    ax.set_title(f"{title}\nscore={score:.2f} |c|={abs(c):.1f} m/s", fontsize=7.5)
    ax.tick_params(labelsize=6)


def montage(sts_titles, path, suptitle, ncols=4):
    n = len(sts_titles); ncols = min(ncols, n); nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.9 * nrows), squeeze=False)
    for k, (st, title) in enumerate(sts_titles):
        ax = axs[k // ncols][k % ncols]
        draw(ax, st, title)
        if k % ncols == 0:
            ax.set_ylabel("r [mm]", fontsize=6)
        if k // ncols == nrows - 1:
            ax.set_xlabel("t [ms]", fontsize=6)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=135); plt.close(fig)


def recipe_label(r):
    return (f"{r['quantity'][:4]}/{r['prefilter']}/svd{r['svd']}/{r['motion']}/{r['spatial']}/"
            f"{r['temporal']}/dir:{r['direction'][:4]}/{r['N']}{r['agg'][:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=r"D:\Luuk van Knippenberg\Claude\invivo_sw")
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    cfg = rc.load_config(os.path.join(_ROOT, "configs", "passive.yaml"))
    out = os.path.join(a.folder, "output"); cfg["data"]["root"] = out
    iq_path = rc.hdf5_path(cfg, 0)
    bmode = os.path.join(out, os.path.basename(iq_path)); mldir = os.path.join(out, "mlines")
    outdir = os.path.join(rc.outdir(cfg, out), "search2"); os.makedirs(outdir, exist_ok=True)

    n_grid = (len(IQ_PREFILTER) * len(SVD_RANKS) * len(QUANTITIES) * len(SPATIALS) * len(MLINE_N)
              * len(MLINE_AGG) * len(MOTIONS) * len(TEMPORALS) * len(DIRECTIONS))
    print(f"grid = {n_grid} combos/window")
    acq = load_acquisition(iq_path); prf, dz, dx = acq.prf, acq.dz, acq.dx
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
    w = windows[a.window]
    ml = _ensure_mline(os.path.join(mldir, f"passive_win{a.window}_mline.npz"), bmode,
                       _frame_at_time(acq.t, w.t_peak), f"win{a.window}", n_samples=cfg["mline"]["n_samples"])
    i0 = _frame_at_time(acq.t, w.t0 - 0.02); i1 = _frame_at_time(acq.t, w.t1 + 0.02) + 1
    acq_w = replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1]); tf = np.asarray(acq_w.t, float)

    t0 = time.time()
    rows, base = search_window(acq_w, ml, prf, dz, dx, tf)
    rows.sort(key=lambda r: -r["score"])
    print(f"win{a.window} {a.label}: {len(rows)} combos in {time.time()-t0:.0f}s; "
          f"best score={rows[0]['score']:.3f} |c|={abs(rows[0]['speed']):.1f} [{recipe_label(rows[0])}]")

    with open(os.path.join(outdir, f"results_win{a.window}.csv"), "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)

    # top montage
    top = rows[:16]
    montage([(build_st(base, acq_w, ml, prf, dz, dx, tf, r), recipe_label(r)) for r in top],
            os.path.join(outdir, f"top_win{a.window}.png"),
            f"win{a.window} @ {w.t_peak*1e3:.0f}ms ({a.label}) -- top 16 by propagation-clarity score")

    # per-axis marginal montages: vary ONE axis, hold the rest at the best combo
    best = rows[0]
    for axis in AXES:
        panels = []
        for lvl in LEVELS[axis]:
            rec = dict(best); rec[axis] = lvl
            st = build_st(base, acq_w, ml, prf, dz, dx, tf, rec)
            panels.append((st, f"{axis}={lvl}"))
        montage(panels, os.path.join(outdir, f"marg_{axis}_win{a.window}.png"),
                f"win{a.window} ({a.label}) -- vary {axis}; others = best "
                f"[{recipe_label(best)}]", ncols=min(4, len(panels)))

    # leaderboard
    with open(os.path.join(outdir, f"leaderboard_win{a.window}.md"), "w", encoding="utf-8") as f:
        f.write(f"# win{a.window} ({a.label}) -- top 30 by propagation-clarity score\n\n")
        f.write("| rank | score | |c| | recipe |\n|---|---|---|---|\n")
        for i, r in enumerate(rows[:30], 1):
            f.write(f"| {i} | {r['score']:.3f} | {abs(r['speed']):.1f} | {recipe_label(r)} |\n")
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
