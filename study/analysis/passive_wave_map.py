"""2D wave map of a passive event: the shear wave on the buffer-4 B-mode, as a slow-motion video.

For one event window of the manual study (``<folder>/output/swp_passive_manual``), the same
processing as its "velocity gauss" space-time (configs/passive_manual.yaml: Loupas frame-to-frame
axial velocity, 15-150 Hz, Gaussian 0.6 x 1.2 mm, mean 3) is run on a box around the event's
M-line, and the full 2D field (``PipelineResult.field``) is kept instead of only its samples
along the line. Read only: nothing is written to the folder.

Per video frame:
  left   the buffer-4 B-mode of that frame, with the axial tissue velocity overlaid (red = away
         from the probe, blue = towards it; opacity grows with |v|, so still tissue stays grey),
         the event M-line (star = r = 0) and a dot where the HAND slope puts the wavefront
  right  the space-time along the M-line with the hand line and a cursor at the current time

    python study/analysis/passive_wave_map.py --folder <folder> --window <i> [--margin-mm 12]
           [--fps 15] [--out study/montages/passive_wave_map]

Writes <subject>_<event label>_w<i>.mp4 (H.264) and a .gif, plus a still of the clearest frame.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
import matplotlib                                               # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.manual import store as S                               # noqa: E402

PAD_S = 0.020                    # as the worker (swp.manual.worker.PAD_S)
EDGE_S = 0.012                   # frames this close to the record ends are filter transients: not shown
SURF, INK = "#fcfcfb", "#0b0b0b"


def compute(folder, i, margin_m):
    """The event's 2D velocity field, B-mode envelope, line and hand slope."""
    from swp.passive import _build_views
    from swp.viz import runconfig as rc
    from swp.viz.io import load_acquisition
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline

    p = S.Paths(folder)
    w = S.event_windows(p)["windows"][i]
    pts = S.load_points(p.event_npz(i))
    lo, hi = pts.min(axis=0) - margin_m, pts.max(axis=0) + margin_m
    acq = load_acquisition(p.bmode(4), roi=(lo[0], hi[0], lo[1], hi[1]))
    i0 = int(np.argmin(np.abs(acq.t - (w["t0"] - PAD_S))))
    i1 = int(np.argmin(np.abs(acq.t - (w["t1"] + PAD_S)))) + 1
    acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
    cfg = rc.load_config(S.CONFIG)
    cfg["data"]["root"] = p.output
    view = dict(_build_views(cfg, acq_w))["velocity gauss"]
    ml = mline_from_points(pts, S.N_SAMPLES)
    res = run_pipeline(acq_w, ml, view, focus=None)
    env = np.abs(acq_w.iq)
    # field times are frame midpoints, one fewer than the IQ frames: B-mode of the later frame
    k_env = np.clip(np.searchsorted(acq_w.t, res.times), 0, env.shape[0] - 1)
    slope = (S.read_json(p.slopes_json) or {}).get(str(i)) or {}
    return dict(field=np.asarray(res.field, np.float32), t=np.asarray(res.times), env=env[k_env],
                x=acq.x, z=acq.z, ml=ml, st=res.st, window=w, slope=slope.get("shared") or {},
                confidence=slope.get("confidence"),
                label=w.get("label"), subject=os.path.basename(os.path.dirname(folder)))


def bmode_db(env, dr=50.0):
    ref = np.percentile(env, 99.9)
    return np.clip(20 * np.log10(env / ref + 1e-12), -dr, 0)


def render(d, out_base, fps=15):
    import imageio.v2 as imageio
    from matplotlib.colors import Normalize

    keep = (d["t"] >= d["t"][0] + EDGE_S) & (d["t"] <= d["t"][-1] - EDGE_S)
    f, t = d["field"][keep], d["t"][keep]
    x_mm, z_mm = d["x"] * 1e3, d["z"] * 1e3
    db = bmode_db(d["env"][keep])
    # temporal smoothing of the B-mode (3 frames) for a steadier background
    db = np.stack([db[max(k - 1, 0):k + 2].mean(axis=0) for k in range(db.shape[0])])
    tissue = db.mean(axis=0) > -35                      # skip blood / noise for the clim
    # the overlay fades out where the mean B-mode is dark (blood pool, shadow): weight 0 at -45 dB, 1 at -35 dB
    tissue_w = np.clip((db.mean(axis=0) + 45.0) / 10.0, 0.0, 1.0)
    clim = float(np.percentile(np.abs(f[:, tissue]), 99))
    norm = Normalize(-clim, clim)
    cmap = plt.get_cmap("RdBu_r")
    ext = [x_mm[0], x_mm[-1], z_mm[-1], z_mm[0]]
    ml = d["ml"]
    sl = d["slope"]
    fig = plt.figure(figsize=(13, 6.0), facecolor=SURF)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.22)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    im_b = ax.imshow(db[0], cmap="gray", vmin=-50, vmax=0, extent=ext, aspect="equal", interpolation="bilinear")
    rgba = np.zeros(f.shape[1:] + (4,))
    im_v = ax.imshow(rgba, extent=ext, aspect="equal", interpolation="bilinear")
    ax.plot(ml.x * 1e3, ml.z * 1e3, "-", color="#e5d200", lw=1.4, alpha=0.9)
    ax.plot([ml.x[0] * 1e3], [ml.z[0] * 1e3], "*", color="#e5d200", ms=13, mec=INK)
    dot, = ax.plot([], [], "o", ms=10, mfc="none", mec="lime", mew=2.2)
    ttl = ax.set_title("", fontsize=10, loc="left")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("depth z [mm]")
    st = d["st"]
    v_st = np.asarray(st.data)
    lim_st = float(np.percentile(np.abs(v_st), 98))
    ax2.imshow(v_st.T, aspect="auto", cmap="RdBu_r", vmin=-lim_st, vmax=lim_st,
               extent=[st.t[0] * 1e3, st.t[-1] * 1e3, st.r[-1] * 1e3, st.r[0] * 1e3])
    have_line = sl.get("anchor_t_ms") is not None
    if have_line:
        rr = np.array([st.r[0], st.r[-1]]) * 1e3
        ax2.plot(sl["anchor_t_ms"] + (rr - sl["anchor_r_mm"]) / sl["speed_m_s"], rr, "-", color="lime", lw=1.6)
    cur = ax2.axvline(t[0] * 1e3, color=INK, lw=1.4)
    for a in (d["window"]["t0"], d["window"]["t1"]):
        ax2.axvline(a * 1e3, color=INK, lw=0.8, ls=":")
    ax2.set_xlabel("t [ms] (buffer-4 clock)")
    ax2.set_ylabel("r along the M-line [mm]")
    ax2.set_title(f"space-time along the M-line (velocity gauss)"
                  + (f", hand slope {sl['speed_m_s']:+.2f} m/s" if have_line else ""), fontsize=10, loc="left")
    fig.suptitle(f"{d['subject']}  {d['label']} event"
                 + (f" (your confidence {d['confidence']})" if d["confidence"] is not None else "")
                 + f" - axial tissue velocity on the buffer-4 B-mode, slow motion x{1 / (np.median(np.diff(t)) * fps):.0f}\n"
                 f"red: away from the probe, blue: towards it (+-{clim * 1e3:.1f} mm/s); yellow: M-line, star r = 0; "
                 f"green circle: where your hand slope puts the wavefront", fontsize=10, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.11)
    frames, energy = [], []
    for k in range(f.shape[0]):
        im_b.set_data(db[k])
        a = np.clip(np.abs(f[k]) / clim, 0, 1) ** 0.8 * 0.85 * tissue_w
        rgba = cmap(norm(f[k]))
        rgba[..., 3] = a
        im_v.set_data(rgba)
        tk = t[k] * 1e3
        cur.set_xdata([tk, tk])
        ttl.set_text(f"t = {tk:.1f} ms")
        if have_line:
            r_mm = sl["anchor_r_mm"] + sl["speed_m_s"] * (tk - sl["anchor_t_ms"])
            if st.r[0] * 1e3 <= r_mm <= st.r[-1] * 1e3:
                j = int(np.argmin(np.abs(ml.r * 1e3 - r_mm)))
                dot.set_data([ml.x[j] * 1e3], [ml.z[j] * 1e3])
            else:
                dot.set_data([], [])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        energy.append(float(np.mean(f[k][tissue] ** 2)))
    plt.close(fig)
    with imageio.get_writer(out_base + ".mp4", fps=fps, codec="libx264", quality=8,
                            macro_block_size=8) as wr:
        for fr in frames:
            wr.append_data(fr)
    step = 2
    imageio.mimsave(out_base + ".gif", [fr[::2, ::2] for fr in frames[::step]], duration=1000 * step / fps, loop=0)
    k_best = int(np.argmax(energy))
    imageio.imwrite(out_base + "_still.png", frames[k_best])
    return len(frames), k_best


def _dist_to_line(px, pz, ml):
    """Distance of points to the M-line polyline (m) and the arc length r of the nearest sample."""
    d = np.hypot(px[..., None] - ml.x, pz[..., None] - ml.z)
    k = np.argmin(d, axis=-1)
    return np.min(d, axis=-1), ml.r[k]


def arrival_map(d, ref_r_m=0.005, band_m=0.006, cc_min=0.7, max_lag_s=0.025):
    """Arrival time of the event at every pixel, against the M-line's first ``ref_r_m``.

    tau(x, z) = the lag of maximum (same-polarity) cross-correlation of the pixel's velocity trace
    with the reference trace (mean over the tissue within 2 mm of the line's r < ref_r_m), over the
    window minus the filter edges; cc = that correlation. A plane t = a + b.(x, z) fitted to the
    reliable pixels (cc >= cc_min, tissue, within ``band_m`` of the line) gives the 2D speed 1/|b|
    and the propagation direction b/|b|; its angle with the M-line predicts the apparent speed
    along the line, c / cos(theta). The earliest 2 % of the reliable pixels give the origin.
    """
    keep = (d["t"] >= d["t"][0] + EDGE_S) & (d["t"] <= d["t"][-1] - EDGE_S)
    f = d["field"][keep].astype(np.float32)
    nt, nz, nx = f.shape
    dt = float(np.median(np.diff(d["t"])))
    db = bmode_db(d["env"][keep]).mean(axis=0)
    tissue_w = np.clip((db + 45.0) / 10.0, 0.0, 1.0)
    X, Z = np.meshgrid(d["x"], d["z"])
    ml = d["ml"]
    dist, rnear = _dist_to_line(X, Z, ml)
    refm = (dist < 0.002) & (rnear <= ref_r_m) & (tissue_w > 0.5)
    ref = f[:, refm].mean(axis=1)
    P = f.reshape(nt, -1)
    P = P - P.mean(axis=0)
    ref = ref - ref.mean()
    n = 1 << int(np.ceil(np.log2(2 * nt)))
    xc = np.fft.irfft(np.fft.rfft(P, n, axis=0) * np.conj(np.fft.rfft(ref, n))[:, None], n, axis=0)
    xc /= (np.linalg.norm(P, axis=0) * np.linalg.norm(ref) + 1e-30)
    L = int(round(max_lag_s / dt))
    lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
    xc = np.concatenate([xc[:L + 1], xc[n - L:]], axis=0)        # lags 0..L, -L..-1
    k = np.argmax(xc, axis=0)
    cc = xc[k, np.arange(xc.shape[1])]
    # parabolic sub-sample refinement (neighbours in lag order)
    order = np.argsort(lags)
    pos = np.argsort(order)[k]
    xs = xc[order]
    lo, hi = np.clip(pos - 1, 0, 2 * L), np.clip(pos + 1, 0, 2 * L)
    y0, y1, y2 = xs[lo, np.arange(xs.shape[1])], cc, xs[hi, np.arange(xs.shape[1])]
    den = y0 - 2 * y1 + y2
    off = np.where((den < 0) & (lo != pos) & (hi != pos), 0.5 * (y0 - y2) / np.where(den == 0, 1, den), 0.0)
    tau = ((lags[k] + off) * dt).reshape(nz, nx)
    cc = cc.reshape(nz, nx)
    ok = (cc >= cc_min) & (tissue_w >= 0.5) & (dist <= band_m)
    out = dict(tau=tau, cc=cc, ok=ok, tissue_w=tissue_w, db=db, ref_pixels=int(refm.sum()))
    if ok.sum() < 50:
        return out
    A = np.c_[np.ones(ok.sum()), X[ok], Z[ok]]
    y, wgt = tau[ok], cc[ok] ** 2
    good = np.ones(y.size, bool)
    for _ in range(2):                                              # weighted LS + one MAD rejection
        coef = np.linalg.lstsq(A[good] * wgt[good, None], y[good] * wgt[good], rcond=None)[0]
        res = y - A @ coef
        mad = np.median(np.abs(res[good] - np.median(res[good]))) + 1e-9
        good = np.abs(res) < 3 * 1.4826 * mad
    b = coef[1:]
    res = (y - A @ coef)[good]
    r2 = 1 - np.sum(res ** 2) / np.sum((y[good] - y[good].mean()) ** 2)
    c2d = 1.0 / np.linalg.norm(b)
    u_prop = b / np.linalg.norm(b)
    u_line = np.array([ml.x[-1] - ml.x[0], ml.z[-1] - ml.z[0]])
    u_line /= np.linalg.norm(u_line)
    cos_t = float(u_prop @ u_line)
    # along the line: tau at the line samples, robust linear fit
    iz = np.clip(np.searchsorted(d["z"], ml.z), 0, nz - 1)
    ix = np.clip(np.searchsorted(d["x"], ml.x), 0, nx - 1)
    tl, cl = tau[iz, ix], cc[iz, ix]
    m = cl >= cc_min
    slope_line = np.polyfit(ml.r[m], tl[m], 1)[0] if m.sum() > 10 else np.nan
    # origin: the earliest 2 % of the reliable pixels (anywhere in the box)
    rel = (cc >= cc_min) & (tissue_w >= 0.5)
    thr = np.percentile(tau[rel], 2)
    om = rel & (tau <= thr)
    ox, oz = float(X[om].mean()), float(Z[om].mean())
    od, orr = _dist_to_line(np.array(ox), np.array(oz), ml)
    out.update(c2d=float(c2d), dir=u_prop, cos_theta=cos_t, theta_deg=float(np.degrees(np.arccos(np.clip(cos_t, -1, 1)))),
               c_line_pred=float(c2d / cos_t) if abs(cos_t) > 0.05 else np.inf, r2=float(r2),
               n_fit=int(good.sum()), c_line_tau=float(1 / slope_line) if slope_line and np.isfinite(slope_line) else np.nan,
               frac_coherent=float(ok.sum() / max(int(((tissue_w >= 0.5) & (dist <= band_m)).sum()), 1)),
               origin=(ox, oz), origin_to_star_mm=float(np.hypot(ox - ml.x[0], oz - ml.z[0]) * 1e3),
               origin_off_line_mm=float(od * 1e3), origin_r_mm=float(orr * 1e3),
               tau_line=(ml.r, tl, cl))
    return out


def arrival_figure(d, m, out_base):
    """Arrival-time map (isochrones) on the mean B-mode, with the plane-fit direction and origin."""
    ext = [d["x"][0] * 1e3, d["x"][-1] * 1e3, d["z"][-1] * 1e3, d["z"][0] * 1e3]
    ml = d["ml"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax.imshow(m["db"], cmap="gray", vmin=-50, vmax=0, extent=ext, aspect="equal")
    rel = (m["cc"] >= 0.7) & (m["tissue_w"] >= 0.5)
    tau_ms = np.where(rel, m["tau"] * 1e3, np.nan)
    lim = np.nanpercentile(np.abs(tau_ms), 98) if np.isfinite(tau_ms).any() else 10
    im = ax.imshow(tau_ms, cmap="viridis", vmin=-lim / 3, vmax=lim, extent=ext, aspect="equal", alpha=0.75)
    X, Z = np.meshgrid(d["x"] * 1e3, d["z"] * 1e3)
    if np.isfinite(tau_ms).sum() > 50:
        ax.contour(X, Z, np.where(rel, m["tau"] * 1e3, np.nan), levels=np.arange(-20, 30, 2), colors="white",
                   linewidths=0.6)
    ax.plot(ml.x * 1e3, ml.z * 1e3, "-", color="#e5d200", lw=1.5)
    ax.plot([ml.x[0] * 1e3], [ml.z[0] * 1e3], "*", color="#e5d200", ms=13, mec=INK)
    title = f"{d['subject']} {d['label']} - arrival time vs the line's first 5 mm [ms]"
    if "c2d" in m:
        mid = np.array([ml.x[len(ml.x) // 2], ml.z[len(ml.z) // 2]]) * 1e3
        ax.annotate("", xy=mid + 8 * m["dir"], xytext=mid, arrowprops=dict(arrowstyle="->", color="#eb6834", lw=2.5))
        ax.plot([m["origin"][0] * 1e3], [m["origin"][1] * 1e3], "X", color="#eb6834", ms=12, mec="white")
        title += (f"\nplane fit: {m['c2d']:.2f} m/s, {m['theta_deg']:.0f} deg to the line (R2 {m['r2']:.2f}) -> along the "
                  f"line {m['c_line_pred']:.2f} m/s; origin (X) {m['origin_off_line_mm']:.0f} mm off the line")
    ax.set_title(title, fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.035, label="arrival [ms]")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("depth z [mm]")
    if "tau_line" in m:
        r, tl, cl = m["tau_line"]
        ok = cl >= 0.7
        ax2.plot(r[ok] * 1e3, tl[ok] * 1e3, ".", color=INK, ms=4, label="arrival on the line (cc >= 0.7)")
        ax2.plot(r[~ok] * 1e3, tl[~ok] * 1e3, ".", color="0.75", ms=3, label="cc < 0.7")
        sl = d["slope"]
        if sl.get("speed_m_s"):
            rr = np.array([r[0], r[-1]]) * 1e3
            ax2.plot(rr, (rr - rr[0]) / sl["speed_m_s"] + np.nanmedian(tl[ok][:10]) * 1e3, "-", color="lime", lw=1.5,
                     label=f"hand slope {sl['speed_m_s']:+.2f} m/s (offset to fit)")
        ax2.set_xlabel("r along the M-line [mm]")
        ax2.set_ylabel("arrival [ms]")
        ax2.legend(fontsize=8, frameon=False)
        ax2.set_title(f"arrival along the line: {m['c_line_tau']:.2f} m/s (linear fit)", fontsize=9, loc="left")
    for a in (ax, ax2):
        a.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(out_base + "_arrival.png", dpi=90, facecolor=SURF)
    plt.close(fig)


def run_one(folder, window, margin_m, fps, out, video=True):
    import json
    d = compute(folder, window, margin_m)
    os.makedirs(out, exist_ok=True)
    base = os.path.join(out, f"{d['subject']}_{d['label']}_w{window}")
    m = arrival_map(d)
    arrival_figure(d, m, base)
    rec = dict(subject=d["subject"], folder=folder, window=window, label=d["label"], confidence=d["confidence"],
               hand_m_s=(d["slope"] or {}).get("speed_m_s"), ref_pixels=m["ref_pixels"],
               **{k: m.get(k) for k in ("c2d", "theta_deg", "c_line_pred", "c_line_tau", "r2", "n_fit",
                                         "frac_coherent", "origin_to_star_mm", "origin_off_line_mm", "origin_r_mm")})
    with open(base + "_metrics.json", "w") as fh:
        json.dump(rec, fh, indent=1, default=float)
    if video:
        n, kb = render(d, base, fps)
        print(f"{n} frames -> {base}.mp4 / .gif / _still.png (strongest frame {kb})", flush=True)
    return rec


def _run_job(args):
    try:
        return run_one(*args)
    except Exception as exc:                                         # noqa: BLE001
        import traceback
        traceback.print_exc()
        return dict(folder=args[0], window=args[1], error=f"{type(exc).__name__}: {exc}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder")
    ap.add_argument("--window", type=int)
    ap.add_argument("--batch", help="CSV with columns folder, window (e.g. a filtered passive_manual.py export "
                                    "with subject + folder name columns)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--no-video", action="store_true", help="only the arrival map + metrics")
    ap.add_argument("--margin-mm", type=float, default=12.0)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--out", default=os.path.join(REPO, "study", "montages", "passive_wave_map"))
    a = ap.parse_args()
    if a.batch:
        import pandas as pd
        from concurrent.futures import ProcessPoolExecutor
        b = pd.read_csv(a.batch)
        root = os.environ.get("SWP_RAW_DATA", "Z:/raw_data")
        jobs = [((r.folder if os.path.isabs(str(r.folder)) else f"{root}/{r.subject}/{r.folder}"), int(r.window),
                 a.margin_mm * 1e-3, a.fps, a.out, not a.no_video) for r in b.itertuples()]
        with ProcessPoolExecutor(a.jobs) as ex:
            recs = list(ex.map(_run_job, jobs))
        pd.DataFrame(recs).to_csv(os.path.join(a.out, "metrics.csv"), index=False)
        print(f"{len(recs)} event(s), {sum('error' in r for r in recs)} failed -> {a.out}/metrics.csv")
        return
    run_one(a.folder, a.window, a.margin_mm * 1e-3, a.fps, a.out, not a.no_video)


if __name__ == "__main__":
    main()
