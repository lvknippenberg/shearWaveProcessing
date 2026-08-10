"""Matplotlib figure builders for the GUI (return Figure objects; the app calls st.pyplot)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swp.viz.core.geometry import robust_clim


def spacetime_png_datauri(cell, clim, dpi=110):
    """The space-time image alone (no axes/margins) as a base64 PNG data URI, using the SAME colormap
    and colour limits as the quantity grid (``RdBu_r``, ±clim) with the fixed r0 line baked in - so the
    manual-speed panel's backdrop matches the other plots exactly. The Plotly component overlays only the
    draggable line on top, mapped 1:1 to the r/t axes."""
    import base64
    import io
    unit = QUNITS[cell["quantity"]][0]
    r = cell["r"] * 1e3
    t = cell["t"] * 1e3
    fig = plt.figure(figsize=(6.0, 3.2), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.imshow(cell["data"] * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
              vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    ax.axvline(cell["r0"] * 1e3, color="0.15", ls="--", lw=1.3, alpha=0.75)  # fixed r0 symmetry line
    buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def fig_spacetime_with_lines(cell, clim, lines, title=""):
    """A publication-style space-time (same RdBu_r/clim as the grid) with the manual speed line(s)
    overlaid and labelled - saved as the record of a manual measurement."""
    unit, ulab = QUNITS[cell["quantity"]]
    r = cell["r"] * 1e3
    t = cell["t"] * 1e3
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    im = ax.imshow(cell["data"] * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                   vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    ax.axvline(cell["r0"] * 1e3, color="0.15", ls="--", lw=1.2, alpha=0.7, label="r0")
    for i, ((r0m, t0m), (r1m, t1m)) in enumerate(lines, 1):
        dr, dt = r1m - r0m, t1m - t0m
        spd = abs(dr / dt) if abs(dt) > 1e-6 else float("inf")
        ax.plot([r0m, r1m], [t0m, t1m], color="k", lw=2.4)
        ax.plot([r0m, r1m], [t0m, t1m], color="w", lw=0.8, ls=":")
        ax.annotate(f"L{i}: {spd:.2f} m/s", xy=((r0m + r1m) / 2, (t0m + t1m) / 2),
                    xytext=(4, 4), textcoords="offset points", fontsize=8, color="k",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", alpha=0.85))
    ax.set_xlabel("r along M-line [mm]"); ax.set_ylabel("t [ms]")
    ax.set_title(title or f"{cell['quantity']} — manual speed", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=ulab)
    fig.tight_layout()
    return fig


def fig_spacetime(res, title="", show_speed=True, clim=None):
    """Space-time D(s,t) with the r0 origin and (optionally) the fitted wavefront overlay."""
    st = res.st
    r = st.r * 1e3
    t = st.t * 1e3
    unit = 1e6 if st.quantity == "displacement" else 1e3
    img = st.data * unit
    if clim is None:
        rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        clim = robust_clim(st.data, rc, pct=97) * unit or float(np.nanpercentile(np.abs(img), 97))
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.imshow(img, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r", vmin=-clim, vmax=clim,
              aspect="auto", origin="upper")
    ax.axvline(res.r0 * 1e3, color="0.15", ls="--", lw=1.0, alpha=0.7)
    if show_speed and res.speed is not None:
        sp = res.speed
        for tp in (sp.t_pred_pos, sp.t_pred_neg):
            if tp is not None and np.isfinite(tp).any():
                ax.plot(r, np.asarray(tp) * 1e3, color="k", lw=1.2)
    ax.set_xlabel("r along M-line [mm]")
    ax.set_ylabel("t [ms]")
    ax.set_title(title, fontsize=10)
    unit_lbl = "µm" if st.quantity == "displacement" else "mm/s"
    fig.colorbar(ax.images[0], ax=ax, label=f"{st.quantity} [{unit_lbl}]", fraction=0.046)
    fig.tight_layout()
    return fig, float(clim)


# ---- multi-quantity (displacement / velocity / acceleration) displays ----------------------------
QORDER = ["displacement", "velocity", "acceleration"]
QUNITS = {"displacement": (1e6, "µm"), "velocity": (1e3, "mm/s"), "acceleration": (1.0, "m/s²")}


def cell_of(res):
    """Pack a pipeline result into a display cell (arrays copied so it survives in session state)."""
    s = res.st
    return dict(data=s.data.copy(), r=s.r.copy(), t=s.t.copy(), quantity=s.quantity, r0=float(res.r0))


def _draw_cell(ax, c, clim):
    unit = QUNITS[c["quantity"]][0]
    r = c["r"] * 1e3; t = c["t"] * 1e3
    im = ax.imshow(c["data"] * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                   vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    ax.axvline(c["r0"] * 1e3, color="0.15", ls="--", lw=1.0, alpha=0.7)
    ax.tick_params(labelsize=7)
    return im


def _col_clim(cells):
    """Shared colour limit for one quantity column (over the rows that have it)."""
    cl = []
    for c in cells:
        if c is None:
            continue
        unit = QUNITS[c["quantity"]][0]
        rc = (c["r"] > 0.1 * c["r"][-1]) & (c["r"] < 0.9 * c["r"][-1])
        cl.append((robust_clim(c["data"], rc, pct=97) * unit) or
                  float(np.nanpercentile(np.abs(c["data"] * unit), 97)))
    return float(np.percentile(cl, 75)) if cl else 1.0


def fig_quantity_grid(rows, row_labels, title="", col_titles=None, scale=1.0, speed_line=None,
                      per_cell=False):
    """Grid of space-times: one ROW per entry in ``rows`` (each a dict quantity->cell), one COLUMN per
    quantity (displacement / velocity / acceleration). ``scale`` sizes the figure; every panel gets a
    colour bar. Colour scaling: by default shared per quantity column (rows directly comparable);
    ``per_cell=True`` gives every panel its own scale (so a big previous row can't squash the current).
    ``speed_line`` = ((r0_mm,t0_ms),(r1_mm,t1_ms)) overlays the manual speed line on the post-push rows."""
    nrow = len(rows)
    fig, axs = plt.subplots(nrow, 3, figsize=(2.9 * 3 * scale, 2.05 * nrow * scale), squeeze=False)
    for cj, q in enumerate(QORDER):
        shared = None if per_cell else _col_clim([row.get(q) for row in rows])
        for ri, row in enumerate(rows):
            ax = axs[ri][cj]
            c = row.get(q)
            if c is not None:
                clim = _col_clim([c]) if per_cell else shared
                im = _draw_cell(ax, c, clim)
                cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
                cb.ax.tick_params(labelsize=6)
                if speed_line is not None:
                    (x0, y0), (x1, y1) = speed_line
                    tmin, tmax = c["t"].min() * 1e3, c["t"].max() * 1e3   # panel time range [ms]
                    if not (max(y0, y1) < tmin or min(y0, y1) > tmax):    # skip pre-push (line is post-push t)
                        ax.plot([x0, x1], [y0, y1], "k-", lw=1.6, alpha=0.9)
            else:
                ax.axis("off")
            if ri == 0:
                ax.set_title((col_titles[cj] if col_titles else f"{q}  [{QUNITS[q][1]}]"), fontsize=10)
            if ri == nrow - 1:
                ax.set_xlabel("r along M-line [mm]", fontsize=8)
            if cj == 0:
                ax.set_ylabel(f"{row_labels[ri]}\nt [ms]", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95) if title else (0, 0, 1, 1))
    return fig


def fig_quantity_row(res_by_q, title="", scale=1.0):
    """Single row: displacement / velocity / acceleration space-times for the current run."""
    row = {q: cell_of(res_by_q[q]) for q in QORDER if q in res_by_q}
    return fig_quantity_grid([row], [""], title=title, scale=scale)


def fig_svd_spectrum(S, n_remove, n_high_remove, domain="IQ", scale=1.0):
    """Singular-value spectrum (log) for tuning the SVD clutter cutoffs. The removed low ranks (clutter)
    and high ranks (noise) are shaded; the kept middle band is the 'signal'. The steep initial drop is
    clutter→signal; the flat tail is signal→noise — set the cutoffs at those two knees."""
    S = np.asarray(S, float)
    N = len(S)
    idx = np.arange(1, N + 1)
    sn = S / (S[0] + 1e-30)
    lo, hi = n_remove + 1, N - n_high_remove          # 1-indexed kept range
    fig, ax = plt.subplots(figsize=(4.6 * scale, 2.7 * scale))
    ax.semilogy(idx, sn, "-o", ms=3, color="0.25", zorder=3)
    nlo, nhi = min(n_remove, N), min(n_high_remove, N)          # clamp to the axis on over-removal
    if nlo > 0:
        ax.axvspan(0.5, nlo + 0.5, color="tab:red", alpha=0.18)
        ax.text(nlo / 2 + 0.5, 0.9, "clutter", color="tab:red", fontsize=7, ha="center", va="top")
    if nhi > 0:
        ax.axvspan(N - nhi + 0.5, N + 0.5, color="0.5", alpha=0.22)
        ax.text(N - nhi / 2 + 0.5, 0.9, "noise", color="0.35", fontsize=7, ha="center", va="top")
    if hi >= lo:                                                # kept 'signal' band (empty if over-removed)
        ax.axvspan(nlo + 0.5, N - nhi + 0.5, color="tab:green", alpha=0.10)
        ax.text((lo + hi) / 2, sn[hi - 1], "signal", color="tab:green", fontsize=7, ha="center", va="bottom")
    ax.set_xlabel("singular-value index (rank)", fontsize=8)
    ax.set_ylabel("σ / σ₁  (log)", fontsize=8)
    kept = f"keep {lo}–{hi}" if hi >= lo else "keep NONE"
    ax.set_title(f"SVD spectrum · {domain} · N={N} · {kept}", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def fig_speed_plotly(cell, clim, line=None, scale=1.0):
    """Interactive Plotly space-time for 2-click manual speed picking (data in mm / ms). A faint
    selectable point grid overlays the heatmap so a click anywhere returns a nearby (r,t); ``line`` =
    ((r0_mm,t0_ms),(r1_mm,t1_ms)) draws the current measurement."""
    import numpy as np
    import plotly.graph_objects as go
    unit = QUNITS[cell["quantity"]][0]
    z = cell["data"] * unit
    r = cell["r"] * 1e3
    t = cell["t"] * 1e3
    fig = go.Figure(go.Heatmap(z=z, x=r, y=t, colorscale="RdBu", reversescale=True, zmid=0,
                               zmin=-clim, zmax=clim, showscale=False, hoverinfo="x+y"))
    rr = r[:: max(1, len(r) // 70)]
    tt = t[:: max(1, len(t) // 50)]
    gx, gy = np.meshgrid(rr, tt)
    fig.add_trace(go.Scatter(x=gx.ravel(), y=gy.ravel(), mode="markers",
                             marker=dict(size=7, color="rgba(60,60,60,0.05)"),
                             hoverinfo="x+y", showlegend=False, name="click"))
    if line is not None:
        (x0, y0), (x1, y1) = line
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines+markers",
                                 line=dict(color="black", width=3), marker=dict(size=9, color="black"),
                                 hoverinfo="skip", showlegend=False))
    fig.add_vline(x=cell["r0"] * 1e3, line=dict(color="gray", dash="dash", width=1))
    fig.update_xaxes(title_text="r along M-line [mm]")
    fig.update_yaxes(title_text="t [ms]", autorange="reversed")
    fig.update_layout(height=int(400 * scale + 40), margin=dict(l=48, r=10, t=8, b=42),
                      dragmode=False, clickmode="event+select")
    return fig


def fig_bmode_mline(img_u8, extent_mm, mline, push_xz=None, n_offsets=1, offset_step_m=0.0, scale=1.0):
    """B-mode with the resampled **spline** M-line, the parallel **offset lines** actually used for
    averaging, the clicked **anchor points**, and the push focus overlaid."""
    fig, ax = plt.subplots(figsize=(3.7 * scale, 4.0 * scale))
    ax.imshow(img_u8, cmap="gray", extent=extent_mm, aspect="auto")
    # offset lines used for averaging (faint), then the central spline (bright)
    lines = mline.offset_lines(n_offsets, offset_step_m)
    for ox, oz in lines:
        ax.plot(ox * 1e3, oz * 1e3, "-", color="deepskyblue", lw=0.6, alpha=0.5)
    ax.plot(mline.x * 1e3, mline.z * 1e3, "-", color="cyan", lw=2.0, label="M-line (spline)")
    if len(lines) > 1:
        ax.plot([], [], "-", color="deepskyblue", lw=0.8, alpha=0.7,
                label=f"{len(lines)} offset lines")
    if getattr(mline, "points", None) is not None:
        ax.plot(mline.points[:, 0] * 1e3, mline.points[:, 1] * 1e3, "o", color="yellow",
                ms=5, mec="k", label="anchors")
    if push_xz is not None and push_xz[0] is not None:
        ax.plot(push_xz[0] * 1e3, push_xz[1] * 1e3, "x", color="orange", ms=11, mew=2, label="push")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title("B-mode + M-line", fontsize=10)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def _to_rgb(fig):
    """Rasterise a Matplotlib figure to an (H, W, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return buf[..., :3].copy()


def frame_rgb(img_u8, extent_mm, ml, res, meas, met, push_xz=None, n_offsets=1, offset_step_m=0.0,
              clim=None):
    """One animation frame: B-mode + spline M-line + offsets (left) and the space-time plot (right)
    for one push, at a **shared** colour ``clim``. Returns an (H, W, 3) uint8 array."""
    st = res.st
    unit = 1e6 if st.quantity == "displacement" else 1e3
    if clim is None:
        rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        clim = (robust_clim(st.data, rc, pct=97) * unit) or float(np.nanpercentile(np.abs(st.data * unit), 97))
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.0))
    # B-mode + M-line
    axs[0].imshow(img_u8, cmap="gray", extent=extent_mm, aspect="auto")
    for ox, oz in ml.offset_lines(n_offsets, offset_step_m):
        axs[0].plot(ox * 1e3, oz * 1e3, "-", color="deepskyblue", lw=0.6, alpha=0.5)
    axs[0].plot(ml.x * 1e3, ml.z * 1e3, "-", color="cyan", lw=2.0)
    if getattr(ml, "points", None) is not None:
        axs[0].plot(ml.points[:, 0] * 1e3, ml.points[:, 1] * 1e3, "o", color="yellow", ms=5, mec="k")
    if push_xz is not None and push_xz[0] is not None:
        axs[0].plot(push_xz[0] * 1e3, push_xz[1] * 1e3, "x", color="orange", ms=11, mew=2)
    axs[0].set_title(f"meas {meas} · B-mode + M-line", fontsize=10)
    axs[0].set_xlabel("x [mm]"); axs[0].set_ylabel("z [mm]")
    # space-time
    r = st.r * 1e3; t = st.t * 1e3
    axs[1].imshow(st.data * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                  vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    axs[1].axvline(res.r0 * 1e3, color="0.15", ls="--", lw=1.0, alpha=0.7)
    axs[1].set_title(f"space-time · oc={met['origin_coherence']:.2f} · "
                     f"amp={met['amp_p95']:.0f}{met['amp_unit']}", fontsize=10)
    axs[1].set_xlabel("r [mm]"); axs[1].set_ylabel("t [ms]")
    fig.tight_layout()
    return _to_rgb(fig)


def fig_history_strip(entries):
    """Space-time comparison strip of recent runs (newest first). Each entry is a dict with keys
    ``data`` (nt,nr), ``r``, ``t``, ``quantity``, ``r0``, ``label``, ``S`` (or nan), ``oc``. All
    panels share one colour scale so the effect of a change is directly comparable."""
    n = len(entries)
    if n == 0:
        return None
    unit = 1e6 if entries[0]["quantity"] == "displacement" else 1e3
    clims = []
    for e in entries:
        r = e["r"]; rc = (r > 0.1 * r[-1]) & (r < 0.9 * r[-1])
        clims.append((robust_clim(e["data"], rc, pct=97) * unit) or
                     float(np.nanpercentile(np.abs(e["data"] * unit), 97)))
    clim = float(np.percentile(clims, 75))
    fig, axs = plt.subplots(1, n, figsize=(4.4 * n, 4.2), squeeze=False)
    for j, e in enumerate(entries):
        ax = axs[0][j]
        r = e["r"] * 1e3; t = e["t"] * 1e3
        ax.imshow(e["data"] * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                  vmin=-clim, vmax=clim, aspect="auto", origin="upper")
        ax.axvline(e["r0"] * 1e3, color="0.15", ls="--", lw=1.0, alpha=0.7)
        tag = "current" if j == 0 else f"−{j}"
        lbl = e["label"]
        lbl = lbl if len(lbl) <= 46 else lbl[:44] + "…"
        ax.set_title(f"[{tag}] oc={e['oc']:.2f}\n{lbl}", fontsize=7)
        ax.set_xlabel("r [mm]", fontsize=8)
        if j == 0:
            ax.set_ylabel("t [ms]", fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("Comparison — current vs previous runs (shared colour scale)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def build_gif(frames, fps=2):
    """Assemble RGB frames into a looping GIF (bytes)."""
    import io
    import imageio.v2 as imageio
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF", duration=1.0 / max(fps, 0.1), loop=0)
    return buf.getvalue()


def fig_push_vs_nopush(res_push, res_nopush, m_push, m_nopush):
    """Side-by-side space-time of the recipe on the tracking (push) vs pre-push reference window,
    same colour scale, so you can see if the 'wave' is really the push or cardiac motion."""
    st = res_push.st
    unit = 1e6 if st.quantity == "displacement" else 1e3
    rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
    clim = (robust_clim(st.data, rc, pct=97) * unit) or float(np.nanpercentile(np.abs(st.data * unit), 97))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, res, lbl, mt in ((axs[0], res_push, "PUSH (tracking)", m_push),
                             (axs[1], res_nopush, "NO-PUSH (reference)", m_nopush)):
        s = res.st
        r = s.r * 1e3; t = s.t * 1e3
        ax.imshow(s.data * unit, extent=(r[0], r[-1], t[-1], t[0]), cmap="RdBu_r",
                  vmin=-clim, vmax=clim, aspect="auto", origin="upper")
        ax.axvline(res.r0 * 1e3, color="0.15", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(f"{lbl}\noc={mt['origin_coherence']:.2f}  amp95={mt['amp_p95']:.1f}{mt['amp_unit']}",
                     fontsize=9)
        ax.set_xlabel("r [mm]")
    axs[0].set_ylabel("t [ms]")
    fig.suptitle("Same recipe: push window vs no-push reference window (shared colour scale)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
