"""Matplotlib figure builders for the GUI (return Figure objects; the app calls st.pyplot)."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swp.viz.core.geometry import robust_clim


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


def fig_bmode_mline(img_u8, extent_mm, mline, push_xz=None, n_offsets=1, offset_step_m=0.0):
    """B-mode with the resampled **spline** M-line, the parallel **offset lines** actually used for
    averaging, the clicked **anchor points**, and the push focus overlaid."""
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
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
