"""Montage of space-time panels for comparing method/filter combinations."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.geometry import robust_clim
from ..speed.spacetime import SpaceTime
from ..speed.tof import SpeedResult


def draw_spacetime_panel(ax, st: SpaceTime, speed: Optional[SpeedResult] = None,
                         r0_mm: Optional[float] = None, title: str = "",
                         clim: Optional[float] = None, transpose: bool = False):
    """Draw one space-time panel. ``transpose=True`` uses the M-mode orientation (x = time,
    y = along-line position, r=0 at top) — the convention in cardiac natural-SWE papers, in which a
    propagating wave reads as a clear diagonal; default (False) keeps r on x, t on y (active side)."""
    unit = 1e3 if st.quantity == "velocity" else 1e6
    img = st.data * unit
    if clim is None:
        rc = (st.r > 0.1 * st.r[-1]) & (st.r < 0.9 * st.r[-1])
        clim = robust_clim(st.data, rc, pct=97) * unit
    tmin, tmax = st.t[0] * 1e3, st.t[-1] * 1e3
    if transpose:
        ax.imshow(img.T, extent=[tmin, tmax, st.r[-1] * 1e3, st.r[0] * 1e3], cmap="RdBu_r",
                  vmin=-clim, vmax=clim, aspect="auto", origin="upper")
        if speed is not None:
            for tp in (speed.t_pred_pos, speed.t_pred_neg):
                m = np.isfinite(tp) & (tp * 1e3 >= tmin) & (tp * 1e3 <= tmax)
                if m.sum() > 2:
                    ax.plot(tp[m] * 1e3, st.r[m] * 1e3, "k", lw=1.3, alpha=0.85)
        if r0_mm is not None:
            ax.axhline(r0_mm, color="0.2", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlim(tmin, tmax); ax.set_ylim(st.r[-1] * 1e3, st.r[0] * 1e3)
    else:
        ax.imshow(img, extent=st.extent_ms_mm(), cmap="RdBu_r", vmin=-clim, vmax=clim,
                  aspect="auto", origin="upper")
        if speed is not None:
            for tp in (speed.t_pred_pos, speed.t_pred_neg):
                m = np.isfinite(tp) & (tp * 1e3 >= tmin) & (tp * 1e3 <= tmax)
                if m.sum() > 2:
                    ax.plot(st.r[m] * 1e3, tp[m] * 1e3, "k", lw=1.3, alpha=0.85)
        if r0_mm is not None:
            ax.axvline(r0_mm, color="0.2", ls="--", lw=0.8, alpha=0.6)
        ax.set_ylim(tmax, tmin)                 # keep axis to the data's time span
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7)


def draw_bmode_mline_panel(ax, row):
    """B-mode frame with the M-line used for that row of a montage.

    ``row``: dict with ``img`` (z, x) uint8, ``extent`` [x0, x1, z1, z0] in mm, ``x``/``z`` line
    samples in mm, optional ``r0_mm`` (marker at that arc length), ``title`` and ``margin_mm``
    (zoom around the line; default 25, None = whole frame). r = 0 is marked with a yellow dot.
    """
    ax.imshow(row["img"], cmap="gray", extent=row["extent"], aspect="equal", vmin=0, vmax=255)
    x, z = np.asarray(row["x"]), np.asarray(row["z"])
    ax.plot(x, z, "-", color="cyan", lw=1.6)
    ax.plot(x[0], z[0], "o", color="yellow", ms=5, mec="k", mew=0.5)
    if row.get("r0_mm") is not None and len(x) > 1:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(z)))])
        k = int(np.argmin(np.abs(s - row["r0_mm"])))
        ax.plot(x[k], z[k], "+", color="0.9", ms=9, mew=1.5)
    margin = row.get("margin_mm", 25.0)
    if margin is not None:
        ax.set_xlim(x.min() - margin, x.max() + margin)
        ax.set_ylim(z.max() + margin, z.min() - margin)
    ax.set_title(row.get("title", ""), fontsize=8)
    ax.set_xlabel("x [mm]", fontsize=7); ax.set_ylabel("z [mm]", fontsize=7)
    ax.tick_params(labelsize=7)


def spacetime_montage(results, out_path: str, ncols: int = 4,
                      suptitle: str = "", panel_titles: Optional[List[str]] = None,
                      transpose: bool = False, row_bmodes: Optional[list] = None):
    """Grid of space-time panels from a list of PipelineResult-like objects.

    Each item must expose ``.st`` (SpaceTime), ``.speed`` (SpeedResult), ``.r0`` (m),
    and ``.config.label()``. ``transpose=True`` -> M-mode orientation (x=time, y=position).
    ``row_bmodes`` (one dict per row, see :func:`draw_bmode_mline_panel`; None entries allowed)
    adds a leading column with the B-mode frame and the M-line of that row.
    """
    n = len(results)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    lead = 1 if row_bmodes else 0
    fig, axs = plt.subplots(nrows, ncols + lead, figsize=(3.4 * (ncols + lead), 3.1 * nrows),
                            squeeze=False)
    if lead:
        for k in range(nrows):
            row = row_bmodes[k] if k < len(row_bmodes) else None
            if row is None:
                axs[k][0].axis("off")
            else:
                draw_bmode_mline_panel(axs[k][0], row)
        axs = [r[1:] for r in axs]
    xlab, ylab = ("t [ms]", "r [mm]") if transpose else ("r [mm]", "t [ms]")
    for i, r in enumerate(results):
        ax = axs[i // ncols][i % ncols]
        title = panel_titles[i] if panel_titles else r.config.label()
        title = f"{title}\n{r.speed.label()}"
        draw_spacetime_panel(ax, r.st, r.speed, r0_mm=r.r0 * 1e3, title=title, transpose=transpose)
        if i % ncols == 0:
            ax.set_ylabel(ylab, fontsize=7)
        if i // ncols == nrows - 1:
            ax.set_xlabel(xlab, fontsize=7)
    for j in range(n, nrows * ncols):
        axs[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98 if suptitle else 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
