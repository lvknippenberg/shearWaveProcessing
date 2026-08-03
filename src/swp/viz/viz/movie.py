"""Propagation movies of the wavefield."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from ..core.acquisition import Acquisition
from ..mline.mline import MLine


def save_propagation_movie(
    acq: Acquisition,
    field: np.ndarray,
    out_path: str,
    quantity: str = "velocity",
    unit_scale: float = 1e3,
    unit_label: str = "mm/s",
    clim: Optional[float] = None,
    fps: int = 12,
    mline: Optional[MLine] = None,
    bmode_underlay: bool = True,
):
    """Animate a (n_frames, nz, nx) field and save as an animated GIF.

    Uses PillowWriter so no external ffmpeg dependency is required.
    """
    extent = acq.extent_mm()
    n = field.shape[0]
    if clim is None:
        clim = np.percentile(np.abs(field * unit_scale), 99) + 1e-9

    fig, ax = plt.subplots(figsize=(5, 5.5))
    if bmode_underlay:
        ax.imshow(acq.bmode(0, db=True), extent=extent, cmap="gray",
                  vmin=-40, vmax=0, aspect="equal")
    im = ax.imshow(field[0] * unit_scale, extent=extent, cmap="RdBu_r",
                   vmin=-clim, vmax=clim, alpha=0.8, aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"axial {quantity} [{unit_label}]")
    if mline is not None:
        ax.plot(mline.x * 1e3, mline.z * 1e3, "-", color="lime", lw=1.5)
    ax.set_xlabel("lateral x [mm]")
    ax.set_ylabel("depth z [mm]")
    ttl = ax.set_title("")

    def update(k):
        im.set_data(field[k] * unit_scale)
        tk = acq.t[min(k, acq.t.size - 1)] * 1e3
        ttl.set_text(f"{acq.source}/{acq.grid}  frame {k}  t={tk:.2f} ms")
        return im, ttl

    anim = FuncAnimation(fig, update, frames=n, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path
