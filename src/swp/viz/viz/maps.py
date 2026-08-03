"""Displacement / velocity map figures with a B-mode underlay."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..core.acquisition import Acquisition
from ..mline.mline import MLine


def plot_frame_map(
    acq: Acquisition,
    field: np.ndarray,
    frame: int,
    out_path: str,
    quantity: str = "velocity",
    unit_scale: float = 1e3,
    unit_label: str = "mm/s",
    clim: Optional[float] = None,
    mline: Optional[MLine] = None,
    bmode_ref_frame: int = 0,
):
    """Render one frame of a (n_frames, nz, nx) field over the B-mode envelope."""
    extent = acq.extent_mm()
    fig, ax = plt.subplots(figsize=(5, 5.5))

    ax.imshow(acq.bmode(bmode_ref_frame, db=True), extent=extent, cmap="gray",
              vmin=-40, vmax=0, aspect="equal")

    img = field[frame] * unit_scale
    if clim is None:
        clim = np.percentile(np.abs(field * unit_scale), 99) + 1e-9
    m = ax.imshow(img, extent=extent, cmap="RdBu_r", vmin=-clim, vmax=clim,
                  alpha=0.75, aspect="equal")
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"axial {quantity} [{unit_label}]")

    if mline is not None:
        ax.plot(mline.x * 1e3, mline.z * 1e3, "-", color="lime", lw=1.5)

    t_ms = acq.t[min(frame, acq.t.size - 1)] * 1e3
    ax.set_title(f"{acq.source}/{acq.grid}  frame {frame}  t={t_ms:.2f} ms")
    ax.set_xlabel("lateral x [mm]")
    ax.set_ylabel("depth z [mm]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path
