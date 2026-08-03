"""Space-time plot figures."""
from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..speed.spacetime import SpaceTime


def plot_spacetime(
    st: SpaceTime,
    out_path: str,
    title: str = "",
    clim: Optional[float] = None,
    r0_mm: Optional[float] = None,
):
    """Render a space-time plot (arc length vs time).

    A tilted wavefront ridge has slope dr/dt = shear-wave speed.  ``r0_mm`` optionally
    marks the push origin along the M-line.
    """
    data = st.data
    unit = 1e3 if st.quantity == "velocity" else 1e6  # mm/s or um
    ulabel = "mm/s" if st.quantity == "velocity" else "um"
    img = data * unit
    if clim is None:
        clim = np.percentile(np.abs(img), 99) + 1e-9

    fig, ax = plt.subplots(figsize=(6, 5))
    m = ax.imshow(img, extent=st.extent_ms_mm(), cmap="RdBu_r",
                  vmin=-clim, vmax=clim, aspect="auto", origin="upper")
    cb = fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(f"axial {st.quantity} [{ulabel}]")
    if r0_mm is not None:
        ax.axvline(r0_mm, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("position along M-line r [mm]")
    ax.set_ylabel("time t [ms]")
    ax.set_title(title or "space-time plot")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
