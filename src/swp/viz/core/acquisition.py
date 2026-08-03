"""Container for a single shear-wave measurement (one push + tracking ensemble)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Acquisition:
    """One ARF-SWE measurement in the zea HDF5 layout.

    Arrays follow (frame, z, x) ordering; ``iq`` and ``ref_iq`` are complex (I + jQ).
    Axes ``x`` and ``z`` are 1-D, monotonically increasing, in metres.
    """

    iq: np.ndarray            # (n_frames, nz, nx) complex, post-push tracking
    ref_iq: Optional[np.ndarray]  # (n_ref, nz, nx) complex pre-push ref; None for passive SWE
    x: np.ndarray             # (nx,) lateral axis [m]
    z: np.ndarray             # (nz,) axial (depth) axis [m]
    t: np.ndarray             # (n_frames,) tracking timestamps [s]
    prf: float                # tracking frame rate [Hz]
    f_demod: float            # IQ demodulation freq [Hz] -> displacement scaling
    f0: float                 # transmit/center frequency [Hz]
    c: float                  # speed of sound [m/s]
    dz: float                 # axial pixel spacing [m]
    dx: float                 # lateral pixel spacing [m]
    grid: str = "unknown"     # "normal" | "fine"
    source: str = "unknown"   # "phantom" | "invivo"
    t_ref: Optional[np.ndarray] = None   # (n_ref,) reference times [s] (in-vivo)
    coords: Optional[np.ndarray] = None  # (nz, nx, 3) Cartesian (x,y,z) [m]
    push_x: Optional[float] = None       # ARF push focal point, lateral x [m] (from HDF5)
    push_z: Optional[float] = None       # ARF push focal point, axial z [m] (from HDF5)
    meta: dict = field(default_factory=dict)

    # --- shapes -----------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.iq.shape[0]

    @property
    def nz(self) -> int:
        return self.iq.shape[1]

    @property
    def nx(self) -> int:
        return self.iq.shape[2]

    @property
    def dt(self) -> float:
        """Slow-time frame interval [s]."""
        return 1.0 / self.prf

    # --- convenience ------------------------------------------------------
    def bmode(self, frame: int = 0, ref: bool = False, db: bool = True) -> np.ndarray:
        """Envelope (|IQ|) of a frame, optionally in dB (for background display)."""
        data = self.ref_iq if ref else self.iq
        env = np.abs(data[frame])
        if db:
            env = 20.0 * np.log10(env / (env.max() + 1e-12) + 1e-6)
        return env

    def extent_mm(self):
        """(xmin, xmax, zmax, zmin) in mm for imshow with origin='upper'."""
        return (
            self.x[0] * 1e3, self.x[-1] * 1e3,
            self.z[-1] * 1e3, self.z[0] * 1e3,
        )

    def summary(self) -> str:
        ref = "none" if self.ref_iq is None else str(self.ref_iq.shape)
        return (
            f"Acquisition[{self.source}/{self.grid}] "
            f"iq{self.iq.shape} ref{ref} "
            f"x[{self.x[0]*1e3:.1f},{self.x[-1]*1e3:.1f}]mm "
            f"z[{self.z[0]*1e3:.1f},{self.z[-1]*1e3:.1f}]mm "
            f"prf={self.prf:.0f}Hz f_demod={self.f_demod/1e6:.3f}MHz "
            f"dz={self.dz*1e6:.0f}um dx={self.dx*1e6:.0f}um"
        )
