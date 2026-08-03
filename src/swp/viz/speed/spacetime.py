"""Assemble the space-time plot (position-along-M-line vs slow time)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mline.mline import MLine, sample_along_mline


@dataclass
class SpaceTime:
    """Wavefield sampled along the M-line: ``data`` is (n_time, n_r)."""
    data: np.ndarray       # (n_time, n_samples) velocity [m/s] or displacement [m]
    r: np.ndarray          # (n_samples,) arc length [m]
    t: np.ndarray          # (n_time,) time [s]
    quantity: str = "velocity"

    def extent_ms_mm(self):
        """(rmin, rmax, tmax, tmin) in (mm, ms) for imshow(origin='upper')."""
        return (self.r[0] * 1e3, self.r[-1] * 1e3,
                self.t[-1] * 1e3, self.t[0] * 1e3)


def build_spacetime(
    field: np.ndarray,
    z_axis: np.ndarray,
    x_axis: np.ndarray,
    mline: MLine,
    t: np.ndarray,
    quantity: str = "velocity",
    n_offsets: int = 1,
    offset_step_m: float = 0.0,
) -> SpaceTime:
    """Build a :class:`SpaceTime` by sampling ``field`` (n_frames, nz, nx) along ``mline``.

    ``t`` must have length equal to ``field.shape[0]`` (e.g. interval-centre times for
    a velocity field, or frame times for a displacement field).
    """
    data = sample_along_mline(field, z_axis, x_axis, mline,
                              n_offsets=n_offsets, offset_step_m=offset_step_m)
    t = np.asarray(t, dtype=np.float64)
    if t.size != data.shape[0]:
        raise ValueError(f"t length {t.size} != field frames {data.shape[0]}")
    return SpaceTime(data=data, r=mline.r.copy(), t=t, quantity=quantity)
