"""Metadata passed to field filters so the registry can call them uniformly."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class FilterCtx:
    dz: float
    dx: float
    prf: float
    t: np.ndarray                  # (n_frames,) times of the field being filtered [s]
    x: np.ndarray                  # (nx,) lateral axis [m]
    z: np.ndarray                  # (nz,) axial axis [m]
    focus_ix: Optional[int] = None  # lateral index of push focus (for directional)
    focus_x: Optional[float] = None
    # pre-push reference trajectory, for reference-frame motion compensation:
    ref_disp: Optional[np.ndarray] = None  # (n_ref, nz, nx) displacement of each ref frame [m]
    t_ref: Optional[np.ndarray] = None     # (n_ref,) reference times [s], rel. to tracking frame 0
