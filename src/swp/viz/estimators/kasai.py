"""Kasai 1-D autocorrelator: phase shift at the fixed demodulation frequency.

This is the Loupas estimator with the local center-frequency correction disabled
(``local_frequency=False``), i.e. a constant carrier ``f_demod``.  Kept as a fast sanity
baseline (see docs/literature_review.md sec. 2.1).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .loupas import loupas_displacement


def kasai_displacement(iq: np.ndarray, dz: float, dx: float, c: float, f_demod: float,
                       prf: float, kernel_z_m: float = 1.0e-3, kernel_x_m: float = 0.0,
                       mode: str = "frame_to_frame", reference: Optional[np.ndarray] = None):
    res = loupas_displacement(iq, dz=dz, dx=dx, c=c, f_demod=f_demod, prf=prf,
                              kernel_z_m=kernel_z_m, kernel_x_m=kernel_x_m,
                              local_frequency=False, mode=mode, reference=reference)
    res.estimator = "kasai"
    return res
