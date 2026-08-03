"""M3 clutter suppression in IQ space (SVD / eigen-based).

Removes the dominant low-rank slow-time subspace (bulk tissue / clutter) before displacement
estimation, as in ultrafast-Doppler clutter filtering.  See docs/literature_review.md sec. 3.4.
"""
from __future__ import annotations

import numpy as np


def svd_clutter(iq: np.ndarray, n_remove: int = 1, n_high_remove: int = 0) -> np.ndarray:
    """Zero the first ``n_remove`` (clutter) and last ``n_high_remove`` (noise) singular
    components of the slow-time IQ ensemble.

    iq : (n_frames, nz, nx) complex.  Returns the filtered ensemble (same shape/dtype).
    """
    n_frames, nz, nx = iq.shape
    casorati = iq.reshape(n_frames, nz * nx)           # (frames, space)
    U, S, Vh = np.linalg.svd(casorati, full_matrices=False)
    keep = np.ones_like(S)
    if n_remove > 0:
        keep[:n_remove] = 0.0
    if n_high_remove > 0:
        keep[-n_high_remove:] = 0.0
    filtered = (U * (S * keep)) @ Vh
    return filtered.reshape(n_frames, nz, nx)
