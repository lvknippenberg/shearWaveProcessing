"""Load zea-format ARF-SWE HDF5 files and manual M-line .npz files."""
from __future__ import annotations

import os
from typing import Optional

import h5py
import numpy as np

from ..core.acquisition import Acquisition

_BF = "tracks/track_0/data/beamformed_data"


def _scalar(f: h5py.File, key: str, default: Optional[float] = None) -> Optional[float]:
    if key in f:
        return float(np.array(f[key]).ravel()[0])
    return default


def _iq(values: np.ndarray) -> np.ndarray:
    """(..., 2) real [I, Q] -> complex."""
    return values[..., 0].astype(np.float64) + 1j * values[..., 1].astype(np.float64)


def load_acquisition(path: str) -> Acquisition:
    """Load one measurement HDF5 into an :class:`Acquisition`.

    Handles both normal-grid and fine-grid files and phantom vs in-vivo
    (in-vivo carries ``custom/t_reference``).
    """
    with h5py.File(path, "r") as f:
        values = np.array(f[f"{_BF}/values"])          # (n_frames, nz, nx, 2)
        coords = np.array(f[f"{_BF}/coordinates"])     # (nz, nx, 3): (x, y, z)
        t = np.array(f[f"{_BF}/timestamps"]).astype(np.float64)

        # Pre-push reference is present for active SWE (buffer 2) but ABSENT for passive
        # SWE (buffer 4, continuous natural waves). Tolerate its absence so the passive
        # path can share this loader; passive uses frame_to_frame displacement.
        ref_values = None
        if "custom/reference_iq" in f:
            ref_values = np.array(f["custom/reference_iq"])  # (n_ref, nz, nx, 2)
        t_ref = None
        if "custom/t_reference" in f:
            t_ref = np.array(f["custom/t_reference"]).astype(np.float64)

        prf = _scalar(f, "custom/prf")
        f_demod = _scalar(f, "custom/demodulation_frequency")
        # normal grid: transmit_frequency; fine grid: center_frequency
        f0 = _scalar(f, "custom/transmit_frequency") or _scalar(f, "custom/center_frequency")
        c = _scalar(f, "custom/sound_speed", 1540.0)
        dz = _scalar(f, "custom/dz")
        dx = _scalar(f, "custom/dx")
        # ARF push focal point (metres), written by the acquisition stage so the wave
        # origin r0 can be anchored from the data instead of a hard-coded assumption.
        push_x = _scalar(f, "custom/push_focus_x")
        push_z = _scalar(f, "custom/push_focus_z")

    iq = _iq(values)          # (n_frames, nz, nx) complex
    ref_iq = _iq(ref_values) if ref_values is not None else None  # (n_ref, nz, nx) complex

    # 1-D axes from the per-pixel Cartesian coordinates. Channel order (x, y, z).
    x = coords[0, :, 0].astype(np.float64)   # varies along lateral index
    z = coords[:, 0, 2].astype(np.float64)   # varies along axial index

    # Guarantee monotonically-increasing axes; flip data if the grid is stored reversed.
    if x[0] > x[-1]:
        x = x[::-1].copy()
        iq = iq[:, :, ::-1]
        if ref_iq is not None:
            ref_iq = ref_iq[:, :, ::-1]
        coords = coords[:, ::-1, :]
    if z[0] > z[-1]:
        z = z[::-1].copy()
        iq = iq[:, ::-1, :]
        if ref_iq is not None:
            ref_iq = ref_iq[:, ::-1, :]
        coords = coords[::-1, :, :]

    name = os.path.basename(path).lower()
    grid = "fine" if "fine" in name else "normal"
    # passive (buffer 4) has no pre-push reference; active in-vivo carries t_reference.
    if ref_iq is None:
        source = "passive"
    elif "phantom" in path.lower() or t_ref is None:
        source = "phantom"
    else:
        source = "invivo"

    return Acquisition(
        iq=iq, ref_iq=ref_iq, x=x, z=z, t=t,
        prf=prf, f_demod=f_demod, f0=f0, c=c, dz=dz, dx=dx,
        grid=grid, source=source, t_ref=t_ref, coords=coords,
        push_x=push_x, push_z=push_z,
        meta={"path": path},
    )


def load_mline(path: str):
    """Load a manual M-line ``.npz`` (anchor points + sample count).

    Returns ``(points, n_samples)`` where ``points`` is (k, 2) as (x, z) in metres.
    """
    d = np.load(path)
    points = np.asarray(d["points"], dtype=np.float64)  # (k, 2) = (x, z)
    n_samples = int(d["n_samples"]) if "n_samples" in d else 250
    return points, n_samples
