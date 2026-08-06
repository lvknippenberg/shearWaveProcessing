"""Fine-grid local re-beamforming of the active shear-wave RF around the M-line.

The stored per-measurement IQ is beamformed on the coarse PData grid (~lambda/2 pixel pitch, 2
pixels/wavelength) - fine enough for phase-based Loupas/Kasai, but too coarse for **RF cross-
correlation**, whose correlation peak needs many samples per wavelength to localize. This module
re-beamforms buffer 2 for one measurement onto a **fine axial grid over a tight ROI** (the M-line
bounding box + margin), returning an :class:`Acquisition` carrying both the fine complex IQ and a
reconstructed real **RF** volume (carrier re-inserted from the baseband IQ), so any estimator -
Loupas/Kasai/xcorr on the fine IQ, or ``rf_ncc`` on the RF - can run on it.

It reuses the Stage-A beamformer (:mod:`swp.acquisition.beamform`) and frame assembly
(:mod:`swp.acquisition.sequence`); only the grid (ROI + pixels-per-wavelength) differs. Heavy
(a GPU beamform of ~100 frames), so the GUI caches the result per (folder, meas, ROI, density).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

from ..viz.core.acquisition import Acquisition
from .beamform import find_mat, read_buffer, beamform_frames
from .sequence import (SPEC_BY_INDEX, read_swi_meta, assemble_tracking_frames, BufferGrid)


def _roi_from_mline(points_xz, base_grid: BufferGrid, margin_m=3e-3):
    """Bounding box (xlims, zlims) around the M-line points + margin, clipped to the base grid."""
    p = np.asarray(points_xz, float)
    xlims = (p[:, 0].min() - margin_m, p[:, 0].max() + margin_m)
    zlims = (p[:, 1].min() - margin_m, p[:, 1].max() + margin_m)
    xlims = (max(xlims[0], base_grid.xlims[0]), min(xlims[1], base_grid.xlims[1]))
    zlims = (max(zlims[0], base_grid.zlims[0]), min(zlims[1], base_grid.zlims[1]))
    return xlims, zlims


def _apply_fine_grid(params, xlims, zlims, ppw):
    """Fix a fine cartesian beamforming grid on a Parameters object (like beamform.apply_grid
    but with a caller-chosen pixels-per-wavelength for a fine axial pitch)."""
    params.n_ch = 1
    params.grid_type = "cartesian"
    params.pixels_per_wavelength = float(ppw)
    params.xlims = tuple(float(v) for v in xlims)
    params.zlims = tuple(float(v) for v in zlims)
    return params


def _reconstruct_rf(iq, z_axis, f_demod, c):
    """Real RF from baseband IQ by re-inserting the carrier: rf = Re{iq * exp(j 2pi f tau)},
    tau = 2 z / c (fast-time at depth z). Gives a fine-grid RF line for RF cross-correlation."""
    phase = 2.0 * np.pi * f_demod * (2.0 * np.asarray(z_axis) / c)   # (nz,)
    carrier = np.exp(1j * phase)[None, :, None]                      # (1, nz, 1)
    return (np.asarray(iq) * carrier).real


def finegrid_acquisition(folder, meas: int, points_xz=None, ppw: float = 8.0,
                         margin_m: float = 3e-3, roi=None, pi_mode: str = "sliding",
                         verbose: bool = True) -> Acquisition:
    """Re-beamform buffer 2 for one measurement on a fine ROI grid.

    Args:
        folder: measurement folder (containing the Verasonics ``.mat`` / converted RF).
        meas: measurement index (0-based).
        points_xz: M-line anchor points (k, 2) = (x, z) metres, used to size the ROI. If None,
            ``roi`` or the SW push-focus ROI is used.
        ppw: pixels per wavelength of the fine grid (axial density; 8-16 is typical for RF NCC).
        margin_m: ROI margin around the M-line bounding box.
        roi: explicit ((xmin,xmax),(zmin,zmax)) metres, overriding ``points_xz``.
        pi_mode: pulse-inversion recombination (matches Stage A).

    Returns:
        An :class:`Acquisition` (grid="fine") whose ``iq``/``ref_iq`` are the fine complex IQ, with
        the reconstructed real RF volumes in ``meta['rf']`` / ``meta['ref_rf']`` (see :func:`as_rf`).
    """
    from zea import init_device
    init_device(verbose=False)

    folder = Path(folder)
    mat_path = find_mat(folder)
    meta = read_swi_meta(mat_path)
    spec = SPEC_BY_INDEX[1]                                  # buffer 2 (active_sw)
    base_grid = meta.grids[1]
    sw = meta.sw

    raw, params = _read_active_buffer(mat_path, folder, spec.index)

    tf = assemble_tracking_frames(raw, n_reference=sw.n_reference, n_tracking=sw.n_tracking,
                                  na=sw.na, harmonic=sw.harmonic, pri=sw.pri, pi_mode=pi_mode)
    if not 0 <= meas < tf.n_meas:
        raise IndexError(f"meas {meas} out of range (0..{tf.n_meas - 1})")

    if roi is not None:
        xlims, zlims = roi
    elif points_xz is not None:
        xlims, zlims = _roi_from_mline(points_xz, base_grid, margin_m)
    else:
        xlims, zlims = sw.roi_xlims, sw.roi_zlims
    _apply_fine_grid(params, xlims, zlims, ppw)
    params.set_transmits([0])
    coords = np.asarray(params.grid, dtype=np.float32)       # (nz, nx, 3)

    def _bf(block):                                          # (n_frames, n_ax, n_el, 1) -> (n_fr, z, x) complex
        flat = block[:, None]                                # add tx axis
        iq, _ = beamform_frames(flat, params)                # (n_fr, z, x, 2)
        return iq[..., 0] + 1j * iq[..., 1]

    if verbose:
        print(f"  [finegrid] meas{meas} ROI x[{xlims[0]*1e3:.1f},{xlims[1]*1e3:.1f}] "
              f"z[{zlims[0]*1e3:.1f},{zlims[1]*1e3:.1f}]mm ppw={ppw} grid {coords.shape[:2]}")
    trk_iq = _bf(tf.tracking[meas])
    ref_iq = _bf(tf.reference[meas])

    x = coords[0, :, 0].astype(np.float64)
    z = coords[:, 0, 2].astype(np.float64)
    if x[0] > x[-1]:
        x = x[::-1].copy(); trk_iq = trk_iq[:, :, ::-1]; ref_iq = ref_iq[:, :, ::-1]; coords = coords[:, ::-1]
    if z[0] > z[-1]:
        z = z[::-1].copy(); trk_iq = trk_iq[:, ::-1]; ref_iq = ref_iq[:, ::-1]; coords = coords[::-1]

    f0 = float(np.asarray(params.center_frequency).ravel()[0])
    c = float(meta.sound_speed)
    dz = float(np.median(np.diff(z))); dx = float(np.median(np.diff(x)))
    rf = _reconstruct_rf(trk_iq, z, f0, c)
    ref_rf = _reconstruct_rf(ref_iq, z, f0, c)

    return Acquisition(
        iq=trk_iq, ref_iq=ref_iq, x=x, z=z, t=tf.t_tracking.astype(np.float64),
        prf=float(tf.prf), f_demod=f0, f0=f0, c=c, dz=dz, dx=dx, grid="fine",
        source="invivo" if tf.reference.shape[1] and meta.sw else "phantom",
        t_ref=tf.t_reference.astype(np.float64), coords=coords,
        push_x=float(sw.focus_x), push_z=float(sw.focus_z),
        meta={"rf": rf, "ref_rf": ref_rf, "ppw": ppw, "folder": str(folder), "meas": meas},
    )


def as_rf(acq: Acquisition) -> Acquisition:
    """A view of a fine-grid acquisition whose ``iq``/``ref_iq`` are the real RF volumes (for
    the ``rf_ncc`` estimator). Requires ``meta['rf']`` (produced by :func:`finegrid_acquisition`)."""
    import dataclasses
    if "rf" not in acq.meta:
        raise ValueError("as_rf needs a fine-grid acquisition from finegrid_acquisition()")
    return dataclasses.replace(acq, iq=acq.meta["rf"], ref_iq=acq.meta["ref_rf"])


# --- read the active buffer, reusing the converted RF file if it exists ---
def _read_active_buffer(mat_path, folder, buffer_index):
    """(raw, params) for the active buffer: from the converted zea file if present (cheap
    read-back, no ``.mat`` decode), else straight from the Verasonics workspace."""
    converted = Path(folder) / "output" / "converted" / f"{Path(mat_path).stem}_buffer2.hdf5"
    if converted.is_file():
        return read_buffer(None, buffer_index, converted_path=converted)
    from zea.data.convert.verasonics import VerasonicsFile
    with VerasonicsFile(str(mat_path)) as vf:
        return read_buffer(vf, buffer_index, converted_path=None)
