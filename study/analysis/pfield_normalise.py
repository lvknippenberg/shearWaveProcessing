"""Divide the focused B-mode by its compounded transmit sensitivity (pfield normalisation).

The phantom showed the line-spacing ripple largely SURVIVES envelope summing (only 3.1x down),
which points at a multiplicative transmit-field dip - amplitude scalloping - rather than
interference between transmits. If that is right, dividing the image by the compounded transmit
sensitivity should flatten the ripple, and unlike REFoCUS it should NOT cost resolution, because
it is a smooth per-pixel scalar.

This is NOT what ``Beamform(enable_pfield=True)`` does. That *weights* each transmit's
contribution inside the DAS sum (and with ``norm=True`` the weights are normalised per pixel, so
overall brightness is untouched). Here we want the unnormalised compound sensitivity
``S(pixel) = sum_tx |A_tx(pixel)|`` as a correction map, and then ``image / S``.

.. warning::

   ``compute_pfield``'s default ``downsample=10`` evaluates the field on a grid 10x coarser -
   about 3.9 mm here - which cannot represent the ~1.5 mm scalloping we are trying to correct.
   It is set to 1 below; that is the whole point of the exercise.
"""
import os
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, r"D:/Luuk van Knippenberg/Github/shearWaveProcessing/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np

from zea import File, init_device
from zea.beamform.pfield import compute_pfield

from swp.acquisition.beamform import apply_grid
from swp.acquisition.sequence import read_swi_meta

ROOT = r"D:\swp_res\Resolution phantom\DefaultPatient_SW_data_18-June-2026_13-52-51"
OUT = os.path.dirname(os.path.abspath(__file__))
DOWNSAMPLE = 1          # MUST be fine enough to resolve the scalloping (see warning)

init_device(verbose=True)
meta = read_swi_meta(os.path.join(ROOT, "CombinedData.mat"))
conv = os.path.join(ROOT, "output", "converted", "CombinedData_buffer3.hdf5")
with File(conv) as f:
    params = f.load_parameters()
apply_grid(params, meta.grids[2])

grid = np.asarray(params.grid, dtype=np.float32)
print(f"grid {grid.shape}  ({grid.shape[0] * grid.shape[1]} pixels)")

bw = getattr(params, "probe_bandwidth_percent", None)
bw = float(bw) if bw is not None else 60.0
print(f"probe bandwidth: {bw}%")

t0 = time.perf_counter()
pf = compute_pfield(
    sound_speed=float(params.sound_speed),
    center_frequency=float(np.asarray(params.center_frequency).reshape(-1)[0]),
    probe_bandwidth_percent=bw,
    n_el=int(params.n_el),
    probe_geometry=np.asarray(params.probe_geometry, np.float32),
    tx_apodizations=np.asarray(params.tx_apodizations, np.float32),
    grid=grid,
    t0_delays=np.asarray(params.t0_delays, np.float32),
    downsample=DOWNSAMPLE,
    norm=False,                    # we want the UNNORMALISED field to build a sensitivity map
)
pf = np.asarray(pf.cpu() if hasattr(pf, "cpu") else pf)   # compute_pfield returns a device tensor
print(f"compute_pfield -> {pf.shape} in {time.perf_counter() - t0:.0f}s")
np.save(os.path.join(OUT, "pfield_buffer3.npy"), pf)

# Compound sensitivity: sum the per-transmit field magnitudes over transmits.
# compute_pfield returns (n_tx, nz, nx) - the transmit axis is FIRST.
assert pf.shape[1:] == grid.shape[:2], f"pfield {pf.shape} vs grid {grid.shape}"
S = np.abs(pf).sum(axis=0)
S = S / np.median(S[S > 0])
print(f"sensitivity map: min {S.min():.3f}  median {np.median(S):.3f}  max {S.max():.3f}  "
      f"({20 * np.log10(S.max() / max(S[S > 0].min(), 1e-9)):.1f} dB span)")
np.save(os.path.join(OUT, "pfield_sensitivity.npy"), S)
print("saved pfield_sensitivity.npy")
