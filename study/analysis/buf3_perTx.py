"""Buffer 3: beamform each of the 73 focused transmits separately, then composite.

Tests the user's hypothesis directly. Current pipeline coherently sums ALL 73 transmits into
every pixel, including beams steered up to 80 deg away that never insonified it. Here each
transmit is beamformed alone and pixels are composited with an angular weight centred on that
transmit's steering direction, so a pixel is built only from the beams that actually covered it.

Saves the per-transmit IQ so different composite rules can be compared without re-beamforming.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

import h5py
import numpy as np

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate

from swp.acquisition.beamform import apply_grid, _ensure_cpu_t_peak
from swp.acquisition.sequence import read_swi_meta

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
CONV = os.path.join(FOLDER, "output", "converted", "CombinedData_buffer3.hdf5")
MAT = os.path.join(FOLDER, "CombinedData.mat")
FRAME = 0
OUT = os.path.dirname(os.path.abspath(__file__))

init_device(verbose=True)
meta = read_swi_meta(MAT)
grid = meta.grids[2]

with File(CONV) as fh:
    raw = np.asarray(fh.data.raw_data[FRAME:FRAME + 1])
    params = fh.load_parameters()
n_tx = raw.shape[1]
print(f"raw {raw.shape}  n_tx={n_tx}")

apply_grid(params, grid)
_ensure_cpu_t_peak(params)
coords = np.asarray(params.grid, dtype=np.float32)
np.save(os.path.join(OUT, "buf3_coords.npy"), coords)

# Steering angle of each transmit, from the .mat (the converted file does not carry Steer).
with h5py.File(MAT, "r") as f:
    def refs(g, k):
        return np.atleast_1d(np.array(g[k]).squeeze())

    TX = f["TX"]
    foc = np.array([float(np.array(f[r]).squeeze()) for r in refs(TX, "focus")])
    steer = np.array([np.array(f[r]).squeeze().reshape(-1) for r in refs(TX, "Steer")])
    apod = [np.array(f[r]).squeeze().reshape(-1) for r in refs(TX, "Apod")]
n_act = np.array([int((np.abs(a) > 0).sum()) for a in apod])
sel = np.where((np.round(foc, 2) == 160.0) & (n_act == 79))[0]
az_all = steer[sel, 0]
az = np.unique(np.round(az_all, 9))
print(f"focused family: {len(sel)} tx -> {len(az)} unique angles, "
      f"{np.degrees(az).min():.1f}..{np.degrees(az).max():.1f} deg")
assert len(az) == n_tx, f"{len(az)} angles vs {n_tx} stored transmits"
np.save(os.path.join(OUT, "buf3_az.npy"), az)

pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                     Beamform(beamformer="delay_and_sum", num_patches=1, enable_pfield=False)],
                    with_batch_dim=True, jit_options=None)

per_tx = np.zeros((n_tx,) + coords.shape[:2] + (2,), np.float32)
for i in range(n_tx):
    params.set_transmits([i])
    bf_in = pipe.prepare_parameters(params)
    iq = pipe(data=raw[:, i:i + 1], **bf_in)[pipe.output_key]
    iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
    per_tx[i] = iq[0]
    if i % 12 == 0:
        print(f"  tx {i}/{n_tx}")

np.save(os.path.join(OUT, "buf3_per_tx.npy"), per_tx)
print(f"saved per-transmit IQ {per_tx.shape} -> buf3_per_tx.npy")
