"""Beamforming time: standard vs REFoCUS adjoint, buffers 3 and 1.

GPU beamform only - reading the converted RF off the network share dominates a real run's wall
clock and is identical for both methods.

Two things the first attempt got wrong and this one does not:
* after ``Refocus`` the transmit axis is n_el (80 virtual elements), not n_tx (73), so the patch
  budget has to be planned for 80 or it OOMs;
* the GPU is emptied between methods, otherwise the second one runs in whatever the first left.

Each method gets a one-frame warm-up that is not timed, so kernel autotuning does not land on one
method's stopwatch.
"""
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, r"D:/Luuk van Knippenberg/Github/shearWaveProcessing/src")

import numpy as np
import torch

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate, Refocus

from swp.acquisition.beamform import (_ensure_cpu_t_peak, apply_grid, beamform_frames,
                                      plan_patches_and_chunk)
from swp.acquisition.sequence import SPEC_BY_INDEX, read_swi_meta

FOLDER = r"Z:\raw_data\C000000001\SWE_01_SW_data_21-April-2026_12-12-54"
OUT = os.path.join(FOLDER, "output")


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


init_device(verbose=False)
meta = read_swi_meta(Path(FOLDER) / "CombinedData.mat")

for b in (3, 1):
    with File(os.path.join(OUT, "converted", f"CombinedData_buffer{b}.hdf5")) as fh:
        raw = np.asarray(fh.data.raw_data[:])
        params = fh.load_parameters()
    apply_grid(params, meta.grids[b - 1])
    spec = SPEC_BY_INDEX[b - 1]
    n_f, n_tx, n_el = raw.shape[0], raw.shape[1], raw.shape[3]
    grid_shape = tuple(np.asarray(params.grid).shape[:2])
    print(f"\nbuffer {b} ({spec.name}): {n_f} frames x {n_tx} tx x {n_el} el, grid {grid_shape}, "
          f"pfield={spec.pfield}")

    clear()
    beamform_frames(raw[:1], params, enable_pfield=spec.pfield)          # warm-up, untimed
    clear()
    t0 = time.perf_counter()
    beamform_frames(raw, params, enable_pfield=spec.pfield)
    t_std = time.perf_counter() - t0
    print(f"  standard (pfield={str(spec.pfield):5s})  {t_std:7.1f} s   "
          f"{t_std / n_f * 1000:6.0f} ms/frame")
    clear()

    # Refocus expands the transmit axis to n_el virtual transmits - budget for that, not n_tx.
    _ensure_cpu_t_peak(params)
    num_patches, chunk = plan_patches_and_chunk(params, n_el, n_el)
    chunk = max(1, min(chunk, 2))
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(), Refocus(method="adjoint"),
                         Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                  enable_pfield=False)],
                        with_batch_dim=True, jit_options=None)
    bf_in = pipe.prepare_parameters(params)
    pipe(data=raw[:1], **bf_in)                                          # warm-up, untimed
    clear()
    t0 = time.perf_counter()
    for s in range(0, n_f, chunk):
        pipe(data=raw[s:s + chunk], **bf_in)
    t_ref = time.perf_counter() - t0
    print(f"  REFoCUS adjoint            {t_ref:7.1f} s   {t_ref / n_f * 1000:6.0f} ms/frame"
          f"   ({t_ref / t_std:.2f}x standard, {num_patches} patches, chunk {chunk})")
    del pipe, raw
    clear()
