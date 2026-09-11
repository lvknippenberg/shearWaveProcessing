"""Reconstruct a focused B-mode buffer with INCOHERENT (envelope) transmit compounding.

Coherently summing the 73 steered focused transmits - what the normal pipeline does - leaves a
residual interference pattern at the transmit-angle sampling period (~1.111 deg here), visible
as fine radial striations. Measured on C000000001 buffer 3, frame 0, 45-85 mm band, as the
fraction of angular-ripple power at the line spacing:

    coherent sum of all 73    2.27%
    nearest single beam      10.31%      (strip stitching - worse)
    INCOHERENT sum of all 73  0.08%      (28x lower than coherent)

Incoherent compounding sums |IQ| per transmit instead of the complex IQ, so there is no
cross-transmit phase interference and the striations essentially vanish. It is not free: it
discards the coherent gain, so speckle contrast and lateral resolution drop. That trade is
defensible for buffer 3, which is an orientation B-mode and never feeds the shear-wave
processing - it is NOT appropriate for buffers 2/4, whose phase the estimators depend on.

Writes ``<stem>_buffer<k>_incoh_iq.hdf5`` (envelope stored in the I channel, Q = 0) plus a
real-time GIF beside it, so the result sits next to the normal output for comparison rather
than replacing it.

Usage:
    python scripts/incoherent_bmode.py <measurement folder> [--buffer 3] [--max-frames N]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np

import zea
from zea import File, init_device
from zea.ops import Beamform, Cast, Demodulate

from swp.acquisition.beamform import (DEFAULT_COMPRESSION, _ensure_cpu_t_peak, _save_beamformed,
                                      apply_grid, find_mat, plan_patches_and_chunk)
from swp.acquisition.sequence import SPEC_BY_INDEX, read_swi_meta


def incoherent_beamform(raw, params, frame_chunk=8):
    """Envelope-sum each transmit's own reconstruction -> (n_frames, z, x).

    Beamforms ONE transmit at a time over a chunk of frames and accumulates |IQ|, so peak
    memory stays at one chunk rather than n_tx x n_frames images.
    """
    n_frames, n_tx = raw.shape[0], raw.shape[1]
    _ensure_cpu_t_peak(params)
    num_patches, _ = plan_patches_and_chunk(params, 1, raw.shape[3])
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                         Beamform(beamformer="delay_and_sum", num_patches=num_patches,
                                  enable_pfield=False)],
                        with_batch_dim=True, jit_options=None)
    out = None
    t0 = time.perf_counter()
    for i in range(n_tx):
        params.set_transmits([i])
        bf_in = pipe.prepare_parameters(params)
        for s in range(0, n_frames, frame_chunk):
            block = raw[s:s + frame_chunk, i:i + 1]
            iq = pipe(data=block, **bf_in)[pipe.output_key]
            iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
            env = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
            if out is None:
                out = np.zeros((n_frames,) + env.shape[1:], np.float32)
            out[s:s + env.shape[0]] += env
        if i % 10 == 0:
            print(f"    transmit {i}/{n_tx}  ({time.perf_counter() - t0:.0f}s)")
    print(f"    incoherent compound done in {time.perf_counter() - t0:.0f}s")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--buffer", type=int, default=3, help="MATLAB buffer number (default 3)")
    p.add_argument("--max-frames", type=int, default=None, help="limit frames (for a quick look)")
    a = p.parse_args()

    folder = Path(a.folder)
    mat = find_mat(folder)
    out_dir = folder / "output"
    conv = out_dir / "converted" / f"{mat.stem}_buffer{a.buffer}.hdf5"
    if not conv.is_file():
        raise SystemExit(f"converted RF not found: {conv}\nRun the beamform stage first.")

    init_device(verbose=True)
    meta = read_swi_meta(mat)
    spec = SPEC_BY_INDEX[a.buffer - 1]
    grid = meta.grids[a.buffer - 1]
    fps = meta.fps.get(a.buffer - 1)

    with File(str(conv)) as fh:
        raw = np.asarray(fh.data.raw_data[:a.max_frames] if a.max_frames
                         else fh.data.raw_data[:])
        params = fh.load_parameters()
    print(f"buffer {a.buffer} ({spec.name}): raw {raw.shape}, {fps:.1f} FPS")

    apply_grid(params, grid)
    env = incoherent_beamform(raw, params)
    coords = np.asarray(params.grid, dtype=np.float32)

    # Store the envelope in I with Q=0 so the file stays a normal beamformed_data stack that
    # the GIF renderer (which takes sqrt(I^2+Q^2)) reads unchanged.
    iq = np.zeros(env.shape + (2,), np.float32)
    iq[..., 0] = env
    out_path = out_dir / f"{mat.stem}_buffer{a.buffer}_incoh_iq.hdf5"
    _save_beamformed(out_path, iq, coords, fps=fps,
                     description=(f"{spec.name}: INCOHERENT (envelope) transmit compounding - "
                                  f"no cross-transmit phase interference"),
                     compression=DEFAULT_COMPRESSION)
    print(f"  wrote {out_path.name}")

    from swp.acquisition.gifs import gif_for_file
    gif_for_file(out_path)
    print(f"\nCompare: {out_path.with_suffix('.gif').name}  vs  "
          f"{mat.stem}_buffer{a.buffer}_iq.gif")


if __name__ == "__main__":
    main()
