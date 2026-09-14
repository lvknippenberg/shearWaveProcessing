"""Reconstruct a focused B-mode buffer via REFoCUS (retrospective transmit beamforming).

REFoCUS inverts the transmit encoding to recover the multistatic (full-matrix) dataset, then
beamforms that. Because the recovered dataset has uniform synthetic-aperture transmit coverage,
it removes the per-line transmit structure entirely - rather than smoothing over it the way
incoherent compounding does - so it should suppress the line-spacing striations WITHOUT the
resolution penalty. See docs/focused_bmode_striations.md.

Uses zea's ``Refocus`` operation, which also rewrites the downstream transmit parameters
(t0_delays -> zeros, tx_apodizations -> identity, pfield reset) so ``Beamform`` stays consistent.

.. warning::

   The encoding here is **under-determined**: 73 transmits recovering 80 virtual elements. The
   pseudo-inverse is therefore rank-deficient and the inversion method matters - hence the
   ``--method`` sweep. ``adjoint`` is the stable matched-filter option; the SVD-based methods
   (``tikhonov`` / ``tsvd`` / ``rsvd``) invert harder and can amplify noise. Compare before
   trusting any of them.

Writes ``<stem>_buffer<k>_refocus-<method>_iq.hdf5`` + GIF beside the normal output, and reports
the line-spacing ripple metric so the result is directly comparable to the coherent and
incoherent reconstructions.

Usage:
    python scripts/refocus_bmode.py <folder> [--buffer 3] [--method adjoint] [--max-frames N]
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
from zea.ops import Beamform, Cast, Demodulate, Refocus

from swp.acquisition.beamform import (DEFAULT_COMPRESSION, _ensure_cpu_t_peak, _save_beamformed,
                                      apply_grid, find_mat, plan_patches_and_chunk)
from swp.acquisition.sequence import SPEC_BY_INDEX, read_swi_meta


def line_spacing_power(env, coords, r_lo=0.045, r_hi=0.085, spacing_deg=1.1111):
    """Fraction of angular-ripple power at the line spacing, 45-85 mm band (see the doc)."""
    from scipy.ndimage import uniform_filter1d
    x = coords[..., 0]
    z = coords[..., -1]
    x0, x1 = np.nanmin(x), np.nanmax(x)
    z0, z1 = np.nanmin(z), np.nanmax(z)
    nz, nx = env.shape[-2:]
    th = np.radians(np.linspace(-36, 36, 1501))
    r = np.linspace(r_lo, r_hi, 200)
    R, TH = np.meshgrid(r, th, indexing="ij")
    jj = np.clip(((R * np.sin(TH) - x0) / (x1 - x0) * (nx - 1)).astype(int), 0, nx - 1)
    ii = np.clip(((R * np.cos(TH) - z0) / (z1 - z0) * (nz - 1)).astype(int), 0, nz - 1)
    frames = env if env.ndim == 3 else env[None]
    acc = np.zeros(th.size)
    for fr in frames:
        pol = fr[ii, jj]
        pol = pol / (np.median(pol, axis=1, keepdims=True) + 1e-12)
        acc += pol.mean(axis=0)
    prof = acc / len(frames)
    dth = np.degrees(th[1] - th[0])
    trend = uniform_filter1d(prof, int(round(10.0 / dth)) | 1, mode="nearest")
    rip = (prof - trend) / (trend + 1e-12)
    spec = np.abs(np.fft.rfft(rip * np.hanning(rip.size))) ** 2
    freq = np.fft.rfftfreq(rip.size, d=dth)
    tot = spec[(freq > 0.05) & (freq < 5)].sum()
    fl = 1.0 / spacing_deg
    return spec[(freq > fl * 0.9) & (freq < fl * 1.1)].sum() / tot * 100, rip.std() * 100


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--buffer", type=int, default=3)
    p.add_argument("--method", default="adjoint",
                   choices=["adjoint", "tikhonov", "tsvd", "rsvd"],
                   help="REFoCUS inversion method (default adjoint)")
    p.add_argument("--param", type=float, default=None,
                   help="regularisation / ramp parameter (see zea.ops.Refocus)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--spacing-deg", type=float, default=None,
                   help="transmit angular spacing for the ripple metric (default: read from the "
                        "buffer - 1.111 for the focused buffer 3, 4.0 for the widebeam buffers)")
    a = p.parse_args()

    folder = Path(a.folder)
    mat = find_mat(folder)
    out_dir = folder / "output"
    conv = out_dir / "converted" / f"{mat.stem}_buffer{a.buffer}.hdf5"
    if not conv.is_file():
        raise SystemExit(f"converted RF not found: {conv}")

    init_device(verbose=True)
    meta = read_swi_meta(mat)
    spec = SPEC_BY_INDEX[a.buffer - 1]
    grid = meta.grids[a.buffer - 1]
    fps = meta.fps.get(a.buffer - 1)

    with File(str(conv)) as fh:
        raw = np.asarray(fh.data.raw_data[:a.max_frames] if a.max_frames
                         else fh.data.raw_data[:])
        params = fh.load_parameters()
    n_tx, n_el = raw.shape[1], raw.shape[3]
    print(f"buffer {a.buffer} ({spec.name}): raw {raw.shape}  {fps:.1f} FPS")
    print(f"  encoding matrix H is ({n_tx} tx x {n_el} el)"
          f"{'  -- UNDER-DETERMINED, inversion is rank-deficient' if n_tx < n_el else ''}")

    apply_grid(params, grid)
    _ensure_cpu_t_peak(params)
    # After decoding there are n_el virtual transmits, so size the patches for that.
    num_patches, _ = plan_patches_and_chunk(params, n_el, n_el)
    pipe = zea.Pipeline(
        [Cast(dtype="float32"),
         Refocus(method=a.method, param=a.param),
         Demodulate(),
         Beamform(beamformer="delay_and_sum", num_patches=num_patches, enable_pfield=False)],
        with_batch_dim=True, jit_options=None,
    )
    bf_in = pipe.prepare_parameters(params)

    out, t0 = [], time.perf_counter()
    for k in range(raw.shape[0]):
        iq = pipe(data=raw[k:k + 1], **bf_in)[pipe.output_key]
        iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
        out.append(iq[0])
        if k % 5 == 0:
            print(f"    frame {k}/{raw.shape[0]}  ({time.perf_counter() - t0:.0f}s)")
    iq = np.stack(out)
    print(f"  REFoCUS[{a.method}] + beamform: {raw.shape[0]} frames in "
          f"{time.perf_counter() - t0:.0f}s -> IQ {iq.shape}")

    coords = np.asarray(params.grid, dtype=np.float32)
    tag = f"refocus-{a.method}"
    out_path = out_dir / f"{mat.stem}_buffer{a.buffer}_{tag}_iq.hdf5"
    _save_beamformed(out_path, iq, coords, fps=fps,
                     description=f"{spec.name}: REFoCUS ({a.method}) retrospective transmit "
                                 f"beamforming -> synthetic aperture",
                     compression=DEFAULT_COMPRESSION)
    env = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
    spacing = a.spacing_deg or (1.1111 if a.buffer == 3 else 4.0)
    pw, rms = line_spacing_power(env, coords, spacing_deg=spacing)
    print(f"  ripple power at the {spacing:.3f} deg transmit spacing: {pw:.2f}%   "
          f"total ripple RMS {rms:.1f}%")
    print(f"  (coherent all-73 ~5%, incoherent ~0.08% - see docs/focused_bmode_striations.md)")

    from swp.acquisition.gifs import gif_for_file
    gif_for_file(out_path)
    print(f"  wrote {out_path.name} + .gif")


if __name__ == "__main__":
    main()
