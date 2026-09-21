"""Reconstruct a widebeam B-mode buffer with a per-transmit PIXEL-INCLUSION window.

The widebeam buffers fire 21 beams 4 deg apart from a virtual source 123 mm behind the
array. One beam opens ~9.3-11.6 deg, so at 100 mm depth only about **5 of the 21 transmits
insonify a given pixel** - yet the pipeline compounds all 21 into every pixel. Measured in
vivo (``docs/widebeam_bmode_reconstruction.md``), the 16 transmits whose cone excludes a
pixel contribute as much amplitude there as the 5 that include it; on the resolution phantom,
which has the same geometry but no reverberating chest wall, they are 18 dB down. The
difference is clutter, and it lands in the echo-free regions.

Restricting each transmit to its own cone (zea's ``AlignedApodization``, fed
``swp.acquisition.txwindow``) raises dynamic range by +2.6 to +5.9 dB on every subject and
frame tested, at **no measurable resolution cost** (-0.01% lateral -6 dB width, 95% CI
[-0.16, +0.15], paired over 66 wire targets), for +12% beamforming time.

It removes real clutter, not only noise. Two independent measurements agree on where that
clutter comes from: an echo-free chamber receives its energy almost uniformly across all 21
transmits, peaking at the ones aimed at a bright specular arc on the OPPOSITE side of the
sector; and over the cardiac cycle the part the cone removes tracks that arc (r = +0.88) while
what it keeps tracks the static near field (r = +0.74). The cone flips the chamber from
arc-driven clutter to chest-wall reverberation, which no aperture weighting can address.

``("rect", 1.0)`` is the PIPELINE DEFAULT for buffers 1 and 5 since 2026-09-21, so this script
is now for exploring alternatives rather than for producing the standard image. ``--scale 1.5``
is the wider, provably-zero-resolution-change setting.

Writes ``<stem>_buffer<k>_txwin-<window><scale>_iq.hdf5`` + GIF beside the normal output, so
the result sits next to the standard reconstruction rather than replacing it.

Usage:
    python scripts/widebeam_bmode.py <folder> [--buffer 1] [--window rect] [--scale 1.0]
                                     [--beamformer ...] [--max-frames N] [--no-gif]
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
from swp.acquisition.txwindow import (DEFAULT_SCALE, DEFAULT_WINDOW, WINDOWS, coverage,
                                      flat_window, transmit_window)

# Buffers whose phase feeds the displacement estimators: never reweight these.
PHASE_CRITICAL = (2, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder")
    p.add_argument("--buffer", type=int, default=1, help="MATLAB buffer number (default 1)")
    p.add_argument("--window", default=DEFAULT_WINDOW, choices=sorted(WINDOWS),
                   help="cone edge profile (default rect: a hard cone)")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                   help="cone width as a multiple of the geometric half-opening "
                        f"(default {DEFAULT_SCALE})")
    p.add_argument("--beamformer", default="delay_and_sum",
                   choices=["delay_and_sum", "generalized_coherence_factor",
                            "delay_multiply_and_sum"],
                   help="OPEN QUESTION, not a recommendation. "
                        "'generalized_coherence_factor' suppresses the residual clutter far "
                        "harder than the cone alone (specular-arc contrast 51.7 -> 66.9 dB) and "
                        "is BETTER on phantom point targets (-0.9%% lateral, +2.6 dB wire CNR) "
                        "at no extra runtime - but in vivo it drops speckle SNR 1.06 -> 0.72 and "
                        "does not improve gCNR, i.e. it changes the speckle statistics rather "
                        "than only removing clutter. 'delay_multiply_and_sum' is the extreme "
                        "case: -28%% lateral on wires and the WORST in-vivo gCNR of anything "
                        "tested. Judge these on a cine, not on the numbers "
                        "(docs/widebeam_bmode_reconstruction.md S12).")
    p.add_argument("--m-zero", type=int, default=4,
                   help="generalized_coherence_factor only: low spatial-frequency cutoff "
                        "(higher = gentler, more tolerant of aberration). Default 4.")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--no-gif", action="store_true")
    a = p.parse_args()

    if a.buffer in PHASE_CRITICAL:
        raise SystemExit(f"buffer {a.buffer} feeds the displacement estimators; "
                         "reweighting transmits changes its phase. Refusing.")

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
    apply_grid(params, grid)
    _ensure_cpu_t_peak(params)
    coords = np.asarray(params.grid, np.float32)
    n_frames, n_tx, n_el = raw.shape[0], raw.shape[1], raw.shape[3]
    print(f"buffer {a.buffer} ({spec.name}): raw {raw.shape}, {fps:.1f} FPS, "
          f"grid {coords.shape[0]}x{coords.shape[1]}")

    w = transmit_window(params, coords, a.window, a.scale)
    cov = coverage(w)
    inside = cov[cov > 0]
    print(f"  transmit window '{a.window}' x{a.scale:g}: effective transmits per pixel "
          f"median {np.median(inside):.1f}, 5-95 pct {np.percentile(inside, 5):.1f}-"
          f"{np.percentile(inside, 95):.1f} (of {n_tx}); the standard reconstruction uses "
          f"all {n_tx}")

    num_patches, _ = plan_patches_and_chunk(params, n_tx, n_el)
    bf_kwargs = {"m_zero": a.m_zero} if a.beamformer == "generalized_coherence_factor" else {}
    pipe = zea.Pipeline([Cast(dtype="float32"), Demodulate(),
                         Beamform(beamformer=a.beamformer, num_patches=num_patches,
                                  enable_aligned_apodization=True, **bf_kwargs)],
                        with_batch_dim=True, jit_options=None)
    bf_in = pipe.prepare_parameters(params)
    # flat_aligned_apodization is a computed Parameters property, so it is injected here.
    bf_in["flat_aligned_apodization"] = flat_window(params, coords, a.window, a.scale)

    out, t0 = [], time.perf_counter()
    chunk = max(1, min(8, n_frames))
    for s in range(0, n_frames, chunk):
        iq = pipe(data=raw[s:s + chunk], **bf_in)[pipe.output_key]
        iq = np.asarray(iq.cpu() if hasattr(iq, "cpu") else iq, dtype=np.float32)
        out.append(iq)
        print(f"    frame {min(s + chunk, n_frames)}/{n_frames} "
              f"({time.perf_counter() - t0:.0f}s)", end="\r")
    iq = np.concatenate(out)
    print(f"\n  windowed beamform: {n_frames} frames in {time.perf_counter() - t0:.0f}s "
          f"-> IQ {iq.shape}")

    tag = f"txwin-{a.window}{a.scale:g}"
    if a.beamformer != "delay_and_sum":
        tag += "-" + ("gcf%d" % a.m_zero if a.beamformer == "generalized_coherence_factor"
                      else "dmas")
    out_path = out_dir / f"{mat.stem}_buffer{a.buffer}_{tag}_iq.hdf5"
    _save_beamformed(out_path, iq, coords, fps=fps,
                     description=(f"{spec.name}: per-transmit pixel-inclusion window "
                                  f"({a.window}, x{a.scale:g} of the geometric cone) - each "
                                  f"transmit contributes only where it insonified"
                                  + ("" if a.beamformer == "delay_and_sum"
                                     else f"; beamformer={a.beamformer}")),
                     compression=DEFAULT_COMPRESSION)
    print(f"  wrote {out_path.name}")

    if not a.no_gif:
        from swp.acquisition.gifs import gif_for_file
        gif_for_file(out_path)
        print(f"\nCompare: {out_path.with_suffix('.gif').name}  vs  "
              f"{mat.stem}_buffer{a.buffer}_iq.gif")


if __name__ == "__main__":
    main()
