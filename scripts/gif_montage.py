"""Tile several GIFs into one synchronised montage GIF.

Clips of different length are padded with **black frames** to the longest duration, so every
tile restarts at the same instant on every loop. Without that, clips of different length drift
apart after the first repeat and the montage becomes impossible to read.

Clips with different frame delays are resampled onto a common timebase (the shortest delay
present), so mixing e.g. a 20 ms and a 40 ms GIF still lines up in wall-clock time.

Examples
--------
Compare reconstructions of one buffer::

    python scripts/gif_montage.py -o compare.gif --cols 3 --tile-width 420 \\
        --labels "standard,incoherent,REFoCUS adjoint" \\
        <folder>/output/CombinedData_buffer3_iq.gif \\
        <folder>/output/CombinedData_buffer3_incoh_iq.gif \\
        <folder>/output/CombinedData_buffer3_refocus-adjoint_iq.gif

Every subject's widebeam B-mode in one grid (label from the subject folder)::

    python scripts/gif_montage.py -o all_buffer1.gif --cols 8 --tile-width 170 \\
        --label-from subject --glob "Z:/raw_data/C*/*/output/CombinedData_buffer1_iq.gif"
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

LABEL_H = 15          # px reserved under each tile for its caption


def read_gif(path):
    """Return ``(frames as list of L-mode PIL images, per-frame delay in ms)``."""
    im = Image.open(path)
    frames, delays = [], []
    for fr in ImageSequence.Iterator(im):
        frames.append(fr.convert("L").copy())
        delays.append(fr.info.get("duration", im.info.get("duration", 100)))
    # GIFs written by this repo use a constant delay; take the median if it varies.
    return frames, int(np.median(delays))


def resample(frames, delay, target_delay, n_out):
    """Put a clip on the target timebase, padding with black once it has ended."""
    w, h = frames[0].size
    black = Image.new("L", (w, h), 0)
    out = []
    for k in range(n_out):
        t = k * target_delay                     # ms since the montage restarted
        idx = int(t // delay)
        out.append(frames[idx] if idx < len(frames) else black)
    return out


def label_for(path, mode):
    p = Path(path)
    if mode == "subject":
        # <root>/<subject>/<measurement>/output/<file>.gif -> subject
        parts = p.parts
        return parts[-4] if len(parts) >= 4 else p.stem
    if mode == "measurement":
        return p.parts[-3] if len(p.parts) >= 3 else p.stem
    return p.stem


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gifs", nargs="*", help="input GIF paths")
    p.add_argument("--glob", default=None, help="glob pattern for inputs (instead of listing)")
    p.add_argument("-o", "--out", required=True, help="output montage GIF")
    p.add_argument("--cols", type=int, default=None, help="grid columns (default: near-square)")
    p.add_argument("--tile-width", type=int, default=220, help="tile width in px (default 220)")
    p.add_argument("--labels", default=None, help="explicit comma-separated captions")
    p.add_argument("--label-from", default="stem", choices=["stem", "subject", "measurement"],
                   help="where captions come from when --labels is not given")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--title", default=None, help="optional title strip along the top")
    a = p.parse_args()

    paths = list(a.gifs)
    if a.glob:
        paths += sorted(globmod.glob(a.glob))
    if not paths:
        raise SystemExit("no input GIFs")
    print(f"=== montage of {len(paths)} GIF(s) ===")

    clips = [read_gif(q) for q in paths]
    delays = [d for _, d in clips]
    target = min(delays)
    n_out = max(int(np.ceil(len(f) * d / target)) for f, d in clips)
    dur_s = n_out * target / 1000.0
    print(f"  common delay {target} ms; longest clip {dur_s:.2f} s -> {n_out} frames")
    for q, (f, d) in zip(paths, clips):
        pad = n_out - int(np.ceil(len(f) * d / target))
        print(f"    {Path(q).name:48s} {len(f):3d} fr @ {d}ms"
              f"{f'  (+{pad} black)' if pad else ''}")

    # tile geometry from the first clip's aspect ratio
    w0, h0 = clips[0][0][0].size
    tw = a.tile_width
    th = int(round(tw * h0 / w0))
    lab_h = 0 if a.no_labels else LABEL_H
    cols = a.cols or int(np.ceil(np.sqrt(len(paths))))
    rows = int(np.ceil(len(paths) / cols))
    title_h = 18 if a.title else 0
    W, H = cols * tw, rows * (th + lab_h) + title_h
    print(f"  grid {cols} x {rows}, tile {tw}x{th} -> {W}x{H}")

    if a.labels:
        caps = [s.strip() for s in a.labels.split(",")]
        if len(caps) != len(paths):
            raise SystemExit(f"--labels has {len(caps)} entries for {len(paths)} GIFs")
    else:
        caps = [label_for(q, a.label_from) for q in paths]

    series = [resample(f, d, target, n_out) for f, d in clips]

    out_frames = []
    for k in range(n_out):
        canvas = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(canvas)
        if a.title:
            draw.text((6, 4), a.title, fill=255)
        for i, s in enumerate(series):
            r, c = divmod(i, cols)
            x, y = c * tw, title_h + r * (th + lab_h)
            canvas.paste(s[k].resize((tw, th), Image.BILINEAR), (x, y))
            if lab_h:
                draw.text((x + 3, y + th + 2), caps[i][:28], fill=200)
        out_frames.append(canvas)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head, *tail = out_frames
    head.save(out, format="GIF", append_images=tail, save_all=True, loop=0,
              duration=target, optimize=True)
    print(f"  wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, {n_out} frames, "
          f"{dur_s:.2f} s loop)")


if __name__ == "__main__":
    main()
