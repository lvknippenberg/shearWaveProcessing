"""Montage of many subjects rendered against ONE shared dB reference.

The per-folder GIFs written by the beamform stage log-compress each clip against its **own**
maximum, so brightness is not comparable between subjects - fine for judging view quality, no
good for comparing signal levels. This rebuilds the montage straight from the IQ using a single
reference level for every tile, so a dim tile really is a weaker signal.

Frame selection reproduces the real-time GIF rule (``swp.acquisition.gifs``), so the result is
frame-for-frame comparable with the per-clip-normalised montages.

Choosing the reference (``--reference``):

* ``median`` (default) - median across subjects of the per-clip 99.9th percentile. Robust: one
  unusually bright subject cannot darken all the others.
* ``max`` - the brightest subject's 99.9th percentile. Nothing clips, but dim subjects go black.
* a number - an explicit envelope value, for putting **different sets on the same scale** (pass
  the same value to each run).

.. warning::

   Only compare sets that are physically comparable. Buffer 1 and buffer 3 are both plain DAS,
   so a shared reference between them is meaningful. **REFoCUS is not**: it applies H^H with a
   ramp filter, which is not unit gain, so its absolute level differs from standard DAS by a
   decode gain that has nothing to do with image quality. Compare REFoCUS against standard using
   its *own* reference (relative texture/contrast), not a shared one.

Usage:
    python scripts/shared_norm_montage.py -o out.gif --cols 8 --tile-width 170 \\
        --glob "Z:/raw_data/C*/*/output/CombinedData_buffer3_iq.hdf5"
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import h5py
import numpy as np
from PIL import Image, ImageDraw

from swp.viz.tonecurves import apply_curve

MIN_DELAY_CS, MAX_DELAY_CS, TOL = 2, 100, 0.02
LABEL_H = 15


def realtime_indices(n, fps, stretch=1.0):
    """Frame indices + centisecond delay the real-time GIF rule would use."""
    if fps <= 0:
        return np.arange(n), MIN_DELAY_CS
    frame_dt = stretch / fps
    delay_cs = int(round(frame_dt * 100))
    keep = (MIN_DELAY_CS <= delay_cs <= MAX_DELAY_CS
            and abs(delay_cs / 100.0 - frame_dt) <= TOL * frame_dt)
    if keep:
        return np.arange(n), delay_cs
    delay_cs = int(np.clip(delay_cs, MIN_DELAY_CS, MAX_DELAY_CS))
    target_s = stretch * n / fps
    n_shown = max(2, int(round(target_s * 100.0 / delay_cs)))
    return np.linspace(0, n - 1, n_shown).round().astype(int), delay_cs


def load_tiles(path, tile_w):
    """Envelope frames for one file, real-time sampled and resized.

    Returns ``(frames, delay_cs, p99.9, noise_floor)``. Both levels are taken over **in-sector**
    pixels only — outside the sector the envelope is exactly 0, which would drag any low
    percentile to zero and make the noise floor meaningless.
    """
    with h5py.File(path, "r") as f:
        g = f["tracks/track_0/data/beamformed_data"]
        n = g["values"].shape[0]
        ts = np.asarray(g["timestamps"]) if "timestamps" in g else None
        fps = float(1.0 / np.median(np.diff(ts))) if ts is not None and ts.size > 1 else 25.0
        idx, delay = realtime_indices(n, fps)
        iq = np.asarray(g["values"][:])[idx]
    env = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2).astype(np.float32)
    h, w = env.shape[1:]
    tile_h = int(round(tile_w * h / w))
    small = np.stack([np.asarray(Image.fromarray(fr).resize((tile_w, tile_h), Image.BILINEAR))
                      for fr in env])
    ins = env[env > 0]
    return (small, delay, float(np.percentile(ins, 99.9)), float(np.percentile(ins, 5)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--glob", required=True)
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--tile-width", type=int, default=170)
    p.add_argument("--dynamic-range", type=float, default=50.0, help="dB below the reference")
    p.add_argument("--reference", default="median",
                   help="'median' (default), 'max', or an explicit envelope value")
    p.add_argument("--title", default=None)
    p.add_argument("--cache", default=None,
                   help="npz of downscaled tiles. Written on first use, reused after - so "
                        "re-rendering with a different --reference / --dynamic-range costs "
                        "seconds instead of re-reading every IQ file.")
    p.add_argument("--per-subject", action="store_true",
                   help="optimise EACH acquisition separately: its own white point (own p99.9) "
                        "and its own range (down to its own noise floor). Best visibility per "
                        "tile; brightness then carries no cross-subject meaning at all.")
    p.add_argument("--gain-db", type=float, default=0.0,
                   help="brightness, in dB. Lowers the white point, so higher values brighten "
                        "the image and deliberately let the brightest structures CLIP. The "
                        "saturated fraction is reported so the trade is visible.")
    p.add_argument("--curve", default="linear",
                   help="display tone curve: linear (default, what the pipeline does today) or "
                        "gamma1..gamma5 (see swp.viz.tonecurves)")
    p.add_argument("--auto", action="store_true",
                   help="derive BOTH the white point and the range from this set's own data: "
                        "reference = 90th pct of per-subject p99.9, range = down to the median "
                        "in-sector 5th pct (the noise floor). Subjects stay mutually comparable; "
                        "brightness is NOT comparable with a differently-scaled set.")
    a = p.parse_args()

    paths = sorted(globmod.glob(a.glob))
    if not paths:
        raise SystemExit(f"no files match {a.glob!r}")
    print(f"=== shared-normalisation montage of {len(paths)} file(s) ===")

    cache = Path(a.cache) if a.cache else None
    if cache and cache.is_file():
        z = np.load(cache, allow_pickle=True)
        series = list(z["series"]); delays = list(z["delays"])
        levels = list(z["levels"]); labels = list(z["labels"])
        floors = list(z["floors"])
        print(f"  loaded {len(series)} tile set(s) from cache {cache.name}")
    else:
        series, delays, levels, labels, floors = [], [], [], [], []
        for i, q in enumerate(paths, 1):
            fr, d, lvl, flo = load_tiles(q, a.tile_width)
            series.append(fr); delays.append(d); levels.append(lvl); floors.append(flo)
            labels.append(Path(q).parts[-4])
            print(f"  [{i}/{len(paths)}] {labels[-1]}: {len(fr)} fr @ {d}cs, "
                  f"p99.9 {lvl:.0f}, floor {flo:.1f}")
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, series=np.array(series, dtype=object),
                                delays=delays, levels=levels, labels=labels, floors=floors)
            print(f"  cached tiles -> {cache}")

    levels, floors = np.array(levels), np.array(floors)
    if a.auto:
        ref = float(np.percentile(levels, 90))
        a.dynamic_range = float(20 * np.log10(ref / np.median(floors)))
        print(f"  AUTO: reference = {ref:.0f} (90th pct of per-subject p99.9), "
              f"range = {a.dynamic_range:.0f} dB (down to the median noise floor "
              f"{np.median(floors):.1f})")
    elif a.reference == "median":
        ref = float(np.median(levels))
    elif a.reference == "max":
        ref = float(levels.max())
    else:
        ref = float(a.reference)
    print(f"\n  reference ({a.reference}) = {ref:.0f}")
    print(f"  per-subject level spread = {20 * np.log10(levels.max() / levels.min()):.1f} dB "
          f"({20 * np.log10(levels.min() / ref):+.1f} .. {20 * np.log10(levels.max() / ref):+.1f} dB "
          f"about the reference)")

    gain = 10.0 ** (a.gain_db / 20.0)
    refs_i = levels.copy() / gain                           # each subject's own white point
    drs_i = 20 * np.log10(np.maximum(levels, 1e-9) / np.maximum(floors, 1e-9))
    if a.per_subject:
        print(f"  PER-SUBJECT: white points {refs_i.min():.0f}..{refs_i.max():.0f}, "
              f"ranges {drs_i.min():.0f}..{drs_i.max():.0f} dB "
              f"(median {np.median(drs_i):.0f} dB)")
    if a.gain_db:
        ref = ref / gain
        print(f"  gain {a.gain_db:+.1f} dB -> white point lowered by {gain:.2f}x "
              f"(clipping expected)")
    if a.curve != "linear":
        print(f"  tone curve: {a.curve}")

    delay = min(delays)
    n_out = max(int(np.ceil(len(s) * d / delay)) for s, d in zip(series, delays))
    tile_h = series[0].shape[1]
    cols = a.cols
    rows = int(np.ceil(len(paths) / cols))
    title_h = 18 if a.title else 0
    W = cols * a.tile_width
    H = rows * (tile_h + LABEL_H) + title_h
    print(f"  grid {cols}x{rows}, tile {a.tile_width}x{tile_h} -> {W}x{H}, {n_out} frames")

    def frame_at(s, d, k):
        j = int(k * delay // d)
        return s[j] if j < len(s) else None

    sat_total, sat_count = 0.0, 0
    out_frames = []
    for k in range(n_out):
        canvas = Image.new("L", (W, H), 0)
        draw = ImageDraw.Draw(canvas)
        if a.title:
            draw.text((6, 4), a.title, fill=255)
        for i, (s, d) in enumerate(zip(series, delays)):
            r, c = divmod(i, cols)
            x, y = c * a.tile_width, title_h + r * (tile_h + LABEL_H)
            fr = frame_at(s, d, k)
            if fr is not None:
                r_i, dr_i = (refs_i[i], drs_i[i]) if a.per_subject else (ref, a.dynamic_range)
                db = 20.0 * np.log10(fr / r_i + 1e-12)
                norm = np.clip((db + dr_i) / dr_i, 0, 1)
                u8 = apply_curve(norm, a.curve) * 255
                if k == n_out // 2:
                    # Clipping happens at the NORMALISATION, not the output: a tone curve can
                    # cap below 255 (gamma2 tops out at 0.97), so testing the 8-bit value would
                    # report 0% however hard the image is driven.
                    sat_total += float((norm >= 1.0).mean()); sat_count += 1
                canvas.paste(Image.fromarray(u8.astype(np.uint8)), (x, y))
            draw.text((x + 3, y + tile_h + 2), labels[i][:28], fill=200)
        out_frames.append(canvas)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    head, *tail = out_frames
    head.save(out, format="GIF", append_images=tail, save_all=True, loop=0,
              duration=delay * 10, optimize=True)
    print(f"  CLIPPED pixels (mid frame, in-tile, norm>=1): "
          f"{100 * sat_total / max(sat_count, 1):.2f}%")
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {n_out} frames, "
          f"{n_out * delay / 100:.2f} s loop)")
    return ref


if __name__ == "__main__":
    main()
