"""Buffer-3 montage of every subject, aligned on the R-peak with the unwrapped frame times.

The 2026-09 montages (``study/logs/adaptive_montages.sh`` -> ``all_buffer3_refocus_adaptive.gif``)
started every tile at stored frame 0, which - buffer 3 being a rotated circular buffer - was an
arbitrary cardiac phase per subject (docs/buffer3_unwrap.md). Here every tile of an unwrapped
folder is resampled onto a common phase axis: output step k shows the frame whose phase (time since
the preceding R-peak, ``custom/frame_rpeak_phase_ms``, as a fraction of the subject's median RR) is
nearest k/K. So every tile starts at the R-peak and completes exactly one beat per loop, whatever
the heart rate. Folders that are not (resolvably) unwrapped play in stored order, labelled in red.
Unwrapped folders without a trustworthy ECG (pulse-train or sparse R-peak record) play in
chronological order, labelled in orange: their frame order is right, their R-peak is not.

    python study/analysis/buffer3_rpeak_montage.py [--glob ...] [--steps 25]
-> study/montages/all_buffer3_refocus_rpeak_aligned.gif  (+ _R.png: the frame at the R-peak,
   _mid.png: at 40 % of RR)
"""
from __future__ import annotations

import argparse
import glob as globmod
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")

import h5py                                                     # noqa: E402
import hdf5plugin                                               # noqa: E402,F401
from PIL import Image, ImageDraw                                # noqa: E402

_BF = "tracks/track_0/data/beamformed_data"
LABEL_H = 15
COLOURS = {"white": (255, 255, 255), "orange": (240, 160, 40), "red": (230, 60, 60)}


def tile_frames(path, steps):
    """-> (list of uint8 frames, label, label colour: white aligned | orange | red)."""
    from swp.acquisition.gifs import iq_to_bmode
    from swp.acquisition.triggerlog import buffer_timing
    from swp.acquisition.unwrap import read_flag
    folder = Path(path).parent.parent
    with h5py.File(path, "r") as f:
        iq = np.asarray(f[f"{_BF}/values"], np.float32)
        ph = np.asarray(f["custom/frame_rpeak_phase_ms"], float) if "custom/frame_rpeak_phase_ms" in f else None
    from swp.acquisition.rrcheck import assess_rr
    frames = iq_to_bmode(iq, adaptive=True)
    rec = read_flag(path)
    subj = folder.parent.name[-3:]
    unwrapped = rec is not None and rec.get("buffer_unwrapped")
    idx = [k % len(frames) for k in range(steps)]
    if not unwrapped:
        return [frames[i] for i in idx], f"C{subj} stored order ({rec['status'] if rec else 'not unwrapped'})", "red"
    if not assess_rr(str(folder)).trustworthy or ph is None or np.isfinite(ph).sum() < len(ph) // 2:
        return [frames[i] for i in idx], f"C{subj} in order, no valid ECG", "orange"
    bt = buffer_timing(str(folder), 3)
    rr = float(np.median(np.diff(bt.r_peaks_ms)))
    frac = ph / rr
    good = np.isfinite(frac)
    out = []
    for k in range(steps):
        target = k / steps
        d = np.abs(((frac - target) + 0.5) % 1.0 - 0.5)          # circular distance in cycles
        d[~good] = np.inf
        out.append(frames[int(np.argmin(d))])
    return out, f"C{subj} {'trigger' if rec['method'] == 'trigger-count' else 'buffer1'}, RR {rr:.0f}", "white"


def render(tiles, cols, tile_w, title):
    h0, w0 = tiles[0][0][0].shape
    tw, th = tile_w, int(round(tile_w * h0 / w0))
    rows = int(np.ceil(len(tiles) / cols))
    W, H = cols * tw, rows * (th + LABEL_H) + 18
    steps = len(tiles[0][0])
    images = []
    for k in range(steps):
        canvas = Image.new("L", (W, H), 0)
        for j, (frames, _, _) in enumerate(tiles):
            r, c = divmod(j, cols)
            canvas.paste(Image.fromarray(frames[k]).resize((tw, th), Image.BILINEAR),
                         (c * tw, 18 + r * (th + LABEL_H)))
        rgb = canvas.convert("RGB")
        d2 = ImageDraw.Draw(rgb)
        d2.text((4, 2), f"{title}   phase {k / steps:.2f} RR", fill=(255, 255, 255))
        for j, (_, label, colour) in enumerate(tiles):
            r, c = divmod(j, cols)
            d2.text((c * tw + 2, 18 + r * (th + LABEL_H) + th + 1), label[:30], fill=COLOURS[colour])
        images.append(rgb)
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="Z:/raw_data/C*/*/output/CombinedData_buffer3_refocus-adjoint_iq.hdf5")
    ap.add_argument("--steps", type=int, default=25, help="output frames per cardiac cycle")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile-width", type=int, default=170)
    ap.add_argument("--delay-ms", type=int, default=40)
    ap.add_argument("-o", "--out", default=str(_REPO / "study" / "montages" / "all_buffer3_refocus_rpeak_aligned.gif"))
    a = ap.parse_args()
    paths = sorted(globmod.glob(a.glob), key=lambda p: str(p).lower())
    tiles = []
    for p in paths:
        try:
            tiles.append(tile_frames(p, a.steps))
            print(f"  {tiles[-1][1]}  [{tiles[-1][2]}]", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"  {p}: skipped ({exc})")
    n_al = sum(t[2] == "white" for t in tiles)
    title = (f"buffer 3 REFoCUS adjoint, unwrapped, aligned on the R-peak: white {n_al}/{len(tiles)} aligned "
             f"(one loop = one beat); orange = chronological, no valid ECG; red = stored order")
    images = render(tiles, a.cols, a.tile_width, title)
    images[0].save(a.out, save_all=True, append_images=images[1:], duration=a.delay_ms, loop=0)
    base = a.out[:-4]
    images[0].save(base + "_R.png")
    images[int(round(0.4 * a.steps))].save(base + "_mid.png")
    print(f"{len(tiles)} tiles ({n_al} aligned) -> {a.out}")


if __name__ == "__main__":
    main()
