"""Playback montage of buffer 1 or buffer 3 for the N-th SW acquisition of every subject (2026-09-28).

Plain playback: every tile plays ALL of its frames once, in file order, starting together; a shorter
clip holds its last frame (grey label) until the GIF loops. No resampling and NO synchronisation to
the R-peak or between tiles - buffers 1 and 3 are not ECG-triggered, and the point of the buffer-3
unwrap is continuous playback without the wrap jump (docs/buffer3_unwrap.md). Replaces
buffer3_rpeak_montage.py (R-peak phase resampling, which re-introduced jumps and repeats).

- ``--buffer 1``: buffer 1 exactly as stored (never modified).
- ``--buffer 3 --order unwrapped``: buffer 3 as on disk now (chronological after the unwrap;
  ambiguous folders keep the stored order, labelled red with the reason).
- ``--buffer 3 --order stored``: the ORIGINAL stored slot order, rebuilt from the head recorded by
  the unwrap (file frame i = stored slot (first + i) mod n): the pre-correction view.
Buffer 3 uses the REFoCUS adjoint reconstruction: the ``*_buffer3_refocus-adjoint_iq.hdf5`` variant
where it exists (folders beamformed before 2026-09-14), else the main file (REFoCUS by default since).

    python study/analysis/buffer_playback_montage.py --acq 1 --buffer 3 --order unwrapped
    python study/analysis/buffer_playback_montage.py --acq 2 --buffer 1 --tile-width 140 --delay-ms 20
-> study/montages/playback/acq<N>_buffer<b>_<order>.gif (+ _first.png)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")

import h5py                                                     # noqa: E402
import hdf5plugin                                               # noqa: E402,F401
from PIL import Image, ImageDraw                                # noqa: E402

_BF = "tracks/track_0/data/beamformed_data"
ROOT = Path("Z:/raw_data")
OUT = _REPO / "study" / "montages" / "playback"
LABEL_H, TOP = 15, 18
_TS = re.compile(r"_(\d{1,2}-[A-Za-z]+-\d{4}_\d{2}-\d{2}-\d{2})$")
METHOD_TAG = {"trigger-count": "trigger", "combined": "combined", "continuity": "continuity"}


def nth_acquisition(acq):
    """The acq-th SW_data folder (by the timestamp in its name) of every subject that has one."""
    out = []
    for subj in sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("C")):
        sw = sorted((d for d in subj.iterdir() if d.is_dir() and "_sw_data_" in d.name.lower()),
                    key=lambda d: datetime.strptime(_TS.search(d.name).group(1), "%d-%B-%Y_%H-%M-%S")
                    if _TS.search(d.name) else datetime.max)
        if len(sw) >= acq:
            out.append(sw[acq - 1])
    return out


def iq_file(folder, buffer):
    out = folder / "output"
    if buffer == 3:
        var = out / "CombinedData_buffer3_refocus-adjoint_iq.hdf5"
        if var.exists():
            return var
    return out / f"CombinedData_buffer{buffer}_iq.hdf5"


def tile(folder, buffer, order):
    """-> (frames in playback order, label, colour)."""
    from swp.acquisition.gifs import iq_to_bmode
    from swp.acquisition.unwrap import read_flag
    subj = folder.parent.name[-3:]
    path = iq_file(folder, buffer)
    with h5py.File(path, "r") as f:
        iq = np.asarray(f[f"{_BF}/values"], np.float32)
    frames = iq_to_bmode(iq, adaptive=True)
    n = len(frames)
    if buffer == 1:
        return frames, f"C{subj} {n} fr", "white"
    rec = read_flag(path)
    if not (rec and rec.get("buffer_unwrapped")):
        return frames, f"C{subj} {n} fr stored ({rec['status'] if rec else 'not unwrapped'})", "red"
    first = int(rec["first_frame"])
    if order == "unwrapped":
        return frames, f"C{subj} {n} fr {METHOD_TAG.get(rec['method'], rec['method'])}", "white"
    idx = [(s - first) % n for s in range(n)]                      # stored slot s -> file frame
    return [frames[i] for i in idx], f"C{subj} {n} fr, wrap after slot {(first - 1) % n}", "white"


def render(tiles, cols, tile_w, title):
    h0, w0 = tiles[0][0][0].shape
    tw, th = tile_w, int(round(tile_w * h0 / w0))
    rows = int(np.ceil(len(tiles) / cols))
    W, H = cols * tw, rows * (th + LABEL_H) + TOP
    steps = max(len(t[0]) for t in tiles)
    small = [[Image.fromarray(f).resize((tw, th), Image.BILINEAR) for f in fr] for fr, _, _ in tiles]
    colours = {"white": (255, 255, 255), "red": (230, 60, 60)}
    images = []
    for k in range(steps):
        canvas = Image.new("L", (W, H), 0)
        for j, fr in enumerate(small):
            r, c = divmod(j, cols)
            canvas.paste(fr[min(k, len(fr) - 1)], (c * tw, TOP + r * (th + LABEL_H)))
        rgb = canvas.convert("RGB")
        d = ImageDraw.Draw(rgb)
        d.text((4, 2), f"{title}   frame {k}", fill=(255, 255, 255))
        for j, (fr, label, colour) in enumerate(tiles):
            r, c = divmod(j, cols)
            d.text((c * tw + 2, TOP + r * (th + LABEL_H) + th + 1), label[:34],
                   fill=(120, 120, 120) if k >= len(fr) else colours[colour])
        images.append(rgb)
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acq", type=int, default=1, help="which SW acquisition of each subject (1 = first)")
    ap.add_argument("--buffer", type=int, choices=(1, 3), required=True)
    ap.add_argument("--order", choices=("stored", "unwrapped"), default="unwrapped",
                    help="buffer 3 only; buffer 1 always plays as stored")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--tile-width", type=int, default=170)
    ap.add_argument("--delay-ms", type=int, default=40, help="per frame (buffer 3 ~39 ms = real time)")
    a = ap.parse_args()
    order = "stored" if a.buffer == 1 else a.order
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"acq{a.acq}_buffer{a.buffer}_{order}.gif"
    tiles = []
    for f in nth_acquisition(a.acq):
        try:
            tiles.append(tile(f, a.buffer, order))
            print(f"  {tiles[-1][1]}  [{tiles[-1][2]}]", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"  {f}: skipped ({type(exc).__name__}: {exc})", flush=True)
    what = ("buffer 1 widebeam, as stored" if a.buffer == 1 else
            "buffer 3 REFoCUS, ORIGINAL stored slot order (before unwrap)" if order == "stored" else
            "buffer 3 REFoCUS, chronological after unwrap (red = ambiguous, stored order)")
    title = (f"SW acquisition {a.acq}, {what}; every frame in order, not ECG-synchronised; "
             f"grey label = clip ended, holding last frame")
    images = render(tiles, a.cols, a.tile_width, title)
    images[0].save(out, save_all=True, append_images=images[1:], duration=a.delay_ms, loop=0)
    images[0].save(str(out)[:-4] + "_first.png")
    print(f"{len(tiles)} tiles, {len(images)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
