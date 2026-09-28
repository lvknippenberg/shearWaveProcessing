"""Buffer 3 of one folder played with EVERY possible head: one GIF per head + one grid (2026-09-28).

For a visual check of the unwrap: head h = the stored slot shown first, so GIF h plays stored slots
h, h+1, ..., h-1 (cyclic). Exactly one head puts the frames in acquisition order; every other head
has the newest->oldest wrap jump somewhere inside the sequence. Labels: the head, the head chosen by
the unwrap (*), and the weakest frame-to-frame link inside the sequence (low-pass de-meaned log
envelope, relative to the median link; the GIF loop point is excluded).

    python study/analysis/all_heads_gifs.py --folder "Z:/raw_data/C000000029/VIS-031_SW_data_24-July-2026_11-13-37"
-> study/montages/playback/<subject>_<folder>_all_heads/head_XX.gif, all_heads_grid.gif, links.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("KERAS_BACKEND", "torch")

import h5py                                                     # noqa: E402
import hdf5plugin                                               # noqa: E402,F401
from PIL import Image, ImageDraw                                # noqa: E402

_BF = "tracks/track_0/data/beamformed_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--width", type=int, default=420, help="single-GIF width (px)")
    ap.add_argument("--grid-width", type=int, default=170)
    ap.add_argument("--delay-ms", type=int, default=40)
    a = ap.parse_args()
    from swp.acquisition.gifs import iq_to_bmode
    from swp.acquisition.unwrap import read_flag
    folder = Path(a.folder)
    path = folder / "output" / "CombinedData_buffer3_refocus-adjoint_iq.hdf5"
    if not path.exists():
        path = folder / "output" / "CombinedData_buffer3_iq.hdf5"
    with h5py.File(path, "r") as f:
        iq = np.asarray(f[f"{_BF}/values"], np.float32)
    frames = iq_to_bmode(iq, adaptive=True)
    n = len(frames)
    rec = read_flag(path)
    cur = int(rec["first_frame"]) if rec and rec.get("buffer_unwrapped") else 0   # stored slot at file frame 0
    chosen = cur if rec and rec.get("buffer_unwrapped") else None
    stored = [frames[(s - cur) % n] for s in range(n)]           # back to stored slot order

    # motion-sensitive link measure on stored order (from the unwrap_explore cache when present)
    from unwrap_explore_eval import CACHE, feats
    npz = CACHE / f"{folder.parent.name}__{folder.name}.npz"
    X = feats(np.load(npz), "L", True)[1] if npz.exists() else None
    link = np.array([X[i] @ X[(i + 1) % n] for i in range(n)]) if X is not None else None
    med = np.median(link) if link is not None else 1.0

    out = _REPO / "study" / "montages" / "playback" / f"{folder.parent.name[-3:]}_{folder.name}_all_heads"
    out.mkdir(parents=True, exist_ok=True)
    h0, w0 = stored[0].shape
    rows_csv = ["head,chosen,weakest_inside_link,at_step"]
    grid_tiles = []
    for h in range(n):
        seq = [stored[(h + k) % n] for k in range(n)]
        if link is not None:
            inside = [link[(h + k) % n] / med for k in range(n - 1)]
            k_min = int(np.argmin(inside))
            info = f"weakest link {inside[k_min]:+.2f} at {k_min}->{k_min + 1}"
            rows_csv.append(f"{h},{int(h == chosen)},{inside[k_min]:.3f},{k_min}")
        else:
            info = ""
        tag = f"head {h:2d}{' *unwrap' if h == chosen else ''}"
        w, hh = a.width, int(round(a.width * h0 / w0))
        imgs = []
        for k, fr in enumerate(seq):
            im = Image.fromarray(fr).resize((w, hh), Image.BILINEAR).convert("RGB")
            d = ImageDraw.Draw(im)
            d.text((4, 2), f"{folder.parent.name} {folder.name[-19:]}  {tag}", fill=(255, 255, 255))
            d.text((4, 14), f"frame {k:2d} = stored slot {(h + k) % n:2d}   {info}", fill=(255, 220, 120))
            imgs.append(im)
        imgs[0].save(out / f"head_{h:02d}.gif", save_all=True, append_images=imgs[1:], duration=a.delay_ms, loop=0)
        grid_tiles.append((seq, tag, info, h == chosen))

    tw, th, lab, cols = a.grid_width, int(round(a.grid_width * h0 / w0)), 26, 8
    rows = int(np.ceil(n / cols))
    small = [[Image.fromarray(f).resize((tw, th), Image.BILINEAR) for f in seq] for seq, _, _, _ in grid_tiles]
    grid = []
    for k in range(n):
        canvas = Image.new("RGB", (cols * tw, rows * (th + lab) + 18))
        d = ImageDraw.Draw(canvas)
        d.text((4, 2), f"{folder.parent.name} {folder.name}: buffer 3 with every possible head (head = stored slot "
                       f"shown first; * = unwrap's choice)   frame {k}", fill=(255, 255, 255))
        for j, (seq, tag, info, ch) in enumerate(grid_tiles):
            r, c = divmod(j, cols)
            x, y = c * tw, 18 + r * (th + lab)
            canvas.paste(small[j][k], (x, y))
            col = (120, 230, 120) if ch else (255, 255, 255)
            d.text((x + 2, y + th + 1), tag, fill=col)
            d.text((x + 2, y + th + 12), info.replace("weakest link ", "min "), fill=(255, 220, 120))
        grid.append(canvas)
    grid[0].save(out / "all_heads_grid.gif", save_all=True, append_images=grid[1:], duration=a.delay_ms, loop=0)
    (out / "links.csv").write_text("\n".join(rows_csv) + "\n")
    print(f"{n} head GIFs + grid -> {out}")
    print("\n".join(rows_csv))


if __name__ == "__main__":
    main()
