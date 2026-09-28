"""Buffer 3 before/after the unwrap, paired per subject (2026-09-28).

Crops each subject's tile from acq<N>_buffer3_stored.gif and acq<N>_buffer3_unwrapped.gif (made by
buffer_playback_montage.py: same subject order and geometry, every frame in file order, not
ECG-synchronised) and lays them out as [stored | unwrapped] pairs, 4 pairs per row.

    python study/analysis/buffer_playback_pair.py --acq 1
-> study/montages/playback/acq<N>_buffer3_stored_vs_unwrapped.gif
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageSequence

D = Path(__file__).resolve().parents[2] / "study" / "montages" / "playback"
COLS, TW, TOP, PAIRS = 8, 170, 18, 4


def frames(p):
    return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(p))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acq", type=int, default=1)
    ap.add_argument("--n", type=int, required=True, help="number of tiles (subjects) in the montages")
    a = ap.parse_args()
    s, u = frames(D / f"acq{a.acq}_buffer3_stored.gif"), frames(D / f"acq{a.acq}_buffer3_unwrapped.gif")
    cell = (s[0].height - TOP) // int(np.ceil(a.n / COLS))
    W, H = PAIRS * (2 * TW + 10), TOP + int(np.ceil(a.n / PAIRS)) * cell
    out = []
    for k in range(len(s)):
        canvas = Image.new("RGB", (W, H))
        for j in range(a.n):
            r, c = divmod(j, COLS)
            box = (c * TW, TOP + r * cell, (c + 1) * TW, TOP + (r + 1) * cell)
            pr, pc = divmod(j, PAIRS)
            x0, y0 = pc * (2 * TW + 10), TOP + pr * cell
            canvas.paste(s[k].crop(box), (x0, y0))
            canvas.paste(u[k].crop(box), (x0 + TW, y0))
        ImageDraw.Draw(canvas).text((4, 2), f"SW acquisition {a.acq}, buffer 3: stored slot order (left) | unwrapped "
                                            f"(right); every frame in order, not ECG-synchronised   frame {k}",
                                    fill=(255, 255, 255))
        out.append(canvas)
    dst = D / f"acq{a.acq}_buffer3_stored_vs_unwrapped.gif"
    out[0].save(dst, save_all=True, append_images=out[1:], duration=40, loop=0)
    print(dst, f"{dst.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
