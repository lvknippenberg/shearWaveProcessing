"""Tile one frame from several GIFs into a single still PNG.

The animated montages are the right tool for motion, but for comparing reconstructions a still
is better: a fine radial pattern that the eye tracks through a moving loop is much easier to
judge frozen, side by side, at full tile resolution.

Usage:
    python frame_montage.py -o out.png --frame 18 --cols 4 \\
        --labels "a,b,c" gif1 gif2 gif3
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

LABEL_H = 22


def load_frame(path, idx):
    im = Image.open(path)
    n = getattr(im, "n_frames", 1)
    if idx >= n:
        raise SystemExit(f"{os.path.basename(path)} has only {n} frames (asked for {idx})")
    im.seek(idx)
    return im.convert("L"), n


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gifs", nargs="+")
    p.add_argument("-o", "--out", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile-width", type=int, default=500)
    p.add_argument("--labels", default=None, help="comma-separated, one per gif")
    a = p.parse_args()

    labels = a.labels.split(",") if a.labels else [os.path.basename(g) for g in a.gifs]
    if len(labels) != len(a.gifs):
        raise SystemExit(f"{len(labels)} labels for {len(a.gifs)} gifs")

    tiles = []
    for g, lab in zip(a.gifs, labels):
        fr, n = load_frame(g, a.frame)
        w = a.tile_width
        h = round(fr.height * w / fr.width)
        tiles.append(fr.resize((w, h), Image.LANCZOS))
        print(f"  {os.path.basename(g):46s} {n:3d} fr -> frame {a.frame}  [{lab}]")

    tw = a.tile_width
    th = max(t.height for t in tiles)
    cols = a.cols
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("L", (cols * tw, rows * (th + LABEL_H)), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    for i, (t, lab) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, cols)
        y = r * (th + LABEL_H)
        canvas.paste(t, (c * tw, y))
        draw.text((c * tw + 6, y + th + 3), lab, fill=255, font=font)

    canvas.save(a.out)
    print(f"wrote {a.out}  ({canvas.size[0]}x{canvas.size[1]}, {rows}x{cols})")


if __name__ == "__main__":
    main()
