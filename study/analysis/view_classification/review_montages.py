"""One review-montage GIF per patient (to confirm the view labels with a colleague), ranked by confidence.

Each GIF is the review screen as a movie: every SW acquisition of the patient in acquisition order,
its buffer-3 loop playing in real time (25.4 Hz), with the reviewed label above it.

Confidence (independent of the review, so a colleague's attention goes where the model disagrees):
  per loop   probability that the leave-one-subject-out supervised head (``supervised_probe.py``,
             trained WITHOUT this patient) gives the reviewed label; Unclear counts as 0.5, Apical
             uses EchoPrime's apical probability.
  per patient  mean over its loops (the minimum is shown too).
Files are ranked from least to most confident: ``01_<subject>_conf0.71.gif`` is the one to check
first. A loop whose confidence is < 0.5 (the model disagrees with the label) gets a thick red frame.

Usage: python review_montages.py [--out DIR] [--subject C000000007 ...]
Writes the GIFs + index.csv to --out (default D:\\Luuk van Knippenberg\\Claude\\view_classification\\review_montages).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from swp.views import LOGS                     # noqa: E402
from swp.views.review import load_loop         # noqa: E402

OUT = Path(r"D:\Luuk van Knippenberg\Claude\view_classification\review_montages")
TILE_H, HEAD, GAP, NCOLS, TOP = 170, 34, 8, 6, 58
FRAME_MS = 40                                   # buffer-3 frame interval 39.4 ms (GIF delay unit 10 ms)
N_GRAY = 240
COLORS = {"PLAX": (42, 120, 214), "PSAX": (235, 104, 52), "Apical": (31, 158, 110),
          "Unclear": (150, 150, 145), "white": (255, 255, 255), "red": (214, 39, 40),
          "dim": (175, 175, 170), "bg": (16, 16, 16)}
IDX = {k: N_GRAY + i for i, k in enumerate(COLORS)}
GRAY_LUT = (np.arange(256) * (N_GRAY - 1) // 255).astype(np.uint8)


def palette():
    pal = []
    for i in range(N_GRAY):
        g = round(i * 255 / (N_GRAY - 1))
        pal += [g, g, g]
    for c in COLORS.values():
        pal += list(c)
    return pal + [0] * (768 - len(pal))


def font(size, bold=False):
    for name in (("segoeuib.ttf" if bold else "segoeui.ttf"), "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def confidence(rows: pd.DataFrame) -> np.ndarray:
    p = rows.p_plax_sup.to_numpy(float)
    c = np.where(rows.label == "PLAX", p, 1 - p)
    c = np.where(rows.label == "Unclear", 0.5, c)
    return np.where(rows.label == "Apical", rows.p_apical.to_numpy(float), c)


def render(rows: pd.DataFrame, loops: list, rank: int, n_subj: int, score: float, path: Path):
    n = len(loops)
    ncols = min(NCOLS, n)
    nrows = int(np.ceil(n / ncols))
    tw = max(v.shape[2] for v in loops)
    cw, ch = tw + GAP, HEAD + TILE_H + GAP
    W, H = ncols * cw + GAP, TOP + nrows * ch
    static = Image.new("P", (W, H), IDX["bg"])
    static.putpalette(palette())
    dr = ImageDraw.Draw(static)
    subj = rows.subject.iloc[0]
    counts = rows.label.value_counts()
    dr.text((GAP, 6), f"#{rank:02d}/{n_subj}   {subj}   confidence {score:.2f}  (lowest loop {rows.conf.min():.2f})",
            fill=IDX["white"], font=font(17, bold=True))
    x = GAP
    for lab in ("PLAX", "PSAX", "Apical", "Unclear"):
        txt = f"{lab} {int(counts.get(lab, 0))}"
        dr.text((x, 32), txt, fill=IDX[lab], font=font(14, bold=True))
        x += int(dr.textlength(txt, font=font(14, bold=True))) + 18
    dr.text((x, 32), "|  per loop: reviewed label, model confidence in it  |  red frame = model disagrees (< 0.5)",
            fill=IDX["dim"], font=font(13))
    slots = []
    f_t, f_s = font(13, bold=True), font(12)
    for i, (v, (_, r)) in enumerate(zip(loops, rows.iterrows())):
        rr, cc = divmod(i, ncols)
        w = v.shape[2]
        x0 = GAP + cc * cw + (tw - w) // 2
        y0 = TOP + rr * ch + HEAD
        slots.append((y0, x0, w))
        disagree = r.conf < 0.5
        dr.text((x0 + w / 2, y0 - 31), f"{r.folder.split('_')[-1]}   {r.label}", fill=IDX[r.label], font=f_t, anchor="ma")
        dr.text((x0 + w / 2, y0 - 15), f"confidence {r.conf:.2f}", fill=IDX["red" if disagree else "dim"],
                font=f_s, anchor="ma")
        lw = 4 if disagree else 2
        dr.rectangle((x0 - lw, y0 - lw, x0 + w - 1 + lw, y0 + TILE_H - 1 + lw),
                     outline=IDX["red" if disagree else r.label], width=lw)
    base = np.asarray(static).copy()
    frames = []
    for k in range(max(len(v) for v in loops)):
        a = base.copy()
        for (y0, x0, w), v in zip(slots, loops):
            a[y0:y0 + TILE_H, x0:x0 + w] = GRAY_LUT[v[k % len(v)]]
        im = Image.fromarray(a, "P")
        im.putpalette(palette())
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=FRAME_MS, loop=0, optimize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--subject", action="append", default=[])
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(LOGS / "sw_views_manual.csv")
    probe = pd.read_csv(LOGS / "sw_views_probe.csv")[["folder", "p_plax_sup", "p_apical"]]
    gifs = dict(zip(*pd.read_csv(LOGS / "all_sw_views.csv")[["folder", "gif"]].T.values))
    d = man.merge(probe, on="folder", how="left")
    d["time"] = d.folder.str.split("_").str[-1]
    d = d.sort_values(["subject", "time"]).drop(columns="time")
    d["conf"] = confidence(d)
    score = d.groupby("subject").conf.mean().sort_values()
    index = pd.DataFrame({"rank": np.arange(1, len(score) + 1), "subject": score.index, "confidence": score.values.round(3),
                          "lowest_loop": d.groupby("subject").conf.min()[score.index].values.round(3),
                          "n_loops": d.groupby("subject").size()[score.index].values})
    index["file"] = [f"{r:02d}_{s}_conf{c:.2f}.gif" for r, s, c in zip(index["rank"], index.subject, index.confidence)]
    for old in out.glob("*_C0*_conf*.gif"):                  # ranks shift when labels change
        if old.name not in set(index.file) and (not a.subject or old.name.split("_")[1] in a.subject):
            old.unlink()
    index.to_csv(out / "index.csv", index=False)
    index.to_csv(LOGS / "review_montages_index.csv", index=False)
    todo = index[index.subject.isin(a.subject)] if a.subject else index
    for _, r in todo.iterrows():
        rows = d[d.subject == r.subject].reset_index(drop=True)
        loops = [load_loop(gifs[f], h=TILE_H) for f in rows.folder]
        render(rows, loops, int(r["rank"]), len(index), float(r.confidence), out / r.file)
        print(f"{r.file}  {(out / r.file).stat().st_size / 1e6:.1f} MB", flush=True)
    print(out / "index.csv")


if __name__ == "__main__":
    main()
