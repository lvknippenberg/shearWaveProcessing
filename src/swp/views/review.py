"""Review window for the view labels: one subject per screen, every buffer-3 loop playing.

=====================  =====================================================================
mouse / key            action
=====================  =====================================================================
click a loop           cycle its label  PLAX -> PSAX -> Apical -> Unclear
1 / 2 / 3 / 4          set the loop under the mouse to PLAX / PSAX / Apical / Unclear
ENTER                  accept this subject as shown (saved at once) and go to the next
b                      back to the previous subject
space                  pause / play
q                      quit (accepted subjects are already saved)
=====================  =====================================================================

Loops the voters disagree on (``swp.views.vote``) have a thick red frame; the title of each loop
shows the label, ``*`` if it differs from the proposal, and the votes (ep clu_f clu_v sup; L = PLAX,
S = PSAX). Unclear = off-axis / poor window / cannot tell.

Plain Tk (no matplotlib): each loop's frames become Tk images once per subject and playing only
swaps which pre-made image a tile shows (~13 ms per tick on the loaded server; matplotlib redrew
the figure in ~220 ms). The next subject's GIFs are read in the background.
"""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import LABELS

COL = {"PLAX": "#2a78d6", "PSAX": "#eb6834", "Apical": "#1f9e6e", "Unclear": "#8a8983"}
REVIEW_EDGE = "#d62728"
TILE_H, HEAD, GAP, NCOLS, FPS = 190, 30, 6, 6, 12
LABEL_COLUMNS = ["subject", "folder", "proposed", "needs_review", "votes", "label", "changed",
                 "source", "reviewed_at"]


def load_loop(gif, h=TILE_H) -> np.ndarray:
    """Buffer-3 GIF frames, black side margins cropped, resized to height ``h`` (uint8, N x h x w)."""
    from .echoprime import gif_frames
    v = gif_frames(gif)
    cols = np.where(v.max(axis=(0, 1)) > 8)[0]
    v = v[:, :, cols.min():cols.max() + 1] if len(cols) else v
    w = int(round(v.shape[2] * h / v.shape[1]))
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in v])


def read_labels(path: Path) -> pd.DataFrame:
    if not Path(path).is_file():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    d = pd.read_csv(path)
    if "source" not in d:
        d["source"] = "review"
    return d


def save_subject(path: Path, sub: pd.DataFrame, labels, source="review"):
    """Replace one subject's rows in the labels table (columns ``LABEL_COLUMNS``)."""
    rows = sub[["subject", "folder", "proposed", "needs_review", "votes"]].copy()
    rows["label"] = list(labels)
    rows["changed"] = rows.label != rows.proposed
    rows["source"] = source
    rows["reviewed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    old = read_labels(path)
    old = old[old.subject != rows.subject.iloc[0]]
    out = pd.concat([old, rows])[LABEL_COLUMNS].sort_values(["subject", "folder"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


class TkReview:
    """The review window (one for the whole session)."""

    def __init__(self):
        import tkinter as tk
        self.root = tk.Tk()
        self.root.title("SW view review")
        self.header = tk.Label(self.root, font=("Segoe UI", 10), justify="center")
        self.header.pack(side="top", fill="x")
        self.cv = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.cv.pack(side="top")
        self.cv.bind("<Button-1>", self.on_click)
        self.cv.bind("<Motion>", self.on_motion)
        for k in ("1", "2", "3", "4"):
            self.root.bind(k, self.on_number)
        self.root.bind("<Return>", lambda e: self._finish("accept"))
        self.root.bind("b", lambda e: self._finish("back"))
        self.root.bind("q", lambda e: self._finish("quit"))
        self.root.bind("<space>", lambda e: setattr(self, "playing", not self.playing))
        self.root.protocol("WM_DELETE_WINDOW", lambda: self._finish("quit"))
        self.hover, self.after_id = None, None
        self.done = tk.StringVar(master=self.root)

    def show(self, rows: pd.DataFrame, loops: list, title: str):
        """Review one subject; returns (result, labels), result in accept / back / quit."""
        from PIL import ImageTk
        cv = self.cv
        cv.delete("all")
        self.rows = rows.reset_index(drop=True)
        self.labels = list(self.rows.label)
        n = len(loops)
        self.ncols = min(NCOLS, n)
        nrows = int(np.ceil(n / self.ncols))
        tw = max(v.shape[2] for v in loops)
        self.cell_w, self.cell_h = tw + GAP, HEAD + TILE_H + GAP
        cv.config(width=self.ncols * self.cell_w, height=nrows * self.cell_h)
        self.header.config(text=title + "\nclick: cycle label  |  1-4: PLAX / PSAX / Apical / Unclear under "
                                        "the mouse  |  ENTER: accept  |  b: back  |  space: pause  |  q: quit")
        self.frames, self.items, self.texts, self.boxes = [], [], [], []
        for i, v in enumerate(loops):
            r, c = divmod(i, self.ncols)
            x0 = c * self.cell_w + GAP // 2 + (tw - v.shape[2]) // 2
            y0 = r * self.cell_h + HEAD
            self.frames.append([ImageTk.PhotoImage(Image.fromarray(f), master=self.root) for f in v])
            self.items.append(cv.create_image(x0, y0, anchor="nw", image=self.frames[-1][0]))
            self.boxes.append(cv.create_rectangle(x0 - 2, y0 - 2, x0 + v.shape[2] + 1, y0 + TILE_H + 1))
            self.texts.append(cv.create_text(x0 + v.shape[2] / 2, y0 - 3, anchor="s", justify="center",
                                             font=("Segoe UI", 8)))
            self._style(i)
        self.tick_n, self.playing, self.result = 0, True, None
        self._tick()
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.done.set("")
        self.root.wait_variable(self.done)                # Tk event loop until accept / back / quit
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        return self.result, self.labels

    def _style(self, i):
        r, lab = self.rows.iloc[i], self.labels[i]
        changed = lab != r.proposed
        reason = r.reason if isinstance(r.reason, str) and r.reason else ""
        self.cv.itemconfigure(self.texts[i], fill=COL[lab],
                              text=f"{r.folder.split('_')[-1]}  {lab}{' *' if changed else ''}\n"
                                   f"votes {r.votes}{'  ' + reason if reason else ''}")
        self.cv.itemconfigure(self.boxes[i], outline=REVIEW_EDGE if r.needs_review else COL[lab],
                              width=4 if r.needs_review else 2)

    def _tick(self):
        if self.playing:
            self.tick_n += 1
            for item, fr in zip(self.items, self.frames):
                self.cv.itemconfigure(item, image=fr[self.tick_n % len(fr)])
        self.after_id = self.root.after(1000 // FPS, self._tick)

    def _tile_at(self, x, y):
        c, r = int(x // self.cell_w), int(y // self.cell_h)
        i = r * self.ncols + c
        return i if 0 <= c < self.ncols and 0 <= i < len(self.items) else None

    def on_motion(self, e):
        self.hover = self._tile_at(e.x, e.y)

    def on_click(self, e):
        i = self._tile_at(e.x, e.y)
        if i is not None:
            self._set(i, LABELS[(LABELS.index(self.labels[i]) + 1) % len(LABELS)])

    def on_number(self, e):
        if self.hover is not None:
            self._set(self.hover, LABELS[int(e.keysym) - 1])

    def _set(self, i, lab):
        self.labels[i] = lab
        self._style(i)

    def _finish(self, result):
        if self.result is None:
            self.result = result
            self.done.set(result)

    def close(self):
        try:
            self.root.destroy()
        except Exception:                                   # noqa: BLE001 - already gone
            pass


def run(votes: pd.DataFrame, gifs: dict, labels_csv: Path, subjects: list):
    """Review ``subjects`` in order. ``votes``: output of ``vote.vote`` (+ folder order);
    ``gifs``: folder -> GIF path. Existing labels pre-fill the window."""
    pool = ThreadPoolExecutor(max_workers=2)
    pending = {}

    def fetch(s):                                         # background read of a subject's GIFs
        if s not in pending:
            folders = list(votes.folder[votes.subject == s])
            pending[s] = pool.submit(lambda fs: [load_loop(gifs[f]) for f in fs], folders)
        return pending[s]

    ui = TkReview()
    i = 0
    try:
        while 0 <= i < len(subjects):
            s = subjects[i]
            loops = fetch(s).result()
            if i + 1 < len(subjects):
                fetch(subjects[i + 1])
            sub = votes[votes.subject == s].reset_index(drop=True)
            prev = read_labels(labels_csv)
            prev = prev[prev.subject == s].set_index("folder").label
            sub["label"] = [prev.get(f, p) for f, p in zip(sub.folder, sub.proposed)]
            title = f"{s}  ({i + 1}/{len(subjects)}, {int(sub.needs_review.sum())} flagged)"
            print(title, flush=True)
            result, labels = ui.show(sub, loops, title)
            if result == "accept":
                save_subject(labels_csv, sub, labels)
                i += 1
            elif result == "back":
                i = max(i - 1, 0)
            else:
                break
    finally:
        ui.close()
        pool.shutdown(wait=False, cancel_futures=True)
