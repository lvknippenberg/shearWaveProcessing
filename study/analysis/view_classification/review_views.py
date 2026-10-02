"""Review the label-free PLAX/PSAX sorting, one subject per screen, every loop playing.

Shows all SW acquisitions of a subject in acquisition order, each buffer-3 loop animating, with the
proposed view. Loops the four label-free voters disagree on (``label_free_sort.py``) have a thick
red frame - those need a decision; the rest are unanimous and only need a glance.

=====================  =====================================================================
mouse / key            action
=====================  =====================================================================
click a loop           cycle its label  PLAX -> PSAX -> Apical -> Unclear
1 / 2 / 3 / 4          set the loop under the mouse to PLAX / PSAX / Apical / Unclear
ENTER                  accept this subject as shown (saved) and go to the next
b                      back to the previous subject
space                  pause / play
q                      save what is accepted and quit
=====================  =====================================================================

Unclear = off-axis / poor window / cannot tell; it is kept out of training and sorting.
Subjects come in ID order (C1, C2, ...); every accepted subject is saved at once, so stop whenever
you like. Re-running resumes; accepted subjects are skipped unless --all is given.

Speed (it runs over remote desktop on a loaded machine): plain Tk, no matplotlib. Each loop's frames
become Tk images once per subject; playing only swaps which pre-made image a tile shows. The next
subject's GIFs are read from Z: in the background while the current one is reviewed.

Usage:
  python review_views.py                       # study defaults
  python review_views.py --subject C000000020  # one subject (also if already accepted)
Labels: study/logs/view_classification/sw_views_manual.csv (one row per acquisition).
"""
from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[1] / "logs" / "view_classification"
LABELS = ("PLAX", "PSAX", "Apical", "Unclear")
COL = {"PLAX": "#2a78d6", "PSAX": "#eb6834", "Apical": "#1f9e6e", "Unclear": "#8a8983"}
REVIEW_EDGE = "#d62728"
TILE_H = 190                       # px height a loop is shown at
HEAD = 30                          # px strip above each loop for its title
GAP = 6                            # px between tiles
NCOLS = 6
FPS = 12


def load_loop(gif, h=TILE_H):
    """Buffer-3 GIF frames, black side margins cropped, resized to height ``h`` (uint8, N x h x w)."""
    im = Image.open(gif)
    frames = []
    try:
        while True:
            frames.append(np.asarray(im.convert("L")))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    v = np.stack(frames)
    cols = np.where(v.max(axis=(0, 1)) > 8)[0]
    v = v[:, :, cols.min():cols.max() + 1] if len(cols) else v
    w = int(round(v.shape[2] * h / v.shape[1]))
    return np.stack([np.asarray(Image.fromarray(f).resize((w, h), Image.BILINEAR)) for f in v])


def block_fit(score):
    """Protocol prior: one PLAX block, then one PSAX block. ``score`` > 0 leans PLAX; returns the
    PLAX/PSAX label of the best single change point."""
    s = np.asarray(score, dtype=float)
    k = int(np.argmax([s[:j].sum() - s[j:].sum() for j in range(len(s) + 1)]))
    return np.array(["PLAX"] * k + ["PSAX"] * (len(s) - k))


def proposals(cons: pd.DataFrame) -> pd.DataFrame:
    """Proposed label per loop: the consensus where unanimous, else the vote majority; 2-2 ties
    take the subject's protocol block fit (PLAX block then PSAX block) over the vote scores.
    EchoPrime apical -> Apical."""
    cons = cons.copy()
    cons["time"] = cons.folder.str.split("_").str[-1]
    cons = cons.sort_values(["subject", "time"]).drop(columns="time").reset_index(drop=True)
    votes = cons[["ep", "clu_f", "clu_v", "self"]]
    n_plax = (votes == "PLAX").sum(axis=1)
    block = np.concatenate([block_fit((n_plax[g.index] - 2) / 2.0) for _, g in cons.groupby("subject", sort=False)])
    major = np.where(n_plax > 2, "PLAX", np.where(n_plax < 2, "PSAX", block))
    prop = np.where(cons.consensus != "review", cons.consensus, major)
    prop = np.where(cons.review_reason == "apical?", "Apical", prop)
    out = cons[["subject", "folder"]].copy()
    out["proposed"] = prop
    out["needs_review"] = (cons.consensus == "review").values
    out["reason"] = cons.review_reason.fillna("").values
    out["votes"] = votes.apply(lambda r: " ".join(x[1] if x == "PLAX" else "S" for x in r), axis=1).values
    return out


class TkReview:
    """The review window (plain Tk, one for the whole session). Each loop's frames become Tk images
    once per subject; playing only swaps which pre-made image a tile shows, so a tick costs ~nothing
    even on a loaded machine over remote desktop."""

    def __init__(self):
        import tkinter as tk
        self.tk = tk
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
        from PIL import Image, ImageTk
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
        self.cv.itemconfigure(self.texts[i], fill=COL[lab],
                              text=f"{r.folder.split('_')[-1]}  {lab}{' *' if changed else ''}\n"
                                   f"votes {r.votes}{'  ' + r.reason if r.reason else ''}")
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


def save(manual_csv: Path, sub: pd.DataFrame, labels):
    sub = sub[["subject", "folder", "proposed", "needs_review", "votes"]].copy()
    sub["label"] = labels
    sub["changed"] = sub.label != sub.proposed
    sub["reviewed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    old = pd.read_csv(manual_csv) if manual_csv.is_file() else pd.DataFrame(columns=sub.columns)
    old = old[old.subject != sub.subject.iloc[0]]
    pd.concat([old, sub]).sort_values(["subject", "folder"]).to_csv(manual_csv, index=False)


def subject_rows(prop, s, done):
    sub = prop[prop.subject == s].reset_index(drop=True)          # already in acquisition order
    prev = done[done.subject == s].set_index("folder").label if done is not None else None
    sub["label"] = [prev.get(f, p) if prev is not None else p for f, p in zip(sub.folder, sub.proposed)]
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--consensus", default=str(LOGS / "sw_views_consensus.csv"))
    ap.add_argument("--views", default=str(LOGS / "all_sw_views.csv"), help="for the GIF paths")
    ap.add_argument("--out", default=str(LOGS / "sw_views_manual.csv"))
    ap.add_argument("--subject", action="append", default=[])
    ap.add_argument("--all", action="store_true", help="also show subjects already accepted")
    a = ap.parse_args()

    cons = pd.read_csv(a.consensus)
    gifs = dict(zip(*pd.read_csv(a.views)[["folder", "gif"]].T.values))
    prop = proposals(cons)
    out = Path(a.out)
    done = pd.read_csv(out) if out.is_file() else None

    flagged = prop.groupby("subject").needs_review.sum()
    subjects = a.subject or sorted(flagged.index)
    if not a.subject and done is not None and not a.all:
        subjects = [s for s in subjects if s not in set(done.subject)]
    print(f"{len(subjects)} subject(s) to review "
          f"({int((flagged[subjects] > 0).sum())} with flagged loops, {int(flagged[subjects].sum())} flagged loops)")

    pool = ThreadPoolExecutor(max_workers=2)
    pending = {}

    def fetch(s):                                                 # background read of a subject's GIFs
        if s not in pending:
            folders = list(prop.folder[prop.subject == s])
            pending[s] = pool.submit(lambda fs: [load_loop(gifs[f]) for f in fs], folders)
        return pending[s]

    ui = TkReview()
    i = 0
    while 0 <= i < len(subjects):
        s = subjects[i]
        loops = fetch(s).result()
        if i + 1 < len(subjects):
            fetch(subjects[i + 1])
        sub = subject_rows(prop, s, done)
        title = f"{s}  ({i + 1}/{len(subjects)}, {int(sub.needs_review.sum())} flagged)"
        print(title, flush=True)
        result, labels = ui.show(sub, loops, title)
        if result == "accept":
            save(out, sub, labels)
            done = pd.read_csv(out)
            i += 1
        elif result == "back":
            i = max(i - 1, 0)
        else:
            break
    ui.close()
    pool.shutdown(wait=False, cancel_futures=True)
    print(f"labels: {out}")


if __name__ == "__main__":
    main()
