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
Subjects with something to decide come first, then the unanimous ones (stop whenever you like:
every accepted subject is saved at once). Re-running resumes; accepted subjects are skipped
unless --all is given.

Usage:
  python review_views.py                       # study defaults
  python review_views.py --subject C000000020  # one subject (also if already accepted)
Labels: study/logs/view_classification/sw_views_manual.csv (one row per acquisition).
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
LOGS = HERE.parents[1] / "logs" / "view_classification"
LABELS = ("PLAX", "PSAX", "Apical", "Unclear")
COL = {"PLAX": "#2a78d6", "PSAX": "#eb6834", "Apical": "#1f9e6e", "Unclear": "#8a8983"}
REVIEW_EDGE = "#d62728"
TILE_H = 200                       # px height a loop is shown at
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


class SubjectReview:
    def __init__(self, rows: pd.DataFrame, gifs: dict, title: str):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.rows = rows.reset_index(drop=True)
        self.labels = list(self.rows.label)
        self.loops = [load_loop(gifs[f]) for f in self.rows.folder]
        n = len(self.rows)
        self.ncols = min(6, n)
        nrows = int(np.ceil(n / self.ncols))
        w = max(v.shape[2] for v in self.loops)
        self.fig, axs = plt.subplots(nrows, self.ncols, figsize=(self.ncols * w / 90, nrows * (TILE_H + 75) / 90),
                                     squeeze=False)
        self.axes = list(axs.ravel())
        for ax in self.axes:
            ax.set_axis_off()
        self.ims, self.frame, self.playing, self.result = [], 0, True, None
        for i, (ax, v) in enumerate(zip(self.axes, self.loops)):
            self.ims.append(ax.imshow(v[0], cmap="gray", vmin=0, vmax=255))
            ax.set_axis_on()
            ax.set_xticks([]); ax.set_yticks([])
            self._style(i)
        self.fig.suptitle(title + "\nclick: cycle label | 1-4: PLAX/PSAX/Apical/Unclear under mouse | "
                          "ENTER: accept | b: back | space: pause | q: quit", fontsize=9)
        self.fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.5)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", lambda e: self._finish(self.result or "quit"))
        from matplotlib.animation import FuncAnimation
        self.anim = FuncAnimation(self.fig, self._step, interval=1000 // FPS, blit=False, cache_frame_data=False)

    def _style(self, i):
        r, lab, ax = self.rows.iloc[i], self.labels[i], self.axes[i]
        changed = lab != r.proposed
        ax.set_title(f"{r.folder.split('_')[-1]}  {lab}{' *' if changed else ''}\n"
                     f"votes {r.votes}{'  ' + r.reason if r.reason else ''}", fontsize=8, color=COL[lab])
        for s in ax.spines.values():
            s.set_edgecolor(REVIEW_EDGE if r.needs_review else COL[lab])
            s.set_linewidth(4 if r.needs_review else 1.5)

    def _step(self, _):
        if self.playing:
            self.frame += 1
            for im, v in zip(self.ims, self.loops):
                im.set_data(v[self.frame % len(v)])
        return self.ims

    def _tile(self, event):
        return self.axes.index(event.inaxes) if event.inaxes in self.axes[:len(self.loops)] else None

    def on_click(self, event):
        i = self._tile(event)
        if i is not None and event.button == 1:
            self.labels[i] = LABELS[(LABELS.index(self.labels[i]) + 1) % len(LABELS)]
            self._style(i)
            self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ("1", "2", "3", "4"):
            i = self._tile(event)
            if i is not None:
                self.labels[i] = LABELS[int(event.key) - 1]
                self._style(i)
                self.fig.canvas.draw_idle()
        elif event.key == "enter":
            self._finish("accept")
        elif event.key == "b":
            self._finish("back")
        elif event.key == "q":
            self._finish("quit")
        elif event.key == " ":
            self.playing = not self.playing

    def _finish(self, result):
        if self.result is None:
            self.result = result
            self.anim.event_source.stop()
            self.plt.close(self.fig)

    def run(self):
        self.plt.show()
        return self.result or "quit", self.labels


def save(manual_csv: Path, sub: pd.DataFrame, labels):
    sub = sub[["subject", "folder", "proposed", "needs_review", "votes"]].copy()
    sub["label"] = labels
    sub["changed"] = sub.label != sub.proposed
    sub["reviewed_at"] = dt.datetime.now().isoformat(timespec="seconds")
    old = pd.read_csv(manual_csv) if manual_csv.is_file() else pd.DataFrame(columns=sub.columns)
    old = old[old.subject != sub.subject.iloc[0]]
    pd.concat([old, sub]).sort_values(["subject", "folder"]).to_csv(manual_csv, index=False)


def main():
    import matplotlib
    matplotlib.use("TkAgg", force=True)
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

    order = prop.groupby("subject").needs_review.sum().sort_index()
    subjects = list(order[order > 0].index) + list(order[order == 0].index)
    if a.subject:
        subjects = a.subject
    elif done is not None and not a.all:
        subjects = [s for s in subjects if s not in set(done.subject)]
    print(f"{len(subjects)} subject(s) to review "
          f"({int((order[subjects] > 0).sum())} with flagged loops, {int(order[subjects].sum())} flagged loops)")

    i = 0
    while 0 <= i < len(subjects):
        s = subjects[i]
        sub = prop[prop.subject == s].copy()
        sub["time"] = sub.folder.str.split("_").str[-1]
        sub = sub.sort_values("time").drop(columns="time").reset_index(drop=True)
        prev = done[done.subject == s].set_index("folder").label if done is not None else None
        sub["label"] = [prev.get(f, p) if prev is not None else p for f, p in zip(sub.folder, sub.proposed)]
        title = f"{s}  ({i + 1}/{len(subjects)}, {int(sub.needs_review.sum())} flagged)"
        print(title, flush=True)
        result, labels = SubjectReview(sub, gifs, title).run()
        if result == "accept":
            save(out, sub, labels)
            done = pd.read_csv(out)
            i += 1
        elif result == "back":
            i = max(i - 1, 0)
        else:
            break
    print(f"labels: {out}")


if __name__ == "__main__":
    main()
