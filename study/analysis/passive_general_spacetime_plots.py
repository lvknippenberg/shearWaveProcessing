"""Whole-recording space-time along the GENERAL M-line, with expected and detected windows.

One figure per folder of the manual study (from the general-line cache of
passive_general_screen.py; nothing is read from or written to Z:):

  strip   R-peaks; the expected search windows (MVC R+0-150 ms, AVC QS2 +-120 ms); the windows the
          ENERGY detector picked (read, with the reader's confidence 0-3); the windows the
          SEMBLANCE picker picks (swp.passive_screen; outlined = screened, no line asked)
  row 1   velocity ("velocity gauss" view), one colour scale for the whole recording
  row 2   the same, normalised by its 100 ms moving RMS - a weak wave is not hidden by a strong
          event elsewhere in the recording, so windows could be chosen by eye
  row 3   the semblance track the picker uses, with the 0.3 screen threshold

x = time on the buffer-4 clock (ms), y = distance along the general line from its first point.
Outputs: study/montages/passive_general/<subject>__<folder>.png + all_folders.pdf
"""
from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages            # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402
from scipy.ndimage import uniform_filter1d                      # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.mline.select import phase_targets                      # noqa: E402
from swp.passive_screen import pick_windows, screen_track       # noqa: E402

CACHE = os.path.join(REPO, "study", "analysis", "general_screen_cache")
LOGS = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
OUT = os.path.join(REPO, "study", "montages", "passive_general")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COL = {"MVC": "#2a78d6", "AVC": "#eb6834", "": "#8a8983"}       # categorical slots 1, 2; neutral
SCREEN_MIN = 0.3


def track_for(npz):
    """The semblance track of a cached folder (computed once, cached next to it)."""
    tp = npz.replace(".npz", "_track.npz")
    if os.path.exists(tp):
        z = np.load(tp)
        return {k: z[k] for k in ("t", "sem", "c", "burst")}
    z = np.load(npz, allow_pickle=True)
    tr = screen_track(z["v_data"], z["v_r"], z["v_t"])
    np.savez_compressed(tp, **tr)
    return tr


def _bar(ax, y, t0, t1, color, text=None, filled=True, h=0.7, text_color="white"):
    ax.add_patch(Rectangle((t0, y - h / 2), t1 - t0, h, facecolor=color if filled else "none",
                           edgecolor=color, lw=1.6, alpha=0.9 if filled else 1.0))
    if text is not None:
        ax.text((t0 + t1) / 2, y, text, ha="center", va="center", fontsize=8,
                color=text_color if filled else color, fontweight="bold")


def figure(npz, old):
    z = np.load(npz, allow_pickle=True)
    folder = str(z["folder"])
    v = np.asarray(z["v_data"], float)                 # (n_t, n_r)
    t_ms, r_mm = np.asarray(z["v_t"]) * 1e3, np.asarray(z["v_r"]) * 1e3
    rp = np.asarray(z["r_peaks_s"], float)
    rr = float(z["rr_ms"]) * 1e-3
    tr = track_for(npz)
    new = pick_windows(tr, rp if rp.size else None, rr if rp.size else None,
                       energy=(z["ov_t"], z["e_masked"]), screen_min=SCREEN_MIN)

    fig = plt.figure(figsize=(16, 9.2), facecolor=SURF)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.25, 3, 3, 1.1], hspace=0.12)
    ax = [fig.add_subplot(gs[k]) for k in range(4)]
    for a in ax[1:]:
        a.sharex(ax[0])
    xlim = (t_ms[0], t_ms[-1])

    # strip
    s = ax[0]
    lanes = {"expected search window": 2, "energy detector (your score)": 1, "semblance picker": 0}
    if rp.size:
        for name, lo, hi in phase_targets(rp, rr):
            _bar(s, 2, lo * 1e3, hi * 1e3, COL[name], name, filled=False)
    for _, o in old.iterrows():
        lab = o.label if o.label in ("MVC", "AVC") else ""
        conf = "–" if pd.isna(o.confidence) else f"{int(o.confidence)}"
        _bar(s, 1, o.t_peak_ms - 50, o.t_peak_ms + 50, COL[lab], conf,
             text_color="white" if lab else INK)
    for w in new:
        _bar(s, 0, w.t0 * 1e3, w.t1 * 1e3, COL[w.expect], f"{w.screen:.2f}", filled=not w.screened,
             text_color="white" if w.expect else INK)
    s.set_ylim(-0.6, 2.6)
    s.set_yticks(list(lanes.values()), list(lanes.keys()), fontsize=8)
    s.set_xlim(*xlim)
    subj = os.path.basename(os.path.dirname(folder))
    hr = f"HR {60 / rr:.0f} bpm" if rp.size else "no usable R-peaks"
    s.set_title(f"{subj}  {os.path.basename(folder)}   -   general M-line, {r_mm[-1]:.0f} mm, {hr}   "
                f"(bars: blue MVC, orange AVC, grey other; bottom lane: number = semblance, outline = "
                f"screened < {SCREEN_MIN})", loc="left", fontsize=9, color=INK)
    s.tick_params(axis="x", labelbottom=False)
    for sp in ("top", "right"):
        s.spines[sp].set_visible(False)

    # space-times
    lim = np.nanpercentile(np.abs(v), 99.5) or 1.0
    ms = np.sqrt(uniform_filter1d(np.mean(v ** 2, axis=1), size=max(3, int(0.1 / np.median(np.diff(z["v_t"])))),
                                  mode="nearest"))
    v_n = v / (ms[:, None] + 1e-30)
    lim_n = np.nanpercentile(np.abs(v_n), 99.5) or 1.0
    ext = [t_ms[0], t_ms[-1], r_mm[-1], r_mm[0]]
    for a, data, l, name in ((ax[1], v, lim, "velocity, one scale"),
                             (ax[2], v_n, lim_n, "velocity / 100 ms moving RMS")):
        a.imshow(data.T, aspect="auto", cmap="RdBu_r", vmin=-l, vmax=l, extent=ext, interpolation="nearest")
        a.set_ylabel("along line [mm]", fontsize=8)
        a.text(0.005, 0.96, name, transform=a.transAxes, va="top", fontsize=8, color=INK,
               bbox=dict(facecolor=SURF, alpha=0.8, lw=0))
        a.tick_params(axis="x", labelbottom=False)

    # semblance track
    a = ax[3]
    a.plot(tr["t"] * 1e3, tr["sem"], color=INK, lw=1.3)
    a.axhline(SCREEN_MIN, color=INK2, lw=0.8, ls="--")
    a.text(xlim[1], SCREEN_MIN, f" screen {SCREEN_MIN}", va="center", fontsize=7, color=INK2)
    a.set_ylim(0, 1)
    a.set_ylabel("semblance", fontsize=8)
    a.set_xlabel("time on the buffer-4 clock [ms]")
    a.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        a.spines[sp].set_visible(False)

    for a in ax:
        for r in rp[(rp * 1e3 >= xlim[0]) & (rp * 1e3 <= xlim[1])]:
            a.axvline(r * 1e3, color=INK2, lw=0.8, ls=":")
        a.set_facecolor(SURF)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.95, bottom=0.06)
    return fig, subj, os.path.basename(folder)


def main():
    old_all = pd.read_csv(os.path.join(LOGS, "windows.csv"))
    os.makedirs(OUT, exist_ok=True)
    files = sorted(f for f in glob.glob(os.path.join(CACHE, "*.npz")) if not f.endswith("_track.npz"))
    with PdfPages(os.path.join(OUT, "all_folders.pdf")) as pdf:
        for f in files:
            folder = str(np.load(f, allow_pickle=True)["folder"])
            fig, subj, name = figure(f, old_all[old_all.folder == folder])
            fig.savefig(os.path.join(OUT, f"{subj}__{name}.png"), dpi=110, facecolor=SURF)
            pdf.savefig(fig, facecolor=SURF)
            plt.close(fig)
            print("  ", subj, flush=True)
    print(f"{len(files)} figure(s) -> {OUT}")


if __name__ == "__main__":
    main()
