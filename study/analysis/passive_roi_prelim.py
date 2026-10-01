"""Preliminary check of the ROIs marked by eye (scripts/passive_roi.py), and the 100 ms windows.

Read only: rois.json of every folder whose ROIs are done (the general-line cache gives the
space-time), and study/logs/passive_manual_prelim/windows.csv (the detector's windows with the
reader's scores from 2026-09-29).

1. Integrity: every stored ROI against the cache it was drawn on (time range, phase, duration).
2. A common 100 ms window per ROI. A ROI shorter than 100 ms is EXPANDED (the window contains the
   whole ROI); a longer one is TRIMMED (the window lies inside the ROI). Three placements:
     centre     centred on the ROI midpoint (shifted inside the recording at the ends);
     energy     the admissible position with the most along-line velocity energy;
     semblance  the admissible position with the highest general-line slant-stack semblance
                (the cached screen track, 5 ms steps, swp.passive_screen).
   How far apart they are says whether the placement matters.
3. Against the detector: does a detector window (t_peak +- 50 ms) cover each ROI, and which
   detector windows the reader scored >= 2 got no ROI.
4. Edge check: ROIs starting in the first 30 ms, where the zero-phase filters have no history
   (swp.passive_screen.EDGE_S); the mean RMS-vs-time over ALL cached folders shows whether the
   recording start is systematically loud.

Outputs: study/logs/passive_roi_prelim/{rois.csv, summary.txt}, study/montages/passive_roi_prelim/
{roi_windows.png, summary.png}.
"""
from __future__ import annotations

import glob
import os
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import passive_roi as PR                                        # noqa: E402
from swp.manual.roi_gui import COL, describe, expected_windows  # noqa: E402

W = 0.100
EDGE = 0.030
LOGS = os.path.join(REPO, "study", "logs", "passive_roi_prelim")
FIGS = os.path.join(REPO, "study", "montages", "passive_roi_prelim")
DETECTOR = os.path.join(REPO, "study", "logs", "passive_manual_prelim", "windows.csv")
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
PLACE_COL = {"centre": "#e5d200", "energy": "#1f9e6e", "semblance": "#8a3ffc"}


def admissible(t0, t1, rec):
    """Range of window starts [lo, hi]: the window contains the ROI (expand) or lies inside it
    (trim), and stays inside the recording."""
    if t1 - t0 <= W:
        lo, hi = t1 - W, t0
    else:
        lo, hi = t0, t1 - W
    return max(lo, rec[0]), min(hi, rec[1] - W)


def place(t0, t1, t, v, track):
    rec = (t[0], t[-1])
    lo, hi = admissible(t0, t1, rec)
    if lo > hi:                                     # ROI longer than the record allows: centre
        lo = hi = float(np.clip(0.5 * (t0 + t1) - W / 2, rec[0], rec[1] - W))
    out = dict(centre=float(np.clip(0.5 * (t0 + t1) - W / 2, lo, hi)))
    starts = np.arange(lo, hi + 1e-9, 0.001) if hi > lo else np.array([lo])
    e = np.mean(v ** 2, axis=1)
    ce = np.concatenate([[0], np.cumsum(e)])
    i0 = np.searchsorted(t, starts)
    i1 = np.searchsorted(t, starts + W)
    out["energy"] = float(starts[np.argmax((ce[i1] - ce[i0]) / np.maximum(i1 - i0, 1))])
    sem = np.interp(starts + W / 2, track["t"], np.nan_to_num(track["sem"], nan=-1), left=-1, right=-1)
    out["semblance"] = float(starts[np.argmax(sem)]) if np.max(sem) > -1 else np.nan
    return out


def track_at(track, key, ws):
    if not np.isfinite(ws):
        return np.nan
    k = int(np.argmin(np.abs(track["t"] - (ws + W / 2))))
    return float(track[key][k]) if abs(track["t"][k] - (ws + W / 2)) <= 0.0026 else np.nan


def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    det = pd.read_csv(DETECTOR)
    rows, crops, problems, rms_curves = [], [], [], []
    for f, c, h in PR._cached(SimpleNamespace(folder=[], subject=None)):
        d = PR._load(c)
        t, v = np.asarray(d["t_s"], float), np.asarray(d["v"], float)
        rms_curves.append((t, np.sqrt(np.mean(v ** 2, axis=1)) / np.sqrt(np.mean(v ** 2))))
        st, rec, _ = PR._state(f, h)
        if st != "done":
            continue
        tz = np.load(c.replace(".npz", "_track.npz"))
        track = {k: tz[k] for k in ("t", "sem", "c", "burst")}
        exp = expected_windows(d["r_peaks_s"], d["rr_s"]) if d["r_peaks_s"].size else []
        subj = os.path.basename(os.path.dirname(f))
        dw = det[det.folder == f]
        for i, q in enumerate(rec["rois"]):
            info = describe(q["t0"], q["t1"], d["r_peaks_s"], exp)
            if not (t[0] - 1e-3 <= q["t0"] < q["t1"] <= t[-1] + 1e-3):
                problems.append(f"{subj} roi{i}: outside the recording")
            if info["phase_ms"] is not None and abs(info["phase_ms"] - q["phase_ms"]) > 0.2:
                problems.append(f"{subj} roi{i}: phase {q['phase_ms']} != {info['phase_ms']}")
            if abs(q["duration_ms"] - (q["t1"] - q["t0"]) * 1e3) > 0.02:
                problems.append(f"{subj} roi{i}: duration")
            pl = place(q["t0"], q["t1"], t, v, track)
            row = dict(subject=subj, folder=f, roi=i, label=q["label"], t0_ms=q["t0"] * 1e3,
                       t1_ms=q["t1"] * 1e3, dur_ms=q["duration_ms"], phase_ms=q["phase_ms"],
                       ecg=d["rr_s"] is not None, starts_in_edge=q["t0"] < t[0] + EDGE)
            row["change_ms"] = W * 1e3 - q["duration_ms"]          # + expand, - trim
            for k, ws in pl.items():
                row[f"{k}_t0_ms"] = ws * 1e3
                inter = max(0.0, min(q["t1"], ws + W) - max(q["t0"], ws)) if np.isfinite(ws) else np.nan
                row[f"{k}_roi_cover"] = inter / (q["t1"] - q["t0"])
                row[f"{k}_sem"] = track_at(track, "sem", ws)
                row[f"{k}_c"] = track_at(track, "c", ws)
            row["shift_energy_ms"] = (pl["energy"] - pl["centre"]) * 1e3
            row["shift_semblance_ms"] = (pl["semblance"] - pl["centre"]) * 1e3
            row["centre_clipped"] = abs(pl["centre"] - (0.5 * (q["t0"] + q["t1"]) - W / 2)) > 1e-6
            # the detector: its windows are t_peak +- 50 ms
            ov = [(max(0, min(q["t1"] * 1e3, w.t_peak_ms + 50) - max(q["t0"] * 1e3, w.t_peak_ms - 50))
                   / (q["duration_ms"]), w) for w in dw.itertuples()]
            best = max(ov, key=lambda x: x[0]) if ov else (0.0, None)
            row["det_cover"] = best[0]
            row["det_label"] = best[1].label if best[1] is not None else None
            row["det_conf"] = best[1].confidence if best[1] is not None else np.nan
            row["det_offset_ms"] = (best[1].t_peak_ms - row["centre_t0_ms"] - 50) if best[1] is not None else np.nan
            rows.append(row)
            crops.append((row, t, v, d["r_m"], pl, rec.get("clim_pct", 99.5)))
        # detector windows the reader found usable but that got no ROI
        for w in dw.itertuples():
            hit = any(max(0, min(q["t1"] * 1e3, w.t_peak_ms + 50) - max(q["t0"] * 1e3, w.t_peak_ms - 50)) > 0
                      for q in rec["rois"])
            if not hit and w.confidence >= 2:
                problems.append(f"{subj}: detector window {w.label} at {w.t_peak_ms:.0f} ms scored "
                                f"{int(w.confidence)} has no ROI")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(LOGS, "rois.csv"), index=False)

    # ---------------------------------------------------------------- text summary
    L = []
    L.append(f"{df.folder.nunique()} folders done, {len(df)} ROIs "
             f"({', '.join(f'{k} {v}' for k, v in df.label.value_counts().items())})")
    L.append(f"integrity: {'no problems' if not any('roi' in p for p in problems) else ''}")
    L += ["  " + p for p in problems if " roi" in p]
    L.append(f"duration ms: median {df.dur_ms.median():.0f}, IQR {df.dur_ms.quantile(.25):.0f}-"
             f"{df.dur_ms.quantile(.75):.0f}, range {df.dur_ms.min():.0f}-{df.dur_ms.max():.0f}; "
             f"{(df.dur_ms < 100).sum()} expand, {(df.dur_ms > 100).sum()} trim")
    L.append(f"to 100 ms: change median {df.change_ms.median():+.0f} ms, |change| > 20 ms in "
             f"{(df.change_ms.abs() > 20).sum()}")
    tr = df[df.dur_ms > 100]
    if len(tr):
        L.append(f"trimmed ROIs keep (centre placement) {100 * tr.centre_roi_cover.median():.0f} % of the ROI "
                 f"(min {100 * tr.centre_roi_cover.min():.0f} %)")
    for k in ("energy", "semblance"):
        s = df[f"shift_{k}_ms"].abs()
        L.append(f"{k} vs centre placement: |shift| median {s.median():.0f} ms, > 10 ms in {(s > 10).sum()}, "
                 f"max {s.max():.0f} ms")
    L.append(f"centre window clipped at a record end: {int(df.centre_clipped.sum())}")
    L.append(f"ROIs starting in the first {EDGE * 1e3:.0f} ms (filter edge): {int(df.starts_in_edge.sum())} "
             f"({', '.join(df[df.starts_in_edge].subject + ' ' + df[df.starts_in_edge].label)})")
    L.append(f"general-line semblance at the centre window: median {df.centre_sem.median():.2f}; "
             f"< 0.3 (the old screen) in {(df.centre_sem < 0.3).sum()}")
    L.append(f"covered >= 50 % by a detector window: {(df.det_cover >= .5).sum()} of {len(df)}; "
             f"no detector window at all: {(df.det_cover == 0).sum()}")
    m = df[df.det_cover > 0]
    L.append(f"  where both: detector centre - ROI centre median {m.det_offset_ms.median():+.0f} ms, "
             f"|.| median {m.det_offset_ms.abs().median():.0f} ms")
    L += ["  " + p for p in problems if " roi" not in p]
    # edge loudness over ALL cached folders
    grid = np.arange(0.002, 0.95, 0.001)
    R = np.array([np.interp(grid, tt, rr) for tt, rr in rms_curves])
    first = R[:, grid < EDGE].mean()
    L.append(f"mean normalised RMS over all {len(R)} cached folders: first {EDGE * 1e3:.0f} ms "
             f"{first:.2f}, 30-100 ms {R[:, (grid >= .03) & (grid < .1)].mean():.2f}, 100-950 ms "
             f"{R[:, grid >= .1].mean():.2f}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(LOGS, "summary.txt"), "w") as fh:
        fh.write(txt + "\n")

    # ---------------------------------------------------------------- montage of every ROI
    n = len(crops)
    nc = 6
    nr = int(np.ceil(n / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(3.1 * nc, 2.35 * nr), facecolor=SURF, squeeze=False)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, (row, t, v, r, pl, pct) in zip(axs.ravel(), crops):
        lo, hi = row["t0_ms"] - 70, row["t1_ms"] + 70
        m = (t * 1e3 >= lo) & (t * 1e3 <= hi)
        lim = np.percentile(np.abs(v), pct)
        ax.imshow(v[m].T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest",
                  extent=[t[m][0] * 1e3, t[m][-1] * 1e3, r[-1] * 1e3, r[0] * 1e3])
        for x in (row["t0_ms"], row["t1_ms"]):
            ax.axvline(x, color=INK, lw=1.4, ls="--")
        ax.axvspan(pl["centre"] * 1e3, pl["centre"] * 1e3 + 100, color=PLACE_COL["centre"], alpha=0.22, lw=0)
        yl = ax.get_ylim()
        for k, yy in (("centre", 0.03), ("energy", 0.10), ("semblance", 0.17)):
            ws = pl[k]
            if np.isfinite(ws):
                y = yl[0] + (yl[1] - yl[0]) * (1 - yy)
                ax.plot([ws * 1e3, ws * 1e3 + 100], [y, y], "-", color=PLACE_COL[k], lw=3,
                        solid_capstyle="butt")
        ax.set_title(f"{row['subject'][-2:]} #{row['roi']} {row['label']}  {row['dur_ms']:.0f} ms"
                     f"  sem {row['centre_sem']:.2f}", fontsize=8,
                     color=COL.get(row["label"], INK) if row["label"] != "other" else INK)
        ax.tick_params(labelsize=6)
    fig.suptitle("Every ROI (black dashed) with the 100 ms window: centre (shaded yellow + bar), energy "
                 "(green bar), semblance (purple bar); title: subject, ROI, label, ROI length, general-line "
                 "semblance of the centre window", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(os.path.join(FIGS, "roi_windows.png"), dpi=90, facecolor=SURF)
    plt.close(fig)

    # ---------------------------------------------------------------- summary figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), facecolor=SURF)
    for lab in ("MVC", "AVC", "other"):
        s = df[df.label == lab]
        ax[0].scatter(s.phase_ms.fillna(-50), s.dur_ms, color=COL[lab], label=lab, s=28)
    ax[0].axhline(100, color=INK2, ls="--", lw=1)
    ax[0].set_xlabel("ROI centre since the preceding R-peak [ms] (no ECG: -50)")
    ax[0].set_ylabel("ROI length [ms]")
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title("ROI length vs phase (dashed: the common 100 ms)", fontsize=9, loc="left")
    b = np.arange(-60, 61, 5)
    ax[1].hist([df.shift_energy_ms, df.shift_semblance_ms], bins=b, color=[PLACE_COL["energy"],
               PLACE_COL["semblance"]], label=["energy", "semblance"])
    ax[1].set_xlabel("window start minus the centred window's start [ms]")
    ax[1].set_ylabel("ROIs")
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].set_title("Does the 100 ms placement matter?", fontsize=9, loc="left")
    gm = R.mean(axis=0)
    ax[2].plot(grid * 1e3, gm, color=INK, lw=1.4)
    ax[2].fill_between(grid * 1e3, np.percentile(R, 25, axis=0), np.percentile(R, 75, axis=0),
                       color=INK2, alpha=0.2, lw=0)
    ax[2].axvline(EDGE * 1e3, color=INK2, ls="--", lw=1)
    ax[2].set_xlabel("time on the buffer-4 clock [ms]")
    ax[2].set_ylabel("RMS(t) / RMS of the record")
    ax[2].set_title(f"Velocity RMS over time, all {len(R)} cached folders (mean, IQR)", fontsize=9, loc="left")
    for a in ax:
        a.set_facecolor(SURF)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "summary.png"), dpi=100, facecolor=SURF)
    plt.close(fig)
    print(f"-> {LOGS}, {FIGS}")


if __name__ == "__main__":
    main()
