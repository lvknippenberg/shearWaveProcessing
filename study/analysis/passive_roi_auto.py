"""Automatic MVC / AVC windows against the ROIs marked by eye (scripts/passive_roi.py).

Ground truth: in every folder with a usable ECG, each expected search window (MVC R+0-150 ms,
AVC QS2 +- 120 ms; swp.manual.roi_gui.expected_windows) that lies at least half inside the
recording, and the reader's ROI(s) of that label overlapping it (none = "no event marked").

Pickers place ONE fixed-length window (100 or 120 ms) per search window, its centre inside the
search window:
  phase       a fixed delay: MVC at R + d, AVC at the QS2 centre + d (d = median of the OTHER
              subjects' ROIs: leave-one-subject-out)
  energy      the centre of maximum along-line velocity energy
  prop        the same with the flat, in-phase band removed (per-time spatial mean subtracted)
  sem         maximum slant-stack semblance (swp.passive_screen)
  sem*prop    product of the two (each normalised to its search-window maximum)
  prior*X     X times a Gaussian prior around the phase pick (sigma, leave-one-subject-out)

Scored per marked ROI: coverage = the fraction of the ROI inside the automatic window
(>= 0.95 "fully inside"), and the centre error. Scores are from passive_roi_auto_features.py.

Outputs: study/logs/passive_roi_auto/{picks.csv, summary.csv, summary.txt}.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(REPO, "src"), os.path.join(REPO, "scripts")]
import passive_roi as PR                                        # noqa: E402
from swp.manual.roi_gui import expected_windows                 # noqa: E402

FEAT = os.path.join(REPO, "study", "analysis", "roi_auto_cache")
LOGS = os.path.join(REPO, "study", "logs", "passive_roi_auto")
MIN_INSIDE = 0.5
SIGMA = {"MVC": 0.030, "AVC": 0.045}
PEAK_S = 0.020


def ground_truth():
    """One row per search window: folder, event, search window, R-peak, matched ROIs."""
    rows, feats = [], {}
    for f, c, h in PR._cached(SimpleNamespace(folder=[], subject=None)):
        st, rec, _ = PR._state(f, h)
        d = PR._load(c)
        if st != "done" or d["rr_s"] is None:
            continue
        fz = np.load(os.path.join(FEAT, os.path.basename(c)), allow_pickle=True)
        if str(fz["general_hash"]) != h:
            raise SystemExit(f"stale features: {c}")
        feats[f] = {k: fz[k] for k in fz.files}
        t = np.asarray(d["t_s"], float)
        # short-time energy (PEAK_S box): the sharp valve burst, which 100-120 ms scores smear out
        e = np.mean(np.asarray(d["v"], float) ** 2, axis=1)
        es = uniform_filter1d(e, max(1, int(round(PEAK_S / np.median(np.diff(t))))), mode="nearest")
        feats[f]["peak"] = np.interp(feats[f]["tc"], t, es)
        rp = np.asarray(d["r_peaks_s"], float)
        for name, lo, hi in expected_windows(rp, d["rr_s"]):
            inside = max(0.0, min(hi, t[-1]) - max(lo, t[0])) / (hi - lo)
            if inside < MIN_INSIDE:
                continue
            rois = [q for q in rec["rois"] if q["label"] == name and min(q["t1"], hi) > max(q["t0"], lo)]
            r_peak = rp[rp <= lo + 1e-9].max() if name == "MVC" else rp[rp <= lo + 1e-9].max()
            rows.append(dict(folder=f, subject=os.path.basename(os.path.dirname(f)), event=name, lo=lo, hi=hi,
                             search_mid=(lo + hi) / 2, r_peak=r_peak, rec0=t[0], rec1=t[-1],
                             rois=[(q["t0"], q["t1"]) for q in rois]))
    return pd.DataFrame(rows), feats


def _anchor(row):
    """The reference the phase delay is measured from: the R-peak (MVC) or the QS2 centre (AVC)."""
    return row.r_peak if row.event == "MVC" else row.search_mid


def phase_delays(gt):
    """Per event, per held-out subject: the median delay of the OTHER subjects' ROI centres."""
    out = {}
    for ev in ("MVC", "AVC"):
        g = gt[(gt.event == ev) & (gt.rois.str.len() > 0)]
        delays = pd.Series([np.mean([(a + b) / 2 for a, b in r.rois]) - _anchor(r) for r in g.itertuples()],
                           index=g.subject.values)
        for s in gt.subject.unique():
            out[(ev, s)] = float(np.median(delays[delays.index != s]))
    return out


def pick(row, F, w, method, delay):
    k = f"{int(round(w * 1e3))}"
    tc = F["tc"]
    lo, hi = max(row.lo, row.rec0), min(row.hi, row.rec1)
    m = (tc >= lo) & (tc <= hi)
    cand = tc[m]
    prior_c = float(np.clip(_anchor(row) + delay, lo, hi))
    if method == "phase":
        return prior_c, np.nan

    def norm(x):
        x = np.nan_to_num(np.asarray(x, float)[m], nan=0.0)
        return x / x.max() if x.max() > 0 else x
    prior, base = method.split("*", 1) if method.startswith("prior") else ("", method)
    if base == "sem*prop":
        s = norm(F[f"sem{k}"]) * norm(F[f"prop{k}"])
    else:
        s = norm(F[base] if base == "peak" else F[f"{base}{k}"])
    if prior:
        sig = float(prior[5:]) * 1e-3 if prior[5:] else SIGMA[row.event]
        s = s * np.exp(-0.5 * ((cand - prior_c) / sig) ** 2)
    i = int(np.argmax(s))
    return float(cand[i]), float(np.nan_to_num(F[f"sem{k}"][m][i]))


def window_of(c, w, row):
    """The window [a, a + w] centred on c, shifted inside the recording."""
    a = float(np.clip(c - w / 2, row.rec0, row.rec1 - w))
    return a, a + w


def score(row, a, b):
    if not row.rois:
        return dict(cover=np.nan, full=np.nan, err_ms=np.nan)
    covs = [(max(0.0, min(b, t1) - max(a, t0)) / (t1 - t0), abs((a + b) / 2 - (t0 + t1) / 2)) for t0, t1 in row.rois]
    cov, err = max(covs, key=lambda x: x[0])
    return dict(cover=cov, full=float(cov >= 0.95), err_ms=err * 1e3)


METHODS = ["phase", "energy", "prop", "sem", "sem*prop", "prior*energy", "prior*prop", "prior*sem", "prior*sem*prop",
           "peak", "prior30*peak", "prior45*peak", "prior60*peak", "prior90*peak"]


def peak_offsets(gt, feats, delays):
    """For the peak pickers: the window centre = the picked burst + an offset, the median of the
    OTHER subjects' (ROI centre - picked burst). Returns {(method, event, subject): offset_s}."""
    out = {}
    for meth in [m for m in METHODS if m.endswith("peak")]:
        for ev in ("MVC", "AVC"):
            g = gt[(gt.event == ev) & (gt.rois.str.len() > 0)]
            d = pd.Series([np.mean([(a + b) / 2 for a, b in r.rois])
                           - pick(r, feats[r.folder], 0.1, meth, delays[(ev, r.subject)])[0] for r in g.itertuples()],
                          index=g.subject.values)
            for subj in gt.subject.unique():
                out[(meth, ev, subj)] = float(np.median(d[d.index != subj]))
    return out


def main():
    os.makedirs(LOGS, exist_ok=True)
    gt, feats = ground_truth()
    delays = phase_delays(gt)
    offsets = peak_offsets(gt, feats, delays)
    rows = []
    for r in gt.itertuples():
        for w in (0.100, 0.120):
            for meth in METHODS:
                c, sem = pick(r, feats[r.folder], w, meth, delays[(r.event, r.subject)])
                if meth.endswith("peak"):
                    c += offsets[(meth, r.event, r.subject)]
                    F = feats[r.folder]
                    sem = float(np.interp(c, F["tc"], np.nan_to_num(F[f"sem{int(w * 1e3)}"])))
                a, b = window_of(c, w, r)
                rows.append(dict(subject=r.subject, folder=r.folder, event=r.event, lo=r.lo, marked=bool(r.rois),
                                 n_rois=len(r.rois), w_ms=int(w * 1e3), method=meth, t0=a, t1=b, sem=sem,
                                 **score(r, a, b)))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(LOGS, "picks.csv"), index=False)
    M = P[P.marked]
    S = (M.groupby(["event", "w_ms", "method"])
         .agg(n=("cover", "size"), fully_inside=("full", "mean"), cover_ge_80=("cover", lambda x: (x >= 0.8).mean()),
              cover_median=("cover", "median"), err_median_ms=("err_ms", "median"), err_p90_ms=("err_ms", lambda x: x.quantile(.9)))
         .reset_index())
    S.to_csv(os.path.join(LOGS, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    txt = [f"ground truth: {len(gt)} search windows ({', '.join(f'{e} {n}' for e, n in gt.event.value_counts().items())}); "
           f"marked: {', '.join(f'{e} {int(n)}' for e, n in gt.assign(m=gt.rois.str.len() > 0).groupby('event').m.sum().items())}",
           S.round(3).to_string(index=False)]
    # presence: semblance of the pick in marked vs unmarked AVC search windows
    for meth in ("prior*sem*prop", "prior45*peak", "prior60*peak"):
        q = P[(P.event == "AVC") & (P.w_ms == 120) & (P.method == meth)]
        txt.append(f"AVC presence ({meth}, 120 ms): semblance marked {np.round(np.sort(q[q.marked]["sem"].values), 2).tolist()}"
                   f" | unmarked {np.round(q[~q.marked]["sem"].values, 2).tolist()}")
    out = "\n".join(txt)
    print(out)
    with open(os.path.join(LOGS, "summary.txt"), "w") as fh:
        fh.write(out + "\n")


if __name__ == "__main__" and "--figure" not in sys.argv:
    main()


def figure():
    """Every folder with an ECG: the whole-recording space-time with the reader's ROIs (filled)
    and the automatic windows of swp.passive_valves (outlined; dashed = screened)."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from swp.manual.roi_gui import COL
    auto = list(csv.DictReader(open(os.path.join(REPO, "study", "logs", "passive_auto_windows.csv"))))
    items = [(f, c, h) for f, c, h in PR._cached(SimpleNamespace(folder=[], subject=None))
             if any(a["folder"] == f for a in auto)]
    fig, axs = plt.subplots(len(items), 1, figsize=(16, 1.25 * len(items)), facecolor="#fcfcfb")
    for ax, (f, c, h) in zip(axs, items):
        d = PR._load(c)
        _, rec, _ = PR._state(f, h)
        t, v, r = d["t_s"] * 1e3, d["v"], d["r_m"] * 1e3
        lim = np.percentile(np.abs(v), 99.5)
        ax.imshow(v.T, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[t[0], t[-1], r[-1], r[0]],
                  interpolation="nearest")
        for q in (rec or {}).get("rois", []):
            if q["label"] in ("MVC", "AVC"):
                ax.axvspan(q["t0"] * 1e3, q["t1"] * 1e3, ymin=0.0, ymax=0.18, color=COL[q["label"]], alpha=0.9)
        for a in (a for a in auto if a["folder"] == f):
            t0, t1 = float(a["t0"]) * 1e3, float(a["t1"]) * 1e3
            ax.add_patch(Rectangle((t0, r[0]), t1 - t0, r[-1] - r[0], fill=False, lw=2, edgecolor="#0b0b0b",
                                   ls="--" if a["screened"] == "True" else "-"))
            ax.text(t0 + 2, r[0], f"{a['label']} {float(a['sem']):.2f}", va="top", fontsize=7,
                    bbox=dict(facecolor="white", lw=0, pad=0.5))
        ax.set_yticks([])
        ax.set_ylabel(os.path.basename(os.path.dirname(f))[-2:], rotation=0, ha="right", va="center", fontsize=9)
        ax.set_xlim(0, 1200)
        ax.tick_params(labelsize=7, labelbottom=ax is axs[-1])
    axs[-1].set_xlabel("time on the buffer-4 clock [ms]")
    fig.suptitle("Automatic 120 ms windows (black box, label + semblance; dashed = screened < 0.3) vs your ROIs "
                 "(coloured bar at the bottom: blue MVC, orange AVC)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(REPO, "study", "montages", "passive_roi_prelim", "auto_vs_rois.png")
    fig.savefig(out, dpi=80, facecolor="#fcfcfb")
    print("->", out)


if __name__ == "__main__" and "--figure" in sys.argv:
    figure()
