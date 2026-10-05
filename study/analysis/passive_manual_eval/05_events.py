"""Point 5: event (window) placement - what the review changed and how the detector could follow it.

Compares, per folder, the automatic valves windows (incl. screened ones) with the reviewed windows,
and the reviewed windows with where the reader put the wavefront (slope anchor):
  * kept / moved / deleted / added / relabelled, per label;
  * timing of the accepted windows vs the R-peak (MVC) and the Weissler QS2 (AVC) -> refit the
    detector's timing priors; AK timing vs the next R-peak;
  * the hand anchor vs the window centre (is the wave centred in its window?);
  * confidence of reviewed windows by origin (kept automatic / moved / added / screened-but-kept);
  * leave-one-subject-out re-run of the timing prior on the cached general-line space-times
    (general_st.npz) with refitted priors, scored against the reviewed windows.

    python 05_events.py [--snapshot <stamp>]
Outputs: results/<snap>/05_events/
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as C

MATCH_MS = 40.0         # an automatic and a reviewed window are "the same event" within this


def centre(df):
    return 0.5 * (df.t0 + df.t1) * 1e3


def match(snap):
    t = C.tables(snap)
    R, W, F = t["review"], t["windows"], t["folders"]
    rows = []
    for (s, f), g in R.groupby(["subject", "folder"]):
        a = g[g.kind == "auto"].copy()
        r = g[g.kind == "reviewed"].copy()
        if not len(r):
            continue
        a["c"], r["c"] = centre(a), centre(r)
        used = set()
        for _, rr in r.iterrows():
            d = (a.c - rr.c).abs() if len(a) else pd.Series(dtype=float)
            d = d[~d.index.isin(used)]
            j = d.idxmin() if len(d) and d.min() <= MATCH_MS else None
            row = dict(subject=s, folder=f, window=int(rr.idx), label=rr.label, c_rev=rr.c,
                       t_peak_rev=rr.t_peak * 1e3, phase_ms=rr.phase_ms, rr_ms=rr.rr_ms, qs2_ms=rr.qs2_ms,
                       proposed_from=rr.proposed_from)
            if j is not None:
                used.add(j)
                aa = a.loc[j]
                row.update(c_auto=aa.c, auto_label=aa.label, screened=bool(aa.screened), screen=aa.screen,
                           dc_ms=rr.c - aa.c, status="kept" if abs(rr.c - aa.c) <= 3 else "moved",
                           relabelled=aa.label != rr.label)
            else:
                row.update(status="added")
            rows.append(row)
        for j, aa in a[~a.index.isin(used)].iterrows():
            rows.append(dict(subject=s, folder=f, window=None, label=None, auto_label=aa.label,
                             c_auto=aa.c, screened=bool(aa.screened), screen=aa.screen,
                             phase_ms=aa.phase_ms, status="deleted"))
    m = pd.DataFrame(rows)
    m = m.merge(F[["subject", "folder", "ecg_trustworthy", "hr_bpm", "rr_median_ms"]], on=["subject", "folder"],
                how="left")
    w = W[W.state == "slope"][["subject", "folder", "window", "confidence", "anchor_t_ms", "anchor_r_mm",
                                "speed", "t0_ms", "t1_ms"]]
    m = m.merge(w, on=["subject", "folder", "window"], how="left")
    m["anchor_minus_centre_ms"] = m.anchor_t_ms - m.c_rev
    # the anchor is an arbitrary point on the line; the time the line crosses the line's middle is not
    return m, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap, "05_events")
    m, t = match(snap)
    m.to_csv(out / "window_matching.csv", index=False)
    S = {}
    ecg = m[m.ecg_trustworthy == True]                                  # noqa: E712
    S["n_folders_reviewed"] = int(m.folder.nunique())
    S["n_folders_with_ecg"] = int(ecg.folder.nunique())
    S["status_by_label_ecg"] = pd.crosstab(ecg.label.fillna(ecg.auto_label), ecg.status).to_dict()
    S["status_by_label_noecg"] = pd.crosstab(m[m.ecg_trustworthy != True].label.fillna("-"),     # noqa: E712
                                             m[m.ecg_trustworthy != True].status).to_dict()
    S["screened_auto_windows"] = pd.crosstab(ecg[ecg.screened == True].auto_label,               # noqa: E712
                                             ecg[ecg.screened == True].status).to_dict()
    mv = ecg[ecg.status == "moved"]
    S["moved_shift_ms"] = mv.groupby("label").dc_ms.describe().round(1).to_dict()
    S["relabelled"] = int(ecg.relabelled.fillna(False).sum())
    # confidence by origin
    conf = ecg[ecg.confidence.notna()].copy()
    conf["origin"] = np.where(conf.status == "added", "added",
                              np.where(conf.screened == True, "screened, kept", conf.status))     # noqa: E712
    S["confidence_by_origin"] = conf.groupby(["label", "origin"]).confidence.agg(
        n="size", usable=lambda x: (x >= 2).mean(), clear=lambda x: (x == 3).mean()).round(2).reset_index().to_dict("records")
    # timing priors refitted from the accepted windows (t_peak = peak 20 ms energy inside the window)
    tim = {}
    acc = ecg[ecg.status != "deleted"].copy()
    acc["tp_phase"] = acc.phase_ms                                      # review phases are of t_peak
    acc["c_phase"] = acc.phase_ms + (acc.c_rev - acc.t_peak_rev)        # phase of the window centre
    mvc = acc[acc.label == "MVC"]
    avc = acc[acc.label == "AVC"].copy()
    avc["rel_qs2"] = avc.tp_phase - avc.qs2_ms
    ak = acc[acc.label == "AK"].copy()
    ak["before_next_R"] = ak.rr_ms - ak.tp_phase
    tim["MVC_tpeak_after_R_ms"] = mvc.tp_phase.describe().round(1).to_dict()
    tim["MVC_centre_after_R_ms"] = mvc.c_phase.describe().round(1).to_dict()
    tim["AVC_tpeak_minus_QS2_ms"] = avc.rel_qs2.describe().round(1).to_dict()
    tim["AK_tpeak_before_next_R_ms"] = ak.before_next_R.describe().round(1).to_dict()
    tim["AK_tpeak_after_R_frac_RR"] = (ak.tp_phase / ak.rr_ms).describe().round(3).to_dict()
    # the detector priors (configs: MVC R+43, AVC QS2-28, sigma 45 ms) for comparison
    tim["detector_priors"] = dict(MVC_after_R=43, AVC_minus_QS2=-28, sigma=45)
    usable = acc[acc.confidence >= 2]
    tim["usable_MVC_tpeak_after_R"] = usable[usable.label == "MVC"].tp_phase.describe().round(1).to_dict()
    tim["usable_AVC_tpeak_minus_QS2"] = (usable[usable.label == "AVC"].tp_phase
                                        - usable[usable.label == "AVC"].qs2_ms).describe().round(1).to_dict()
    # AVC timing vs heart rate: does QS2 scale it right?
    if len(avc) > 10:
        ok = avc.rel_qs2.notna() & avc.hr_bpm.notna()
        b = np.polyfit(avc.hr_bpm[ok], avc.tp_phase[ok], 1)
        b2 = np.polyfit(avc.hr_bpm[ok], avc.rel_qs2[ok], 1)
        tim["AVC_tpeak_vs_HR"] = dict(slope_ms_per_bpm=b[0], residual_vs_QS2_slope_ms_per_bpm=b2[0],
                                     n=int(ok.sum()), sd_rel_qs2=float(avc.rel_qs2[ok].std()),
                                     sd_abs_phase=float(avc.tp_phase[ok].std()))
    S["timing"] = tim
    # wavefront position inside the window: anchor relative to the window centre is not meaningful
    # by itself (an anchor can sit anywhere on the line), so use the time the hand line crosses the
    # middle of the M-line.
    w = t["windows"]
    w = w[w.state == "slope"].copy()
    w["t_mid_ms"] = w.anchor_t_ms + (w.mline_length_mm / 2 - w.anchor_r_mm) / w.speed.where(w.speed.abs() > 0.05)
    w["mid_minus_centre_ms"] = w.t_mid_ms - 0.5 * (w.t0_ms + w.t1_ms)
    w["mid_minus_tpeak_ms"] = w.t_mid_ms - w.t_peak_ms
    S["hand_line_vs_window"] = dict(
        mid_minus_centre=w.groupby("label").mid_minus_centre_ms.describe().round(1).to_dict(),
        mid_minus_tpeak=w.groupby("label").mid_minus_tpeak_ms.describe().round(1).to_dict(),
        frac_outside_window=float(((w.t_mid_ms < w.t0_ms) | (w.t_mid_ms > w.t1_ms)).mean()))
    w[["subject", "folder", "window", "label", "confidence", "t_mid_ms", "mid_minus_centre_ms",
       "mid_minus_tpeak_ms"]].to_csv(out / "hand_line_timing.csv", index=False)
    # missed / extra events per folder: expected events from the R-peaks inside the record
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=lambda x: round(float(x), 2)))
    figures(m, acc, w, out)


def figures(m, acc, w, out):
    fig, axs = plt.subplots(1, 4, figsize=(20, 4.5))
    ax = axs[0]
    q = acc[acc.label == "MVC"]
    ax.hist(q.tp_phase, bins=np.arange(-60, 260, 10), color="C0", alpha=0.6, label="MVC accepted")
    q = acc[(acc.label == "MVC") & (acc.confidence >= 2)]
    ax.hist(q.tp_phase, bins=np.arange(-60, 260, 10), histtype="step", color="k", label="MVC usable")
    ax.axvline(43, color="r", ls="--", label="detector prior R+43")
    ax.set_xlabel("energy peak after R [ms]")
    ax.legend(fontsize=8)
    ax = axs[1]
    q = acc[acc.label == "AVC"]
    ax.hist(q.tp_phase - q.qs2_ms, bins=np.arange(-200, 200, 10), color="C1", alpha=0.6, label="AVC accepted")
    q = q[q.confidence >= 2]
    ax.hist(q.tp_phase - q.qs2_ms, bins=np.arange(-200, 200, 10), histtype="step", color="k", label="AVC usable")
    ax.axvline(-28, color="r", ls="--", label="detector prior QS2-28")
    ax.set_xlabel("energy peak - Weissler QS2 [ms]")
    ax.legend(fontsize=8)
    ax = axs[2]
    q = acc[acc.label == "AVC"]
    ax.scatter(q.hr_bpm, q.tp_phase, s=12, c="C1", label="AVC energy peak")
    hr = np.linspace(45, 110, 50)
    ax.plot(hr, 546 - 2.1 * hr - 28, "r--", label="QS2 (Weissler) - 28")
    ax.set_xlabel("heart rate [bpm]")
    ax.set_ylabel("after R [ms]")
    ax.legend(fontsize=8)
    ax = axs[3]
    for lab, cl in (("MVC", "C0"), ("AVC", "C1"), ("AK", "C2")):
        q = w[w.label == lab]
        ax.hist(q.mid_minus_centre_ms.clip(-100, 100), bins=np.arange(-100, 101, 5), histtype="step",
                color=cl, label=lab)
    ax.axvspan(-60, 60, color="0.92", zorder=0)
    ax.set_xlabel("hand line at mid M-line - window centre [ms] (grey: inside 120 ms)")
    ax.legend(fontsize=8)
    C.savefig(fig, out / "fig1_event_timing.png")


if __name__ == "__main__":
    main()
