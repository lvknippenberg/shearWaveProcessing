"""Points 1 + 2: reproducibility of hand speeds within one acquisition (beat to beat) and between the
subsequent acquisitions of a subject, plus the re-reading noise of the same beat.

All on log(speed) (errors scale with speed; a hand slope fixes delay, not speed), reported as CV %.
Main set: confidence >= 2, positive speed. Sensitivity subsets: confidence 3 only; "resolved"
(the hand line takes >= 5 frames to cross the M-line, i.e. not a lower bound); "tilted" (the reader
moved the slider away from the automatic start - an untilted slope is the automatic speed).

    python 02_reproducibility.py [--snapshot <stamp>] [--boot 300]
Outputs: results/<snap>/02_reproducibility/ (summary.json, tables, figures)
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

import common as C
import stats as ST

FS = 925.9                       # buffer-4 frame rate [Hz]


def load(snap):
    t = C.tables(snap)
    w = t["windows"]
    f = t["folders"]
    s = w[w.state == "slope"].copy()
    s = s.merge(f[["subject", "folder", "acq_order", "acq_time", "hr_bpm"]], on=["subject", "folder"],
                how="left", suffixes=("", "_folder"))
    s["logc"] = np.log(s.speed.where(s.speed > 0))
    s["cross_frames"] = s.mline_length_mm / s.speed.abs() * FS / 1e3
    s["resolved"] = s.cross_frames >= 5
    s["tilted"] = ~s.untilted.astype(bool)
    s["usable"] = (s.confidence >= 2) & (s.speed > 0)
    return s, t


SUBSETS = {
    "conf>=2": lambda d: d[d.usable],
    "conf=3": lambda d: d[d.usable & (d.confidence == 3)],
    "conf>=2 resolved": lambda d: d[d.usable & d.resolved],
    "conf>=2 tilted": lambda d: d[d.usable & d.tilted],
    "conf>=1": lambda d: d[(d.confidence >= 1) & (d.speed > 0)],
}


# ------------------------------------------------------------------ point 1: within acquisition
def pairs_within(d, label):
    """All pairs of same-label slopes in one folder, ordered in time."""
    rows = []
    for (s, f), g in d[d.label == label].groupby(["subject", "folder"]):
        g = g.sort_values("t_peak_ms")
        for (_, a), (_, b) in combinations(g.iterrows(), 2):
            rows.append(dict(subject=s, folder=f, label=label, w1=a.window, w2=b.window,
                             c1=a.speed, c2=b.speed, conf1=a.confidence, conf2=b.confidence,
                             dt_ms=b.t_peak_ms - a.t_peak_ms, both_untilted=bool(a.untilted and b.untilted),
                             same_line=a.line_hash == b.line_hash,
                             dlog=np.log(b.speed) - np.log(a.speed)))
    return pd.DataFrame(rows)


def ba_stats(p):
    if len(p) < 3:
        return dict(n_pairs=len(p))
    dl = p.dlog.values
    sd_w = float(np.std(dl, ddof=1) / np.sqrt(2))
    return dict(n_pairs=len(p), n_subjects=int(p.subject.nunique()),
                bias_pct=100 * (np.exp(np.mean(dl)) - 1),
                loa_pct=[100 * (np.exp(np.mean(dl) - 1.96 * np.std(dl, ddof=1)) - 1),
                         100 * (np.exp(np.mean(dl) + 1.96 * np.std(dl, ddof=1)) - 1)],
                within_sd_log=sd_w, within_cv_pct=ST.pct(sd_w),
                median_abs_diff_pct=float(np.median(np.abs(p.c2 - p.c1) / ((p.c1 + p.c2) / 2)) * 100),
                rc_pct=100 * (np.exp(2.77 * sd_w) - 1))


def point1(s, out):
    res, all_pairs = {}, []
    for name, fn in SUBSETS.items():
        d = fn(s)
        for lab in ("MVC", "AVC", "AK"):
            p = pairs_within(d, lab)
            if len(p):
                p["subset"] = name
                all_pairs.append(p)
            res[f"{name}|{lab}"] = ba_stats(p)
            sd, dfree = ST.within_sd(d[d.label == lab].logc, d[d.label == lab].folder)
            res[f"{name}|{lab}"].update(pooled_within_sd_log=sd, pooled_df=dfree,
                                        pooled_within_cv_pct=ST.pct(sd) if np.isfinite(sd) else None)
    pairs = pd.concat(all_pairs) if all_pairs else pd.DataFrame()
    pairs.to_csv(out / "pairs_within_acquisition.csv", index=False)
    # MVC pairs: does sharing the line or both untilted make them agree better?
    p = pairs[(pairs.subset == "conf>=2") & (pairs.label == "MVC")]
    extra = {}
    for key in ("same_line", "both_untilted"):
        for v in (True, False):
            q = p[p[key] == v]
            extra[f"MVC {key}={v}"] = ba_stats(q)
    # MVC vs AVC in the same acquisition (folder means)
    d = SUBSETS["conf>=2"](s)
    fm = d[d.label.isin(["MVC", "AVC"])].groupby(["subject", "folder", "label"]).logc.mean().unstack()
    fm = fm.dropna()
    if len(fm) > 3:
        r = fm.AVC - fm.MVC
        sub = r.groupby(level=0).mean()                     # one value per subject
        extra["AVC_vs_MVC_same_folder"] = dict(n_folders=len(r), n_subjects=len(sub),
                                               ratio_median=float(np.exp(np.median(r))),
                                               ratio_subject_mean=float(np.exp(sub.mean())),
                                               wilcoxon_p_subjects=float(wilcoxon(sub).pvalue) if len(sub) > 5 else None)
    res["extra"] = extra
    # figure: Bland-Altman MVC / AVC within folder
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, lab in zip(axs, ("MVC", "AVC", "AK")):
        q = pairs[(pairs.subset == "conf>=2") & (pairs.label == lab)]
        if not len(q):
            ax.set_title(f"{lab}: no pairs")
            continue
        m = np.exp((np.log(q.c1) + np.log(q.c2)) / 2)
        diff = 100 * (np.exp(q.dlog) - 1)
        col = np.where(q.conf1.eq(3) & q.conf2.eq(3), "C0", "C1")
        ax.scatter(m, diff, c=col, s=18)
        b = res[f"conf>=2|{lab}"]
        ax.axhline(b.get("bias_pct", 0), color="k", lw=1)
        for lo in b.get("loa_pct", []):
            ax.axhline(lo, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("geometric mean speed [m/s]")
        ax.set_ylabel("later / earlier - 1 [%]")
        ax.set_title(f"{lab} beat-to-beat, same acquisition: n={len(q)}, within-CV "
                     f"{b.get('within_cv_pct', np.nan):.0f} %", fontsize=9)
    axs[0].text(0.02, 0.98, "blue: both clear (3)\norange: otherwise", transform=axs[0].transAxes,
                va="top", fontsize=8)
    C.savefig(fig, out / "fig1_within_acquisition_BA.png")
    return res


# ------------------------------------------------------------------ point 2: between acquisitions
def point2(s, t, out, n_boot):
    res = {}
    rows = []
    for name in ("conf>=2", "conf=3", "conf>=2 resolved", "conf>=2 tilted"):
        d = SUBSETS[name](s)
        for lab in ("MVC", "AVC"):
            q = d[d.label == lab].dropna(subset=["logc"])
            fit = ST.reml_nested(q.logc, q.subject, q.folder)
            ci = None
            if name == "conf>=2" and n_boot:
                def fn(df):
                    r = ST.reml_nested(df.logc, df.subject, df.folder)
                    return [np.sqrt(r["var_subject"]), np.sqrt(r["var_acq"]), np.sqrt(r["var_resid"])]
                est, lo, hi = ST.cluster_bootstrap(q, "subject", fn, n=n_boot)
                ci = dict(sd_subject=[lo[0], hi[0]], sd_acq=[lo[1], hi[1]], sd_resid=[lo[2], hi[2]])
            vs, va, ve = fit["var_subject"], fit["var_acq"], fit["var_resid"]
            k = q.groupby("folder").size().mean()
            r = dict(subset=name, label=lab, n=fit["n"], n_subj=fit["n_subj"], n_acq=fit["n_acq"],
                     n_subj_multi=int((q.groupby("subject").folder.nunique() > 1).sum()),
                     geo_mean=float(np.exp(fit["mu"])),
                     sd_subject=np.sqrt(vs), sd_acq=np.sqrt(va), sd_resid=np.sqrt(ve),
                     cv_subject=ST.pct(np.sqrt(vs)), cv_acq=ST.pct(np.sqrt(va)), cv_resid=ST.pct(np.sqrt(ve)),
                     cv_between_acq_total=ST.pct(np.sqrt(va + ve)),
                     icc_single_beat=vs / (vs + va + ve),
                     icc_acq_mean=vs / (vs + va + ve / k), beats_per_acq=float(k), ci=ci)
            rows.append(r)
            res[f"{name}|{lab}"] = r
    comp = pd.DataFrame(rows)
    comp.drop(columns="ci").to_csv(out / "variance_components.csv", index=False)
    # how many beats / acquisitions for a given precision of a subject's mean (main set)
    plan = {}
    for lab in ("MVC", "AVC"):
        r = res[f"conf>=2|{lab}"]
        va, ve = r["sd_acq"] ** 2, r["sd_resid"] ** 2
        grid = {}
        for n_acq in (1, 2, 3, 6):
            for n_beat in (1, 2, 4):
                grid[f"{n_acq} acq x {n_beat} beats"] = ST.pct(np.sqrt(va / n_acq + ve / (n_acq * n_beat)))
        plan[lab] = grid
    res["precision_of_subject_mean_cv_pct"] = plan
    # acquisition order (subjects with >= 3 acquisitions): within-subject trend of folder means
    d = SUBSETS["conf>=2"](s)
    fm = d[d.label.isin(["MVC", "AVC"])].groupby(["subject", "folder", "label"]).agg(
        logc=("logc", "mean"), acq_order=("acq_order", "first"), n=("logc", "size")).reset_index()
    trend = {}
    for lab in ("MVC", "AVC"):
        q = fm[fm.label == lab]
        q = q[q.groupby("subject").folder.transform("nunique") >= 3].copy()
        q["dev"] = q.logc - q.groupby("subject").logc.transform("mean")
        q["ord"] = q.acq_order - q.groupby("subject").acq_order.transform("mean")
        if len(q) > 5:
            rho, pv = spearmanr(q.ord, q.dev)
            slope = np.polyfit(q.ord, q.dev, 1)[0]
            trend[lab] = dict(n_folders=len(q), n_subjects=q.subject.nunique(), spearman=rho, p=pv,
                              pct_per_acquisition=100 * (np.exp(slope) - 1))
    res["acquisition_order"] = trend
    fm.to_csv(out / "folder_means.csv", index=False)
    # does a change of line geometry / HR between acquisitions explain the difference?
    lines = t["lines"]
    ev = lines[lines.kind == "event"]
    geo = ev.groupby(["subject", "folder"]).agg(depth=("zm_mm", "mean"), lat=("xm_mm", "mean"),
                                               angle=("angle_deg", "mean"), length=("length_mm", "mean")).reset_index()
    hr = s.groupby(["subject", "folder"]).hr_bpm.first().reset_index()
    geo = geo.merge(hr, on=["subject", "folder"], how="left")
    cov = {}
    for lab in ("MVC", "AVC"):
        q = fm[fm.label == lab].merge(geo, on=["subject", "folder"])
        prs = []
        for sbj, g in q.groupby("subject"):
            for (_, a), (_, b) in combinations(g.iterrows(), 2):
                prs.append(dict(subject=sbj, dlog=abs(a.logc - b.logc), ddepth=abs(a.depth - b.depth),
                                dlat=abs(a.lat - b.lat), dangle=abs(a.angle - b.angle),
                                dlength=abs(a.length - b.length), dhr=abs(a.hr_bpm - b.hr_bpm)))
        prs = pd.DataFrame(prs)
        if len(prs) < 10:
            continue
        cc = {}
        rng = np.random.default_rng(0)
        for k in ("ddepth", "dlat", "dangle", "dlength", "dhr"):
            ok = prs[k].notna()
            rho = spearmanr(prs.dlog[ok], prs[k][ok])[0]
            # permutation within subject (pairs are not independent across subjects)
            null = []
            for _ in range(500):
                perm = prs[ok].groupby("subject")[k].transform(lambda x: rng.permutation(x.values))
                null.append(spearmanr(prs.dlog[ok], perm)[0])
            cc[k] = dict(rho=float(rho), p_perm=float(np.mean(np.abs(null) >= abs(rho))), n_pairs=int(ok.sum()))
        cov[lab] = cc
    res["geometry_vs_difference"] = cov
    # figure: speeds per subject x acquisition
    d = SUBSETS["conf>=2"](s)
    multi = sorted(d.groupby("subject").folder.nunique().loc[lambda x: x > 1].index)
    fig, axs = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    for ax, lab in zip(axs, ("MVC", "AVC")):
        q = d[(d.label == lab) & d.subject.isin(multi)]
        for i, sbj in enumerate(multi):
            g = q[q.subject == sbj]
            for _, rr in g.iterrows():
                x = i + (rr.acq_order - 1) / 12.0 - 0.4
                ax.scatter(x, rr.speed, s=14 if rr.confidence == 2 else 26,
                           c=f"C{int(rr.acq_order - 1) % 10}", marker="o" if rr.resolved else "^",
                           edgecolors="k" if rr.confidence == 3 else "none", lw=0.5)
            ax.axvline(i + 0.5, color="0.9", lw=0.6)
        ax.set_yscale("log")
        ax.set_ylabel(f"{lab} speed [m/s]")
        r = res[f"conf>=2|{lab}"]
        ax.set_title(f"{lab}: CV subject {r['cv_subject']:.0f} %, between acquisitions {r['cv_acq']:.0f} %, "
                     f"beat/reading residual {r['cv_resid']:.0f} %  (REML, conf >= 2)", fontsize=9)
        ax.set_yticks([1, 2, 3, 5, 8, 12])
        ax.set_yticklabels(["1", "2", "3", "5", "8", "12"])
    axs[1].set_xticks(range(len(multi)))
    axs[1].set_xticklabels([C.subj_short(x) for x in multi])
    axs[1].set_xlabel("subject (colour = acquisition order; triangle = < 5 frames, lower bound; "
                      "black edge = clear)")
    C.savefig(fig, out / "fig2_between_acquisitions.png")
    return res


def reading_noise(t, out):
    """Same beat read twice (energy-era reading, archived 2026-10-01, vs the current one)."""
    rt = t.get("retest")
    if rt is None or not len(rt):
        return {}
    q = rt[(rt.anchor_dt_ms < 30) & (rt.label_old == rt.label_new)].copy()
    q["dlog"] = np.log(q.speed_new.abs()) - np.log(q.speed_old.abs())
    res = {}
    for name, m in (("both conf>=2", (q.confidence_old >= 2) & (q.confidence_new >= 2)),
                    ("both conf=3", (q.confidence_old == 3) & (q.confidence_new == 3))):
        g = q[m & (q.speed_new > 0) & (q.speed_old > 0)]
        if len(g) < 3:
            continue
        sd = float(np.std(g.dlog, ddof=1) / np.sqrt(2))
        res[name] = dict(n=len(g), n_subjects=int(g.subject.nunique()), single_reading_sd_log=sd,
                         single_reading_cv_pct=ST.pct(sd), bias_pct=100 * (np.exp(g.dlog.mean()) - 1),
                         median_abs_diff_pct=float(np.median(np.abs(np.exp(g.dlog) - 1)) * 100))
    # confidence agreement (same beat, same reader)
    cm = pd.crosstab(q.confidence_old, q.confidence_new)
    cm.to_csv(out / "retest_confidence_crosstab.csv")
    res["confidence_exact_agreement"] = float((q.confidence_old == q.confidence_new).mean())
    res["confidence_within_one"] = float(((q.confidence_old - q.confidence_new).abs() <= 1).mean())
    res["n_matched"] = len(q)
    try:
        from sklearn.metrics import cohen_kappa_score
        res["confidence_weighted_kappa"] = float(cohen_kappa_score(q.confidence_old.astype(int),
                                                                   q.confidence_new.astype(int), weights="quadratic"))
    except Exception:                                  # noqa: BLE001
        pass
    q.to_csv(out / "retest_pairs.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap, "02_reproducibility")
    s, t = load(snap)
    summary = dict(snapshot=snap.name, n_slopes=len(s), n_usable=int(s.usable.sum()))
    summary["point1_within_acquisition"] = point1(s, out)
    summary["point2_between_acquisitions"] = point2(s, t, out, a.boot)
    summary["reading_noise_same_beat"] = reading_noise(t, out)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
