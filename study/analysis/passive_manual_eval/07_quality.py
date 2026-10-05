"""Point 7: how well does the current space-time metric predict the reader's confidence, can a better
metric be defined, and which windows look mis-scored (or scored inconsistently over sessions)?

Reference = the reader's confidence (0/1 merged as "not usable / guess", 2 plausible, 3 clear).
Current metrics: the slant-stack semblance of the event line ("velocity gauss", processed.json, the
number behind the slider start) and the detection screen on the general line (gen_sem).
New metrics, all reader-independent (computed before a slope is drawn):
  simple   mean z-score of four features, signs fixed in advance, z-scores from the TRAINING folds:
           best one-polarity line strength (velocity median and acceleration), semblance
           (acceleration), minus the spread of the five views' automatic speeds
  model    L2 logistic regressions for confidence >= 2 and = 3 on a dozen features
Both validated leave-one-subject-out (LOSO): every window is scored by a metric fitted without its
subject. Expected confidence E = 1 + P(>=2) + P(=3).

Outliers = windows whose given confidence is far from E (LOSO). Listed per group with the path of
their slope snapshot, plus contact sheets of those snapshots, for re-checking by eye.
Drift = the residual (given - E) per reading day / per session position.

    python 07_quality.py [--snapshot <stamp>] [--sheets]
Outputs: results/<snap>/07_quality/
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

SHORT = {"displacement gauss": "disp", "velocity median": "vmed", "velocity gauss": "vg",
         "Keijzer velocity": "keij", "acceleration": "acc"}
SIMPLE = {"vmed_bests_score": +1, "acc_bests_score": +1, "acc_sem": +1, "c_spread": -1}
MODEL = ["vmed_bests_score", "vg_bests_score", "acc_bests_score", "keij_bests_score", "acc_sem", "vmed_sem",
         "vg_sem", "c_spread", "neg_auto", "vg_burst", "acc_flat", "vmed_best_coh", "is_avc", "is_ak", "gen_sem",
         "log_xframes", "fast_auto"]


def load(snap):
    t = C.tables(snap)
    w = t["windows"]
    f = pd.read_csv(C.OUT / snap.name / "features.csv")
    d = w.merge(f, on=["subject", "folder", "window"])
    d = d[d.state == "slope"].copy()
    for v, k in SHORT.items():
        d[f"{k}_sem"] = d[f"auto_sem|{v}"]
        d[f"{k}_logc"] = np.log(d[f"auto_c|{v}"].abs().clip(0.5, 20))
    d["c_spread"] = d[[f"{k}_logc" for k in SHORT.values()]].std(axis=1)
    d["neg_auto"] = (d["auto_c|velocity gauss"] < 0).astype(float)
    # how many frames the automatic (velocity median) line needs to cross the M-line: < 5 = a
    # near-vertical band, a lower bound at best (docs/passive_manual_prelim_2026-09-29.md)
    xf = d.mline_length_mm / d["auto_c|velocity median"].abs().clip(0.3, 20) * 0.926
    d["log_xframes"] = np.log(xf)
    d["fast_auto"] = (xf < 5).astype(float)
    d["is_avc"] = (d.label == "AVC").astype(float)
    d["is_ak"] = (d.label == "AK").astype(float)
    d["y"] = d.confidence.clip(lower=1).astype(int)            # 0 and 1 merged
    d["date"] = pd.to_datetime(d.slope_time).dt.date.astype(str)
    d = d.sort_values("slope_time").reset_index(drop=True)
    return d, t


def loso(d, fit_predict):
    out = pd.Series(np.nan, index=d.index)
    for s in d.subject.unique():
        tr, te = d[d.subject != s], d[d.subject == s]
        out.loc[te.index] = fit_predict(tr, te)
    return out


def simple_metric(tr, te):
    z = 0
    for k, sgn in SIMPLE.items():
        mu, sd = tr[k].mean(), tr[k].std()
        z = z + sgn * (te[k] - mu) / sd
    return z / len(SIMPLE)


def model_probs(tr, te, target):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    X = tr[MODEL].fillna(tr[MODEL].median())
    m.fit(X, target(tr))
    return m.predict_proba(te[MODEL].fillna(tr[MODEL].median()))[:, 1]


def calibrated_E(tr, te, score_col):
    """Expected confidence from a 1-D score: two logistic fits (>=2, =3) on the training folds."""
    p = []
    for tgt in (lambda q: q.y >= 2, lambda q: q.y == 3):
        m = LogisticRegression(max_iter=1000).fit(tr[[score_col]].fillna(tr[score_col].median()), tgt(tr))
        p.append(m.predict_proba(te[[score_col]].fillna(tr[score_col].median()))[:, 1])
    return 1 + p[0] + p[1]


def evaluate(d, col):
    x = d[col]
    ok = x.notna()
    return dict(spearman=float(spearmanr(x[ok], d.y[ok])[0]),
                auc_usable=float(roc_auc_score(d.y[ok] >= 2, x[ok])),
                auc_clear=float(roc_auc_score(d.y[ok] == 3, x[ok])), n=int(ok.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--sheets", action="store_true", help="contact sheets of the outliers' slope PNGs (reads Z:)")
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap, "07_quality")
    d, t = load(snap)
    S = dict(n=len(d), confidence_counts=d.confidence.value_counts().sort_index().to_dict())
    # single features
    feats = [c for c in d.columns if any(c.endswith(s) for s in ("_sem", "_burst", "_flat", "_best_score",
                                                                   "_bests_score", "_best_coh", "_bests_coh"))]
    feats += ["c_spread", "neg_auto"]
    single = pd.DataFrame([dict(feature=c, **evaluate(d, c)) for c in dict.fromkeys(feats)]).sort_values("spearman", ascending=False)
    single.to_csv(out / "single_features.csv", index=False)
    S["current_metric_event_line_semblance"] = evaluate(d, "vg_sem")
    S["current_metric_general_line_screen"] = evaluate(d, "gen_sem")
    # new metrics (LOSO)
    d["simple"] = loso(d, simple_metric)
    S["simple_metric_loso"] = evaluate(d, "simple")
    d["p_usable"] = loso(d, lambda tr, te: model_probs(tr, te, lambda q: q.y >= 2))
    d["p_clear"] = loso(d, lambda tr, te: model_probs(tr, te, lambda q: q.y == 3))
    d["E_model"] = 1 + d.p_usable + d.p_clear
    S["model_loso"] = evaluate(d, "E_model")
    S["model_loso"]["auc_usable_p"] = float(roc_auc_score(d.y >= 2, d.p_usable))
    S["model_loso"]["auc_clear_p"] = float(roc_auc_score(d.y == 3, d.p_clear))
    d["E_simple"] = loso(d, lambda tr, te: calibrated_E(tr, te, "simple"))
    d["E_current"] = loso(d, lambda tr, te: calibrated_E(tr, te, "vg_sem"))
    # agreement of the rounded expected class with the given one
    for k in ("E_current", "E_simple", "E_model"):
        cls = np.clip(np.round(d[k]), 1, 3)
        S[f"{k}_exact"] = float((cls == d.y).mean())
        S[f"{k}_mae"] = float((d[k] - d.y).abs().mean())
    # reference: same reader, same beat, other session (02_reproducibility retest)
    rt = C.OUT / snap.name / "02_reproducibility" / "summary.json"
    if rt.exists():
        r = json.loads(rt.read_text()).get("reading_noise_same_beat", {})
        S["reader_retest_confidence"] = {k: r.get(k) for k in ("confidence_exact_agreement", "confidence_within_one",
                                                                "confidence_weighted_kappa", "n_matched")}
    # model coefficients on all data (for interpretation only)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    m.fit(d[MODEL].fillna(d[MODEL].median()), d.y == 3)
    S["model_clear_coefficients_std"] = dict(zip(MODEL, np.round(m[-1].coef_[0], 3).tolist()))
    # outliers per given class
    d["resid"] = d.y - d.E_model
    d["resid_simple"] = d.y - d.E_simple
    flag = ((d.y == 3) & (d.E_model < 1.9)) | ((d.y == 2) & ((d.E_model < 1.45) | (d.E_model > 2.75))) \
        | ((d.y == 1) & (d.E_model > 2.35))
    both = flag & ((d.y - d.E_simple).abs() >= 0.6)                # the simple metric agrees it is off
    d["outlier"] = flag
    d["outlier_both_metrics"] = both
    cols = ["subject", "folder", "window", "label", "confidence", "E_model", "E_simple", "E_current", "p_usable",
            "p_clear", "speed", "anchor_view", "untilted", "date", "slope_time", "slope_png"]
    ol = d[flag].sort_values(["confidence", "resid"])[cols + ["outlier_both_metrics"]]
    ol.to_csv(out / "outliers.csv", index=False)
    S["outliers"] = dict(n=int(flag.sum()), n_both_metrics=int(both.sum()),
                         by_given=d[flag].confidence.value_counts().sort_index().to_dict(),
                         rule="given 3 & E<1.9 | given 2 & (E<1.45 | E>2.75) | given 0-1 & E>2.35 (E = LOSO model)")
    # drift over the reading days / sessions
    S["drift_by_date"] = d.groupby("date").agg(n=("y", "size"), mean_given=("y", "mean"),
                                               mean_expected=("E_model", "mean"),
                                               mean_resid=("resid", "mean"),
                                               frac_clear=("y", lambda x: (x == 3).mean())).round(3).reset_index().to_dict("records")
    groups = [g.resid.values for _, g in d.groupby("date")]
    S["drift_kruskal_p"] = float(kruskal(*groups).pvalue) if len(groups) > 1 else None
    d["anchor_acc"] = d.anchor_view == "acceleration"
    S["anchor_view_by_date"] = pd.crosstab(d.date, d.anchor_view).to_dict()
    S["clear_vs_anchor_view"] = d.groupby("anchor_view").agg(n=("y", "size"), frac_clear=("y", lambda x: (x == 3).mean()),
                                                             mean_E=("E_model", "mean")).round(3).reset_index().to_dict("records")
    d.to_csv(out / "windows_scored.csv", index=False)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: v for k, v in S.items() if k not in ("anchor_view_by_date",)}, indent=1,
                     default=lambda x: round(float(x), 3)))
    figures(d, out)
    if a.sheets:
        sheets(ol, out)


def figures(d, out):
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, col, title in zip(axs, ["vg_sem", "simple", "E_model"],
                              ["current: event-line semblance (velocity gauss)", "new simple metric (LOSO z-score)",
                               "new model: expected confidence (LOSO)"]):
        data = [d[d.y == k][col].dropna() for k in (1, 2, 3)]
        ax.boxplot(data, tick_labels=["0-1", "2", "3"], showfliers=False)
        for k, x in zip((1, 2, 3), data):
            ax.scatter(np.random.default_rng(k).normal(k, 0.06, len(x)), x, s=6, alpha=0.5)
        ax.set_xlabel("given confidence")
        r = evaluate(d, col)
        ax.set_title(f"{title}\nSpearman {r['spearman']:.2f}, AUC usable {r['auc_usable']:.2f}, clear {r['auc_clear']:.2f}",
                     fontsize=9)
    C.savefig(fig, out / "fig1_metrics_vs_confidence.png")
    fig, ax = plt.subplots(figsize=(9, 4))
    for k, (dt, g) in enumerate(d.groupby("date")):
        ax.scatter(np.arange(len(g)) + k * 0 + g.index.min(), g.resid, s=8, label=dt)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("reading order (slope accept time)")
    ax.set_ylabel("given - expected confidence")
    ax.legend(fontsize=8)
    C.savefig(fig, out / "fig2_drift.png")


def sheets(ol, out):
    """Contact sheets of the outliers' slope snapshots (read from Z:), 6 per page."""
    import matplotlib.image as mpimg
    sd = out / "outlier_sheets"
    sd.mkdir(exist_ok=True)
    for grp, g in ol.groupby("confidence"):
        g = g.reset_index(drop=True)
        for page in range(0, len(g), 6):
            sub = g.iloc[page:page + 6]
            fig, axs = plt.subplots(3, 2, figsize=(20, 20))
            for ax, (_, r) in zip(axs.ravel(), sub.iterrows()):
                ax.axis("off")
                if isinstance(r.slope_png, str) and os.path.exists(r.slope_png):
                    ax.imshow(mpimg.imread(r.slope_png))
                ax.set_title(f"{C.subj_short(r.subject)} {r.folder[-8:]} w{int(r.window)} {r.label}: given {int(r.confidence)}, "
                             f"expected {r.E_model:.1f} (simple {r.E_simple:.1f}), {r.speed:.1f} m/s", fontsize=11)
            for ax in axs.ravel()[len(sub):]:
                ax.axis("off")
            C.savefig(fig, sd / f"given{int(grp)}_p{page // 6 + 1}.png", dpi=55)


if __name__ == "__main__":
    main()
