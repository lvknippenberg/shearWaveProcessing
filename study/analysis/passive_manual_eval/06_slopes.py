"""Point 6: automatic slope fitting vs the hand slopes, and candidate improvements.

Reference = the hand speed. Because the slider starts at the automatic (velocity gauss) speed, an
UNTILTED slope equals the automatic speed by construction, so the main reference set is the TILTED
slopes with confidence >= 2 (the reader actively corrected the start). Lines crossing the M-line in
< 5 frames are lower bounds; they are reported separately.

Estimators (all on the stored space-times, window +-20 ms):
  cur_<view>    the current automatic slant stack (processed.json; velocity gauss = slider start)
  cur_median    median of the five current automatic speeds
  ss_flat_<v>   slant stack with the per-time spatial mean removed (remove_flat=True), window only
  best_<v>      best straight line, mean |signal| along the line (features.csv)
  bests_<v>     best straight line, |mean signal| along the line = one polarity band (features.csv)
  lag_<v>       arrival delay vs r by cross-correlation with the basal 5 mm, Theil-Sen fit on the
                part that correlates (corr >= 0.5) and lies beyond the in-phase basal block
  anch_<v>      SEMI-AUTOMATIC: the reader's anchor click + the best |mean signal| tilt through it
  consensus     median over views of bests_ (velocity gauss, Keijzer, acceleration)
LOSO selection: the estimator chosen on the other subjects, scored on the held-out subject.

    python 06_slopes.py [--snapshot <stamp>] [--jobs 6]
Outputs: results/<snap>/06_slopes/
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import theilslopes

import common as C
import stlib as L

FS = 925.9
SHORT = {"displacement gauss": "disp", "velocity median": "vmed", "velocity gauss": "vg",
         "Keijzer velocity": "keij", "acceleration": "acc"}


def lag_speed(d, t, r, t0, t1, block_corr=0.9, min_corr=0.5):
    lag, cc = L.lag_profile(d, t, r, t0, t1)
    rr = r * 1e3
    # skip the in-phase basal block: leading samples with |lag| < 1 frame and corr >= block_corr
    dt = t[1] - t[0]
    k = 0
    while k < len(rr) and abs(lag[k]) < dt and cc[k] >= block_corr:
        k += 1
    ok = (np.arange(len(rr)) >= max(k - 5, 0)) & (cc >= min_corr) & np.isfinite(lag)
    if ok.sum() < 20 or np.ptp(rr[ok]) < 8:
        return np.nan, k * (rr[1] - rr[0])
    sl, ic, lo, hi = theilslopes(lag[ok] * 1e3, rr[ok])          # ms per mm
    if abs(sl) < 1e-3:
        return np.inf, k * (rr[1] - rr[0])
    return float(1.0 / sl), float(k * (rr[1] - rr[0]))         # mm/ms = m/s


def anchored(d, t, r, ta, ra, speeds=L.SPEEDS):
    best, bc = -1, np.nan
    for c in speeds:
        v = L.sample_line(d, t, r, ta, ra, c)
        ok = np.isfinite(v)
        if ok.mean() < 0.5:
            continue
        s = abs(np.nanmean(v))
        if s > best:
            best, bc = s, c
    return bc


def estimate(args):
    snap, row = args
    from swp.viz.metrics import slant_stack_speed
    from swp.viz.speed.spacetime import SpaceTime
    out = dict(subject=row["subject"], folder=row["folder"], window=int(row["window"]))
    st = C.load_st(C.st_path(snap, row["subject"], row["folder"], row["window"]))
    t0, t1 = row["t0_ms"] * 1e-3, row["t1_ms"] * 1e-3
    for v in st["views"]:
        k = SHORT[v["name"]]
        d, t, r = v["data"], v["t"], v["r"]
        m = (t >= t0) & (t <= t1)
        sem, c = slant_stack_speed(SpaceTime(d[m], r, t[m], v["quantity"]), None, cmin=1.0, cmax=20.0,
                                   remove_flat=True)
        out[f"ss_flat_{k}"] = c
        out[f"lag_{k}"], out[f"block_mm_{k}"] = lag_speed(d, t, r, t0, t1)
        if np.isfinite(row.get("anchor_t_ms", np.nan)):
            out[f"anch_{k}"] = anchored(d, t, r, row["anchor_t_ms"] * 1e-3, row["anchor_r_mm"] * 1e-3)
    return out


def score(est, ref, length_mm=None):
    """log-ratio stats of an estimator vs the hand speed (positive speeds; inf/neg -> miss), and the
    DELAY error across the whole M-line, |L/c_est - L/c_hand| in ms (a wrong sign counts as its delay
    error too) - the natural unit of a hand slope, insensitive to the speed blow-up of near-vertical
    bands."""
    e = np.asarray(est, float)
    h = np.asarray(ref, float)
    ok = np.isfinite(h) & (h > 0)
    Lm = np.asarray(length_mm, float)[ok] if length_mm is not None else None
    e, h = e[ok], h[ok]
    extra = {}
    if Lm is not None:
        se = np.where(np.isfinite(e) & (e != 0), 1.0 / np.where(e == 0, np.nan, e), 0.0)
        derr = np.abs(Lm * (se - 1.0 / h))                    # ms
        extra = dict(delay_err_median_ms=float(np.median(derr)),
                     delay_within_1frame=float(np.mean(derr <= 1e3 / FS)),
                     delay_within_2frames=float(np.mean(derr <= 2e3 / FS)))
    good = np.isfinite(e) & (e > 0)
    lr = np.full(len(e), np.inf)
    lr[good] = np.log(e[good] / h[good])
    a = np.abs(lr)
    return dict(n=int(len(e)), median_abs_err_pct=float(100 * (np.exp(np.median(a)) - 1)),
                within15=float(np.mean(a <= np.log(1.15))), within30=float(np.mean(a <= np.log(1.30))),
                bias_pct=float(100 * (np.exp(np.median(lr[np.isfinite(lr)])) - 1)) if np.isfinite(lr).any() else np.nan,
                wrong_sign_or_none=float(np.mean(~good)), **extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap, "06_slopes")
    t = C.tables(snap)
    w = t["windows"]
    w = w[w.state == "slope"].copy()
    f = pd.read_csv(C.OUT / snap.name / "features.csv")
    cache = out / "estimates.csv"
    if not cache.exists():
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(estimate, [(snap, r) for r in w.to_dict("records")], chunksize=4))
        pd.DataFrame(rows).to_csv(cache, index=False)
    e = pd.read_csv(cache)
    d = w.merge(f, on=["subject", "folder", "window"]).merge(e, on=["subject", "folder", "window"])
    d["cross_frames"] = d.mline_length_mm / d.speed.abs() * FS / 1e3
    d["resolved"] = d.cross_frames >= 5
    d["tilted"] = ~d.untilted.astype(bool)
    for v, k in SHORT.items():
        d[f"cur_{k}"] = d[f"auto_c|{v}"]
        d[f"best_{k}"] = d[f"{k}_best_c"]
        d[f"bests_{k}"] = d[f"{k}_bests_c"]
    d["cur_median"] = d[[f"cur_{k}" for k in SHORT.values()]].median(axis=1)
    d["consensus"] = d[["bests_vg", "bests_keij", "bests_acc"]].median(axis=1)
    d["consensus_lag"] = d[["lag_vg", "lag_keij", "lag_acc"]].median(axis=1)
    d["anch_own"] = [r[f"anch_{SHORT[r['anchor_view']]}"] if isinstance(r["anchor_view"], str) else np.nan
                     for _, r in d.iterrows()]
    ests = (["cur_median", "consensus", "consensus_lag", "anch_own"]
            + [f"{p}_{k}" for p in ("cur", "ss_flat", "best", "bests", "lag", "anch") for k in SHORT.values()])
    d.to_csv(out / "slopes_with_estimates.csv", index=False)
    sets = {
        "conf>=2 tilted resolved": d[(d.confidence >= 2) & d.tilted & d.resolved & (d.speed > 0)],
        "conf>=2 tilted": d[(d.confidence >= 2) & d.tilted & (d.speed > 0)],
        "conf=3 tilted resolved": d[(d.confidence == 3) & d.tilted & d.resolved & (d.speed > 0)],
        "conf>=2 untilted": d[(d.confidence >= 2) & ~d.tilted & (d.speed > 0)],
        "conf>=2 all": d[(d.confidence >= 2) & (d.speed > 0)],
        "conf=1": d[(d.confidence == 1) & (d.speed > 0)],
    }
    tab = []
    for name, q in sets.items():
        for est in ests:
            if est in q:
                tab.append(dict(set=name, estimator=est, **score(q[est], q.speed, q.mline_length_mm)))
    tab = pd.DataFrame(tab)
    tab.to_csv(out / "estimator_scores.csv", index=False)
    S = dict(n=len(d), sets={k: len(v) for k, v in sets.items()})
    main_set = "conf>=2 tilted resolved"
    top = tab[tab.set == main_set].sort_values("median_abs_err_pct")
    S["ranking_main_set"] = top.head(15).round(3).to_dict("records")
    S["current_vg_all_sets"] = tab[tab.estimator == "cur_vg"].round(3).to_dict("records")
    # LOSO selection among fully automatic estimators (no anchor)
    auto_ests = [x for x in ests if not x.startswith("anch_")]
    q = sets[main_set]
    errs = []
    for sbj in q.subject.unique():
        tr, te = q[q.subject != sbj], q[q.subject == sbj]
        best = min(auto_ests, key=lambda x: score(tr[x], tr.speed)["median_abs_err_pct"])
        lr = np.log(te[best].where(te[best] > 0) / te.speed)
        errs += list(np.abs(lr.fillna(np.inf)))
    errs = np.array(errs)
    S["loso_selected_auto"] = dict(median_abs_err_pct=float(100 * (np.exp(np.median(errs)) - 1)),
                                   within15=float(np.mean(errs <= np.log(1.15))),
                                   within30=float(np.mean(errs <= np.log(1.30))))
    # where does the current fit fail? (main set, |err| > 30 %)
    q = sets["conf>=2 tilted"].copy()
    q["err_cur"] = np.log(q.cur_vg.where(q.cur_vg > 0) / q.speed)
    q["fail"] = ~(q.err_cur.abs() <= np.log(1.3))
    S["current_failures"] = dict(
        n=int(q.fail.sum()), of=len(q),
        too_fast=int((q.err_cur > np.log(1.3)).sum()), too_slow=int((q.err_cur < -np.log(1.3)).sum()),
        wrong_sign=int((q.cur_vg <= 0).sum()), railed=int((q.cur_vg.abs() >= 19).sum()),
        block_mm_median_fail=float(q[q.fail].block_mm_vg.median()), block_mm_median_ok=float(q[~q.fail].block_mm_vg.median()),
        flat_median_fail=float(q[q.fail].vg_flat.median()), flat_median_ok=float(q[~q.fail].vg_flat.median()))
    # how much did the reader tilt? (all conf>=2)
    u = d[(d.confidence >= 2) & (d.speed > 0)]
    S["tilt"] = dict(frac_untilted=float(u.untilted.mean()),
                     tilt_ratio_median=float(np.median(u.speed / u.slider_init)),
                     tilted_hand_over_start_quartiles=np.percentile((u.speed / u.slider_init)[u.tilted], [25, 50, 75]).tolist(),
                     untilted_by_conf=d.groupby("confidence").untilted.mean().round(2).to_dict(),
                     untilted_by_date=d.groupby(pd.to_datetime(d.slope_time).dt.date.astype(str)).untilted.mean().round(2).to_dict())
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=lambda x: round(float(x), 3)))
    figures(d, sets, out)


def figures(d, sets, out):
    q = sets["conf>=2 tilted"]
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    for ax, est, title in zip(axs, ["cur_vg", "bests_acc", "consensus", "anch_acc"],
                              ["current (slant stack, velocity gauss)", "best one-polarity line, acceleration",
                               "consensus of best lines (vg, Keijzer, acc)", "reader's anchor + auto tilt (acc)"]):
        x, y = q.speed, q[est].where(q[est] > 0)
        col = np.where(q.resolved, "C0", "C3")
        ax.scatter(x, y.clip(upper=25).fillna(0.6), c=col, s=14)
        ax.plot([0.5, 25], [0.5, 25], "k--", lw=0.7)
        for f_ in (1.3, 1 / 1.3):
            ax.plot([0.5, 25], [0.5 * f_, 25 * f_], "k:", lw=0.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.5, 25)
        ax.set_ylim(0.5, 25)
        ax.set_xlabel("hand speed [m/s]")
        ax.set_title(title, fontsize=9)
    axs[0].set_ylabel("estimator [m/s] (no estimate / wrong sign at 0.6)")
    axs[0].text(0.02, 0.98, "conf >= 2, tilted\nred: < 5 frames (lower bound)", transform=axs[0].transAxes,
                va="top", fontsize=8)
    C.savefig(fig, out / "fig1_estimators_vs_hand.png")


if __name__ == "__main__":
    main()
