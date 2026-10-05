"""Point 3, part D: how much does the speed change when the event line is placed differently?

For a sample of clear (confidence 3), well-resolved windows, the worker's five space-times are
recomputed (same code path as swp.manual.worker.process, read-only, outputs here) on the drawn
line and on perturbed copies: shifted across the wall by +-1 and +-2 mm, rotated +-5 deg about
its centre. Reported: the change of the automatic speeds (slant stack velocity median /
acceleration, best one-polarity line on acceleration) against the unperturbed line, and the
change of the DELAY of the reader's own line position (lag fit on the hand-anchored band).
This sets the accuracy an automatic line has to reach.

    python 03b_mline_sensitivity.py [--snapshot <stamp>] [--n 24] [--jobs 3]
Outputs: results/<snap>/03_mline/sensitivity.csv, sensitivity_summary.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import common as C
import stlib as L

PERTURB = [("ref", 0.0, 0.0), ("perp+1", 1.0, 0.0), ("perp-1", -1.0, 0.0), ("perp+2", 2.0, 0.0),
           ("perp-2", -2.0, 0.0), ("rot+5", 0.0, 5.0), ("rot-5", 0.0, -5.0)]


def perturb(pts, dperp, dang):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    a = np.radians(dang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = c + (pts - c) @ R.T
    u = (q[-1] - q[0]) / np.linalg.norm(q[-1] - q[0])
    n = np.array([-u[1], u[0]])
    return q + dperp * n


def job(a):
    from swp.manual import store as S
    from swp.manual import worker as Wk
    from swp.passive import _build_views
    from swp.viz.metrics import slant_stack_speed
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline
    path, row = a
    p = S.Paths(path)
    pts_m = np.asarray(json.loads(row["points4_mm"]), float) * 1e-3
    lines = {k: perturb(pts_m * 1e3, dp, da) * 1e-3 for k, dp, da in PERTURB}
    cfg = Wk._cfg()
    cfg["data"]["root"] = p.output
    acq = Wk._load(p, lines.values())
    views = _build_views(cfg, acq)
    t0, t1 = row["t0_ms"] * 1e-3, row["t1_ms"] * 1e-3
    i0 = int(np.argmin(np.abs(acq.t - (t0 - Wk.PAD_S))))
    i1 = int(np.argmin(np.abs(acq.t - (t1 + Wk.PAD_S)))) + 1
    acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
    out = []
    for k, _, _ in PERTURB:
        ml = mline_from_points(lines[k], S.N_SAMPLES)
        rec = dict(subject=row["subject"], folder=row["folder"], window=row["window"], label=row["label"],
                   perturb=k, hand=row["speed"])
        for name, vcfg in views:
            if name not in ("velocity median", "velocity gauss", "acceleration"):
                continue
            res = run_pipeline(acq_w, ml, vcfg, focus=None)
            sem, c = slant_stack_speed(res.st, res.r0, cmin=1.0, cmax=20.0, remove_flat=False)
            short = {"velocity median": "vmed", "velocity gauss": "vg", "acceleration": "acc"}[name]
            rec[f"ss_{short}"], rec[f"sem_{short}"] = c, sem
            d, t, r = np.asarray(res.st.data, float), np.asarray(res.st.t), np.asarray(res.st.r)
            b = L.best_line(d, t, r, t0, t1, objective="signed")
            rec[f"bests_{short}"] = b["c"]
            rec[f"rms_{short}"] = L.panel_rms(d, t, t0, t1)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap, "03_mline")
    t = C.tables(snap)
    W, Ls, F = t["windows"], t["lines"], t["folders"]
    w = W[(W.state == "slope") & (W.confidence == 3) & (W.speed > 0)].copy()
    w["cross"] = w.mline_length_mm / w.speed * 0.926
    w = w[w.cross >= 5]
    # one window per folder, balanced MVC / AVC, fixed seed
    w = w.groupby("folder").sample(1, random_state=0)
    w = pd.concat([w[w.label == "MVC"].sample(min(a.n // 2, (w.label == "MVC").sum()), random_state=0),
                   w[w.label == "AVC"].sample(min(a.n // 2, (w.label == "AVC").sum()), random_state=0)])
    jobs = []
    for r in w.itertuples():
        e = Ls[(Ls.subject == r.subject) & (Ls.folder == r.folder) & (Ls.kind == "event") & (Ls.window == r.window)].iloc[0]
        path = F[(F.subject == r.subject) & (F.folder == r.folder)].path.iloc[0]
        row = dict(r._asdict(), points4_mm=e.points4_mm)
        jobs.append((path, {k: v for k, v in row.items() if k != "Index"}))
    with ProcessPoolExecutor(a.jobs) as ex:
        res = [x for rr in ex.map(job, jobs) for x in rr]
    d = pd.DataFrame(res)
    d.to_csv(out / "sensitivity.csv", index=False)
    S = {}
    for est in ("ss_vmed", "ss_vg", "ss_acc", "bests_acc", "bests_vg"):
        ref = d[d.perturb == "ref"].set_index(["folder", "window"])[est]
        rows = {}
        for k, _, _ in PERTURB[1:]:
            q = d[d.perturb == k].set_index(["folder", "window"])[est]
            j = pd.concat([ref.rename("ref"), q.rename("p")], axis=1).dropna()
            ok = (j.ref > 0) & (j.p > 0)
            lr = np.abs(np.log(j.p[ok] / j.ref[ok]))
            rows[k] = dict(n=int(len(j)), median_abs_change_pct=float(100 * (np.exp(lr.median()) - 1)),
                           p75_abs_change_pct=float(100 * (np.exp(lr.quantile(0.75)) - 1)),
                           sign_flip=float((~ok).mean()))
        S[est] = rows
    # amplitude drop when leaving the wall
    for k, _, _ in PERTURB[1:]:
        ref = d[d.perturb == "ref"].set_index(["folder", "window"]).rms_vg
        q = d[d.perturb == k].set_index(["folder", "window"]).rms_vg
        S.setdefault("rms_ratio_vg", {})[k] = float((q / ref).median())
    (out / "sensitivity_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1, default=lambda x: round(float(x), 3)))


if __name__ == "__main__":
    main()
