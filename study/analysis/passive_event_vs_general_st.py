"""Would the MVC speed change if the general M-line were used instead of the MVC event line?

Companion of passive_event_vs_general.py (geometry). For every MVC window whose event line
DIFFERS from the general line and has a hand slope (confidence >= 1), the five space-times are
computed on the general line with the manual study's own pipeline (swp.manual.worker settings,
same window + pads), and compared with the event-line space-times:

* ``corr``  - correlation of the two space-times on common ground: each event-line sample is
  projected onto the general line and the general space-time is read there (overlap only);
* ``auto_ev`` / ``auto_gen`` - the slant-stack speed of each (as stored by the worker);
* ``anch_ev`` / ``anch_gen`` - the best speed through the reader's anchor: the line through
  (anchor time, anchor position) with the highest tracking score within x0.63..x1.6 of the hand
  speed. On the general line the anchor is the projection of the same tissue point. This is the
  speed a reader starting from the same wavefront point would most likely settle on;
* ``track_ev`` / ``track_gen`` - tracking score of the reader's own line (anchor + speed) on each.

The event-line space-time is recomputed too and checked against the stored st_win<i>.npz
(``repro`` = correlation, should be 1.0) so both come from the identical code path.

Writes study/logs/passive_event_vs_general_st.csv.

Usage:  python study/analysis/passive_event_vs_general_st.py [--root Z:/raw_data] [--label MVC]
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
os.environ.setdefault("KERAS_BACKEND", "torch")
from swp.manual import store as S    # noqa: E402
from swp.manual import worker as W   # noqa: E402

OUT = os.path.join(REPO, "study", "logs", "passive_event_vs_general_st.csv")
LOCAL = np.geomspace(1 / 1.6, 1.6, 81)      # speeds searched around the reader's: x0.63 .. x1.6
VIEWS = ("velocity gauss", "displacement gauss")


def _score(d, r, t, ta, ra, c):
    """Tracking score of the line r = ra + c (t - ta): mean |d| along it / panel RMS."""
    rr = ra + c * (t - ta)
    ok = (rr >= r[0]) & (rr <= r[-1])
    if ok.sum() < 5:
        return np.nan
    ri = np.clip(np.round(np.interp(rr[ok], r, np.arange(r.size))).astype(int), 0, r.size - 1)
    return float(np.mean(np.abs(d[ri, np.arange(t.size)[ok]])) / (np.sqrt(np.mean(d ** 2)) + 1e-30))


def _anchored_best(d, r, t, ta, ra, c0):
    """The best-tracking speed through the anchor within x0.63..x1.6 of ``c0`` (same sign): where
    a reader tilting from the same wavefront point would settle. (An unrestricted search through
    one point locks onto near-horizontal lines along the strongest sample.)"""
    speeds = c0 * LOCAL
    s = np.array([_score(d, r, t, ta, ra, c) for c in speeds])
    if not np.isfinite(s).any():
        return np.nan, np.nan
    k = int(np.nanargmax(s))
    return float(speeds[k]), float(s[k])


def _project(x, z, gx, gz, gr):
    """Arc length on the general line (gx, gz, gr) of the nearest point to each (x, z)."""
    d2 = (x[:, None] - gx[None, :]) ** 2 + (z[:, None] - gz[None, :]) ** 2
    k = np.argmin(d2, axis=1)
    return gr[k], np.sqrt(d2[np.arange(k.size), k])


def _orient(d, r, t):
    d = np.asarray(d, float)
    return d.T if d.shape == (t.size, r.size) else d          # (n_r, n_t)


def folder_rows(folder, label):
    from swp.passive import _build_views
    from swp.viz.metrics import slant_stack_speed
    from swp.viz.mline import mline_from_points
    from swp.viz.pipeline import run_pipeline

    p = S.Paths(folder)
    gen = S.read_json(p.general_json)
    ew = S.event_windows(p)
    ev = S.read_json(p.events_json) or {}
    slopes = S.read_json(p.slopes_json) or {}
    proc = S.read_json(p.processed_json) or {}
    if not gen or gen.get("skipped") or ew is None or ev.get("windows_hash") != ew["hash"]:
        return []
    g_pts = S.load_points(p.general_npz)
    todo = []
    for k, e in ev["events"].items():
        i = int(k)
        sl = slopes.get(k) or {}
        if (e.get("skipped") or ew["windows"][i].get("label") != label or sl.get("skipped")
                or not sl.get("confidence") or sl.get("st_hash") != (proc.get(k) or {}).get("st_hash")):
            continue
        pts = S.load_points(p.event_npz(i))
        if pts.shape == g_pts.shape and np.allclose(pts, g_pts):
            continue                                              # the general line itself
        todo.append((i, pts, sl))
    if not todo:
        return []
    cfg = W._cfg()
    cfg["data"]["root"] = p.output
    acq = W._load(p, [g_pts] + [q for _, q, _ in todo])
    views = [(n, v) for n, v in _build_views(cfg, acq) if n in VIEWS]
    mg = mline_from_points(g_pts, S.N_SAMPLES)
    rows = []
    for i, pts, sl in todo:
        w = ew["windows"][i]
        me = mline_from_points(pts, S.N_SAMPLES)
        i0 = int(np.argmin(np.abs(acq.t - (w["t0"] - W.PAD_S))))
        i1 = int(np.argmin(np.abs(acq.t - (w["t1"] + W.PAD_S)))) + 1
        acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        stored = np.load(p.st_npz(i), allow_pickle=False)
        names = [str(stored[f"v{j}_name"]) for j in range(int(stored["n_views"]))]
        sh = sl.get("shared") or {}
        # same tissue point on the general line; flip the sign if the general line runs the other way
        rg_of_e, dist = _project(me.x, me.z, mg.x, mg.z, mg.r)
        sign = np.sign(np.polyfit(me.r, rg_of_e, 1)[0]) or 1.0
        row = dict(subject=os.path.basename(os.path.dirname(folder)), acq=os.path.basename(folder),
                   window=i, label=label, confidence=sl.get("confidence"),
                   hand_speed=sh.get("speed_m_s"), phase_ms=(ew["phases"][i] or {}).get("phase_ms"),
                   len_ev_mm=me.r[-1] * 1e3, len_gen_mm=mg.r[-1] * 1e3,
                   proj_dist_mean_mm=float(dist.mean() * 1e3), sign=sign)
        for name, vcfg in views:
            key = "vel" if name.startswith("velocity") else "disp"
            res_e = run_pipeline(acq_w, me, vcfg, focus=None)
            res_g = run_pipeline(acq_w, mg, vcfg, focus=None)
            r_e, t_e = np.asarray(res_e.st.r, float), np.asarray(res_e.st.t, float)
            r_g = np.asarray(res_g.st.r, float)
            de, dg = _orient(res_e.st.data, r_e, t_e), _orient(res_g.st.data, r_g, t_e)
            j = names.index(name)
            ds = _orient(stored[f"v{j}_data"], np.asarray(stored[f"v{j}_r"]), np.asarray(stored[f"v{j}_t"]))
            row[f"{key}_repro"] = float(np.corrcoef(de.ravel(), ds.ravel())[0, 1])
            # general st read at the projections of the event samples (event r grid)
            rg_at = np.interp(r_e, me.r, rg_of_e)
            inside = (rg_at >= r_g[0]) & (rg_at <= r_g[-1]) & (np.interp(r_e, me.r, dist) < 2e-3)
            dg_on_e = np.array([np.interp(rg_at, r_g, dg[:, k]) for k in range(t_e.size)]).T
            row[f"{key}_overlap"] = float(inside.mean())
            row[f"{key}_corr"] = (float(np.corrcoef(de[inside].ravel(), dg_on_e[inside].ravel())[0, 1])
                                  if inside.sum() > 10 else np.nan)
            row[f"{key}_auto_ev"] = (proc[str(i)]["auto"].get(name) or {}).get("speed_m_s")
            row[f"{key}_auto_gen"] = float(slant_stack_speed(res_g.st, res_g.r0, cmin=1.0, cmax=W.SPEED_CMAX,
                                                             remove_flat=False)[1])
            if sh.get("speed_m_s") is not None:
                ta = sh["anchor_t_ms"] * 1e-3
                if not (t_e[0] - 0.05 <= ta <= t_e[-1] + 0.05):
                    ta += t_e[0]
                ra = sh["anchor_r_mm"] * 1e-3
                ra_g = float(np.interp(ra, me.r, rg_of_e))
                c = sh["speed_m_s"]
                row[f"{key}_track_ev"] = _score(de, r_e, t_e, ta, ra, c)
                row[f"{key}_track_gen"] = _score(dg, r_g, t_e, ta, ra_g, sign * c)
                row[f"{key}_anch_ev"], _ = _anchored_best(de, r_e, t_e, ta, ra, c)
                cg, _ = _anchored_best(dg, r_g, t_e, ta, ra_g, sign * c)
                row[f"{key}_anch_gen"] = sign * cg
        rows.append(row)
        print(f"  {row['subject']} w{i}: corr vel {row.get('vel_corr', np.nan):.2f}  "
              f"anch {row.get('vel_anch_ev', np.nan):+.2f} -> {row.get('vel_anch_gen', np.nan):+.2f}  "
              f"repro {row.get('vel_repro', np.nan):.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--label", default="MVC")
    a = ap.parse_args()
    folders = sorted({os.path.dirname(os.path.dirname(os.path.dirname(j)))
                      for j in glob.glob(os.path.join(a.root, "*", "*", "output", S.OUTDIR, "slopes.json"))})
    rows = []
    t0 = time.perf_counter()
    for f in folders:
        try:
            rows += folder_rows(f, a.label)
        except Exception as exc:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  {f}: {exc}")
    df = pd.DataFrame(rows)
    out = OUT if a.label == "MVC" else OUT.replace(".csv", f"_{a.label}.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} windows from {len(folders)} folders in {time.perf_counter() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
