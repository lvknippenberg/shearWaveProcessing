"""Point 3: M-line placement - how lines are drawn, how they vary, and whether they can be automated.

A  geometry of the drawn lines; how far each EVENT line is from its folder's GENERAL line
   (the reader's correction for cardiac motion between the R-peak and the event + redraw scatter);
B  can the event line be predicted? swp.mline.transfer registration of the buffer-4 frame at the
   general line onto the frame at the event, applied to the general line, vs the drawn event line
   (baseline: the general line unchanged);
C  can the GENERAL line be predicted?
   C1  the previous acquisition of the same subject: its line registered onto this R-peak frame;
   C2  a wall search ("snap"): the line shifted / rotated to maximise bright wall between dark
       cavities, started from C1 or from the population prior (leave-one-subject-out);
   reference scatter: the September general lines (buffer 1) vs today's (44 folders) = the same
   reader drawing the same septum twice, months apart.
Line distance = mean perpendicular distance of the DRAWN line's middle 70 % to the predicted line
(infinite extension; length / end shifts reported separately) + orientation difference (mod 180).

    python 03_mline.py [--snapshot <stamp>] [--jobs 4]
Outputs: results/<snap>/03_mline/
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
from scipy.ndimage import map_coordinates

import common as C


# ------------------------------------------------------------------ geometry
def orient(p):
    p = np.asarray(p, float)
    d = p[-1] - p[0]
    return np.degrees(np.arctan2(d[1], d[0])) % 180.0


def dist_to_line(ref, test, frac=(0.15, 0.85)):
    """Mean |perpendicular distance| of the ref polyline's middle part to test's chord (infinite)."""
    ref, test = np.asarray(ref, float), np.asarray(test, float)
    seg = np.diff(ref, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    s = np.linspace(frac[0], frac[1], 30) * cum[-1]
    P = np.stack([np.interp(s, cum, ref[:, 0]), np.interp(s, cum, ref[:, 1])], 1)
    # distance to the test polyline (segments extended at the ends)
    best = np.full(len(P), np.inf)
    for a, b in zip(test[:-1], test[1:]):
        u = (b - a) / np.linalg.norm(b - a)
        n = np.array([-u[1], u[0]])
        best = np.minimum(best, np.abs((P - a) @ n))
    return float(best.mean())


def compare(ref, test):
    da = abs(orient(ref) - orient(test)) % 180
    da = min(da, 180 - da)
    ref, test = np.asarray(ref, float), np.asarray(test, float)
    return dict(dist_mm=dist_to_line(ref, test), dangle=da,
                d_r0_mm=float(np.hypot(*(ref[0] - test[0]))),
                dlen_mm=float(np.hypot(*np.diff(test[[0, -1]], axis=0)[0]) - np.hypot(*np.diff(ref[[0, -1]], axis=0)[0])))


# ------------------------------------------------------------------ wall score / snap
def wall_score(db, x, z, pts, cav=(5.0, 9.0)):
    """mean dB on the line (+-1 mm) minus the mean dB of both cavities (cav mm either side)."""
    pts = np.asarray(pts, float)
    p0, p1 = pts[0], pts[-1]
    u = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-u[1], u[0]])
    s = np.linspace(0.15, 0.85, 25)
    base = p0[None] + s[:, None] * (p1 - p0)[None]

    def at(off):
        Q = base[:, None, :] + np.asarray(off)[None, :, None] * n[None, None, :]
        xi = (Q[..., 0] - x[0]) / (x[1] - x[0])
        zi = (Q[..., 1] - z[0]) / (z[1] - z[0])
        return map_coordinates(db, [zi.ravel(), xi.ravel()], order=1, cval=np.nan).reshape(Q.shape[:2])
    on = np.nanmean(at(np.linspace(-1, 1, 5)))
    c1 = np.nanmean(at(np.linspace(-cav[1], -cav[0], 5)))
    c2 = np.nanmean(at(np.linspace(cav[0], cav[1], 5)))
    return float(on - 0.5 * (c1 + c2) - 0.5 * abs(c1 - c2))   # both cavities dark, symmetric-ish


def transform_line(pts, dperp, dang):
    pts = np.asarray(pts, float)
    c = pts.mean(0)
    a = np.radians(dang)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    q = c + (pts - c) @ R.T
    u = (q[-1] - q[0]) / np.linalg.norm(q[-1] - q[0])
    n = np.array([-u[1], u[0]])
    return q + dperp * n


def snap(db, x, z, pts, perp=np.arange(-8, 8.01, 0.5), angs=np.arange(-15, 15.01, 2.5), prior=None):
    best, bp = -np.inf, pts
    for da in angs:
        for dp in perp:
            q = transform_line(pts, dp, da)
            s = wall_score(db, x, z, q)
            if prior is not None:
                s += prior(q)
            if np.isfinite(s) and s > best:
                best, bp = s, q
    return bp, best


def to_db(env):
    return 20 * np.log10(env / (np.percentile(env, 99.5) + 1e-30) + 1e-6)


def register_ends(T, g, A, B, frac=0.35):
    """Each END of the line registered on its own (translation of the end segment, frac of the
    length), so the line can rotate and stretch with the septum; interior points get the shift
    interpolated along the line."""
    g = np.asarray(g, float)
    seg = np.diff(g, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    Lt = cum[-1]
    shifts = []
    for a, b in ((0.0, frac), (1 - frac, 1.0)):
        s = np.linspace(a, b, 5) * Lt
        sub = np.stack([np.interp(s, cum, g[:, 0]), np.interp(s, cum, g[:, 1])], 1)
        r = T.transfer_line(sub, (A.env, A.x_mm, A.z_mm), (B.env, B.x_mm, B.z_mm), check=False,
                            margins=(6.0, 9.0, 12.0))
        shifts.append(r.transform.apply(sub.mean(0)[None])[0] - sub.mean(0))
    w = cum / Lt
    return g + (1 - w)[:, None] * shifts[0][None] + w[:, None] * shifts[1][None]


# ------------------------------------------------------------------ per folder work
def folder_job(job):
    from swp.manual import frames as FR
    from swp.manual._light import transfer
    T = transfer()
    path, subj, fold, gen, events, prev = job
    out = dict(events=[], general={})
    b4 = f"{path}/output/CombinedData_buffer4_iq.hdf5"
    g = np.array(json.loads(gen["points4_mm"]))
    kg = int(gen["frame4"])
    try:
        A = FR.load_panel(path, b4, 4, kg, "general")
    except Exception as exc:                                  # noqa: BLE001
        out["error"] = str(exc)
        return out
    # B: event lines
    for e in events:
        ep = np.array(json.loads(e["points4_mm"]))
        row = dict(subject=subj, folder=fold, window=e["window"], label=e.get("label"),
                   reused=bool(e["auto_reuse"]), identical=bool(e["hash"] == gen["hash"]),
                   phase_ms=e.get("phase4_ms"))
        row.update({f"keep_{k}": v for k, v in compare(ep, g).items()})
        if not e["auto_reuse"] and np.isfinite(e.get("frame4", np.nan)):
            try:
                Bp = FR.load_panel(path, b4, 4, int(e["frame4"]), "event")
                r = T.transfer_line(g, (A.env, A.x_mm, A.z_mm), (Bp.env, Bp.x_mm, Bp.z_mm), check=True)
                row.update({f"reg_{k}": v for k, v in compare(ep, r.points).items()})
                row.update(reg_reliable=r.reliable(), reg_shift_mm=r.shift_mm)
                re = register_ends(T, g, A, Bp)
                if re is not None:
                    row.update({f"regends_{k}": v for k, v in compare(ep, re).items()})
                q, _ = snap(to_db(Bp.env), Bp.x_mm, Bp.z_mm, r.points, perp=np.arange(-3, 3.01, 0.5),
                            angs=np.arange(-6, 6.01, 2))
                row.update({f"regsnap_{k}": v for k, v in compare(ep, q).items()})
            except Exception as exc:                          # noqa: BLE001
                row["reg_error"] = str(exc)
        out["events"].append(row)
    # C1: previous acquisition's line registered onto this R-peak frame; C2: snap
    gr = dict(subject=subj, folder=fold)
    dbA = to_db(A.env)
    gr["drawn_wall_score"] = wall_score(dbA, A.x_mm, A.z_mm, g)
    if prev is not None:
        ppath, pgen = prev
        pp = np.array(json.loads(pgen["points4_mm"]))
        try:
            P = FR.load_panel(ppath, f"{ppath}/output/CombinedData_buffer4_iq.hdf5", 4, int(pgen["frame4"]), "prev")
            gr.update({f"prevraw_{k}": v for k, v in compare(g, pp).items()})
            r = T.transfer_line(pp, (P.env, P.x_mm, P.z_mm), (A.env, A.x_mm, A.z_mm), check=True,
                                margins=(12.0, 20.0, 30.0), max_shift_mm=25.0)
            gr.update({f"prevreg_{k}": v for k, v in compare(g, r.points).items()})
            gr.update(prevreg_reliable=r.reliable(), prevreg_shift=r.shift_mm)
            q, sc = snap(dbA, A.x_mm, A.z_mm, r.points)
            gr.update({f"prevsnap_{k}": v for k, v in compare(g, q).items()})
            q2, _ = snap(dbA, A.x_mm, A.z_mm, pp)
            gr.update({f"prevrawsnap_{k}": v for k, v in compare(g, q2).items()})
        except Exception as exc:                              # noqa: BLE001
            gr["prev_error"] = str(exc)
    # population prior (passed in gen["prior_pts"]) + snap with a wide search
    if gen.get("prior_pts"):
        pr = np.array(gen["prior_pts"])
        gr.update({f"prior_{k}": v for k, v in compare(g, pr).items()})
        q, _ = snap(dbA, A.x_mm, A.z_mm, pr, perp=np.arange(-20, 20.01, 1.0), angs=np.arange(-25, 25.01, 2.5))
        gr.update({f"priorsnap_{k}": v for k, v in compare(g, q).items()})
    # the reader's own line, snapped: how far would the snap move a correct line?
    q, _ = snap(dbA, A.x_mm, A.z_mm, g, perp=np.arange(-4, 4.01, 0.5), angs=np.arange(-6, 6.01, 2))
    gr.update({f"selfsnap_{k}": v for k, v in compare(g, q).items()})
    # September line of this folder (buffer 1 coordinates ~ buffer 4 grid)
    leg = f"{path}/output/mlines/passive_general_mline.npz"
    try:
        lp = np.load(leg)["points"] * 1e3
        gr.update({f"sept_{k}": v for k, v in compare(g, lp).items()})
    except Exception:                                         # noqa: BLE001
        pass
    out["general"] = gr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    snap_ = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap_, "03_mline")
    t = C.tables(snap_)
    L, F, W = t["lines"], t["folders"], t["windows"]
    L = L.merge(W[["subject", "folder", "window", "label"]], on=["subject", "folder", "window"], how="left")
    gens = L[(L.kind == "general") & L.points4_mm.notna()].merge(F[["subject", "folder", "path", "acq_time"]],
                                                                    on=["subject", "folder"])
    gens = gens.sort_values(["subject", "acq_time"])
    # LOSO population prior: median centre, orientation and length of the other subjects' lines
    geo = []
    for r in gens.itertuples():
        p = np.array(json.loads(r.points4_mm))
        geo.append(dict(subject=r.subject, cx=p[:, 0].mean(), cz=p[:, 1].mean(), ang=orient(p),
                        length=float(np.hypot(*(p[-1] - p[0])))))
    geo = pd.DataFrame(geo)
    cache = out / "raw.json"
    if not cache.exists():
        jobs = []
        for s, g in gens.groupby("subject"):
            prev = None
            for r in g.itertuples():
                o = geo[geo.subject != s]
                ang = np.radians(o.ang.median())
                half = o.length.median() / 2
                c = np.array([o.cx.median(), o.cz.median()])
                u = np.array([np.cos(ang), np.sin(ang)])
                prior_pts = [list(c + half * u), list(c - half * u)]
                gd = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r._asdict().items()}
                gd["prior_pts"] = prior_pts
                ev = L[(L.subject == s) & (L.folder == r.folder) & (L.kind == "event") & L.points4_mm.notna()]
                evs = [{k: (None if isinstance(v, float) and np.isnan(v) and k != "frame4" else v)
                        for k, v in e.items()} for e in ev.to_dict("records")]
                jobs.append((r.path, s, r.folder, gd, evs, prev))
                prev = (r.path, gd)
        with ProcessPoolExecutor(a.jobs) as ex:
            res = list(ex.map(folder_job, jobs))
        cache.write_text(json.dumps(res, default=float))
    res = json.loads(cache.read_text())
    ev = pd.DataFrame([e for r in res for e in r["events"]])
    gn = pd.DataFrame([r["general"] for r in res if r.get("general")])
    ev.to_csv(out / "event_lines.csv", index=False)
    gn.to_csv(out / "general_lines.csv", index=False)
    S = summarise(ev, gn, L, geo)
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=lambda x: round(float(x), 3)))
    figures(ev, gn, out)


def _succ(df, p, d=1.5, a=5.0):
    m = df[f"{p}_dist_mm"].notna()
    q = df[m]
    return dict(n=int(m.sum()), dist_median=float(q[f"{p}_dist_mm"].median()),
                dist_p75=float(q[f"{p}_dist_mm"].quantile(0.75)),
                dangle_median=float(q[f"{p}_dangle"].median()),
                within_1p5mm_5deg=float(((q[f"{p}_dist_mm"] <= d) & (q[f"{p}_dangle"] <= a)).mean()),
                within_1mm=float((q[f"{p}_dist_mm"] <= 1.0).mean()))


def summarise(ev, gn, L, geo):
    S = {}
    gl = L[L.kind == "general"]
    S["general_lines"] = dict(n=len(gl), two_points=float((gl.n_points == 2).mean()),
                              length_mm=gl.length_mm.describe().round(1).to_dict(),
                              depth_mid_mm=gl.zm_mm.describe().round(1).to_dict(),
                              orientation_deg=geo.ang.describe().round(1).to_dict())
    drawn = ev[~ev.reused & ~ev.identical]
    S["event_lines"] = dict(n=len(ev), reused=int(ev.reused.sum()), identical_to_general_not_reused=int((ev.identical & ~ev.reused).sum()),
                            redrawn=len(drawn))
    S["event_vs_general_redrawn"] = {lab: _succ(g, "keep") for lab, g in drawn.groupby("label")}
    S["event_vs_general_redrawn"]["all"] = _succ(drawn, "keep")
    S["predict_event_line"] = dict(keep_general=_succ(drawn, "keep"), registered=_succ(drawn, "reg"),
                                   registered_ends=_succ(drawn, "regends"),
                                   registered_snapped=_succ(drawn, "regsnap"))
    if "reg_reliable" in drawn:
        rel = drawn[drawn.reg_reliable == True]               # noqa: E712
        S["predict_event_line"]["registered_trusted_only"] = _succ(rel, "reg")
        S["predict_event_line"]["frac_trusted"] = float(drawn.reg_reliable.mean())
    for lab in ("MVC", "AVC", "AK"):
        q = drawn[drawn.label == lab]
        if len(q):
            S["predict_event_line"][f"{lab}"] = dict(keep=_succ(q, "keep"), reg=_succ(q, "reg"),
                                                     regends=_succ(q, "regends"))
    S["predict_general_line"] = {p: _succ(gn, p) for p in ("sept", "selfsnap", "prevraw", "prevreg", "prevsnap",
                                                           "prevrawsnap", "prior", "priorsnap") if f"{p}_dist_mm" in gn}
    return S


def figures(ev, gn, out):
    drawn = ev[~ev.reused & ~ev.identical]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axs[0]
    for p, lab in (("keep", "general line kept"), ("reg", "general line registered"),
                   ("regends", "each end registered"), ("regsnap", "registered + snapped")):
        x = np.sort(drawn[f"{p}_dist_mm"].dropna())
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=lab)
    ax.set_xlim(0, 6)
    ax.set_xlabel("drawn event line vs prediction: mean distance [mm]")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=8)
    ax.set_title(f"event lines redrawn by the reader (n={len(drawn)})", fontsize=9)
    ax = axs[1]
    for p, lab in (("sept", "September line (same reader)"), ("prevraw", "previous acquisition, as drawn"),
                   ("prevreg", "previous acq., registered"), ("prevsnap", "previous acq., registered + snapped"),
                   ("priorsnap", "population prior + wall search")):
        if f"{p}_dist_mm" in gn:
            x = np.sort(gn[f"{p}_dist_mm"].dropna())
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=f"{lab} (n={len(x)})")
    ax.set_xlim(0, 15)
    ax.set_xlabel("drawn general line vs prediction: mean distance [mm]")
    ax.legend(fontsize=8)
    C.savefig(fig, out / "fig1_line_prediction.png")


if __name__ == "__main__":
    main()
