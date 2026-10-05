"""Point 4: can the septal thickness be estimated from these B-modes?

For every folder with a general line, the wall thickness across the line (septum.py) is measured on:
  ED      buffers 1, 3, 4 at the general line's R-peak frames (end-diastole, the IVSd phase);
  beats   buffer 1 at EVERY R-peak it holds (same line) -> beat-to-beat repeatability;
  adj     buffer 1 one frame either side of the R-peak frame -> pure measurement noise;
  AVC/MVC buffers 1, 3, 4 at the event frames of the event lines (AVC ~ end-systole -> thickening).
Read-only on Z: (B-mode IQ files). ~1 min per 10 folders.

    python 04_septum.py [--snapshot <stamp>] [--jobs 6]
Outputs: results/<snap>/04_septum/ (measurements.csv, summary.json, figures)
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

import common as C
import septum as SP
import stats as ST


def _img(path, b, frame):
    from swp.manual import frames as FR
    env, x, z, k, n = FR.read_env(path, int(frame), 9 if b == 4 else 1)
    return env, x, z, k, n


def measure_folder(job):
    from swp.manual import frames as FR
    folder, subj, fold, lines = job
    out = []
    gen = [l for l in lines if l["kind"] == "general"]
    if not gen:
        return out

    def rec(kind, b, frame, pts, extra=None):
        path = f"{folder}/output/CombinedData_buffer{b}_iq.hdf5"
        try:
            env, x, z, k, n = _img(path, b, frame)
            r = SP.thickness(env, x, z, pts)
        except Exception as exc:                            # noqa: BLE001
            r = dict(error=str(exc))
            k = frame
        prof = r.pop("mean_profile", None)
        row = dict(subject=subj, folder=fold, kind=kind, buffer=b, frame=int(k), **(extra or {}), **r)
        if prof is not None:
            row["profile"] = json.dumps(np.round(prof, 2).tolist())
        out.append(row)

    g = gen[0]
    pts = json.loads(g["points4_mm"])
    for b in (1, 3, 4):
        fr = g.get(f"frame{b}")
        if fr is not None and np.isfinite(fr):
            rec("ED", b, fr, pts, dict(phase_ms=g.get(f"phase{b}_ms")))
    # buffer 1: every R-peak, and the neighbours of the general frame
    f1 = g.get("frame1")
    try:
        t, rp = FR.centre_times(folder, 1)
    except Exception:                                       # noqa: BLE001
        t, rp = None, None
    if t is not None and rp is not None and len(rp):
        for j, R in enumerate(rp):
            k = int(np.argmin(np.abs(t - R)))
            if abs(t[k] - R) < 15:
                rec("beat", 1, k, pts, dict(beat=j, phase_ms=float(t[k] - R)))
    if f1 is not None and np.isfinite(f1):
        for dk in (-1, 1):
            rec("adj", 1, int(f1) + dk, pts, dict(adj=dk))
    for l in lines:
        if l["kind"] != "event" or l.get("skipped") or not isinstance(l.get("points4_mm"), str):
            continue
        lab = l.get("label")
        if lab not in ("AVC", "MVC"):
            continue
        p = json.loads(l["points4_mm"])
        for b in (1, 3, 4):
            fr = l.get(f"frame{b}")
            if fr is not None and np.isfinite(fr):
                rec(lab, b, fr, p, dict(window=l["window"], phase_ms=l.get(f"phase{b}_ms")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot)
    out = C.out_dir(snap, "04_septum")
    t = C.tables(snap)
    L, F, W = t["lines"], t["folders"], t["windows"]
    L = L.merge(W[["subject", "folder", "window", "label"]], on=["subject", "folder", "window"], how="left")
    meas = out / "measurements.csv"
    if not meas.exists():
        jobs = []
        for r in F.itertuples():
            ls = L[(L.subject == r.subject) & (L.folder == r.folder)].to_dict("records")
            ls = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in l.items()} for l in ls]
            jobs.append((r.path, r.subject, r.folder, ls))
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = [x for res in ex.map(measure_folder, jobs) for x in res]
        pd.DataFrame(rows).to_csv(meas, index=False)
    m = pd.read_csv(meas)
    summarise(m, out, F)


def summarise(m, out, F):
    S = {}
    m["valid"] = m.frac_valid >= 0.5
    m["th"] = m.t_half
    S["n_images"] = len(m)
    # feasibility: how often is the wall measurable, per buffer (ED, general line)
    ed = m[m.kind == "ED"]
    S["ED_valid_fraction_by_buffer"] = ed.groupby("buffer").valid.mean().round(3).to_dict()
    S["ED_contrast_median_dB"] = ed.groupby("buffer")[["contrast_rv", "contrast_lv"]].median().round(1).to_dict()
    S["ED_thickness_mm_valid"] = ed[ed.valid].groupby("buffer").th.describe().round(2).to_dict()
    S["ED_thickness_grad_mm_valid"] = ed[ed.valid].groupby("buffer").t_grad.describe().round(2).to_dict()
    # buffer agreement at ED (same folder)
    p = ed[ed.valid].pivot_table(index=["subject", "folder"], columns="buffer", values="th")
    agree = {}
    for a_, b_ in ((1, 3), (1, 4), (3, 4)):
        if a_ in p and b_ in p:
            q = p[[a_, b_]].dropna()
            if len(q) > 3:
                d = q[b_] - q[a_]
                agree[f"b{b_}-b{a_}"] = dict(n=len(q), bias_mm=float(d.mean()), sd_mm=float(d.std()),
                                            r=float(np.corrcoef(q[a_], q[b_])[0, 1]))
    S["ED_buffer_agreement"] = agree
    # adjacent frames (pure measurement noise) and beat-to-beat (buffer 1)
    for kind in ("adj", "beat"):
        q = pd.concat([m[(m.kind == kind) & m.valid], m[(m.kind == "ED") & (m.buffer == 1) & m.valid]]) \
            if kind == "adj" else m[(m.kind == kind) & m.valid]
        sd, dfree = ST.within_sd(q.th, q.folder)
        S[f"{kind}_within_folder_sd_mm"] = dict(sd=sd, df=dfree, n_folders=int(q.folder.nunique()))
    # between acquisitions of a subject (ED buffer 1, valid): REML subject / acquisition / beat
    q = m[(m.kind == "beat") & m.valid].dropna(subset=["th"])
    if len(q) > 10:
        fit = ST.reml_nested(q.th, q.subject, q.folder)
        S["beats_reml_mm"] = dict(mean=fit["mu"], sd_subject=np.sqrt(fit["var_subject"]),
                                  sd_acq=np.sqrt(fit["var_acq"]), sd_beat=np.sqrt(fit["var_resid"]),
                                  n=fit["n"], n_subj=fit["n_subj"], n_acq=fit["n_acq"])
    # systolic thickening: AVC (end-systole) vs ED in the same folder & buffer
    th = {}
    for b in (1, 3, 4):
        e = m[(m.kind == "ED") & (m.buffer == b) & m.valid].set_index("folder").th
        s_ = m[(m.kind == "AVC") & (m.buffer == b) & m.valid].groupby("folder").th.median()
        mv = m[(m.kind == "MVC") & (m.buffer == b) & m.valid].groupby("folder").th.median()
        j = pd.concat([e.rename("ED"), s_.rename("AVC"), mv.rename("MVC")], axis=1)
        jj = j[["ED", "AVC"]].dropna()
        jm = j[["ED", "MVC"]].dropna()
        th[f"b{b}"] = dict(n_AVC=len(jj), AVC_over_ED_median=float((jj.AVC / jj.ED).median()) if len(jj) else None,
                           frac_AVC_thicker=float((jj.AVC > jj.ED).mean()) if len(jj) else None,
                           n_MVC=len(jm), MVC_over_ED_median=float((jm.MVC / jm.ED).median()) if len(jm) else None)
    S["thickening"] = th
    # where is the drawn line on the wall? (offset of the wall middle from the line, + = deeper/LV)
    v = m[m.valid & m.kind.isin(["ED", "AVC", "MVC"])]
    S["line_vs_wall_middle_mm"] = v.groupby("buffer").wall_mid_offset.describe().round(2).to_dict()
    S["line_vs_rv_edge_mm"] = v.groupby("buffer").rv_edge.describe().round(2).to_dict()
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps(S, indent=1, default=lambda x: round(float(x), 3)))
    figures(m, out)


def figures(m, out):
    ed = m[m.kind == "ED"]
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axs[0]
    for b in (1, 3, 4):
        q = ed[(ed.buffer == b) & ed.valid]
        ax.hist(q.th, bins=np.arange(0, 20, 0.75), histtype="step", lw=1.5,
                label=f"b{b}: {len(q)}/{(ed.buffer == b).sum()} measurable")
    ax.axvspan(6, 11, color="0.9", zorder=0, label="normal IVSd (adult, ~6-11 mm)")
    ax.set_xlabel("ED wall thickness, half-level edges [mm]")
    ax.legend(fontsize=8)
    ax = axs[1]
    p = ed[ed.valid].pivot_table(index="folder", columns="buffer", values="th")
    if 1 in p and 3 in p:
        ax.scatter(p[1], p[3], s=12, label="b3 vs b1")
    if 1 in p and 4 in p:
        ax.scatter(p[1], p[4], s=12, label="b4 vs b1")
    ax.plot([0, 20], [0, 20], "k--", lw=0.7)
    ax.set_xlabel("buffer 1 ED [mm]")
    ax.set_ylabel("other buffer ED [mm]")
    ax.legend(fontsize=8)
    ax = axs[2]
    for b, cl in ((1, "C0"), (3, "C1")):
        e = m[(m.kind == "ED") & (m.buffer == b) & m.valid].set_index("folder").th
        s_ = m[(m.kind == "AVC") & (m.buffer == b) & m.valid].groupby("folder").th.median()
        j = pd.concat([e.rename("ED"), s_.rename("AVC")], axis=1).dropna()
        ax.scatter(j.ED, j.AVC, s=12, c=cl, label=f"b{b} (n={len(j)})")
    ax.plot([0, 20], [0, 20], "k--", lw=0.7)
    ax.set_xlabel("ED (R-peak) [mm]")
    ax.set_ylabel("at AVC (~end-systole) [mm]")
    ax.legend(fontsize=8)
    ax.set_title("systolic thickening expected above the diagonal", fontsize=9)
    C.savefig(fig, out / "fig1_thickness.png")
    # example profiles: 12 folders, buffer 3 ED with the edges
    q = ed[(ed.buffer == 3) & ed.profile.notna()].sample(min(12, (ed.buffer == 3).sum()), random_state=1)
    fig, axs = plt.subplots(3, 4, figsize=(16, 9))
    for ax, (_, r) in zip(axs.ravel(), q.iterrows()):
        pr = np.array(json.loads(r.profile))
        ax.plot(SP.OFF, pr, "k", lw=0.8)
        for k, cl in (("rv_edge", "C0"), ("lv_edge", "C3")):
            if np.isfinite(r.get(k, np.nan)):
                ax.axvline(r[k], color=cl, lw=1)
        ax.axvline(0, color="0.6", ls=":")
        ax.set_title(f"{C.subj_short(r.subject)} {r.folder[-8:]}: {r.th:.1f} mm, valid {r.frac_valid:.0%}"
                     if np.isfinite(r.th) else f"{C.subj_short(r.subject)} {r.folder[-8:]}: not measurable", fontsize=8)
    fig.supxlabel("offset from the M-line [mm] (+ = deeper, LV side); blue/red = RV/LV edge (median)")
    C.savefig(fig, out / "fig2_profiles_b3.png")


if __name__ == "__main__":
    main()
