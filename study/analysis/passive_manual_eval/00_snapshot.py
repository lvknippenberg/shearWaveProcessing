"""Freeze the manual passive reading into a dated snapshot, then build flat tables from it.

    python 00_snapshot.py                 # copy Z:/raw_data/*/*/output/swp_passive_manual -> snapshots/<stamp>/
    python 00_snapshot.py --tables-only --snapshot <stamp>   # rebuild the tables of an existing snapshot

Copied per folder (PNG snapshots are NOT copied; their path on Z: is kept in the tables):
  every *.json / *.jsonl / *.npz of the manual dir, and the same of each archive_* sub-dir.
Read-only on Z: and on the repo. The session may be writing while this runs: every file is written
atomically there, and only records whose hashes chain up (store.state logic) count as current.

Tables (snapshots/<stamp>/tables/):
  folders.csv  one row per folder with a general line (subject, acquisition order, view, ECG, stage)
  lines.csv    one row per line: general + one per event (geometry, preload, reuse, frames)
  windows.csv  one row per reviewed event window: label, timing, line, hand slope, confidence,
               automatic speeds/semblance of the five views, slider start, untilted flag
  review.csv   one row per proposed/automatic/reviewed window (event placement)
  log.csv      every log.jsonl record
  retest.csv   slopes of the earlier (energy-detector) reading archived by `redetect`, matched to
               the current reading of the same event (same reader, other session)
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import common as C
from swp.manual import store as S

VIEWS5 = ["displacement gauss", "velocity median", "velocity gauss", "Keijzer velocity", "acceleration"]
MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july",
                                       "august", "september", "october", "november", "december"], 1)}


def acq_time(folder_name):
    m = re.search(r"(\d{1,2})-([A-Za-z]+)-(\d{4})_(\d{2})-(\d{2})-(\d{2})", folder_name)
    if not m:
        return None
    d, mon, y, hh, mm, ss = m.groups()
    return datetime(int(y), MONTHS[mon.lower()], int(d), int(hh), int(mm), int(ss))


# ------------------------------------------------------------------ copy
def copy_folder(src: Path, dst: Path):
    n = 0
    for f in src.iterdir():
        if f.is_file() and f.suffix in (".json", ".jsonl", ".npz") and not f.name.endswith(".tmp"):
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / f.name)
            n += 1
        elif f.is_dir() and f.name.startswith("archive_"):
            n += copy_folder(f, dst / f.name)
    return n


def snapshot(root: Path) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M")
    snap = C.SNAPS / stamp
    data = snap / "data"
    folders = [Path(f) for f in S.find_folders(str(root))]
    done = []
    t0 = time.time()
    for f in folders:
        src = f / "output" / S.OUTDIR
        if not (src / "general.json").exists():
            continue
        n = copy_folder(src, data / f.parent.name / f.name)
        done.append(dict(subject=f.parent.name, folder=f.name, path=str(f), files=n))
    manifest = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), root=str(root), n_folders=len(done),
                    repo=str(C.REPO), repo_commit=_git(C.REPO), seconds=round(time.time() - t0, 1),
                    folders=done)
    (snap / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"snapshot {snap.name}: {len(done)} folders in {time.time() - t0:.0f} s")
    return snap


def _git(repo):
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:            # noqa: BLE001
        return None


# ------------------------------------------------------------------ tables
def _len_angle(pts_mm):
    pts = np.asarray(pts_mm, float)
    seg = np.diff(pts, axis=0)
    L = float(np.hypot(*seg.T).sum()) if len(pts) > 1 else 0.0
    d = pts[-1] - pts[0]
    ang = float(np.degrees(np.arctan2(d[1], d[0])))        # angle of r=0 -> end, in (x, z) image coords
    return L, ang


def _line_row(base, rec, kind, window=None):
    pts = rec.get("points4_mm")
    row = dict(base, kind=kind, window=window, hash=rec.get("hash"), time=rec.get("time"),
               source_buffer=rec.get("source_buffer"), preload=rec.get("preload"),
               motion_correction=rec.get("motion_correction"), auto_reuse=bool(rec.get("auto_reuse")),
               skipped=bool(rec.get("skipped")), excluded=rec.get("excluded"))
    if pts:
        L, ang = _len_angle(pts)
        p = np.asarray(pts, float)
        row.update(points4_mm=json.dumps(np.round(p, 3).tolist()), n_points=len(p), length_mm=L,
                   angle_deg=ang, x0_mm=p[0, 0], z0_mm=p[0, 1], x1_mm=p[-1, 0], z1_mm=p[-1, 1],
                   xm_mm=p[:, 0].mean(), zm_mm=p[:, 1].mean())
    fr = rec.get("frames") or {}
    for b in ("1", "3", "4"):
        if b in fr:
            row[f"frame{b}"] = fr[b].get("frame")
            row[f"phase{b}_ms"] = fr[b].get("phase_ms")
    ar = rec.get("auto_reuse") or {}
    if isinstance(ar, dict):
        row.update(reuse_perp_mm=ar.get("perp_mm"), reuse_shift_mm=ar.get("shift_mm"))
    rc = rec.get("reuse_check") or {}
    row.update(reuse_check_why=rc.get("why"), reuse_check_perp_mm=rc.get("perp_mm"),
               reuse_check_shift_mm=rc.get("shift_mm"))
    return row


def build_tables(snap: Path):
    man = json.loads((snap / "MANIFEST.json").read_text())
    views_manual = pd.read_csv(C.REPO / "study" / "logs" / "view_classification" / "sw_views_manual.csv")
    vlab = {(r.subject, r.folder): r.label for r in views_manual.itertuples()}
    F, Ls, W, R, LOG, RT = [], [], [], [], [], []
    for m in man["folders"]:
        subj, fold, path = m["subject"], m["folder"], m["path"]
        d = snap / "data" / subj / fold
        gen = C.read_json(d / "general.json")
        win = C.read_json(d / "windows.json")
        rev = C.read_json(d / "review.json")
        evs = C.read_json(d / "events.json") or {}
        proc = C.read_json(d / "processed.json") or {}
        slopes = C.read_json(d / "slopes.json") or {}
        ecg = (win or {}).get("ecg") or {}
        at = acq_time(fold)
        base = dict(subject=subj, folder=fold)
        # -------- folder
        stage = "general"
        if gen.get("skipped"):
            stage = "excluded" if gen.get("excluded") else "skipped"
        ew = None
        if win is not None and win.get("key", {}).get("general_hash") == gen.get("hash"):
            if not win.get("needs_review"):
                ew = dict(windows=win["windows"], hash=win["hash"], phases=win.get("window_phases"),
                          reviewed=False)
            elif rev is not None and rev.get("windows_hash") == win["hash"]:
                ew = dict(windows=rev["windows"], hash=rev["hash"], phases=rev.get("phases"),
                          reviewed=True, status=rev.get("status"))
        F.append(dict(base, path=path, acq_time=at.isoformat() if at else None,
                      view=vlab.get((subj, fold)), stage=stage, general_hash=gen.get("hash"),
                      general_time=gen.get("time"), general_source=gen.get("source_buffer"),
                      general_preload=gen.get("preload"),
                      picker=((win or {}).get("key", {}).get("detect") or {}).get("picker"),
                      reviewed=bool(ew and ew.get("reviewed")), review_status=(ew or {}).get("status"),
                      n_windows=len(ew["windows"]) if ew else None,
                      ecg_trustworthy=ecg.get("status") in ("ok", "corrected"), ecg_status=ecg.get("status"),
                      ecg_quality=ecg.get("quality"), hr_bpm=ecg.get("hr_bpm"),
                      rr_median_ms=ecg.get("rr_median_ms"), rr_cv=ecg.get("rr_cv")))
        if gen and not gen.get("skipped"):
            Ls.append(_line_row(base, gen, "general"))
        if ew is None:
            continue
        # -------- review / placement
        if ew.get("reviewed"):
            for k, w in enumerate(win.get("windows") or []):
                R.append(dict(base, kind="auto", idx=k, t_peak=w.get("t_peak"), t0=w.get("t0"),
                              t1=w.get("t1"), label=w.get("label"), screened=w.get("screened"),
                              screen=w.get("screen"), auto_speed=w.get("speed_m_s"),
                              phase_ms=((win.get("window_phases") or [{}] * 99)[k] or {}).get("phase_ms"),
                              search0=(w.get("search") or [None, None])[0],
                              search1=(w.get("search") or [None, None])[1]))
            for k, w in enumerate(rev.get("proposals") or []):
                R.append(dict(base, kind="proposal", idx=k, t0=w.get("t0"), t1=w.get("t1"),
                              label=w.get("label"), proposed_from=rev.get("proposed_from")))
            for k, w in enumerate(ew["windows"]):
                ph = (ew.get("phases") or [{}] * 99)[k] or {}
                R.append(dict(base, kind="reviewed", idx=k, t_peak=w.get("t_peak"), t0=w.get("t0"),
                              t1=w.get("t1"), label=w.get("label"), expect=w.get("expect"),
                              phase_ms=ph.get("phase_ms"), rr_ms=ph.get("rr_ms"), qs2_ms=ph.get("qs2_ms"),
                              proposed_from=rev.get("proposed_from"), review_time=rev.get("time")))
        # -------- events + slopes
        events = evs.get("events", {}) if evs.get("windows_hash") == ew["hash"] else {}
        for k, w in enumerate(ew["windows"]):
            i = str(k)
            ph = (ew.get("phases") or [{}] * 99)[k] or {}
            row = dict(base, window=k, label=w.get("label"), t_peak_ms=w["t_peak"] * 1e3,
                       t0_ms=w["t0"] * 1e3, t1_ms=w["t1"] * 1e3, phase_ms=ph.get("phase_ms"),
                       rr_ms=ph.get("rr_ms"), hr_bpm=ph.get("hr_bpm"), qs2_ms=ph.get("qs2_ms"),
                       screened=bool(w.get("screened")), ecg_trustworthy=ecg.get("status") in ("ok", "corrected"),
                       reviewed=ew.get("reviewed"))
            e = events.get(i)
            if e is None:
                row["state"] = "no-line"
                W.append(row)
                continue
            Ls.append(_line_row(base, e, "event", k))
            if e.get("skipped"):
                row["state"] = "line-skipped"
                W.append(row)
                continue
            row.update(line_hash=e["hash"], line_reused=bool(e.get("auto_reuse")),
                       line_is_general=(e["hash"] == gen.get("hash")), line_preload=e.get("preload"),
                       line_time=e.get("time"))
            pr = proc.get(i)
            if pr is None or pr.get("line_hash") != e["hash"] or not (d / f"st_win{k}.npz").exists():
                row["state"] = "no-st"
                W.append(row)
                continue
            row.update(st_hash=pr["st_hash"], mline_length_mm=pr.get("mline_length_mm"))
            for v, a in (pr.get("auto") or {}).items():
                row[f"auto_c|{v}"] = a.get("speed_m_s")
                row[f"auto_sem|{v}"] = a.get("semblance")
            auto_vg = (pr.get("auto") or {}).get("velocity gauss", {}).get("speed_m_s", 3.0)
            init = auto_vg if (auto_vg is not None and np.isfinite(auto_vg) and 1.05 < abs(auto_vg) < 19.5) else 3.0
            row["slider_init"] = float(np.clip(init, -12, 12))
            sl = slopes.get(i)
            if sl is None or sl.get("st_hash") != pr["st_hash"]:
                row["state"] = "no-slope"
                W.append(row)
                continue
            if sl.get("skipped"):
                row.update(state="slope-skipped", slope_time=sl.get("time"))
                W.append(row)
                continue
            sh, dp = sl.get("shared") or {}, sl.get("disp") or {}
            sp = sh.get("speed_m_s")
            row.update(state="slope", confidence=sl.get("confidence"), speed=sp,
                       anchor_t_ms=sh.get("anchor_t_ms"), anchor_r_mm=sh.get("anchor_r_mm"),
                       anchor_view=sh.get("anchor_view"), disp_unlinked=bool(sl.get("disp")),
                       speed_disp=(dp.get("speed_m_s") if sl.get("disp") else sp),
                       disp_anchor_t_ms=dp.get("anchor_t_ms"), disp_anchor_r_mm=dp.get("anchor_r_mm"),
                       slope_time=sl.get("time"),
                       untilted=(sp is not None and abs(sp - row["slider_init"]) < 0.005),
                       slope_png=str(Path(path) / "output" / S.OUTDIR / f"slope{k}.png"))
            W.append(row)
        # -------- log
        lp = d / "log.jsonl"
        if lp.exists():
            for line in lp.read_text().splitlines():
                if line.strip():
                    LOG.append(dict(base, **{k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                                             for k, v in json.loads(line).items()}))
        # -------- retest: the archived energy-detector reading of the same folder
        for arch in sorted(d.glob("archive_*redetect_valves")):
            asl = C.read_json(arch / "slopes.json") or {}
            for i, s in asl.items():
                if s.get("skipped") or not s.get("shared"):
                    continue
                RT.append(dict(base, archive=arch.name, window_old=int(i), label_old=s.get("label"),
                               t_peak_ms_old=s.get("t_peak_ms"), confidence_old=s.get("confidence"),
                               speed_old=(s.get("shared") or {}).get("speed_m_s"),
                               anchor_t_ms_old=(s.get("shared") or {}).get("anchor_t_ms"),
                               anchor_r_mm_old=(s.get("shared") or {}).get("anchor_r_mm"),
                               auto_vg_old=((s.get("auto") or {}).get("velocity gauss") or {}).get("speed_m_s"),
                               time_old=s.get("time")))
    tdir = snap / "tables"
    tdir.mkdir(exist_ok=True)
    folders = pd.DataFrame(F)
    # acquisition order within subject (all folders with a general line, by acquisition clock)
    folders["acq_order"] = folders.groupby("subject")["acq_time"].rank(method="first").astype(int)
    folders.to_csv(tdir / "folders.csv", index=False)
    pd.DataFrame(Ls).to_csv(tdir / "lines.csv", index=False)
    wdf = pd.DataFrame(W)
    wdf.to_csv(tdir / "windows.csv", index=False)
    pd.DataFrame(R).to_csv(tdir / "review.csv", index=False)
    pd.DataFrame(LOG).to_csv(tdir / "log.csv", index=False)
    rt = pd.DataFrame(RT)
    if len(rt):                       # match each old slope to the current window of the same event
        cur = wdf[wdf.state == "slope"]
        rows = []
        for r in rt.itertuples():
            c = cur[(cur.subject == r.subject) & (cur.folder == r.folder)]
            if not len(c):
                continue
            # the old windows were 100 ms around t_peak; the new event may be re-centred
            dt = (c.anchor_t_ms - r.anchor_t_ms_old).abs() if r.anchor_t_ms_old is not None else None
            j = dt.idxmin() if dt is not None and dt.notna().any() else (c.t_peak_ms - r.t_peak_ms_old).abs().idxmin()
            rows.append(dict(r._asdict(), window_new=c.loc[j, "window"], label_new=c.loc[j, "label"],
                             t_peak_ms_new=c.loc[j, "t_peak_ms"], confidence_new=c.loc[j, "confidence"],
                             speed_new=c.loc[j, "speed"], anchor_t_ms_new=c.loc[j, "anchor_t_ms"],
                             anchor_dt_ms=float(abs(c.loc[j, "anchor_t_ms"] - r.anchor_t_ms_old))
                             if r.anchor_t_ms_old is not None else None,
                             untilted_new=c.loc[j, "untilted"], line_is_general_new=c.loc[j, "line_is_general"]))
        pd.DataFrame(rows).drop(columns=["Index"], errors="ignore").to_csv(tdir / "retest.csv", index=False)
    n = wdf.state.value_counts().to_dict() if len(wdf) else {}
    print(f"tables: {len(F)} folders, {len(Ls)} lines, {len(W)} windows {n}, {len(R)} review rows, "
          f"{len(LOG)} log records, {len(RT)} archived slopes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(C.ROOT))
    ap.add_argument("--tables-only", action="store_true")
    ap.add_argument("--snapshot", default=None)
    a = ap.parse_args()
    snap = C.snapshot_dir(a.snapshot) if a.tables_only else snapshot(Path(a.root))
    build_tables(snap)


if __name__ == "__main__":
    main()
