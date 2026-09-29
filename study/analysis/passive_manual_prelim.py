"""Preliminary evaluation of the manual passive study (scripts/passive_manual.py) - collect.

Reads every ``<folder>/output/swp_passive_manual/`` under the root and writes three tables to
``study/logs/passive_manual_prelim/``:

* ``prompts.csv``  - one row per log record (every accept / skip, in order), with how many times
                     that (folder, task, window) was answered - a prompt served twice shows as n>1;
* ``lines.csv``    - one row per drawn line (general + events): source buffer, what was
                     pre-loaded, motion correction, nudge, registration verdicts, frame phases;
* ``windows.csv``  - one row per detected window: detection (energy, phase, where the peak sits in
                     its phase search window), the hand slope + confidence, the automatic fits,
                     and objective space-time measures computed here from st_win<i>.npz.

Objective space-time measures (per view; reported for the default "velocity gauss" and the
displacement view):

* ``track``   - mean |signal| along the hand line / RMS of the panel ("tracking score",
                docs/passive_speed_estimation.md: 1.0 = line on noise, manual lines on clear
                panels ~2). Only when a slope was drawn.
* ``track_best`` - the same score for the best line through the panel over a slowness grid
                (speeds 0.5-20 m/s both directions, every anchor) = how much line-shaped energy
                the panel holds at all, independent of the reader.
* ``flat``    - fraction of panel energy in the per-time spatial mean (a whole-line in-phase
                band = bulk wall motion, not a propagating wave).
* ``burst``   - RMS inside the 100 ms window / RMS in the +-20 ms pads (does the window hold
                more motion than its surroundings).

Usage:  python study/analysis/passive_manual_prelim.py [--root Z:/raw_data]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.manual import store as S   # noqa: E402

OUT = os.path.join(REPO, "study", "logs", "passive_manual_prelim")
MVC_MAX_MS, AVC_TOL_MS = 150.0, 120.0      # the detector's phase search windows (mline.select)


def _subject(folder):
    return os.path.basename(os.path.dirname(folder))


def _read_log(p):
    if not os.path.exists(p.log):
        return []
    with open(p.log) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _line_rows(folder, rec, kind, window=None, label=None):
    if not rec or rec.get("skipped"):
        return dict(folder=folder, subject=_subject(folder), kind=kind, window=window, label=label,
                    skipped=True)
    mo = rec.get("mapping_other") or {}
    fr = rec.get("frames") or {}
    row = dict(folder=folder, subject=_subject(folder), kind=kind, window=window, label=label,
               skipped=False, time=rec.get("time"), preload=rec.get("preload"),
               source_buffer=rec.get("source_buffer"), motion_correction=rec.get("motion_correction"),
               nudge_mm=float(np.hypot(*(rec.get("nudge_mm") or [0, 0]))),
               mapping_reliable=(rec.get("mapping") or {}).get("reliable"),
               mapping_shift_mm=(rec.get("mapping") or {}).get("shift_mm"),
               n_points=len(rec.get("points4_mm") or []))
    src = np.asarray(rec.get("points_src_mm") or [], float)
    p4 = np.asarray(rec.get("points4_mm") or [], float)
    row["src_to_b4_shift_mm"] = (float(np.abs(src - p4).max())
                                 if src.shape == p4.shape and src.size else np.nan)
    for b in ("1", "3"):
        m = mo.get(b) or {}
        row[f"b{b}_reliable"] = m.get("reliable")
        row[f"b{b}_shift_mm"] = m.get("shift_mm")
        row[f"b{b}_agree"] = m.get("agree")
    for b in ("1", "3", "4"):
        f = fr.get(b) or {}
        row[f"b{b}_frame"] = f.get("frame")
        row[f"b{b}_phase_ms"] = f.get("phase_ms")
        row[f"b{b}_note"] = f.get("note")
    return row


# ------------------------------------------------------------------ space-time measures
def _line_score(d, r, t, t_a, r_a, c):
    """mean |d| along r = r_a + c (t - t_a) / RMS(d); c in m/s, r in m, t in s."""
    rr = r_a + c * (t - t_a)
    ok = (rr >= r[0]) & (rr <= r[-1])
    if ok.sum() < 5:
        return np.nan
    ri = np.interp(rr[ok], r, np.arange(r.size))
    ti = np.arange(t.size)[ok]
    v = d[np.clip(np.round(ri).astype(int), 0, r.size - 1), ti]
    return float(np.mean(np.abs(v)) / (np.sqrt(np.mean(d ** 2)) + 1e-30))


def _orient(d, r, t):
    d = np.asarray(d, float)
    if d.shape == (t.size, r.size):
        d = d.T
    return d                                            # (n_r, n_t)


def _st_measures(npz, i, win, slope):
    z = np.load(npz, allow_pickle=True)
    n = int(z["n_views"])
    out = {}
    for j in range(n):
        name = str(z[f"v{j}_name"])
        r, t = np.asarray(z[f"v{j}_r"], float), np.asarray(z[f"v{j}_t"], float)
        d = _orient(z[f"v{j}_data"], r, t)
        key = {"velocity gauss": "vel", "displacement gauss": "disp"}.get(name)
        if key is None:
            continue
        inside = (t >= win["t0"]) & (t <= win["t1"])
        pads = ~inside
        rms_in = np.sqrt(np.mean(d[:, inside] ** 2))
        rms_pad = np.sqrt(np.mean(d[:, pads] ** 2)) if pads.any() else np.nan
        mean_r = d.mean(axis=0, keepdims=True)
        out[f"{key}_flat"] = float(np.sum(np.broadcast_to(mean_r, d.shape) ** 2) / (np.sum(d ** 2) + 1e-30))
        out[f"{key}_burst"] = float(rms_in / (rms_pad + 1e-30))
        # best line anywhere (reader-independent)
        dw, tw = d[:, inside], t[inside]
        best = 0.0
        speeds = np.concatenate([-np.geomspace(0.5, 20, 25), np.geomspace(0.5, 20, 25)])
        for ta in tw[::4]:
            for c in speeds:
                for ra in (r[0], r[r.size // 2], r[-1]):
                    s = _line_score(dw, r, tw, ta, ra, c)
                    if np.isfinite(s) and s > best:
                        best = s
        out[f"{key}_track_best"] = best
        # the reader's line
        sh = (slope or {}).get("shared") or {}
        if slope and slope.get("confidence", 0) > 0 and sh.get("speed_m_s"):
            dd = (slope.get("disp") or {}) if key == "disp" and slope.get("disp") else sh
            ta = dd.get("anchor_t_ms", sh["anchor_t_ms"]) * 1e-3
            ra = dd.get("anchor_r_mm", sh["anchor_r_mm"]) * 1e-3
            c = dd.get("speed_m_s", sh["speed_m_s"])
            # anchor time is on the window clock of the panel (t already absolute?) - handle both
            if not (t[0] - 0.05 <= ta <= t[-1] + 0.05):
                ta = ta + t[0]
            out[f"{key}_track"] = _line_score(d, r, t, ta, ra, c)
    return out


# ------------------------------------------------------------------ collect
def collect(root):
    folders = [f for f in S.find_folders(root)
               if os.path.exists(S.Paths(f).general_json)]
    prompts, lines, wins = [], [], []
    archives = []
    for f in folders:
        p = S.Paths(f)
        log = _read_log(p)
        cnt = Counter((r.get("task"), r.get("window")) for r in log)
        for k, r in enumerate(log):
            prompts.append(dict(folder=f, subject=_subject(f), seq=k, time=r.get("time"),
                                task=r.get("task"), window=r.get("window"), action=r.get("action"),
                                confidence=r.get("confidence"), hash=r.get("hash"),
                                n_same=cnt[(r.get("task"), r.get("window"))]))
        archives += [dict(folder=f, archive=os.path.basename(a))
                     for a in glob.glob(os.path.join(p.dir, "archive_*"))]
        gen = S.read_json(p.general_json)
        lines.append(_line_rows(f, gen, "general"))
        win = S.read_json(p.windows_json)
        if not win or win.get("key", {}).get("general_hash") != (gen or {}).get("hash"):
            continue
        ev = S.read_json(p.events_json) or {}
        events = ev.get("events", {}) if ev.get("windows_hash") == win["hash"] else {}
        proc = S.read_json(p.processed_json) or {}
        slopes = S.read_json(p.slopes_json) or {}
        phases = win.get("window_phases") or [{}] * len(win["windows"])
        ecg = win.get("ecg") or {}
        for i, w in enumerate(win["windows"]):
            k = str(i)
            ph = phases[i] if i < len(phases) else {}
            e = events.get(k)
            lines.append(_line_rows(f, e, "event", i, w.get("label")))
            row = dict(folder=f, subject=_subject(f), window=i, label=w.get("label"),
                       expect=w.get("expect"), t_peak_ms=w["t_peak"] * 1e3, score=w.get("score"),
                       phase_ms=ph.get("phase_ms"), rr_ms=ph.get("rr_ms"), hr_bpm=ph.get("hr_bpm"),
                       qs2_ms=ph.get("qs2_ms"), ecg_status=ecg.get("status"),
                       ecg_quality=ecg.get("quality"), line_skipped=bool(e and e.get("skipped")))
            # where the detected peak sits inside its phase search window (0 / 1 = on the edge)
            phm, qs2 = row["phase_ms"], row["qs2_ms"]
            if row["expect"] == "MVC" and phm is not None:
                row["pos_in_search"] = phm / MVC_MAX_MS
            elif row["expect"] == "AVC" and phm is not None and qs2:
                row["pos_in_search"] = (phm - (qs2 - AVC_TOL_MS)) / (2 * AVC_TOL_MS)
            pr, sl = proc.get(k), slopes.get(k)
            fresh = bool(e and not e.get("skipped") and pr and pr.get("line_hash") == e.get("hash"))
            if fresh:
                row["mline_length_mm"] = pr.get("mline_length_mm")
                for v, a in (pr.get("auto") or {}).items():
                    row[f"auto_{v}"] = a.get("speed_m_s")
                    row[f"sem_{v}"] = a.get("semblance")
            if fresh and sl and sl.get("st_hash") == pr.get("st_hash"):
                row["slope_skipped"] = bool(sl.get("skipped"))
                row["confidence"] = sl.get("confidence")
                sh = sl.get("shared") or {}
                row["speed_m_s"] = sh.get("speed_m_s")
                row["speed_disp_m_s"] = (sl.get("disp") or {}).get("speed_m_s", sh.get("speed_m_s"))
                row["unlinked"] = bool(sl.get("disp"))
                row["anchor_view"] = sh.get("anchor_view")
                row["slope_time"] = sl.get("time")
            if fresh and os.path.exists(p.st_npz(i)):
                try:
                    row.update(_st_measures(p.st_npz(i), i, w, sl if row.get("confidence") is not None else None))
                except Exception as exc:                          # noqa: BLE001
                    row["st_error"] = str(exc)
            wins.append(row)
    return pd.DataFrame(prompts), pd.DataFrame(lines), pd.DataFrame(wins), pd.DataFrame(archives)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    prompts, lines, wins, arch = collect(a.root)
    for name, df in (("prompts", prompts), ("lines", lines), ("windows", wins), ("archives", arch)):
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"{name}: {len(df)} rows")


if __name__ == "__main__":
    main()
