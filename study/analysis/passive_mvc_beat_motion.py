"""Anatomy motion between the general-line frame and the MVC frames (and between two MVC beats).

Companion of passive_event_vs_general.py (which compares the DRAWN lines). This one measures the
motion in the images, independent of the reader: the buffer-4 envelope (9-frame average, as
in the line editor) at the general line's frame is registered onto the buffer-4 envelope at
each MVC event with the line-transfer registration (swp.mline.transfer.transfer_line: rigid
translation in a box around the general line, ensemble of box sizes and a known-shift check).
The MVC beat 1 frame is registered onto the MVC beat 2 frame the same way. Both are the same
cardiac phase one RR apart, so any shift there is beat-to-beat, breathing or probe motion.

Per MVC window: phase (ms after its R-peak), beat (same beat as the general frame or not),
shift_mm / dx / dz of the anatomy, split into perp_mm (across the general line's chord - the
part that takes a line off the septum) and along_mm (slides it along the septum), reliable (ensemble agrees + known shifts recovered), corr0 ->
corr (anatomy correlation before / after the shift), and for drawn lines the distance of the
drawn MVC line to the general line (d_mean, as passive_event_vs_general.py).

Writes study/logs/passive_mvc_beat_motion.csv.

Usage:  python study/analysis/passive_mvc_beat_motion.py [--root Z:/raw_data] [--jobs 4]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swp.manual import frames as F   # noqa: E402
from swp.manual import store as S    # noqa: E402

OUT = os.path.join(REPO, "study", "logs", "passive_mvc_beat_motion.csv")


def _reg(points, a, b):
    from swp.mline.transfer import transfer_line
    r = transfer_line(points, (a.env, a.x_mm, a.z_mm), (b.env, b.x_mm, b.z_mm), check=True)
    t = r.transform
    u = points[-1] - points[0]
    u = u / np.linalg.norm(u)
    return dict(shift_mm=r.shift_mm, dx=t.dx, dz=t.dz, perp_mm=abs(-u[1] * t.dx + u[0] * t.dz),
                along_mm=abs(u[0] * t.dx + u[1] * t.dz), reliable=r.reliable(), agree=r.agree,
                known_err_mm=r.known_err_mm, corr0=t.corr0, corr=t.corr)


def folder_rows(folder):
    from passive_event_vs_general import compare
    p = S.Paths(folder)
    gen = S.read_json(p.general_json)
    if not gen or gen.get("skipped") or not os.path.exists(p.general_npz):
        return []
    ew = S.event_windows(p)
    if ew is None:
        return []
    mvc = [i for i, w in enumerate(ew["windows"]) if w.get("label") == "MVC"]
    if not mvc:
        return []
    g4 = S.load_points(p.general_npz) * 1e3
    path = p.bmode(4)
    t4 = F.buffer4_times(path)
    kg = int(gen["frames"]["4"]["frame"])
    pg = F.load_panel(folder, path, 4, kg, "general")
    ev = S.read_json(p.events_json) or {}
    events = ev.get("events", {}) if ev.get("windows_hash") == ew["hash"] else {}
    rr = next((ph.get("rr_ms") for ph in ew["phases"] if ph.get("rr_ms")), None)
    base = dict(subject=os.path.basename(os.path.dirname(folder)), acq=os.path.basename(folder),
                rr_ms=rr, n_mvc=len(mvc))
    rows, panels = [], {}
    for i in mvc:
        w, ph = ew["windows"][i], ew["phases"][i]
        k = int(np.argmin(np.abs(t4 - w["t_peak"])))
        panels[i] = F.load_panel(folder, path, 4, k, "event")
        gap = abs(t4[k] - t4[kg]) * 1e3
        row = dict(base, kind="general->MVC", window=i, phase_ms=ph.get("phase_ms"), gap_ms=gap,
                   beat=("same" if rr and gap < 0.5 * rr else "other" if rr else None),
                   **_reg(g4, pg, panels[i]))
        e = events.get(str(i))
        if e and not e.get("skipped") and os.path.exists(p.event_npz(i)):
            pts = S.load_points(p.event_npz(i)) * 1e3
            row.update(line_drawn=True, line_is_general=bool(pts.shape == g4.shape and np.allclose(pts, g4)),
                       line_preload=(e.get("preload") or "")[:20], line_d_mean=compare(pts, g4)["d_mean"])
        rows.append(row)
    for a, b in zip(mvc[:-1], mvc[1:]):
        wa, wb = ew["windows"][a], ew["windows"][b]
        row = dict(base, kind="MVC->MVC", window=a, window_b=b,
                   phase_ms=ew["phases"][a].get("phase_ms"), phase_b_ms=ew["phases"][b].get("phase_ms"),
                   gap_ms=abs(wb["t_peak"] - wa["t_peak"]) * 1e3, **_reg(g4, panels[a], panels[b]))
        ea, eb = events.get(str(a)), events.get(str(b))
        if (ea and eb and not ea.get("skipped") and not eb.get("skipped")
                and os.path.exists(p.event_npz(a)) and os.path.exists(p.event_npz(b))):
            la, lb = S.load_points(p.event_npz(a)) * 1e3, S.load_points(p.event_npz(b)) * 1e3
            same = la.shape == lb.shape and np.allclose(la, lb)
            both_gen = same and la.shape == g4.shape and np.allclose(la, g4)
            row.update(line_drawn=True, line_identical=same, line_both_general=both_gen,
                       line_d_mean=0.0 if same else compare(lb, la)["d_mean"],
                       line_angle=0.0 if same else compare(lb, la)["angle"])
        rows.append(row)
    return rows


def _safe(f):
    try:
        return folder_rows(f)
    except Exception as exc:                                       # noqa: BLE001
        print(f"  {f}: {type(exc).__name__}: {exc}", flush=True)
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    folders = sorted({os.path.dirname(os.path.dirname(os.path.dirname(j)))
                      for j in glob.glob(os.path.join(a.root, "*", "*", "output", S.OUTDIR, "general.json"))})
    rows = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for f, r in zip(folders, ex.map(_safe, folders)):
            rows += r
            print(f"  {os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}: {len(r)} rows", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} rows from {len(folders)} folders -> {OUT}")


if __name__ == "__main__":
    main()
