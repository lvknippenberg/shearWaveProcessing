"""How far is an event M-line from the general M-line? (Is redrawing the line per MVC worth it?)

For every folder of the manual passive study (``<folder>/output/swp_passive_manual/``) with a
general line and event lines on the CURRENT event windows, compares each event line with:

* the general line (drawn on the buffer-4 frame nearest an R-peak), and
* the other event line(s) of the same label in the same folder (e.g. MVC beat 1 vs MVC beat 2) -
  two lines drawn at the same cardiac phase, so their difference is the reader's own
  line-to-line variability: the floor below which a "different" line means nothing.

Geometry (all buffer-4 coordinates, mm; lines are polylines resampled to 250 points by arc length):

* ``d_mean`` / ``d_max`` - mean / max distance of the event line's points to the general polyline
  (one-sided: how far the event line lies from the general one); ``d_sym`` = the mean of both ways;
* ``d_mid``  - distance between the two lines' arc-length midpoints;
* ``angle``  - angle between the end-to-end chords (deg); ``cos`` of it scales a speed measured
  along the other line for a wave travelling along this one;
* ``len_ratio`` - event length / general length;
* ``overlap`` - fraction of the event line within 1 mm of the general line.

Timing: ``phase_ms`` = event time since its preceding R-peak (the drawn frame for the event),
``gap_ms`` = |event time - time of the buffer-4 frame the general line was drawn on| (same beat or
beats apart), ``beats`` = that gap in RR intervals.

Writes study/logs/passive_event_vs_general.csv and prints the summary.

Usage:  python study/analysis/passive_event_vs_general.py [--root Z:/raw_data]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from swp.manual import frames as F   # noqa: E402
from swp.manual import store as S    # noqa: E402

OUT = os.path.join(REPO, "study", "logs", "passive_event_vs_general.csv")
N = 250


def resample(pts, n=N):
    pts = np.asarray(pts, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    u = np.linspace(0, s[-1], n)
    return np.c_[np.interp(u, s, pts[:, 0]), np.interp(u, s, pts[:, 1])], s[-1]


def dist_to_polyline(q, pts):
    """Distance of each point in q to the polyline pts."""
    pts = np.asarray(pts, float)
    best = np.full(len(q), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((q - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(q - (a + t[:, None] * ab), axis=1))
    return best


def compare(ev, gen):
    """Geometry of line ``ev`` relative to line ``gen`` (points in mm)."""
    qe, le = resample(ev)
    qg, lg = resample(gen)
    de = dist_to_polyline(qe, gen)
    dg = dist_to_polyline(qg, ev)
    ce, cg = qe[-1] - qe[0], qg[-1] - qg[0]
    ang = np.degrees(np.arccos(np.clip(abs(ce @ cg) / (np.linalg.norm(ce) * np.linalg.norm(cg)), 0, 1)))
    return dict(d_mean=de.mean(), d_max=de.max(), d_sym=0.5 * (de.mean() + dg.mean()),
                d_mid=float(np.linalg.norm(qe[N // 2] - qg[N // 2])), angle=ang,
                cos=np.cos(np.radians(ang)), len_ev=le, len_ref=lg, len_ratio=le / lg,
                overlap=float((de <= 1.0).mean()))


def folder_rows(folder):
    p = S.Paths(folder)
    gen = S.read_json(p.general_json)
    if not gen or gen.get("skipped") or not os.path.exists(p.general_npz):
        return []
    ew = S.event_windows(p)
    ev = S.read_json(p.events_json) or {}
    if ew is None or ev.get("windows_hash") != ew["hash"]:
        return []
    g4 = S.load_points(p.general_npz) * 1e3
    f4 = (gen.get("frames") or {}).get("4", {})
    t4 = F.buffer4_times(p.bmode(4))
    t_gen = float(t4[int(f4["frame"])]) if "frame" in f4 else np.nan
    rr = next((ph.get("rr_ms") for ph in ew["phases"] if ph.get("rr_ms")), None)
    lines = {}
    rows = []
    for k, e in ev["events"].items():
        i = int(k)
        if e.get("skipped") or not os.path.exists(p.event_npz(i)) or i >= len(ew["windows"]):
            continue
        w, ph = ew["windows"][i], ew["phases"][i]
        pts = S.load_points(p.event_npz(i)) * 1e3
        lines[i] = (w.get("label"), pts)
        gap = abs(w["t_peak"] - t_gen) * 1e3
        pre = e.get("preload") or ""
        rows.append(dict(
            subject=os.path.basename(os.path.dirname(folder)), acq=os.path.basename(folder), window=i,
            label=w.get("label"), ref="general", phase_ms=ph.get("phase_ms"),
            gen_phase_ms=f4.get("phase_ms"), gap_ms=gap, rr_ms=rr,
            beats=gap / rr if rr else np.nan,
            preload=("general" if pre.startswith("general") else "september" if pre.startswith("September")
                     else "redo" if pre.startswith("current") else pre),
            unchanged=bool(np.allclose(pts, g4)) if pts.shape == g4.shape else False,
            **compare(pts, g4)))
    # same-label pairs inside the folder: the reader's line-to-line floor
    keys = sorted(lines)
    for a in keys:
        for b in keys:
            if a < b and lines[a][0] == lines[b][0]:
                rows.append(dict(subject=os.path.basename(os.path.dirname(folder)),
                                 acq=os.path.basename(folder), window=a, window_b=b,
                                 label=lines[a][0], ref="same-label pair",
                                 **compare(lines[a][1], lines[b][1])))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="Z:/raw_data")
    a = ap.parse_args()
    folders = sorted({os.path.dirname(os.path.dirname(os.path.dirname(j)))
                      for j in glob.glob(os.path.join(a.root, "*", "*", "output", S.OUTDIR, "events.json"))})
    rows = []
    for f in folders:
        try:
            rows += folder_rows(f)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {f}: {exc}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(folders)} folders -> {OUT}")
    cols = ["d_mean", "d_max", "d_mid", "angle", "len_ratio", "overlap"]
    q = lambda s: f"{s.median():.2f} [{s.quantile(.25):.2f}-{s.quantile(.75):.2f}]"   # noqa: E731
    for (ref, lab), g in df.groupby(["ref", "label"]):
        print(f"\n{ref:16s} {lab}: n={len(g)}  folders={g.acq.nunique()}")
        for c in cols:
            print(f"   {c:10s} {q(g[c])}")
    return df


if __name__ == "__main__":
    main()
