"""Mark regions of interest by eye on the whole-recording space-time of the general M-line.

    python scripts/passive_roi.py session                  # the interactive work (resumes)
    python scripts/passive_roi.py session --all            # walk every folder again, ROIs pre-loaded
    python scripts/passive_roi.py session --folder "<f>"   # one folder (pre-loaded when done)
    python scripts/passive_roi.py session --click-ms 120   # semi-automatic: click = 120 ms window
    python scripts/passive_roi.py status [-v]
    python scripts/passive_roi.py export                   # -> study/logs/passive_rois.csv
    python scripts/passive_roi.py auto                     # automatic MVC/AVC windows -> study/logs/passive_auto_windows.csv
    python scripts/passive_roi.py session --auto           # review: open with the automatic windows pre-drawn

The window choice of the detector (energy pick + semblance screen) is a major source of error, so
here the time windows to investigate are drawn by hand: the "velocity gauss" space-time along the
general line over the whole buffer-4 recording, with the R-peaks and the expected MVC / AVC search
windows. Drag left-right to draw a horizontal line; its time span is one ROI; draw as many as
needed. Keys: swp/manual/roi_gui.py and docs/passive_manual.md ("Marking ROIs by eye").

Input: the general-line cache of study/analysis/passive_general_screen.py (one npz per folder, so
nothing is read from Z: to show a folder). For folders with a general line but no cache yet:

    python study/analysis/passive_general_screen.py --any-general

Output: <folder>/output/swp_passive_manual/rois.json (+ rois.png, a log.jsonl line), keyed by the
general line's hash, so a redrawn general line asks again.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from swp.manual import store as S          # noqa: E402

CACHE = _REPO / "study" / "analysis" / "general_screen_cache"
EXPORT = _REPO / "study" / "logs" / "passive_rois.csv"
AUTO_EXPORT = _REPO / "study" / "logs" / "passive_auto_windows.csv"


def _cached(a):
    """[(folder, cache npz, cached general hash)], filtered by --folder / --subject."""
    out = []
    want = {str(Path(f)).lower() for f in a.folder}
    for c in sorted(glob.glob(str(CACHE / "*.npz"))):
        if c.endswith("_track.npz"):
            continue
        z = np.load(c, allow_pickle=True)
        f = str(z["folder"])
        if want and str(Path(f)).lower() not in want:
            continue
        if a.subject and Path(f).parent.name != a.subject:
            continue
        out.append((f, c, str(z["general_hash"])))
    if not out:
        raise SystemExit(f"no cached folders in {CACHE} (run study/analysis/passive_general_screen.py)")
    return out


def _state(folder, cache_hash):
    """stage: stale-cache | todo | skipped | none | done (+ the record, + the current general hash)"""
    p = S.Paths(folder)
    gen = S.read_json(p.general_json) or {}
    if gen.get("hash") != cache_hash:
        return "stale-cache", None, gen.get("hash")
    rec = S.read_json(p.rois_json)
    if rec is None or rec.get("general_hash") != cache_hash:
        return "todo", None, cache_hash
    return rec["status"], rec, cache_hash


def _load(npz):
    z = np.load(npz, allow_pickle=True)
    rr = float(z["rr_ms"])
    return dict(v=z["v_data"], t_s=z["v_t"], r_m=z["v_r"], r_peaks_s=z["r_peaks_s"],
                rr_s=rr * 1e-3 if np.isfinite(rr) else None)


def _title(folder, data, k, n, rec):
    rr = data["rr_s"]
    hr = f"HR {60 / rr:.0f} bpm" if rr else "NO VALID ECG"
    prev = f"   (done before: {len(rec['rois'])} ROI)" if rec and rec.get("status") == "done" else \
        "   (done before: no ROI)" if rec and rec.get("status") == "none" else ""
    return (f"[{k + 1}/{n}]  {Path(folder).parent.name}  {Path(folder).name}   -   general M-line "
            f"{data['r_m'][-1] * 1e3:.0f} mm, {hr}, velocity gauss (15-150 Hz){prev}")


def _auto(data, include_screened=False):
    """The automatic MVC / AVC windows (swp.passive_valves) of a cached folder."""
    from swp.passive_valves import valve_windows
    ws = valve_windows(data["v"], data["r_m"], data["t_s"], data["r_peaks_s"], data["rr_s"])
    return [w for w in ws if include_screened or not w["screened"]]


def _as_preload(windows, data):
    r_mid = float(data["r_m"][-1]) * 1e3 / 2
    return dict(rois=[dict(t0=w["t0"], t1=w["t1"], label=w["label"], r_mm=r_mid) for w in windows])


def _save(folder, cache_hash, npz, res, auto=None):
    p = S.Paths(folder)
    status = {"accept": "done", "none": "none", "skip": "skipped"}[res["action"]]
    from swp.provenance import provenance
    S.write_json(p.rois_json, dict(
        general_hash=cache_hash, status=status, rois=res.get("rois", []), clim_pct=res.get("clim_pct"),
        source=dict(cache=os.path.basename(npz), view="velocity gauss",
                    config="configs/passive_manual.yaml"),
        auto_proposals=None if auto is None else [{k: w[k] for k in ("label", "t0", "t1", "sem")} for w in auto],
        time=time.strftime("%Y-%m-%d %H:%M:%S"), host=socket.gethostname(),
        provenance=provenance()))
    S.append_log(p, dict(stage="roi", action=res["action"], n_rois=len(res.get("rois", [])),
                         general_hash=cache_hash))


def cmd_session(a):
    import matplotlib
    matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
    from swp.manual.roi_gui import RoiEditor

    items = _cached(a)
    todo = []
    for f, c, h in items:
        st, rec, cur = _state(f, h)
        if st == "stale-cache":
            print(f"  stale cache (general line redrawn), skipped: {f}")
            continue
        if (a.all or a.folder or st == "todo" or (st == "skipped" and a.retry_skipped)):
            todo.append((f, c, h))
    print(f"{len(todo)} folder(s) to mark")
    k = 0
    while 0 <= k < len(todo):
        f, c, h = todo[k]
        _, rec, _ = _state(f, h)
        data = _load(c)
        preload = rec if rec and rec.get("status") in ("done", "none") else None
        auto = _auto(data) if a.auto and preload is None else None
        if auto is not None:
            preload = _as_preload(auto, data)
        ed = RoiEditor(data, _title(f, data, k, len(todo), rec) + ("   [automatic proposals]" if auto else ""),
                       preload=preload, click_ms=a.click_ms)
        res = ed.run(snapshot=os.path.join(S.Paths(f).dir, "rois.png"))
        act = res["action"]
        if act == "quit":
            break
        if act == "back":
            k = max(k - 1, 0)
            continue
        _save(f, h, c, res, auto)
        print(f"  {act:6s} {len(res.get('rois', []))} ROI  {f}", flush=True)
        k += 1
    print("bye")


def cmd_status(a):
    counts = {}
    for f, c, h in _cached(a):
        st, rec, _ = _state(f, h)
        counts[st] = counts.get(st, 0) + 1
        if a.verbose:
            extra = "" if not rec else "  " + ", ".join(
                f"{q['label']} {q['t0'] * 1e3:.0f}-{q['t1'] * 1e3:.0f}" for q in rec.get("rois", []))
            print(f"  {st:11s} {Path(f).parent.name}/{Path(f).name}{extra}")
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    n_gen = sum(1 for f in S.find_folders(a.root) if (S.read_json(S.Paths(f).general_json) or {}).get("hash")
                and not (S.read_json(S.Paths(f).general_json) or {}).get("skipped")) if a.root else None
    if n_gen is not None:
        print(f"  {n_gen} folder(s) under {a.root} have a general line; {n_gen - sum(counts.values())} "
              f"not cached yet (passive_general_screen.py --any-general)")


def cmd_auto(a):
    """Automatic windows of every cached folder -> CSV, with the agreement where ROIs exist."""
    rows = []
    for f, c, h in _cached(a):
        st, rec, _ = _state(f, h)
        data = _load(c)
        rois = [q for q in (rec or {}).get("rois", []) if q["label"] in ("MVC", "AVC")] if st == "done" else []
        ws = _auto(data, include_screened=True)
        if data["rr_s"] is None:
            print(f"  no usable ECG, no automatic windows: {Path(f).parent.name}")
        for w in ws:
            mine = [q for q in rois if q["label"] == w["label"] and q["t1"] > w["search_lo"] and q["t0"] < w["search_hi"]]
            cover = max([max(0.0, min(w["t1"], q["t1"]) - max(w["t0"], q["t0"])) / (q["t1"] - q["t0"])
                         for q in mine], default=np.nan)
            rows.append(dict(subject=Path(f).parent.name, folder=f, roi_status=st,
                             **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in w.items()},
                             n_rois=len(mine), roi_cover=round(cover, 3) if np.isfinite(cover) else None))
    out = Path(a.out) if a.out else AUTO_EXPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    marked = [r for r in rows if r["n_rois"]]
    unmarked = [r for r in rows if r["roi_status"] == "done" and not r["n_rois"]]
    print(f"{len(rows)} automatic window(s) in {len({r['folder'] for r in rows})} folder(s), "
          f"{sum(r['screened'] for r in rows)} screened -> {out}")
    if marked:
        print(f"  vs your ROIs: {sum(r['roi_cover'] >= 0.95 for r in marked)}/{len(marked)} fully inside, "
              f"{sum(r['roi_cover'] >= 0.8 for r in marked)}/{len(marked)} >= 80 %; screened although marked: "
              f"{sum(r['screened'] for r in marked)}; search windows you left empty: {len(unmarked)}, "
              f"of which screened {sum(r['screened'] for r in unmarked)}")


def cmd_export(a):
    rows = []
    for f, c, h in _cached(a):
        st, rec, _ = _state(f, h)
        base = dict(subject=Path(f).parent.name, folder=f, status=st)
        if not rec or not rec.get("rois"):
            rows.append(dict(base, roi=None))
            continue
        for i, q in enumerate(rec["rois"]):
            rows.append(dict(base, roi=i, **q))
    out = Path(a.out) if a.out else EXPORT
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    print(f"{sum(r['roi'] is not None for r in rows)} ROI(s) from {len({r['folder'] for r in rows})} folder(s) -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="session", choices=["session", "status", "export", "auto"])
    ap.add_argument("--folder", action="append", default=[])
    ap.add_argument("--subject", default=None)
    ap.add_argument("--all", action="store_true", help="session: every cached folder, ROIs pre-loaded")
    ap.add_argument("--auto", action="store_true",
                    help="session: folders without ROIs open with the automatic MVC / AVC windows to review")
    ap.add_argument("--click-ms", type=float, default=None,
                    help="session: a plain click places a window of this length centred on it (e.g. 120)")
    ap.add_argument("--retry-skipped", action="store_true", help="session: also offer skipped folders")
    ap.add_argument("--root", default=None, help="status: also count general lines under this root")
    ap.add_argument("--out", default=None, help="export: CSV path")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    {"session": cmd_session, "status": cmd_status, "export": cmd_export, "auto": cmd_auto}[a.command](a)


if __name__ == "__main__":
    main()
