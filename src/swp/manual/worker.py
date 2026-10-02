"""Unattended stages of the manual passive study: burst detection and the five space-times.

Runs next to the interactive session (``scripts/passive_manual.py worker --watch``), picking up
folders as lines are drawn. Several workers may run at once; a folder is claimed with a lock file.

Only a box around the M-line(s) is read from the ~1 GB buffer-4 file (``roi`` in
:func:`swp.viz.io.loader.load_acquisition`), ``MARGIN_M`` wide: detection samples 5 lines x
0.5 mm across the line and the widest spatial filter is a 2 mm median, so 10 mm leaves them well
clear of the box edge.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import time
import traceback

import numpy as np

from . import store as S

MARGIN_M = 10e-3
PAD_S = 0.020            # time context either side of a window fed to the filters (as swp.passive)
WINDOW_MS = 100.0
MAX_EVENTS = 4
OVERVIEW_STRIDE = 2
SPEED_CMAX = 20.0


def _roi(point_sets):
    pts = np.concatenate([np.asarray(p, float) for p in point_sets])
    lo, hi = pts.min(axis=0) - MARGIN_M, pts.max(axis=0) + MARGIN_M
    return (lo[0], hi[0], lo[1], hi[1])


def _cfg():
    from ..viz import runconfig as rc
    return rc.load_config(S.CONFIG)


def _load(p, point_sets):
    from ..viz.io import load_acquisition
    return load_acquisition(p.bmode(4), roi=_roi(point_sets))


def _record_error(p, what, key, exc):
    errs = S.read_json(os.path.join(p.dir, "worker_errors.json"), {})
    errs[what] = dict(key=key, error=f"{type(exc).__name__}: {exc}", time=time.ctime())
    S.write_json(os.path.join(p.dir, "worker_errors.json"), errs)


def failed(p, what, key):
    errs = S.read_json(os.path.join(p.dir, "worker_errors.json"), {})
    return errs.get(what, {}).get("key") == key


# ------------------------------------------------------------------ detection
def _r_peaks_buffer4(folder):
    """(R-peaks on the buffer-4 clock [s], RR [s]) from the trigger log; (empty, None) without a
    usable record (as study/analysis/passive_general_screen.py)."""
    from ..acquisition.triggerlog import buffer_timing, clean_r_peaks
    bt = buffer_timing(folder, 4)
    if bt is not None:
        kept, rr_ms = clean_r_peaks(bt.r_peaks_ms)
        if kept is not None and rr_ms is not None and np.isfinite(rr_ms):
            return (np.asarray(kept, float) - bt.t0_ms) * 1e-3, float(rr_ms) * 1e-3
    return np.array([]), None


def window_phases(windows, r_peaks_s, rr_s):
    """Per window: label + phase of its t_peak since the preceding R-peak (keys as
    swp.acquisition.triggerlog.label_event, which the slope prompt and the export read)."""
    rp = np.asarray(r_peaks_s, float)
    hr = 60.0 / rr_s if rr_s else np.nan
    out = []
    for w in windows:
        before = rp[rp <= w["t_peak"] + 1e-9]
        out.append(dict(label=w.get("label"),
                        phase_ms=round(float((w["t_peak"] - before.max()) * 1e3), 1) if before.size else None,
                        rr_ms=round(rr_s * 1e3, 1) if rr_s else None,
                        hr_bpm=round(hr, 1) if np.isfinite(hr) else None,
                        qs2_ms=round(546.0 - 2.1 * hr, 1) if np.isfinite(hr) else None))
    return out


def _detect_valves(folder, p, gen, cfg, acq, ml):
    """detect.picker "valves": the whole-recording velocity space-time along the general line ->
    automatic MVC / AVC windows (swp.passive_valves) -> windows.json (needs_review) + general_st.npz."""
    from ..passive import _build_views
    from ..passive_valves import WINDOW_S, valve_windows
    from ..viz.pipeline import run_pipeline

    det = cfg.get("detect", {})
    view = str(det.get("screen_view", "velocity gauss"))
    res = run_pipeline(acq, ml, dict(_build_views(cfg, acq))[view], focus=None)
    v, r, t = np.asarray(res.st.data, np.float32), np.asarray(res.st.r), np.asarray(res.st.t)
    rp, rr = _r_peaks_buffer4(folder)
    ws = valve_windows(v, r, t, rp, rr, window_s=float(det.get("window_ms", WINDOW_S * 1e3)) * 1e-3,
                       screen_min=float(det.get("screen_min", 0.3)))
    windows = [dict(t_peak=w["t_burst"], t0=w["t0"], t1=w["t1"], label=w["label"], expect=w["label"],
                    score=None, screen=w["sem"], speed_m_s=w["c"], screened=w["screened"],
                    search=[w["search_lo"], w["search_hi"]]) for w in ws]
    np.savez_compressed(p.general_st, v=v, r=r, t=t, r_peaks_s=rp, rr_s=np.array(np.nan if rr is None else rr),
                        general_hash=np.array(gen["hash"]), view=np.array(view))
    try:
        from ..acquisition.rrcheck import assess_rr
        ecg = assess_rr(folder).as_row()
    except Exception as exc:                                    # noqa: BLE001 - advisory only
        ecg = dict(error=str(exc))
    from ..provenance import provenance
    S.write_json(p.windows_json, dict(
        key=dict(general_hash=gen["hash"], detect=det), needs_review=True,
        hash=S.windows_hash(windows), windows=windows, window_phases=window_phases(windows, rp, rr),
        ecg=ecg, roi=list(acq.meta["roi"]), provenance=provenance(config=cfg)))
    return windows


def detect(folder):
    """General line -> event windows -> windows.json. Default detector since 2026-10-01:
    ``detect.picker: valves`` (reviewed by hand next); otherwise the earlier phase-aware energy
    detector (as configs/passive.yaml)."""
    from ..passive import detect_windows, label_windows
    from ..viz.mline import mline_from_points

    p = S.Paths(folder)
    gen = S.read_json(p.general_json)
    pts = S.load_points(p.general_npz)
    cfg = _cfg()
    cfg["data"]["root"] = p.output
    acq = _load(p, [pts])
    ml = mline_from_points(pts, S.N_SAMPLES)
    det = cfg.get("detect", {})
    if str(det.get("picker", "energy")).lower() == "valves":
        return _detect_valves(folder, p, gen, cfg, acq, ml)
    track = None
    if det.get("screen") or str(det.get("picker", "energy")).lower() == "semblance":
        # score every 100 ms window of the general line (swp.passive_screen); since 2026-09-29
        from ..passive_screen import general_line_track
        track = general_line_track(acq, ml, cfg, str(det.get("screen_view", "velocity gauss")))
    windows = detect_windows(acq, ml, cfg, p.dir, WINDOW_MS, MAX_EVENTS, OVERVIEW_STRIDE, folder=folder,
                             screen=track)
    st = dict(windows=[dataclasses.asdict(w) for w in windows])
    label_windows(folder, st)
    try:
        from ..acquisition.rrcheck import assess_rr
        ecg = assess_rr(folder).as_row()
    except Exception as exc:                                    # noqa: BLE001 - advisory only
        ecg = dict(error=str(exc))
    from ..provenance import provenance
    S.write_json(p.windows_json, dict(
        key=dict(general_hash=gen["hash"], detect=cfg.get("detect", {}), window_ms=WINDOW_MS,
                 max_events=MAX_EVENTS, overview_stride=OVERVIEW_STRIDE),
        hash=S.windows_hash(st["windows"]), windows=st["windows"],
        window_phases=st.get("window_phases"), ecg=ecg, roi=list(acq.meta["roi"]),
        screen_track=None if track is None else {k: np.round(np.asarray(v, float), 4).tolist()
                                                 for k, v in track.items()},
        provenance=provenance(config=cfg)))
    return windows


# ------------------------------------------------------------------ space-times
def _st_hash(arrays):
    h = hashlib.sha1()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:12]


def process(folder, idxs):
    """The five space-times of each window in ``idxs`` -> st_win<i>.npz + processed.json."""
    from ..passive import _build_views
    from ..viz.metrics import slant_stack_speed
    from ..viz.mline import mline_from_points
    from ..viz.pipeline import run_pipeline
    from ._light import display_8bit

    p = S.Paths(folder)
    win = S.event_windows(p)
    ev = S.read_json(p.events_json)["events"]
    lines = {i: S.load_points(p.event_npz(i)) for i in idxs}
    cfg = _cfg()
    cfg["data"]["root"] = p.output
    acq = _load(p, lines.values())
    views = _build_views(cfg, acq)
    proc = S.read_json(p.processed_json, {})
    for i in idxs:
        w = win["windows"][i]
        ml = mline_from_points(lines[i], S.N_SAMPLES)
        i0 = int(np.argmin(np.abs(acq.t - (w["t0"] - PAD_S))))
        i1 = int(np.argmin(np.abs(acq.t - (w["t1"] + PAD_S)))) + 1
        acq_w = dataclasses.replace(acq, iq=acq.iq[i0:i1], t=acq.t[i0:i1])
        out, auto, arrays = {}, {}, []
        for j, (name, vcfg) in enumerate(views):
            res = run_pipeline(acq_w, ml, vcfg, focus=None)
            sem, c = slant_stack_speed(res.st, res.r0, cmin=1.0, cmax=SPEED_CMAX, remove_flat=False)
            d = np.asarray(res.st.data, np.float32)
            out[f"v{j}_data"], out[f"v{j}_r"], out[f"v{j}_t"] = d, np.asarray(res.st.r), np.asarray(res.st.t)
            out[f"v{j}_name"], out[f"v{j}_quantity"] = np.array(name), np.array(res.st.quantity)
            auto[name] = dict(speed_m_s=float(c), semblance=float(sem))
            arrays.append(d)
        # B-mode of the box at the event: mean envelope of 9 frames around the peak
        k = int(np.argmin(np.abs(acq.t - w["t_peak"])))
        env = np.abs(acq.iq[max(k - 4, 0):k + 5]).mean(axis=0)
        out.update(bmode_u8=display_8bit(env),
                   bmode_extent=np.array([acq.x[0], acq.x[-1], acq.z[-1], acq.z[0]]) * 1e3,
                   line_x_mm=ml.x * 1e3, line_z_mm=ml.z * 1e3, n_views=np.array(len(views)))
        h = _st_hash(arrays)
        np.savez_compressed(p.st_npz(i), st_hash=np.array(h), **out)
        proc[str(i)] = dict(line_hash=ev[str(i)]["hash"], st_hash=h, auto=auto,
                            mline_length_mm=float(ml.r[-1] * 1e3), views=[n for n, _ in views],
                            time=time.strftime("%Y-%m-%d %H:%M:%S"))
        S.write_json(p.processed_json, proc)
        print(f"    win{i}: " + ", ".join(f"{n.split()[0][:4]} {a['speed_m_s']:+.1f}" for n, a in auto.items()),
              flush=True)


def load_spacetimes(path):
    """st_win<i>.npz -> dict for :class:`swp.manual.slope_gui.SlopeEditor`."""
    d = np.load(path, allow_pickle=False)
    n = int(d["n_views"])
    views = [dict(name=str(d[f"v{j}_name"]), quantity=str(d[f"v{j}_quantity"]), data=d[f"v{j}_data"],
                  r=d[f"v{j}_r"], t=d[f"v{j}_t"]) for j in range(n)]
    return dict(views=views, st_hash=str(d["st_hash"]), bmode_u8=d["bmode_u8"],
                bmode_extent=list(d["bmode_extent"]), line_x_mm=d["line_x_mm"], line_z_mm=d["line_z_mm"])


# ------------------------------------------------------------------ loop
def work_once(folder, view_filter=True):
    """Do whatever unattended work a folder needs. Returns True if anything was done.
    ``view_filter``: as :func:`store.state` (non-PLAX folders get no detection / processing)."""
    p = S.Paths(folder)
    s = S.state(folder, view_filter)
    if s["stage"] not in ("detecting",) and not s["need_proc"]:
        return False
    with S.folder_lock(p) as got:
        if not got:
            return False
        did = False
        s = S.state(folder, view_filter)
        if s["stage"] == "detecting":
            gh = S.read_json(p.general_json)["hash"]
            if failed(p, "detect", gh):
                return False
            print(f"  detect  {folder}", flush=True)
            t0 = time.perf_counter()
            try:
                ws = detect(folder)
                print(f"    {len(ws)} window(s), {time.perf_counter() - t0:.0f} s", flush=True)
            except Exception as exc:                            # noqa: BLE001
                traceback.print_exc()
                _record_error(p, "detect", gh, exc)
            did = True
            s = S.state(folder, view_filter)
        if s["need_proc"]:
            ev = S.read_json(p.events_json)["events"]
            key = {str(i): ev[str(i)]["hash"] for i in s["need_proc"]}
            todo = [i for i in s["need_proc"] if not failed(p, f"process{i}", key[str(i)])]
            if todo:
                print(f"  process {folder}  windows {todo}", flush=True)
                t0 = time.perf_counter()
                try:
                    process(folder, todo)
                    print(f"    {time.perf_counter() - t0:.0f} s", flush=True)
                except Exception as exc:                        # noqa: BLE001
                    traceback.print_exc()
                    for i in todo:
                        _record_error(p, f"process{i}", key[str(i)], exc)
                did = True
        return did


def run(folders, watch=False, poll_s=20.0, view_filter=True):
    while True:
        did = False
        for f in folders:
            try:
                if work_once(f, view_filter):
                    did = True
                    break              # rescan from the top: earlier folders in the queue first
            except Exception:                                   # noqa: BLE001
                traceback.print_exc()
        if not did:
            if not watch:
                return
            time.sleep(poll_s)
