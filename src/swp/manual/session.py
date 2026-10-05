"""The interactive session: serves general lines, event lines and slopes one after another.

Every accept / skip is written to disk the moment it is made, so a session can be stopped at any
point (q, or close the window) and the next one resumes where it left off. ``b`` goes back one
prompt and re-opens it with the stored answer pre-loaded.

Task order (``mode="auto"``): finish folders first - slopes, then event lines, then window reviews,
then general lines, each in queue order (the September folders first). The window review (detector
"valves", since 2026-10-01) shows the whole-recording space-time of the general line with the
automatic MVC / AVC windows, to be moved / added / deleted before the event lines are drawn. The data for the next prompt is loaded in a
background thread while the current one is open.

An MVC event close to the R-peak reuses the general line without a prompt when the buffer-4
anatomy has not moved since the general line's frame (``events.reuse_general`` in
configs/passive_manual.yaml, :func:`reuse_general`; docs/passive_mvc_line_reuse.md).
"""
from __future__ import annotations

import gc
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import frames as F
from . import store as S

KINDS = {"auto": ("slope", "event", "review", "general"), "lines": ("general",), "events": ("event",),
         "slopes": ("slope",), "windows": ("review",), "lines+events": ("event", "review", "general")}
REVIEW_MS = 120.0
WAITING = ("not-ready", "detecting", "processing")


def setup_matplotlib():
    import matplotlib
    matplotlib.use("TkAgg", force=True)
    import matplotlib.pyplot as plt
    # the editors use these keys; matplotlib's defaults would also pan/zoom/close/save
    for k in ("keymap.back", "keymap.forward", "keymap.home", "keymap.quit", "keymap.save",
              "keymap.fullscreen", "keymap.xscale", "keymap.yscale", "keymap.grid",
              "keymap.grid_minor", "keymap.copy"):
        if k in plt.rcParams:
            plt.rcParams[k] = []
    return plt


def _name(f):
    return f"{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}"


# ------------------------------------------------------------------ preloads
def _legacy_general(p):
    import json
    npz = p.legacy_general()
    src = os.path.join(p.output, "mlines", "passive_general_mline.json")
    if not (os.path.exists(npz) and os.path.exists(src)):
        return None
    with open(src) as fh:
        b = int(json.load(fh).get("buffer", 0))
    if b not in F.BUFFERS:
        return None
    return dict(source=b, points_mm=S.load_points(npz) * 1e3, what=f"September line (buffer {b})")


def _legacy_event(p, t_peak):
    st = S.read_json(p.legacy_windows())
    if not st or not st.get("windows"):
        return None
    d = [abs(w["t_peak"] - t_peak) for w in st["windows"]]
    j = int(np.argmin(d))
    info = (st.get("window_mlines") or {}).get(str(j), {})
    if d[j] > 0.04 or info.get("skipped") or not os.path.exists(p.legacy_event(j)):
        return None
    b = int(info.get("buffer", 1))
    if b not in F.BUFFERS:
        return None
    return dict(source=b, points_mm=S.load_points(p.legacy_event(j)) * 1e3,
                what=f"September line of this event (buffer {b})")


def _onto_buffer4(pre, panels):
    """Pre-loads always start on buffer 4, so ENTER saves exactly the line shown there.

    A line from another buffer (the September lines were drawn on buffer 1) is moved onto the
    buffer-4 anatomy by the same registration the review step uses, when that registration is
    trusted; otherwise it keeps its coordinates. Before 2026-09-29 such a line opened on buffer 1
    and ENTER silently saved the motion-corrected copy on buffer 4 (7 lines, 0.1-2.4 mm).
    """
    if pre is None or pre["source"] == 4 or panels.get(4) is None:
        return pre
    src = panels.get(pre["source"])
    what = pre["what"]
    pts = np.asarray(pre["points_mm"], float)
    if src is not None:
        from ._light import transfer
        try:
            d4 = panels[4]
            r = transfer().transfer_line(pts, (src.env, src.x_mm, src.z_mm),
                                         (d4.env, d4.x_mm, d4.z_mm), check=True)
            if r.reliable():
                return dict(source=4, points_mm=np.asarray(r.points, float),
                            what=f"{what}, registered onto buffer 4 (moved {r.shift_mm:.1f} mm)")
            what += f", registration NOT trusted ({r.shift_mm:.1f} mm) - uncorrected coordinates"
        except Exception as exc:                                    # noqa: BLE001
            what += f", registration failed ({exc}) - uncorrected coordinates"
    return dict(source=4, points_mm=pts, what=f"{what} - CHECK its position on buffer 4")


# ------------------------------------------------------------------ reuse of the general line
def reuse_config():
    """``events.reuse_general`` of configs/passive_manual.yaml (read without the zea stack)."""
    import yaml
    with open(S.CONFIG) as fh:
        cfg = (yaml.safe_load(fh).get("events") or {}).get("reuse_general") or {}
    return cfg if cfg.get("enabled") else None


def reuse_general(f, p, w, ph, cfg, t4=None):
    """Whether event window ``w`` can take the general line without a prompt -> (ok, info).

    The window's label must be in ``cfg['labels']`` and its phase <= ``max_phase_ms``; then the
    buffer-4 envelope at the general line's frame is registered onto the one at the event (rigid,
    in a box around the general line, :func:`swp.mline.transfer.transfer_line`). It passes when
    the registration is trusted, the anatomy moved <= ``max_perp_mm`` ACROSS the line (the
    component that takes the line off the septum; motion along the line only slides it along the
    septum) and <= ``max_shift_mm`` in total. Probe or breathing motion between the general
    frame and the event fails it, and the line is drawn by hand."""
    phase = (ph or {}).get("phase_ms")
    info = dict(label=w.get("label"), phase_ms=phase, max_phase_ms=cfg.get("max_phase_ms"),
                max_perp_mm=cfg.get("max_perp_mm"), max_shift_mm=cfg.get("max_shift_mm"))
    if w.get("label") not in cfg.get("labels", ()) or phase is None or phase > cfg["max_phase_ms"]:
        return False, dict(info, why="label / phase")
    gen = S.read_json(p.general_json) or {}
    kg = ((gen.get("frames") or {}).get("4") or {}).get("frame")
    if kg is None:
        return False, dict(info, why="general frame unknown")
    t4 = F.buffer4_times(p.bmode(4)) if t4 is None else t4
    ke = int(np.argmin(np.abs(np.asarray(t4) - w["t_peak"])))
    from ._light import transfer
    g4 = S.load_points(p.general_npz) * 1e3
    a = F.load_panel(f, p.bmode(4), 4, int(kg), "general")
    b = F.load_panel(f, p.bmode(4), 4, ke, "event")
    try:
        r = transfer().transfer_line(g4, (a.env, a.x_mm, a.z_mm), (b.env, b.x_mm, b.z_mm), check=True)
    except Exception as exc:                                        # noqa: BLE001
        return False, dict(info, why=f"registration failed: {exc}")
    u = g4[-1] - g4[0]
    u = u / np.linalg.norm(u)
    perp = float(abs(-u[1] * r.transform.dx + u[0] * r.transform.dz))
    info.update(frame_general=int(kg), frame_event=ke, gap_ms=float(abs(t4[ke] - t4[int(kg)]) * 1e3),
                shift_mm=r.shift_mm, perp_mm=perp, dx=r.transform.dx, dz=r.transform.dz,
                reliable=r.reliable(), agree=r.agree, known_err_mm=r.known_err_mm)
    ok = r.reliable() and perp <= cfg["max_perp_mm"] and r.shift_mm <= cfg["max_shift_mm"]
    why = ("ok" if ok else "registration not trusted" if not r.reliable()
           else "anatomy moved across the line" if perp > cfg["max_perp_mm"] else "anatomy moved")
    return ok, dict(info, why=why)


# ------------------------------------------------------------------ task data (thread-safe: no GUI)
def load_task(task, reuse=None):
    kind, f, i = task
    p = S.Paths(f)
    if kind == "general":
        targets = F.rpeak_targets(f)
        panels = F.load_panels(f, p, targets)
        cur = S.read_json(p.general_json)
        if cur and not cur.get("skipped"):          # redo: the saved buffer-4 line
            pre = dict(source=4, points_mm=np.asarray(cur["points4_mm"]), what="current line (redo)")
        else:
            pre = _onto_buffer4(_legacy_general(p), panels)
        return dict(panels=panels, preload=pre, reference=None)
    if kind == "review":
        return load_review(p)
    if kind == "event":
        win = S.event_windows(p)
        w = win["windows"][i]
        t4 = F.buffer4_times(p.bmode(4))
        g4 = S.load_points(p.general_npz) * 1e3
        ev = (S.read_json(p.events_json) or {})
        cur = ev.get("events", {}).get(str(i)) if ev.get("windows_hash") == win["hash"] else None
        reuse_info = None
        if reuse and cur is None:                   # never for an answered (or skipped) window
            ok, reuse_info = reuse_general(f, p, w, win["phases"][i], reuse, t4)
            if ok:
                what = (f"general line, reused without a prompt ({w.get('label')} R+"
                        f"{reuse_info['phase_ms']:.0f} ms, anatomy moved {reuse_info['perp_mm']:.1f} mm "
                        f"across the line, {reuse_info['shift_mm']:.1f} mm in total)")
                return dict(auto_reuse=reuse_info, points4_mm=g4, preload=dict(source=4, what=what),
                            window=w, phase=win["phases"][i])
        panels = F.load_panels(f, p, F.event_targets(f, w["t_peak"], t4))
        if cur and not cur.get("skipped"):          # redo: the saved buffer-4 line
            pre = dict(source=4, points_mm=np.asarray(cur["points4_mm"]), what="current line (redo)")
        else:
            pre = _onto_buffer4(_legacy_event(p, w["t_peak"]), panels) or dict(
                source=4, points_mm=g4, what="general line (buffer 4, R-peak anatomy)")
        return dict(panels=panels, preload=pre, reference=("general line (buffer 4)", g4),
                    window=w, phase=win["phases"][i], reuse_check=reuse_info)
    if kind == "slope":
        from .worker import load_spacetimes
        win = S.event_windows(p)
        proc = S.read_json(p.processed_json)[str(i)]
        data = load_spacetimes(p.st_npz(i))
        cur = (S.read_json(p.slopes_json) or {}).get(str(i))
        return dict(data=data, window=win["windows"][i], proc=proc, phase=win["phases"][i],
                    preload=cur if cur and not cur.get("skipped") else None)
    raise ValueError(kind)


def _fixed(t0, t1, w_s, rec):
    """A window of length ``w_s`` centred on [t0, t1], shifted inside the recording ``rec``."""
    a = float(np.clip(0.5 * (t0 + t1) - w_s / 2, rec[0], rec[1] - w_s))
    return a, a + w_s


def load_review(p):
    """Data for the window review: the whole-recording space-time, the proposals and the hints.

    Proposals, first available: the current review (redo) -> the ROIs marked by eye with
    scripts/passive_roi.py on the same general line (MVC / AVC, as fixed windows centred on them)
    -> the automatic windows above the screen. The screened automatic windows are hints."""
    win = S.read_json(p.windows_json)
    z = np.load(p.general_st, allow_pickle=False)
    rr = float(z["rr_s"])
    data = dict(v=z["v"], t_s=z["t"], r_m=z["r"], r_peaks_s=z["r_peaks_s"], rr_s=rr if np.isfinite(rr) else None)
    rec = (float(z["t"][0]), float(z["t"][-1]))
    w_s = REVIEW_MS * 1e-3
    r_mid = float(z["r"][-1]) * 1e3 / 2
    gen = S.read_json(p.general_json) or {}
    cur = S.read_json(p.review_json)
    rois = S.read_json(p.rois_json)
    if cur and cur.get("windows_hash") == win["hash"] and cur.get("status") in ("done", "none"):
        pro, what = cur["windows"], "current review (redo)"
    elif rois and rois.get("general_hash") == gen.get("hash") and rois.get("status") in ("done", "none"):
        pro = [dict(zip(("t0", "t1"), _fixed(q["t0"], q["t1"], w_s, rec)), label=q["label"])
               for q in rois["rois"] if q["label"] in ("MVC", "AVC")]
        what = "your earlier ROIs (passive_roi.py), as fixed windows"
    else:
        pro = [w for w in win["windows"] if not w.get("screened")]
        what = "automatic windows"
    preload = dict(rois=[dict(t0=w["t0"], t1=w["t1"], label=w["label"], r_mm=r_mid) for w in pro],
                   clim_pct=(cur or {}).get("clim_pct"))
    hints = [dict(t0=w["t0"], t1=w["t1"], text=f"{w['label']} {w['screen']:.2f}")
             for w in win["windows"] if w.get("screened")]
    return dict(data=data, preload=preload, preload_what=what, hints=hints, windows_hash=win["hash"],
                proposals=[{k: w[k] for k in ("t0", "t1", "label")} for w in pro])


def review_windows(records, data):
    """Editor records -> event windows. t_peak = the short-time energy peak inside the window: the
    event time the buffer-1 / 3 frames are matched to."""
    from ..passive_valves import short_energy
    t = np.asarray(data["t_s"], float)
    es = short_energy(data["v"], t)
    out = []
    for q in records:
        m = (t >= q["t0"]) & (t <= q["t1"])
        t_peak = float(t[m][int(np.argmax(es[m]))]) if m.any() else q["t_mid"]
        out.append(dict(t_peak=t_peak, t0=q["t0"], t1=q["t1"], label=q["label"], expect=q.get("expected")))
    return out


def review_hash(windows):
    """Hash of the reviewed windows (start, end, peak, label): a moved window makes its event line stale."""
    import hashlib
    import json
    key = [(round(w["t0"], 5), round(w["t1"], 5), round(w["t_peak"], 5), w["label"]) for w in windows]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ session
class Session:
    def __init__(self, folders, mode="auto", retry_skipped=False, redo=(), include_screened=False,
                 reuse=True, view_filter=True):
        """``redo``: tasks ``(kind, folder, window)`` to re-open first, with their answers pre-loaded.
        ``include_screened``: also ask event lines for windows below the detection screen.
        ``reuse``: reuse the general line for MVC events near the R-peak (``events.reuse_general``);
        False always prompts.
        ``view_filter``: leave out folders the manual view review labels other than PLAX / Unclear
        (:data:`store.PASSIVE_VIEWS`)."""
        self.reuse = reuse_config() if reuse else None
        self.view_filter = view_filter
        self.queue = list(redo)
        self.folders = list(folders)
        self.kinds = KINDS[mode]
        self.retry_skipped = retry_skipped
        self.include_screened = include_screened
        self.states = {}
        self.history = []
        self.forced = None
        self.pool = ThreadPoolExecutor(1)
        self.prefetched = {}
        self.n_done = 0
        print(f"reading the state of {len(self.folders)} folder(s) ...", flush=True)
        t0 = time.perf_counter()
        self.refresh()
        print(f"  {time.perf_counter() - t0:.0f} s\n  " + self.summary(), flush=True)

    # ------------------------------------------------ bookkeeping
    def summary(self):
        c = {}
        for s in self.states.values():
            c[s["stage"]] = c.get(s["stage"], 0) + 1
        ev = sum(len(s["need_events"]) for s in self.states.values())
        sl = sum(len(s["need_slopes"]) for s in self.states.values())
        done = sum(len(s["slopes_done"]) for s in self.states.values())
        return (", ".join(f"{k} {v}" for k, v in sorted(c.items()))
                + f"  |  event lines to draw {ev}, slopes to draw {sl}, slopes done {done}")

    def refresh(self, folders=None):
        for f in (folders if folders is not None else self.folders):
            self.states[f] = S.state(f, self.view_filter)

    def refresh_waiting(self):
        self.refresh([f for f, s in self.states.items() if s["stage"] in WAITING or s["need_proc"]])

    def _tasks(self, f, kind):
        s = self.states[f]
        r = self.retry_skipped
        if kind == "general":
            return [("general", f, None)] if (s["stage"] == "need-general"
                                              or (r and s["stage"] == "skipped")) else []
        if kind == "review":
            return [("review", f, None)] if (s["stage"] == "need-review"
                                             or (r and s.get("review_skipped"))) else []
        if kind == "event":
            return [("event", f, i) for i in s["need_events"] + (s["lines_skipped"] if r else [])
                    + (s.get("screened", []) if self.include_screened else [])]
        return [("slope", f, i) for i in s["need_slopes"] + (s["slopes_skipped"] if r else [])]

    def next_task(self, exclude=()):
        for kind in self.kinds:
            for f in self.folders:
                for t in self._tasks(f, kind):
                    if t not in exclude:
                        return t
        return None

    def pending_worker(self):
        return [f for f, s in self.states.items() if s["stage"] in ("detecting", "processing")
                or s["need_proc"]]

    def _get(self, task):
        fut = self.prefetched.pop(task, None)
        if fut is not None:
            try:
                return fut.result()
            except Exception as exc:                                  # noqa: BLE001
                print(f"  (prefetch failed: {exc}; loading again)")
        return load_task(task, self.reuse)

    def _prefetch(self, exclude):
        nxt = self.next_task(exclude=exclude)
        if nxt is not None and nxt not in self.prefetched:
            self.prefetched = {nxt: self.pool.submit(load_task, nxt, self.reuse)}

    # ------------------------------------------------ main loop
    def run(self, wait=True):
        plt = setup_matplotlib()
        # A closed editor leaves Tk objects in reference cycles; an automatic collection in the
        # prefetch thread would finalise them there -> "main thread is not in main loop" (Tcl may
        # then hang or abort). Collect only here, on the Tk thread, after every prompt.
        gc.disable()
        try:
            self._loop(plt, wait)
        finally:
            self.pool.shutdown(wait=False, cancel_futures=True)
            plt.close("all")
            gc.collect()
            gc.enable()

    def _loop(self, plt, wait):
        last_refresh = time.time()
        while True:
            task = self.forced or (self.queue.pop(0) if self.queue else self.next_task())
            self.forced = None
            if task is None:
                self.refresh_waiting()
                last_refresh = time.time()
                task = self.next_task()
            if task is None:
                pend = self.pending_worker()
                if pend and wait:
                    print(f"\r  waiting for the worker ({len(pend)} folder(s)) ...", end="", flush=True)
                    time.sleep(15)
                    continue
                print("\nnothing left to do." if not pend else
                      f"\n{len(pend)} folder(s) still with the worker; run the session again later.")
                break
            try:
                data = self._get(task)
            except Exception as exc:                                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"  cannot load {task[0]} {_name(task[1])} (window {task[2]}): {exc} - skipped "
                      f"for this session")
                self.states[task[1]] = dict(self.states[task[1]], stage="load-failed",
                                            need_events=[], need_slopes=[])
                continue
            self._prefetch(exclude={task})
            action = self.prompt(task, data)
            gc.collect()
            if action == "quit":
                print("stopped - everything accepted so far is saved; run again to continue.")
                break
            if action == "back":
                if self.history:
                    self.forced = self.history.pop()
                    print(f"  back to {self.forced[0]} {_name(self.forced[1])}"
                          f"{'' if self.forced[2] is None else f' window {self.forced[2]}'}")
                continue
            self.history.append(task)
            self.n_done += 1
            self.refresh([task[1]])
            if time.time() - last_refresh > 120:
                self.refresh_waiting()
                last_refresh = time.time()

    def _title(self, task, what):
        k = self.folders.index(task[1]) + 1
        return (f"[{k}/{len(self.folders)}]  {_name(task[1])}   -   {what}\n"
                f"session: {self.n_done} done  |  {self.summary()}")

    # ------------------------------------------------ prompts
    def prompt(self, task, data):
        kind, f, i = task
        p = S.Paths(f)
        if kind == "slope":
            return self._prompt_slope(p, i, data)
        if kind == "review":
            return self._prompt_review(p, data)
        if data.get("auto_reuse"):
            pts = np.asarray(data["points4_mm"]).tolist()
            info = data["auto_reuse"]
            res = dict(action="accept", source_buffer=4, points_src_mm=pts, points4_mm=pts,
                       motion_correction=False, mapping=None, mapping_other={}, nudge_mm=[0.0, 0.0],
                       frames={"4": dict(frame=info["frame_event"], phase_ms=info["phase_ms"])},
                       auto_reuse=info)
            self._save_event(p, i, res, data)
            return "accept"
        from .line_gui import LineEditor

        def reload_panel(b, frame, n_avg):
            ph = F.frame_phase(f, b, frame)
            note = f"frame stepped: R+{ph:.0f} ms" if np.isfinite(ph) else "frame stepped"
            return F.load_panel(f, p.bmode(b), b, frame, note, ph, n_avg)

        if kind == "general":
            what = "GENERAL M-line, all buffers at the R-peak (used to detect the valve events)"
            hint = S.view_hint(f)
            if hint:
                what += f"  |  view: {hint}"
        else:
            w, ph = data["window"], data.get("phase") or {}
            phs = "" if ph.get("phase_ms") is None else f" (R+{ph['phase_ms']:.0f} ms)"
            what = (f"EVENT {i + 1}: {w.get('label') or '?'} at {w['t_peak'] * 1e3:.0f} ms{phs}"
                    " - buffers at the event's cardiac phase")
        ed = LineEditor(data["panels"], self._title(task, what), reload_panel,
                        preload=data.get("preload"), reference=data.get("reference"),
                        exclude=kind == "general")
        snap = os.path.join(p.dir, "general.png" if kind == "general" else f"event{i}.png")
        os.makedirs(p.dir, exist_ok=True)
        res = ed.run(snapshot=snap)
        if res["action"] in ("accept", "skip", "exclude"):
            (self._save_general if kind == "general" else self._save_event)(p, i, res, data)
        return res["action"]

    def _save_general(self, p, _i, res, data):
        cur = S.read_json(p.general_json)
        if res["action"] == "accept":
            pts = np.asarray(res["points4_mm"]) * 1e-3
            h = S.points_hash(pts)
            if cur is not None and cur.get("hash") != h:
                dest = S.archive(p, S.downstream_files(p), "general_redrawn")
                if dest:
                    print(f"  earlier windows / lines / slopes archived -> {os.path.basename(dest)}")
            S.save_line(p.general_npz, pts)
            rec = dict(hash=h, sync="rpeak", preload=(data.get("preload") or {}).get("what"),
                       time=time.strftime("%Y-%m-%d %H:%M:%S"),
                       **{k: v for k, v in res.items() if k != "action"})
            S.write_json(p.general_json, rec)
            S.append_log(p, dict(task="general", action="accept", hash=h, source=res["source_buffer"]))
            mi = res.get("mapping") or {}
            print(f"  general line: buffer {res['source_buffer']}"
                  + (f", moved {mi.get('shift_mm', 0):.1f} mm to buffer 4" if res["motion_correction"] else ""))
        else:
            excl = res["action"] == "exclude"
            if cur is not None:
                dest = S.archive(p, S.downstream_files(p), "general_excluded" if excl else "general_skipped")
                if dest:
                    print(f"  earlier windows / lines / slopes archived -> {os.path.basename(dest)}")
            rec = dict(skipped=True, hash=None, time=time.strftime("%Y-%m-%d %H:%M:%S"))
            if excl:
                rec.update(excluded="not PLAX", view_hint=S.view_hint(p.folder))
            S.write_json(p.general_json, rec)
            S.append_log(p, dict(task="general", action=res["action"]))
            print("  measurement excluded (not PLAX)" if excl else "  folder skipped (no usable septum)")

    def _prompt_review(self, p, data):
        from .roi_gui import RoiEditor
        d = data["data"]
        hr = f"HR {60 / d['rr_s']:.0f} bpm" if d["rr_s"] else "NO VALID ECG (no automatic windows)"
        what = (f"EVENT WINDOWS ({REVIEW_MS:.0f} ms) on the general line, {hr} - proposed: "
                f"{data['preload_what']}; move / add / delete, then ENTER")
        task = ("review", p.folder, None)
        ed = RoiEditor(d, self._title(task, what), preload=data["preload"], fixed_ms=REVIEW_MS,
                       hints=data["hints"], roi_name="event windows")
        res = ed.run(snapshot=p.review_json.replace(".json", ".png"))
        if res["action"] in ("accept", "none", "skip"):
            from .worker import window_phases
            windows = review_windows(res.get("rois", []), d)
            status = {"accept": "done", "none": "none", "skip": "skipped"}[res["action"]]
            S.write_json(p.review_json, dict(
                windows_hash=data["windows_hash"], hash=review_hash(windows), status=status,
                windows=windows, phases=window_phases(windows, d["r_peaks_s"], d["rr_s"]),
                window_ms=REVIEW_MS, proposed_from=data["preload_what"], proposals=data["proposals"],
                clim_pct=res.get("clim_pct"), time=time.strftime("%Y-%m-%d %H:%M:%S")))
            S.append_log(p, dict(task="review", action=res["action"], n=len(windows)))
            print("  windows: " + (", ".join(f"{w['label']} {w['t0'] * 1e3:.0f}-{w['t1'] * 1e3:.0f}"
                                             for w in windows) or status))
            return "accept" if res["action"] == "none" else res["action"]
        return res["action"]

    def _save_event(self, p, i, res, data):
        win = S.event_windows(p)
        ev = S.read_json(p.events_json) or {}
        if ev.get("windows_hash") != win["hash"]:
            if ev:
                S.archive(p, ["events.json"] + [n for n in os.listdir(p.dir) if n.startswith("event")],
                          "stale_events")
            ev = dict(windows_hash=win["hash"], events={})
        w = data["window"]
        base = dict(window=i, t_peak_ms=w["t_peak"] * 1e3, label=w.get("label"),
                    time=time.strftime("%Y-%m-%d %H:%M:%S"))
        if res["action"] == "accept":
            pts = np.asarray(res["points4_mm"]) * 1e-3
            S.save_line(p.event_npz(i), pts)
            rec = dict(base, hash=S.points_hash(pts), preload=(data.get("preload") or {}).get("what"),
                       **{k: v for k, v in res.items() if k != "action"})
            if data.get("reuse_check") is not None:      # prompted although reuse was checked: why
                rec["reuse_check"] = data["reuse_check"]
            if res.get("auto_reuse"):
                print(f"  event {i}: {rec['preload']}")
            else:
                print(f"  event {i}: buffer {res['source_buffer']}"
                      + (f", moved {(res.get('mapping') or {}).get('shift_mm', 0):.1f} mm"
                         if res["motion_correction"] else ""))
        else:
            rec = dict(base, skipped=True)
            print(f"  event {i}: skipped")
        ev["events"][str(i)] = rec
        S.write_json(p.events_json, ev)
        S.append_log(p, dict(task="event", window=i, action=res["action"], hash=rec.get("hash")))

    def _prompt_slope(self, p, i, data):
        from .slope_gui import SlopeEditor
        w, ph, proc = data["window"], data.get("phase") or {}, data["proc"]
        auto = proc.get("auto", {}).get("velocity gauss", {}).get("speed_m_s", 3.0)
        init = auto if (np.isfinite(auto) and 1.05 < abs(auto) < 19.5) else 3.0
        phs = f" R+{ph['phase_ms']:.0f} ms" if ph.get("phase_ms") is not None else ""
        data["data"]["bmode_title"] = (f"buffer 4 at the event (box around the line), "
                                       f"M-line {proc.get('mline_length_mm', 0):.0f} mm")
        what = f"SLOPE, event {i + 1}: {w.get('label') or '?'} at {w['t_peak'] * 1e3:.0f} ms{phs}"
        task = ("slope", p.folder, i)
        ed = SlopeEditor(data["data"], self._title(task, what), preload=data.get("preload"), init_speed=init)
        res = ed.run(snapshot=os.path.join(p.dir, f"slope{i}.png"))
        if res["action"] in ("accept", "skip"):
            slopes = S.read_json(p.slopes_json) or {}
            ev = (S.read_json(p.events_json) or {}).get("events", {}).get(str(i), {})
            rec = dict(window=i, st_hash=data["data"]["st_hash"], t_peak_ms=w["t_peak"] * 1e3,
                       label=w.get("label"), phase=ph, auto=proc.get("auto"),
                       mline_length_mm=proc.get("mline_length_mm"),
                       line_source_buffer=ev.get("source_buffer"),
                       line_motion_corrected=ev.get("motion_correction"),
                       line_mapping_reliable=(ev.get("mapping") or {}).get("reliable"),
                       line_reused_general=bool(ev.get("auto_reuse")),
                       time=time.strftime("%Y-%m-%d %H:%M:%S"))
            if res["action"] == "accept":
                rec.update({k: v for k, v in res.items() if k != "action"})
                sh, dp = res["shared"], res["disp"]
                msg = (f"confidence {res['confidence']} ({res['confidence_meaning']})"
                       + (f", {sh['speed_m_s']:+.2f} m/s" if sh and sh["speed_m_s"] is not None else "")
                       + (f", displacement {dp['speed_m_s']:+.2f} m/s" if dp and dp["speed_m_s"] is not None else ""))
            else:
                rec.update(skipped=True)
                msg = "skipped"
            slopes[str(i)] = rec
            S.write_json(p.slopes_json, slopes)
            S.append_log(p, dict(task="slope", window=i, action=res["action"],
                                 confidence=res.get("confidence")))
            print(f"  slope {i}: {msg}")
        return res["action"]
