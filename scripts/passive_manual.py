"""Manual passive-SWE reading of the whole study: M-lines on buffers 1/3/4, hand slopes on five views.

    python scripts/passive_manual.py session --root "Z:/raw_data"          # the interactive work
    python scripts/passive_manual.py status  --root "Z:/raw_data"
    python scripts/passive_manual.py export  --root "Z:/raw_data"          # -> study/logs/passive_manual_slopes.csv
    python scripts/passive_manual.py archive --tag <why>                   # move all outputs aside, start over
    python scripts/passive_manual.py worker  --root "Z:/raw_data" --watch  # only with session --workers 0

Per folder: (1) a general M-line with buffers 1 | 3 | 4 shown at the R-peak -> (2, unattended)
valve-event detection along it -> (3) one M-line per event with the buffers at that event's cardiac
phase -> (4, unattended) five space-times per event -> (5) the hand slope, drawn once and mirrored
on all five. ``session`` starts two background workers for (2) and (4) and stops them on exit.

Stop any time (q or close the window); everything accepted is on disk and the next session resumes.
``b`` goes back one prompt. To redo a finished step later:

    python scripts/passive_manual.py session --folder "<folder>" --redo general
    python scripts/passive_manual.py session --folder "<folder>" --redo event --window 2
    python scripts/passive_manual.py session --folder "<folder>" --redo slope --window 2

Workflow, keys and file layout: docs/passive_manual.md.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from swp.manual import store as S          # noqa: E402


def _folders(a):
    fs = S.find_folders(a.root, a.folder, a.subject)
    if not fs:
        raise SystemExit("no measurement folders found")
    return fs


def cmd_status(a):
    counts, rows = {}, []
    for f in _folders(a):
        s = S.state(f)
        counts[s["stage"]] = counts.get(s["stage"], 0) + 1
        if a.verbose:
            extra = (f"  events to draw {s['need_events']}, processing {s['need_proc']}, slopes to "
                     f"draw {s['need_slopes']}, done {s['slopes_done']}" if s["n_windows"] else "")
            print(f"  {s['stage']:13s} {Path(f).parent.name}/{Path(f).name}{extra}"
                  + (f"  ERROR {s['error']}" if s.get("error") else ""))
        rows.append(s)
    print("  " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    print(f"  windows {sum(r['n_windows'] for r in rows)}, event lines to draw "
          f"{sum(len(r['need_events']) for r in rows)}, being processed "
          f"{sum(len(r['need_proc']) for r in rows)}, slopes to draw "
          f"{sum(len(r['need_slopes']) for r in rows)}, slopes done "
          f"{sum(len(r['slopes_done']) for r in rows)}, failed processing "
          f"{sum(len(r['proc_failed']) for r in rows)}, screened out (no line asked) "
          f"{sum(len(r['screened']) for r in rows)}")


def _spawn_workers(a, n):
    if n <= 0:
        return []
    logdir = _REPO / "study" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    args = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--watch"]
    if a.root:
        args += ["--root", a.root]
    for f in a.folder:
        args += ["--folder", f]
    if a.subject:
        args += ["--subject", a.subject]
    procs = []
    for k in range(n):
        log = open(logdir / f"passive_manual_worker{k}_{time.strftime('%Y%m%d')}.log", "a")
        log.write(f"\n=== worker {k} started {time.ctime()} ===\n")
        log.flush()
        env = dict(os.environ, SWP_MANUAL_WORKER=str(k))
        procs.append((subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, env=env), log))
    print(f"  {n} background worker(s) -> study/logs/passive_manual_worker*_{time.strftime('%Y%m%d')}.log")
    return procs


def cmd_session(a):
    from swp.manual.session import Session
    folders = _folders(a)
    redo = []
    if a.redo:
        for f in folders[: (1 if a.folder else len(folders))]:
            if a.redo == "general":
                redo.append(("general", f, None))
            else:
                if a.window is None:
                    raise SystemExit("--redo event/slope needs --window")
                redo.append((a.redo, f, a.window))
    workers = _spawn_workers(a, a.workers)
    try:
        Session(folders, mode=a.task, retry_skipped=a.retry_skipped, redo=redo,
                include_screened=a.include_screened).run(wait=not a.no_wait)
    finally:
        for p, log in workers:
            p.terminate()
            log.close()


def cmd_worker(a):
    from swp.manual import worker
    folders = _folders(a)
    # a second worker starts from the other end of the queue so the two rarely meet
    if os.environ.get("SWP_MANUAL_WORKER", "0") not in ("0", ""):
        folders = folders[::-1]
    print(f"worker on {len(folders)} folder(s){' (watching)' if a.watch else ''}", flush=True)
    worker.run(folders, watch=a.watch)


def cmd_export(a):
    rows = []
    for f in _folders(a):
        p = S.Paths(f)
        slopes = S.read_json(p.slopes_json) or {}
        s = S.state(f)
        if not slopes and not s["screened"]:
            continue
        win = S.read_json(p.windows_json) or {}
        ecg = win.get("ecg") or {}
        gen = S.read_json(p.general_json) or {}
        wins = win.get("windows") or []
        phases = win.get("window_phases") or [{}] * len(wins)
        picker = (win.get("key", {}).get("detect") or {}).get("picker", "energy")
        for i in s["screened"]:                     # detected, below the screen: no line, no slope
            w, ph = wins[i], phases[i] if i < len(phases) else {}
            rows.append(dict(subject=Path(f).parent.name, folder=Path(f).name, window=i,
                             label=w.get("label"), t_peak_ms=round(w["t_peak"] * 1e3, 1),
                             phase_ms=ph.get("phase_ms"), rr_ms=ph.get("rr_ms"), hr_bpm=ph.get("hr_bpm"),
                             ecg_trustworthy=ecg.get("trustworthy"), ecg_status=ecg.get("status"),
                             skipped=True, screened=True, detector=picker,
                             screen=w.get("screen"), burst=w.get("burst")))
        for k, r in sorted(slopes.items(), key=lambda kv: int(kv[0])):
            i = int(k)
            if i not in s["slopes_done"] and i not in s["slopes_skipped"]:
                continue                     # stale (line redrawn since) - not exported
            sh, dp = r.get("shared") or {}, r.get("disp") or {}
            ph = r.get("phase") or {}
            row = dict(subject=Path(f).parent.name, folder=Path(f).name, window=i,
                       label=r.get("label"), t_peak_ms=round(r["t_peak_ms"], 1),
                       phase_ms=ph.get("phase_ms"), rr_ms=ph.get("rr_ms"), hr_bpm=ph.get("hr_bpm"),
                       ecg_trustworthy=ecg.get("trustworthy"), ecg_status=ecg.get("status"),
                       skipped=bool(r.get("skipped")), confidence=r.get("confidence"),
                       speed_m_s=sh.get("speed_m_s"),
                       speed_displacement_m_s=(dp.get("speed_m_s") if r.get("disp") else sh.get("speed_m_s")),
                       displacement_unlinked=bool(r.get("disp")), anchor_view=sh.get("anchor_view"),
                       mline_length_mm=r.get("mline_length_mm"),
                       line_source_buffer=r.get("line_source_buffer"),
                       line_motion_corrected=r.get("line_motion_corrected"),
                       line_mapping_reliable=r.get("line_mapping_reliable"),
                       general_source_buffer=gen.get("source_buffer"), time=r.get("time"),
                       screened=False, detector=picker,
                       screen=(wins[i] if i < len(wins) else {}).get("screen"),
                       burst=(wins[i] if i < len(wins) else {}).get("burst"))
            for view, au in (r.get("auto") or {}).items():
                row[f"auto {view} [m/s]"] = au.get("speed_m_s")
            rows.append(row)
    out = Path(a.out) if a.out else _REPO / "study" / "logs" / "passive_manual_slopes.csv"
    if not rows:
        print("no slopes yet")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n = sum(1 for r in rows if not r["skipped"])
    print(f"{len(rows)} window(s) ({n} with a slope) -> {out}")


def cmd_archive(a):
    """Move everything the manual study wrote for a folder into archive_<time>_<tag>/ (never deletes).

    Used to restart the reading from scratch, e.g. after buffer 3 was unwrapped (2026-09-25): the
    lines drawn before were drawn on mis-timed buffer-3 frames. Refuses while a worker holds a
    folder; close the session (q) first - its workers stop with it.
    """
    moved, busy = 0, []
    for f in _folders(a):
        p = S.Paths(f)
        if not os.path.isdir(p.dir):
            continue
        if os.path.exists(p.lock):
            busy.append(f)
            continue
        names = [n for n in os.listdir(p.dir) if not n.startswith("archive_")]
        if not names:
            continue
        dest = S.archive(p, names, a.tag)
        moved += 1
        print(f"  {Path(f).parent.name}/{Path(f).name}: {len(names)} item(s) -> {os.path.basename(dest)}")
    print(f"{moved} folder(s) archived" + (f"; {len(busy)} skipped (a worker holds them - close the "
                                           f"session and run again)" if busy else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["session", "worker", "status", "export", "archive"])
    ap.add_argument("--tag", default="restart", help="archive: suffix of the archive folder name")
    ap.add_argument("--root", default=None)
    ap.add_argument("--folder", action="append", default=[])
    ap.add_argument("--subject", default=None)
    ap.add_argument("--task", default="auto", choices=["auto", "lines", "events", "slopes", "lines+events"],
                    help="session: which prompts to serve (auto: slopes, then event lines, then general)")
    ap.add_argument("--workers", type=int, default=2, help="session: background workers to start (0: none)")
    ap.add_argument("--no-wait", action="store_true",
                    help="session: exit instead of waiting when only the worker has work left")
    ap.add_argument("--retry-skipped", action="store_true", help="session: also offer skipped prompts")
    ap.add_argument("--include-screened", action="store_true",
                    help="session: also ask event lines for windows below the detection screen")
    ap.add_argument("--redo", choices=["general", "event", "slope"], default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--watch", action="store_true", help="worker: keep polling for new work")
    ap.add_argument("--out", default=None, help="export: CSV path")
    ap.add_argument("-v", "--verbose", action="store_true", help="status: one line per folder")
    a = ap.parse_args()
    if not a.root and not a.folder:
        a.root = os.environ.get("SWP_RAW_DATA", "Z:/raw_data")
    {"session": cmd_session, "worker": cmd_worker, "status": cmd_status, "export": cmd_export,
     "archive": cmd_archive}[a.command](a)


if __name__ == "__main__":
    main()
